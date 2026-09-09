"""SBERT Retrieval module for ViFactCheck.

Ports the exact SBERT implementation from ViFactCheck/src/retrieval/SBERT.ipynb.

Model used (from original repo, Cell 5):
    sentence-transformers/paraphrase-xlm-r-multilingual-v1

Key logic (from original repo, Cell 6):
    - mean_pooling on token embeddings
    - CosineSimilarity between claim and each sentence embedding
    - MinMaxScaler to scale similarities to [0, 1]
    - Return top-5 sentences by scaled similarity
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from transformers import AutoModel, AutoTokenizer

# Model used in the original paper (Cell 5 of SBERT.ipynb)
DEFAULT_MODEL = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"


def mean_pooling(model_output, attention_mask) -> torch.Tensor:
    """Mean pooling on token embeddings (ported from original Cell 6)."""
    token_embeddings = model_output[0]  # [B, seq_len, hidden]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class SBERTRetriever:
    """SBERT-based evidence retriever ported from original SBERT.ipynb.

    Encapsulates the tokenizer and model so they are loaded once,
    not on every function call (unlike the original notebook).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print(f"[SBERT] Loading model '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("[SBERT] Model loaded.")

    def similarities(self, context: list[str], claim: str, top_n: int = 5):
        """Retrieve top-N sentences from context by cosine similarity to claim.

        Ported directly from original Cell 6 of SBERT.ipynb.
        """
        sentences = [claim] + context

        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform mean pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

        sims: list[tuple[str, float]] = []
        claim_embedding = sentence_embeddings[0].unsqueeze(0)
        cosine = nn.CosineSimilarity(dim=1, eps=1e-6)

        for i in range(1, len(sentence_embeddings)):
            evidence_embedding = sentence_embeddings[i].unsqueeze(0)
            similarity = cosine(
                claim_embedding.to(self.device),
                evidence_embedding.to(self.device),
            ).item()
            sims.append((sentences[i], similarity))

        # MinMaxScaler on similarity values (from original Cell 6)
        simi_values = [s[1] for s in sims]
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(
            np.array(simi_values).reshape(-1, 1)
        ).flatten()
        sims = [(sentences[i + 1], float(scaled_values[i])) for i in range(len(scaled_values))]

        sims.sort(key=lambda x: x[1], reverse=True)
        top_docs = [item[0] for item in sims[:top_n]]
        top_scores = [item[1] for item in sims[:top_n]]
        return top_docs, top_scores
