from contextlib import nullcontext
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from . import LABELS
from .config import ALL_MODELS


class PairTokenizer:
    def __init__(self, path, max_length=256, claim_max_tokens=96):
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, local_files_only=True)
        self.max_length, self.claim_max_tokens = max_length, claim_max_tokens

    @lru_cache(maxsize=4096)
    def encode(self, claim, sentence):
        claim_ids = self.tokenizer.encode(claim, add_special_tokens=False)
        sentence_ids = self.tokenizer.encode(sentence, add_special_tokens=False)
        special = self.tokenizer.num_special_tokens_to_add(pair=True)
        kept_claim = claim_ids[:self.claim_max_tokens]
        kept_sentence = sentence_ids[:self.max_length - special - len(kept_claim)]
        ids = self.tokenizer.build_inputs_with_special_tokens(kept_claim, kept_sentence)
        return ids, {"claim_truncated": len(claim_ids) > len(kept_claim), "sentence_truncated": len(sentence_ids) > len(kept_sentence), "claim_original_tokens": len(claim_ids), "sentence_original_tokens": len(sentence_ids), "pair_tokens": len(ids)}

    def inputs(self, claim, sentence, device):
        ids, metadata = self.encode(claim, sentence)
        return {"input_ids": torch.tensor([ids], dtype=torch.long, device=device), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device)}, metadata


class PairEncoder(nn.Module):
    def __init__(self, model_path, classes, checkpointing=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True, add_pooling_layer=False)
        self.encoder.config.use_cache = False
        if checkpointing:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.head = nn.Linear(self.encoder.config.hidden_size, classes)

    def encode(self, inputs):
        hidden = self.encoder(**inputs).last_hidden_state[:, 0]
        return self.dropout(hidden).squeeze(0)

    def forward(self, inputs):
        return self.head(self.encode(inputs).unsqueeze(0)).float().squeeze(0)


class CandidateAttention(nn.Module):
    """Contextualize claim-conditioned candidate vectors within one article.

    No positional embeddings: reordering the candidate list reorders its scores
    equivalently. Original sentence IDs still resolve final ranking ties.
    """
    def __init__(self, hidden_size, config):
        super().__init__()
        if hidden_size % config.listwise_heads:
            raise ValueError("Reranker hidden size must be divisible by listwise_heads")
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=config.listwise_heads,
                                       dim_feedforward=config.listwise_feedforward,
                                       dropout=config.listwise_dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(config.listwise_layers)
        ])

    def forward(self, candidates):
        if candidates.ndim != 2 or candidates.shape[0] == 0:
            raise ValueError("List-aware attention requires nonempty candidate vectors")
        hidden = candidates.unsqueeze(0)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden.squeeze(0)


def stable_topk(scores, sentence_ids, k):
    # Sort a detached copy to enforce deterministic sentence-index tie breaking;
    # gather from the original tensor to retain gradients through selected scores.
    positions = sorted(range(len(sentence_ids)), key=lambda i: (-float(scores[i].detach()), sentence_ids[i]))
    return positions[:k], positions


def aggregate(logits, scores=None):
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("At least one sentence prediction is required")
    if scores is None:
        log_weights = torch.full((logits.shape[0],), -torch.log(torch.tensor(float(logits.shape[0]))).item(), device=logits.device)
    else:
        log_weights = F.log_softmax(scores.float(), dim=0)
    log_probabilities = torch.logsumexp(log_weights[:, None] + F.log_softmax(logits.float(), dim=-1), dim=0)
    return log_probabilities, log_weights.exp()


def evidence_loss(scores, row, candidate_ids):
    if row["label"] == 2 or row["alignment"]["status"] != "valid":
        return None
    gold = set(row["alignment"]["gold_sentence_ids"])
    targets = torch.tensor([float(i in gold) for i in candidate_ids], device=scores.device)
    positives, negatives = targets.bool(), ~targets.bool()
    if not positives.any() or not negatives.any():
        return None
    losses = F.binary_cross_entropy_with_logits(scores.float(), targets, reduction="none")
    return (losses[positives].mean() + losses[negatives].mean()) * 0.5


class VerificationSystem(nn.Module):
    def __init__(self, config, mode, tokenizer=None):
        super().__init__()
        if mode not in ALL_MODELS:
            raise ValueError(f"mode must be one of {ALL_MODELS}")
        self.config, self.mode = config, mode
        self.tokenizer = tokenizer or PairTokenizer(config.model_path, config.max_length, config.claim_max_tokens)
        # Verifier is always constructed first for shared seed initialization.
        self.verifier = PairEncoder(config.model_path, 3, config.gradient_checkpointing)
        self.reranker = PairEncoder(config.model_path, 1, config.gradient_checkpointing) if mode != "M0" else None
        if self.reranker is not None and config.reranker_architecture == "listwise":
            self.reranker.contextualizer = CandidateAttention(self.reranker.head.in_features, config)
        self.to(config.device)

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.float16) if self.config.device == "cuda" and self.config.mixed_precision else nullcontext()

    def configure_training(self, stage):
        self.train()
        self.verifier.requires_grad_(stage != "reranker")
        if stage == "reranker":
            self.verifier.eval()
        if self.reranker is not None:
            train_ranker = stage == "reranker" or (stage == "continuation" and self.mode in ("M2", "A2"))
            self.reranker.requires_grad_(train_ranker)
            if not train_ranker:
                self.reranker.eval()

    def rank_scores(self, row, gradients):
        candidates = row["candidate_ids"][:self.config.candidate_count]
        scores, features = [], []
        with nullcontext() if gradients else torch.no_grad():
            for i in candidates:
                inputs, _ = self.tokenizer.inputs(row["claim_segmented"], row["sentences"][i]["segmented"], self.config.device)
                with self.autocast():
                    if self.config.reranker_architecture == "listwise":
                        features.append(self.reranker.encode(inputs))
                    else:
                        scores.append(self.reranker(inputs).squeeze(-1))
            if features:
                with self.autocast():
                    contextualized = self.reranker.contextualizer(torch.stack(features))
                    return candidates, self.reranker.head(contextualized).float().squeeze(-1)
        return candidates, torch.stack(scores)

    def forward(self, row, training_stage=None):
        rank_loss = None
        if self.mode == "M0":
            ranking = row["candidate_ids"][:self.config.candidate_count]
            selected, selected_scores = ranking[:self.config.evidence_count], None
        else:
            rank_gradients = training_stage == "continuation" and self.mode in ("M2", "A2")
            candidates, scores = self.rank_scores(row, rank_gradients)
            positions, ordered = stable_topk(scores, candidates, self.config.evidence_count)
            ranking = [candidates[p] for p in ordered]
            selected, selected_scores = [candidates[p] for p in positions], scores[positions]
            if rank_gradients:
                rank_loss = evidence_loss(scores, row, candidates)
            if self.mode == "A1":
                selected_scores = None
            elif self.mode == "A2":
                # Keep evidence BCE connected to all candidate scores; detach
                # only their contribution to the claim-verification weights.
                selected_scores = selected_scores.detach()
        logits, truncation = [], []
        for i in selected:
            inputs, metadata = self.tokenizer.inputs(row["claim_segmented"], row["sentences"][i]["segmented"], self.config.device)
            with self.autocast():
                logits.append(self.verifier(inputs))
            truncation.append(metadata)
        log_probs, weights = aggregate(torch.stack(logits), selected_scores)
        return {"log_probabilities": log_probs, "weights": weights, "ranking": ranking, "selected": selected, "evidence_loss": rank_loss, "truncation": truncation}

    def prediction(self, row):
        self.eval()
        with torch.no_grad():
            output = self(row)
        probabilities = output["log_probabilities"].exp().cpu().tolist()
        label = max(range(3), key=lambda i: probabilities[i])
        return {"id": row["id"], "configuration": self.mode, "reranker_architecture": self.config.reranker_architecture if self.reranker is not None else None, "label": LABELS[label], "label_id": label, "probabilities": dict(zip(LABELS, probabilities)), "ranking": output["ranking"], "selected_sentences": [{**row["sentences"][i], "weight": float(weight)} for i, weight in zip(output["selected"], output["weights"].detach().cpu())], "truncation": output["truncation"]}
