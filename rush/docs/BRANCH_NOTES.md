# Rush-version contribution

This branch adds the previously implemented M0/M1/M2 research project under
`rush/`. The existing `src/factcheck/` algorithms, root dependency lock and M0
configuration are retained. M1 and M2 are executable here; the existing M1
planning document still describes future integration with the main workflow.

## Protocol and ownership

| Component | Existing M0 | Rush M0/M1/M2 |
|---|---|---|
| Backbone | `vinai/phobert-base-v2` | `vinai/phobert-base` |
| Sentence boundaries | `conservative_vi_v1` | VnCoreNLP |
| Sentence retrieval | BM25S Lucene, word/number tokens | BM25Okapi, segmented tokens |
| Classifier | Hugging Face sequence classification head | Dropout + linear head |
| Optimizer/schedule | AdamW; linear warmup/decay | Adafactor; 10% warmup then constant LR |
| Effective default batch | 16 claims | 8 claims |
| Execution | Float32 | Mixed precision, activation checkpointing |
| Checkpointing | Best model/tokenizer | Model, optimizer, scaler, RNG; epoch resume |

Both M0 implementations aggregate per-sentence probabilities uniformly and apply
verification loss once per claim. Both preserve short-article sentence counts,
256/96 token limits and sentence-order ranking ties. Different preprocessing,
backbone and optimizer settings mean their scores are not interchangeable.

M1 freezes its separately trained reranker and uses softmax weights over its
selected top five sentences. M2 jointly trains both encoders with verification
loss plus eligible balanced evidence BCE. Its verification gradient reaches
selected reranker scores; top-k indices remain discrete. M1 and M2 branch from
the same best initialization checkpoint in Rush. The main repository's plan
specifies the end-of-epoch-three snapshot, which is a protocol detail to reconcile
before future integration.

## Experiment status at import

The running laptop experiment uses 900 randomly sampled training claims (300 per
label), seed 42, fixed learning rate 2e-5 and lambda 0.5. Selection does not filter
on gold alignment; 331 sampled claims qualify for evidence supervision. It keeps
three reranker epochs, three verifier initialization epochs, two continuation
epochs for M1/M2, and five M0 epochs. All 723 dev and 1,447 test claims are used.

This is a reduced-data single-seed pilot. It is not the full 5,062-claim,
three-seed proposal experiment, and its hyperparameters were not tuned. The
original smaller smoke results in `VALIDATION.md` are pipeline checks.

The laptop run continues from the parent workspace. Importing this source into
the branch does not start another GPU job or move the active checkpoints. Running
`python -m vifactcheck pilot` in this subproject starts/resumes a separate run in
its own `runs/` directory. Prepare its assets first as described in the README.

## Reproduction and generated files

- Create `rush/.venv` with `rush/scripts/setup.ps1`; the main repository uses its
  own root environment and `uv.lock`.
- Run preparation, training, tuning, evaluation, prediction and pilot commands
  from `rush/`; its README provides the complete commands.
- Python source is copied without changes from the active experiment.
  `PROVENANCE.json` records all 13 source hashes, pinned revisions, the proposal
  checksum and exact pilot training IDs. The scoped `.gitattributes` preserves
  those Python source bytes across Git checkouts.
- `rush/.gitignore` excludes its environment, data, cache, generated runs and
  checkpoints. The original proposal, code, tests, example, setup and guides are
  versioned; downloaded assets and model weights are not.
- The pilot automatically creates `REPORT.md`, `results.csv`, `comparison.json`,
  per-claim predictions and paired article-cluster bootstrap intervals. These
  intervals are conditional on its one training seed.

Before merging this work into the main M0/M1 architecture, agree on a shared
backbone, sentence/BM25 pipeline, classifier, optimizer, effective batch and fork
checkpoint rule, then rerun the controlled comparison under that protocol.
