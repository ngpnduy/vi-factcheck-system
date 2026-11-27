# Slide-update validation - 29 October 2026

The update adds A1, A2, optional candidate-list attention, checkpoint-parent reuse,
and the five-configuration, three-seed reporting path. It does not provide new
ablation benchmark scores or launch the full study.

The 52-test suite checks the original behavior and the following additions:

- A1 selects the same sentences as M1 and uniformly averages their label probabilities.
- A2's verification gradient is disconnected while its evidence gradient is nonzero;
  NEI supplies no evidence loss and no A2 reranker update.
- Candidate attention propagates information between sentences, handles a single
  candidate, and is equivariant to reordering.
- Both architectures train all five modes with common verifier initialization;
  M1/M2/A2 share warmup, and M0/A1 receive equal verifier budgets.
- Ablation checkpoints reload/resume; external initialization avoids repeating
  reranker/warmup training and preserves source checkpoint bytes.
- Wrong architecture, data, source phase, or configuration is rejected on reuse.
- The CLI routes all five modes and the fifteen-model full protocol. Reports
  require complete paired predictions and produce the four slide contrasts.

Real PhoBERT-base validation used the laptop's 4-GiB RTX 3050 and an eligible
long training claim, with twenty BM25 candidates, five selected sentences, and
the full 256/96 token limits. Each case performed one diagnostic optimizer update,
then predicted a dev claim without gold label/alignment fields. Both encoders
were trainable in M2/A2; A1's complete reranker remained frozen.

| Architecture | Mode | Peak allocated MiB | Peak reserved MiB | Verification-to-reranker gradient norm | Evidence gradient norm |
|---|---|---:|---:|---:|---:|
| Pairwise | A1 | 1980.20 | 2302 | Frozen | Not applied |
| Pairwise | A2 | 2494.87 | 2912 | 0 | 2.012833 |
| List-aware | A1 | 1998.23 | 2320 | Frozen | Not applied |
| List-aware | A2 | 2531.81 | 2954 | 0 | 2.968520 |
| List-aware | M2 | 2531.81 | 2954 | 0.396731 | Included in joint update |

For A2/M2, first and last attention-query matrices changed in both encoders.
For A1, the verifier matrices changed and reranker matrices stayed identical.
List-attention parameters changed in M2/A2 and stayed identical in A1.
The original seed-42 M1 checkpoint reloaded with identical candidate ranking and
zero maximum probability difference from its saved dev prediction.

These memory measurements cover one diagnostic claim per case, not every claim
or the complete accumulated training workload. The diagnostic uses AMP initial
loss scale 128 for a single update; normal training retains dynamic GradScaler
defaults and eight-claim accumulation. PyTorch emitted its existing warning that
memory-efficient attention backward may be nondeterministic. Seed control does
not guarantee bit-for-bit agreement across repeated GPU training runs.

Reproduce using `scripts/validate_slides.py`; see the
[implementation guide](docs/SLIDES_0410_IMPLEMENTATION.md) for commands and scope.
Machine-readable evidence is in `docs/validation/slides-20261029.json`.
Full ablation/list-aware training, final scores, and timing estimates remain to
be measured in separate experiments.
