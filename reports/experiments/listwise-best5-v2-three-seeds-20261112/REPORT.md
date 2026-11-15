# Full M0/M1/A1/M2/A2 experiment

## Protocol

All 5,062 official training claims were used, with all 723 validation and 1,447 test claims. Final models use seeds 13, 42 and 2026.

Seed-42 validation tuning selected learning rate 1e-05 and M2 evidence lambda 0.1. Search candidates were LR {1e-5, 2e-5} and lambda {0.1, 0.5, 1.0}. Test data did not select hyperparameters or checkpoints.

Protocol v2: reranker pretraining is selected by dev MRR. M1/M2/A2 share the final epoch-three verifier/optimizer/scaler/RNG state, then train epochs four and five. Their final checkpoint maximizes dev Macro-F1 over all five epochs, including warmup; ties favor the earlier global epoch. M0/A1 train five verifier epochs from common initial verifier weights. Learning-rate and lambda searches run afresh for the configured architecture on seed-42 validation only; selected M1/M2 tuning checkpoints are reused.

All models keep separate PhoBERT-base encoders where applicable, 256 pair tokens, 96 claim tokens, BM25 top 20 and selected top five. Adafactor, eight-claim accumulation, mixed precision and activation checkpointing remain enabled. Both M2 encoders are trainable. Lambda applies to M2 evidence loss; it has no effect in M0/M1.

A1 keeps M1's frozen reranker and selected sentence set, but trains its verifier with uniform pooling for five epochs from the common initial verifier. A2 branches from the same weighted warmup checkpoint as M1/M2; its verification weights are detached, while evidence BCE still updates the reranker. M2 and A2 use the same selected lambda.

Reranker architecture: {'reranker_architecture': 'listwise', 'listwise_layers': 1, 'listwise_heads': 8, 'listwise_feedforward': 1536, 'listwise_dropout': 0.1}. List-aware attention is a separately configured implementation inspired by the slides, not a reproduction of the complete M-ReRank or RAV systems.

## Aggregate scores

Scores are means and sample standard deviations across three seeds, not a vote ensemble.

| Model | Dev Macro-F1 | Test Macro-F1 | Test accuracy |
|---|---:|---:|---:|
| M0 | 0.6075 +/- 0.0158 | 0.5871 +/- 0.0158 | 0.5911 +/- 0.0105 |
| M1 | 0.6825 +/- 0.0153 | 0.6733 +/- 0.0140 | 0.6750 +/- 0.0128 |
| A1 | 0.6060 +/- 0.0207 | 0.5919 +/- 0.0038 | 0.5941 +/- 0.0054 |
| M2 | 0.7010 +/- 0.0116 | 0.7066 +/- 0.0057 | 0.7081 +/- 0.0055 |
| A2 | 0.6931 +/- 0.0158 | 0.6799 +/- 0.0047 | 0.6809 +/- 0.0049 |

## Per-seed scores

| Seed | Model | Dev Macro-F1 | Test Macro-F1 | Accuracy |
|---|---|---:|---:|---:|
| 42 | M0 | 0.5895 | 0.5692 | 0.5791 |
| 42 | M1 | 0.6818 | 0.6575 | 0.6607 |
| 42 | A1 | 0.5834 | 0.5875 | 0.5881 |
| 42 | M2 | 0.7062 | 0.7016 | 0.7028 |
| 42 | A2 | 0.6765 | 0.6751 | 0.6759 |
| 13 | M0 | 0.6195 | 0.5932 | 0.5957 |
| 13 | M1 | 0.6982 | 0.6785 | 0.6786 |
| 13 | A1 | 0.6240 | 0.5942 | 0.5985 |
| 13 | M2 | 0.7090 | 0.7052 | 0.7077 |
| 13 | A2 | 0.7080 | 0.6801 | 0.6814 |
| 2026 | M0 | 0.6134 | 0.5991 | 0.5985 |
| 2026 | M1 | 0.6676 | 0.6841 | 0.6856 |
| 2026 | A1 | 0.6107 | 0.5941 | 0.5957 |
| 2026 | M2 | 0.6877 | 0.7128 | 0.7139 |
| 2026 | A2 | 0.6949 | 0.6846 | 0.6856 |

## Selected checkpoints

A warmup checkpoint may legitimately win for M2/A2. Gradient and encoder-update checks concern the terminal trained state, independently of the selected prediction checkpoint.

| Seed | Model | Selected global epoch | Selected stage |
|---|---|---:|---|
| 42 | M0 | 3 | m0 |
| 42 | M1 | 4 | continuation |
| 42 | A1 | 5 | a1 |
| 42 | M2 | 5 | continuation |
| 42 | A2 | 5 | continuation |
| 13 | M0 | 3 | m0 |
| 13 | M1 | 5 | continuation |
| 13 | A1 | 3 | a1 |
| 13 | M2 | 4 | continuation |
| 13 | A2 | 5 | continuation |
| 2026 | M0 | 3 | m0 |
| 2026 | M1 | 4 | continuation |
| 2026 | A1 | 3 | a1 |
| 2026 | M2 | 4 | continuation |
| 2026 | A2 | 5 | continuation |

## Per-label F1

| Model | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
|---|---:|---:|---:|
| M0 | 0.6243 +/- 0.0166 | 0.5270 +/- 0.0479 | 0.6101 +/- 0.0083 |
| M1 | 0.7009 +/- 0.0111 | 0.6683 +/- 0.0204 | 0.6508 +/- 0.0157 |
| A1 | 0.6352 +/- 0.0206 | 0.5427 +/- 0.0157 | 0.5979 +/- 0.0144 |
| M2 | 0.7272 +/- 0.0023 | 0.7040 +/- 0.0074 | 0.6885 +/- 0.0087 |
| A2 | 0.7052 +/- 0.0022 | 0.6841 +/- 0.0105 | 0.6505 +/- 0.0080 |

## Retrieval and alignment

Each record reports its eligible aligned SUPPORTED/REFUTED denominator. NEI and unsuccessful or ambiguous alignment are excluded from evidence metrics. Classification keeps all valid claims.

| Seed | Model | Eligible claims | Hit@5 | Hit@20 | Recall@5 | MRR | Complete gold@5 | Alignment coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | M0 | 539 | 0.9777 | 1.0000 | 0.9460 | 0.9433 | 0.9072 | 0.5923 |
| 42 | M1 | 539 | 0.9833 | 1.0000 | 0.9513 | 0.9539 | 0.9128 | 0.5923 |
| 42 | A1 | 539 | 0.9833 | 1.0000 | 0.9513 | 0.9539 | 0.9128 | 0.5923 |
| 42 | M2 | 539 | 0.9796 | 1.0000 | 0.9455 | 0.9499 | 0.9054 | 0.5923 |
| 42 | A2 | 539 | 0.9833 | 1.0000 | 0.9482 | 0.9473 | 0.9072 | 0.5923 |
| 13 | M0 | 539 | 0.9777 | 1.0000 | 0.9460 | 0.9433 | 0.9072 | 0.5923 |
| 13 | M1 | 539 | 0.9870 | 1.0000 | 0.9571 | 0.9585 | 0.9202 | 0.5923 |
| 13 | A1 | 539 | 0.9870 | 1.0000 | 0.9571 | 0.9585 | 0.9202 | 0.5923 |
| 13 | M2 | 539 | 0.9833 | 1.0000 | 0.9518 | 0.9562 | 0.9128 | 0.5923 |
| 13 | A2 | 539 | 0.9852 | 1.0000 | 0.9528 | 0.9491 | 0.9128 | 0.5923 |
| 2026 | M0 | 539 | 0.9777 | 1.0000 | 0.9460 | 0.9433 | 0.9072 | 0.5923 |
| 2026 | M1 | 539 | 0.9870 | 1.0000 | 0.9553 | 0.9557 | 0.9165 | 0.5923 |
| 2026 | A1 | 539 | 0.9870 | 1.0000 | 0.9553 | 0.9557 | 0.9165 | 0.5923 |
| 2026 | M2 | 539 | 0.9907 | 1.0000 | 0.9600 | 0.9540 | 0.9239 | 0.5923 |
| 2026 | A2 | 539 | 0.9889 | 1.0000 | 0.9599 | 0.9560 | 0.9239 | 0.5923 |

## Paired differences

| Comparison | Mean Macro-F1 difference | Sample SD across seeds |
|---|---:|---:|
| A1-M0 | +0.0048 | 0.0121 |
| M1-A1 | +0.0814 | 0.0103 |
| M2-M1 | +0.0332 | 0.0095 |
| M2-A2 | +0.0266 | 0.0016 |

Per-seed test-sample intervals use 2,000 paired article-cluster bootstrap replicates. They are conditional on fitted checkpoints and do not replace training-seed variability.

| Seed | Comparison | Macro-F1 difference [95% CI] | REFUTED F1 difference [95% CI] |
|---|---|---:|---:|
| 42 | A1-M0 | +0.0183 [-0.0065, +0.0426] | +0.0721 [+0.0359, +0.1105] |
| 42 | M1-A1 | +0.0699 [+0.0420, +0.0971] | +0.0987 [+0.0564, +0.1418] |
| 42 | M2-M1 | +0.0442 [+0.0266, +0.0621] | +0.0519 [+0.0256, +0.0780] |
| 42 | M2-A2 | +0.0265 [+0.0090, +0.0438] | +0.0215 [-0.0053, +0.0487] |
| 13 | A1-M0 | +0.0010 [-0.0147, +0.0172] | -0.0084 [-0.0326, +0.0147] |
| 13 | M1-A1 | +0.0843 [+0.0555, +0.1122] | +0.1452 [+0.1051, +0.1851] |
| 13 | M2-M1 | +0.0267 [+0.0055, +0.0475] | +0.0305 [+0.0036, +0.0580] |
| 13 | M2-A2 | +0.0251 [+0.0067, +0.0434] | +0.0221 [-0.0037, +0.0475] |
| 2026 | A1-M0 | -0.0050 [-0.0240, +0.0135] | -0.0166 [-0.0450, +0.0084] |
| 2026 | M1-A1 | +0.0900 [+0.0612, +0.1189] | +0.1328 [+0.0940, +0.1755] |
| 2026 | M2-M1 | +0.0288 [+0.0112, +0.0460] | +0.0248 [-0.0006, +0.0492] |
| 2026 | M2-A2 | +0.0282 [+0.0105, +0.0468] | +0.0163 [-0.0097, +0.0412] |

## Confusion matrices, diagnostics and costs

### Seed 42: M0

Rows are gold labels; columns are SUPPORTED / REFUTED / NOT_ENOUGH_INFO.

| Gold label | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
|---|---:|---:|---:|
| SUPPORTED | 338 | 54 | 116 |
| REFUTED | 136 | 181 | 151 |
| NOT_ENOUGH_INFO | 95 | 57 | 319 |

Per-label precision/recall/F1: {'SUPPORTED': {'precision': 0.5940246045694201, 'recall': 0.6653543307086615, 'f1': 0.627669452181987, 'support': 508}, 'REFUTED': {'precision': 0.6198630136986302, 'recall': 0.38675213675213677, 'f1': 0.4763157894736842, 'support': 468}, 'NOT_ENOUGH_INFO': {'precision': 0.5443686006825939, 'recall': 0.6772823779193206, 'f1': 0.6035950804162725, 'support': 471}}

Error stages: {'correct': 838, 'verification_with_complete_gold': 195, 'alignment_unavailable': 232, 'selection_incomplete_gold': 27, 'nei_verification_error': 152, 'bm25_incomplete_gold': 3}

Truncation: {'claim_count': 1447, 'claim_truncated_count': 3, 'candidate_pair_count': 22962, 'candidate_sentence_truncated_count': 3, 'selected_pair_count': 7231, 'selected_sentence_truncated_count': 3}

Prepared-input inference cost: {'evaluation_seconds': 117.78819400002249, 'seconds_per_claim': 0.08140165445751381, 'peak_cuda_allocated_mib': 766.7763671875, 'peak_cuda_reserved_mib': 782.0, 'prepared_input_inference_seconds': 112.27598780021071, 'prepared_input_seconds_per_claim': 0.07759225141686987}

### Seed 42: M1

Rows are gold labels; columns are SUPPORTED / REFUTED / NOT_ENOUGH_INFO.

| Gold label | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
|---|---:|---:|---:|
| SUPPORTED | 402 | 34 | 72 |
| REFUTED | 123 | 264 | 81 |
| NOT_ENOUGH_INFO | 131 | 50 | 290 |

Per-label precision/recall/F1: {'SUPPORTED': {'precision': 0.6128048780487805, 'recall': 0.7913385826771654, 'f1': 0.6907216494845361, 'support': 508}, 'REFUTED': {'precision': 0.7586206896551724, 'recall': 0.5641025641025641, 'f1': 0.6470588235294118, 'support': 468}, 'NOT_ENOUGH_INFO': {'precision': 0.654627539503386, 'recall': 0.6157112526539278, 'f1': 0.6345733041575492, 'support': 471}}

Error stages: {'correct': 956, 'verification_with_complete_gold': 110, 'nei_verification_error': 181, 'alignment_unavailable': 176, 'selection_incomplete_gold': 21, 'bm25_incomplete_gold': 3}

Truncation: {'claim_count': 1447, 'claim_truncated_count': 3, 'candidate_pair_count': 22962, 'candidate_sentence_truncated_count': 3, 'selected_pair_count': 7231, 'selected_sentence_truncated_count': 3}

Prepared-input inference cost: {'evaluation_seconds': 534.0776452000719, 'seconds_per_claim': 0.3690930512785569, 'peak_cuda_allocated_mib': 1318.99072265625, 'peak_cuda_reserved_mib': 1340.0, 'prepared_input_inference_seconds': 528.6692836999428, 'prepared_input_seconds_per_claim': 0.365355413752552}

### Seed 42: A1

Rows are gold labels; columns are SUPPORTED / REFUTED / NOT_ENOUGH_INFO.

| Gold label | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
|---|---:|---:|---:|
| SUPPORTED | 288 | 107 | 113 |
| REFUTED | 83 | 255 | 130 |
| NOT_ENOUGH_INFO | 63 | 100 | 308 |

Per-label precision/recall/F1: {'SUPPORTED': {'precision': 0.663594470046083, 'recall': 0.5669291338582677, 'f1': 0.6114649681528662, 'support': 508}, 'REFUTED': {'precision': 0.551948051948052, 'recall': 0.5448717948717948, 'f1': 0.5483870967741935, 'support': 468}, 'NOT_ENOUGH_INFO': {'precision': 0.558983666061706, 'recall': 0.6539278131634819, 'f1': 0.6027397260273972, 'support': 471}}

Error stages: {'correct': 851, 'verification_with_complete_gold': 178, 'alignment_unavailable': 223, 'selection_incomplete_gold': 29, 'nei_verification_error': 163, 'bm25_incomplete_gold': 3}

Truncation: {'claim_count': 1447, 'claim_truncated_count': 3, 'candidate_pair_count': 22962, 'candidate_sentence_truncated_count': 3, 'selected_pair_count': 7231, 'selected_sentence_truncated_count': 3}

Prepared-input inference cost: {'evaluation_seconds': 544.6384951001965, 'seconds_per_claim': 0.3763914962682767, 'peak_cuda_allocated_mib': 1318.99072265625, 'peak_cuda_reserved_mib': 1340.0, 'prepared_input_inference_seconds': 538.8441843001638, 'prepared_input_seconds_per_claim': 0.3723871349690143}

### Seed 42: M2

Rows are gold labels; columns are SUPPORTED / REFUTED / NOT_ENOUGH_INFO.

| Gold label | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
|---|---:|---:|---:|
| SUPPORTED | 396 | 33 | 79 |
| REFUTED | 93 | 296 | 79 |
| NOT_ENOUGH_INFO | 96 | 50 | 325 |

Per-label precision/recall/F1: {'SUPPORTED': {'precision': 0.676923076923077, 'recall': 0.7795275590551181, 'f1': 0.7246111619396157, 'support': 508}, 'REFUTED': {'precision': 0.7810026385224275, 'recall': 0.6324786324786325, 'f1': 0.6989374262101535, 'support': 468}, 'NOT_ENOUGH_INFO': {'precision': 0.6728778467908902, 'recall': 0.6900212314225053, 'f1': 0.6813417190775681, 'support': 471}}

Error stages: {'correct': 1017, 'verification_with_complete_gold': 97, 'alignment_unavailable': 162, 'nei_verification_error': 146, 'selection_incomplete_gold': 22, 'bm25_incomplete_gold': 3}

Truncation: {'claim_count': 1447, 'claim_truncated_count': 3, 'candidate_pair_count': 22962, 'candidate_sentence_truncated_count': 3, 'selected_pair_count': 7231, 'selected_sentence_truncated_count': 3}

Prepared-input inference cost: {'evaluation_seconds': 536.1279581999406, 'seconds_per_claim': 0.37050999184515593, 'peak_cuda_allocated_mib': 1318.99072265625, 'peak_cuda_reserved_mib': 1340.0, 'prepared_input_inference_seconds': 530.9967262998689, 'prepared_input_seconds_per_claim': 0.36696387442976425}

### Seed 42: A2

Rows are gold labels; columns are SUPPORTED / REFUTED / NOT_ENOUGH_INFO.

| Gold label | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
|---|---:|---:|---:|
| SUPPORTED | 360 | 58 | 90 |
| REFUTED | 73 | 314 | 81 |
| NOT_ENOUGH_INFO | 80 | 87 | 304 |

Per-label precision/recall/F1: {'SUPPORTED': {'precision': 0.7017543859649122, 'recall': 0.7086614173228346, 'f1': 0.7051909892262488, 'support': 508}, 'REFUTED': {'precision': 0.6840958605664488, 'recall': 0.6709401709401709, 'f1': 0.6774541531823085, 'support': 468}, 'NOT_ENOUGH_INFO': {'precision': 0.64, 'recall': 0.6454352441613588, 'f1': 0.642706131078224, 'support': 471}}

Error stages: {'correct': 978, 'nei_verification_error': 167, 'selection_incomplete_gold': 20, 'alignment_unavailable': 172, 'verification_with_complete_gold': 107, 'bm25_incomplete_gold': 3}

Truncation: {'claim_count': 1447, 'claim_truncated_count': 3, 'candidate_pair_count': 22962, 'candidate_sentence_truncated_count': 3, 'selected_pair_count': 7231, 'selected_sentence_truncated_count': 3}

Prepared-input inference cost: {'evaluation_seconds': 540.634818200022, 'seconds_per_claim': 0.3736246152038853, 'peak_cuda_allocated_mib': 1318.99072265625, 'peak_cuda_reserved_mib': 1340.0, 'prepared_input_inference_seconds': 535.424391400069, 'prepared_input_seconds_per_claim': 0.37002376738083553}

### Seed 13: M0

Rows are gold labels; columns are SUPPORTED / REFUTED / NOT_ENOUGH_INFO.

| Gold label | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
|---|---:|---:|---:|
| SUPPORTED | 323 | 84 | 101 |
| REFUTED | 109 | 236 | 123 |
| NOT_ENOUGH_INFO | 71 | 97 | 303 |

Per-label precision/recall/F1: {'SUPPORTED': {'precision': 0.6421471172962226, 'recall': 0.6358267716535433, 'f1': 0.6389713155291791, 'support': 508}, 'REFUTED': {'precision': 0.565947242206235, 'recall': 0.5042735042735043, 'f1': 0.5333333333333333, 'support': 468}, 'NOT_ENOUGH_INFO': {'precision': 0.5749525616698292, 'recall': 0.643312101910828, 'f1': 0.6072144288577155, 'support': 471}}

Error stages: {'correct': 862, 'verification_with_complete_gold': 179, 'alignment_unavailable': 212, 'nei_verification_error': 168, 'selection_incomplete_gold': 24, 'bm25_incomplete_gold': 2}

Truncation: {'claim_count': 1447, 'claim_truncated_count': 3, 'candidate_pair_count': 22962, 'candidate_sentence_truncated_count': 3, 'selected_pair_count': 7231, 'selected_sentence_truncated_count': 3}

Prepared-input inference cost: {'evaluation_seconds': 126.31618100008927, 'seconds_per_claim': 0.08729521838292278, 'peak_cuda_allocated_mib': 766.7763671875, 'peak_cuda_reserved_mib': 782.0, 'prepared_input_inference_seconds': 121.13425030000508, 'prepared_input_seconds_per_claim': 0.08371406378714932}

### Seed 13: M1

Rows are gold labels; columns are SUPPORTED / REFUTED / NOT_ENOUGH_INFO.

| Gold label | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
|---|---:|---:|---:|
| SUPPORTED | 343 | 78 | 87 |
| REFUTED | 67 | 320 | 81 |
| NOT_ENOUGH_INFO | 63 | 89 | 319 |

Per-label precision/recall/F1: {'SUPPORTED': {'precision': 0.7251585623678647, 'recall': 0.6751968503937008, 'f1': 0.6992864424057085, 'support': 508}, 'REFUTED': {'precision': 0.6570841889117043, 'recall': 0.6837606837606838, 'f1': 0.6701570680628273, 'support': 468}, 'NOT_ENOUGH_INFO': {'precision': 0.6550308008213552, 'recall': 0.6772823779193206, 'f1': 0.6659707724425887, 'support': 471}}

Error stages: {'correct': 982, 'alignment_unavailable': 182, 'verification_with_complete_gold': 110, 'selection_incomplete_gold': 18, 'nei_verification_error': 152, 'bm25_incomplete_gold': 3}

Truncation: {'claim_count': 1447, 'claim_truncated_count': 3, 'candidate_pair_count': 22962, 'candidate_sentence_truncated_count': 3, 'selected_pair_count': 7231, 'selected_sentence_truncated_count': 3}

Prepared-input inference cost: {'evaluation_seconds': 540.5770191000775, 'seconds_per_claim': 0.3735846711127004, 'peak_cuda_allocated_mib': 1318.99072265625, 'peak_cuda_reserved_mib': 1340.0, 'prepared_input_inference_seconds': 535.3195687001571, 'prepared_input_seconds_per_claim': 0.3699513259849047}

### Seed 13: A1

Rows are gold labels; columns are SUPPORTED / REFUTED / NOT_ENOUGH_INFO.

| Gold label | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
|---|---:|---:|---:|
| SUPPORTED | 343 | 80 | 85 |
| REFUTED | 120 | 226 | 122 |
| NOT_ENOUGH_INFO | 87 | 87 | 297 |

Per-label precision/recall/F1: {'SUPPORTED': {'precision': 0.6236363636363637, 'recall': 0.6751968503937008, 'f1': 0.6483931947069943, 'support': 508}, 'REFUTED': {'precision': 0.5750636132315522, 'recall': 0.4829059829059829, 'f1': 0.5249709639953543, 'support': 468}, 'NOT_ENOUGH_INFO': {'precision': 0.5892857142857143, 'recall': 0.6305732484076433, 'f1': 0.6092307692307692, 'support': 471}}

Error stages: {'correct': 866, 'verification_with_complete_gold': 177, 'alignment_unavailable': 208, 'nei_verification_error': 174, 'selection_incomplete_gold': 20, 'bm25_incomplete_gold': 2}

Truncation: {'claim_count': 1447, 'claim_truncated_count': 3, 'candidate_pair_count': 22962, 'candidate_sentence_truncated_count': 3, 'selected_pair_count': 7231, 'selected_sentence_truncated_count': 3}

Prepared-input inference cost: {'evaluation_seconds': 553.7745722001418, 'seconds_per_claim': 0.3827053021424615, 'peak_cuda_allocated_mib': 1318.99072265625, 'peak_cuda_reserved_mib': 1340.0, 'prepared_input_inference_seconds': 548.1494293001015, 'prepared_input_seconds_per_claim': 0.37881785024194986}

### Seed 13: M2

Rows are gold labels; columns are SUPPORTED / REFUTED / NOT_ENOUGH_INFO.

| Gold label | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
|---|---:|---:|---:|
| SUPPORTED | 417 | 57 | 34 |
| REFUTED | 106 | 323 | 39 |
| NOT_ENOUGH_INFO | 113 | 74 | 284 |

Per-label precision/recall/F1: {'SUPPORTED': {'precision': 0.6556603773584906, 'recall': 0.8208661417322834, 'f1': 0.7290209790209791, 'support': 508}, 'REFUTED': {'precision': 0.711453744493392, 'recall': 0.6901709401709402, 'f1': 0.7006507592190889, 'support': 468}, 'NOT_ENOUGH_INFO': {'precision': 0.7955182072829131, 'recall': 0.6029723991507431, 'f1': 0.6859903381642513, 'support': 471}}

Error stages: {'correct': 1024, 'alignment_unavailable': 132, 'nei_verification_error': 187, 'selection_incomplete_gold': 16, 'verification_with_complete_gold': 88}

Truncation: {'claim_count': 1447, 'claim_truncated_count': 3, 'candidate_pair_count': 22962, 'candidate_sentence_truncated_count': 3, 'selected_pair_count': 7231, 'selected_sentence_truncated_count': 1}

Prepared-input inference cost: {'evaluation_seconds': 543.9641908998601, 'seconds_per_claim': 0.3759254947476573, 'peak_cuda_allocated_mib': 1318.99072265625, 'peak_cuda_reserved_mib': 1340.0, 'prepared_input_inference_seconds': 538.7662114000414, 'prepared_input_seconds_per_claim': 0.37233324906706383}

### Seed 13: A2

Rows are gold labels; columns are SUPPORTED / REFUTED / NOT_ENOUGH_INFO.

| Gold label | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
|---|---:|---:|---:|
| SUPPORTED | 380 | 55 | 73 |
| REFUTED | 94 | 305 | 69 |
| NOT_ENOUGH_INFO | 99 | 71 | 301 |

Per-label precision/recall/F1: {'SUPPORTED': {'precision': 0.6631762652705061, 'recall': 0.7480314960629921, 'f1': 0.7030527289546716, 'support': 508}, 'REFUTED': {'precision': 0.7076566125290024, 'recall': 0.6517094017094017, 'f1': 0.67853170189099, 'support': 468}, 'NOT_ENOUGH_INFO': {'precision': 0.6794582392776524, 'recall': 0.6390658174097664, 'f1': 0.6586433260393874, 'support': 471}}

Error stages: {'correct': 986, 'alignment_unavailable': 168, 'selection_incomplete_gold': 22, 'verification_with_complete_gold': 99, 'nei_verification_error': 170, 'bm25_incomplete_gold': 2}

Truncation: {'claim_count': 1447, 'claim_truncated_count': 3, 'candidate_pair_count': 22962, 'candidate_sentence_truncated_count': 3, 'selected_pair_count': 7231, 'selected_sentence_truncated_count': 3}

Prepared-input inference cost: {'evaluation_seconds': 555.6620674000587, 'seconds_per_claim': 0.3840097217692182, 'peak_cuda_allocated_mib': 1318.99072265625, 'peak_cuda_reserved_mib': 1340.0, 'prepared_input_inference_seconds': 550.4202231999952, 'prepared_input_seconds_per_claim': 0.3803871618521045}

### Seed 2026: M0

Rows are gold labels; columns are SUPPORTED / REFUTED / NOT_ENOUGH_INFO.

| Gold label | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
|---|---:|---:|---:|
| SUPPORTED | 268 | 137 | 103 |
| REFUTED | 68 | 292 | 108 |
| NOT_ENOUGH_INFO | 40 | 125 | 306 |

Per-label precision/recall/F1: {'SUPPORTED': {'precision': 0.7127659574468085, 'recall': 0.5275590551181102, 'f1': 0.6063348416289592, 'support': 508}, 'REFUTED': {'precision': 0.5270758122743683, 'recall': 0.6239316239316239, 'f1': 0.5714285714285714, 'support': 468}, 'NOT_ENOUGH_INFO': {'precision': 0.5918762088974855, 'recall': 0.6496815286624203, 'f1': 0.6194331983805668, 'support': 471}}

Error stages: {'correct': 866, 'verification_with_complete_gold': 170, 'alignment_unavailable': 219, 'selection_incomplete_gold': 25, 'nei_verification_error': 165, 'bm25_incomplete_gold': 2}

Truncation: {'claim_count': 1447, 'claim_truncated_count': 3, 'candidate_pair_count': 22962, 'candidate_sentence_truncated_count': 3, 'selected_pair_count': 7231, 'selected_sentence_truncated_count': 3}

Prepared-input inference cost: {'evaluation_seconds': 128.94023509998806, 'seconds_per_claim': 0.08910866281961856, 'peak_cuda_allocated_mib': 766.7763671875, 'peak_cuda_reserved_mib': 782.0, 'prepared_input_inference_seconds': 122.98516370006837, 'prepared_input_seconds_per_claim': 0.08499320228062776}

### Seed 2026: M1

Rows are gold labels; columns are SUPPORTED / REFUTED / NOT_ENOUGH_INFO.

| Gold label | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
|---|---:|---:|---:|
| SUPPORTED | 392 | 37 | 79 |
| REFUTED | 91 | 294 | 83 |
| NOT_ENOUGH_INFO | 109 | 56 | 306 |

Per-label precision/recall/F1: {'SUPPORTED': {'precision': 0.6621621621621622, 'recall': 0.7716535433070866, 'f1': 0.7127272727272728, 'support': 508}, 'REFUTED': {'precision': 0.7596899224806202, 'recall': 0.6282051282051282, 'f1': 0.6877192982456141, 'support': 468}, 'NOT_ENOUGH_INFO': {'precision': 0.6538461538461539, 'recall': 0.6496815286624203, 'f1': 0.6517571884984026, 'support': 471}}

Error stages: {'correct': 992, 'nei_verification_error': 165, 'verification_with_complete_gold': 103, 'alignment_unavailable': 167, 'selection_incomplete_gold': 16, 'bm25_incomplete_gold': 4}

Truncation: {'claim_count': 1447, 'claim_truncated_count': 3, 'candidate_pair_count': 22962, 'candidate_sentence_truncated_count': 3, 'selected_pair_count': 7231, 'selected_sentence_truncated_count': 3}

Prepared-input inference cost: {'evaluation_seconds': 559.9250407000072, 'seconds_per_claim': 0.38695579868694346, 'peak_cuda_allocated_mib': 1318.99072265625, 'peak_cuda_reserved_mib': 1340.0, 'prepared_input_inference_seconds': 554.6450097998604, 'prepared_input_seconds_per_claim': 0.38330684851407076}

### Seed 2026: A1

Rows are gold labels; columns are SUPPORTED / REFUTED / NOT_ENOUGH_INFO.

| Gold label | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
|---|---:|---:|---:|
| SUPPORTED | 338 | 111 | 59 |
| REFUTED | 113 | 273 | 82 |
| NOT_ENOUGH_INFO | 88 | 132 | 251 |

Per-label precision/recall/F1: {'SUPPORTED': {'precision': 0.62708719851577, 'recall': 0.6653543307086615, 'f1': 0.6456542502387774, 'support': 508}, 'REFUTED': {'precision': 0.5290697674418605, 'recall': 0.5833333333333334, 'f1': 0.5548780487804879, 'support': 468}, 'NOT_ENOUGH_INFO': {'precision': 0.6403061224489796, 'recall': 0.5329087048832272, 'f1': 0.5816917728852838, 'support': 471}}

Error stages: {'correct': 862, 'verification_with_complete_gold': 153, 'alignment_unavailable': 194, 'nei_verification_error': 220, 'selection_incomplete_gold': 17, 'bm25_incomplete_gold': 1}

Truncation: {'claim_count': 1447, 'claim_truncated_count': 3, 'candidate_pair_count': 22962, 'candidate_sentence_truncated_count': 3, 'selected_pair_count': 7231, 'selected_sentence_truncated_count': 3}

Prepared-input inference cost: {'evaluation_seconds': 566.8168524999637, 'seconds_per_claim': 0.39171862646853056, 'peak_cuda_allocated_mib': 1318.99072265625, 'peak_cuda_reserved_mib': 1340.0, 'prepared_input_inference_seconds': 561.3836676999927, 'prepared_input_seconds_per_claim': 0.3879638339322686}

### Seed 2026: M2

Rows are gold labels; columns are SUPPORTED / REFUTED / NOT_ENOUGH_INFO.

| Gold label | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
|---|---:|---:|---:|
| SUPPORTED | 404 | 41 | 63 |
| REFUTED | 98 | 311 | 59 |
| NOT_ENOUGH_INFO | 100 | 53 | 318 |

Per-label precision/recall/F1: {'SUPPORTED': {'precision': 0.6710963455149501, 'recall': 0.7952755905511811, 'f1': 0.7279279279279279, 'support': 508}, 'REFUTED': {'precision': 0.7679012345679013, 'recall': 0.6645299145299145, 'f1': 0.7124856815578465, 'support': 468}, 'NOT_ENOUGH_INFO': {'precision': 0.7227272727272728, 'recall': 0.6751592356687898, 'f1': 0.6981339187705817, 'support': 471}}

Error stages: {'correct': 1033, 'nei_verification_error': 153, 'verification_with_complete_gold': 99, 'alignment_unavailable': 148, 'selection_incomplete_gold': 13, 'bm25_incomplete_gold': 1}

Truncation: {'claim_count': 1447, 'claim_truncated_count': 3, 'candidate_pair_count': 22962, 'candidate_sentence_truncated_count': 3, 'selected_pair_count': 7231, 'selected_sentence_truncated_count': 3}

Prepared-input inference cost: {'evaluation_seconds': 567.0077215000056, 'seconds_per_claim': 0.391850533172084, 'peak_cuda_allocated_mib': 1318.99072265625, 'peak_cuda_reserved_mib': 1340.0, 'prepared_input_inference_seconds': 561.7244940001983, 'prepared_input_seconds_per_claim': 0.38819937387712394}

### Seed 2026: A2

Rows are gold labels; columns are SUPPORTED / REFUTED / NOT_ENOUGH_INFO.

| Gold label | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
|---|---:|---:|---:|
| SUPPORTED | 376 | 50 | 82 |
| REFUTED | 80 | 314 | 74 |
| NOT_ENOUGH_INFO | 99 | 70 | 302 |

Per-label precision/recall/F1: {'SUPPORTED': {'precision': 0.6774774774774774, 'recall': 0.7401574803149606, 'f1': 0.7074317968015051, 'support': 508}, 'REFUTED': {'precision': 0.7235023041474654, 'recall': 0.6709401709401709, 'f1': 0.6962305986696231, 'support': 468}, 'NOT_ENOUGH_INFO': {'precision': 0.6593886462882096, 'recall': 0.6411889596602972, 'f1': 0.6501614639397201, 'support': 471}}

Error stages: {'correct': 992, 'nei_verification_error': 169, 'alignment_unavailable': 168, 'verification_with_complete_gold': 104, 'selection_incomplete_gold': 12, 'bm25_incomplete_gold': 2}

Truncation: {'claim_count': 1447, 'claim_truncated_count': 3, 'candidate_pair_count': 22962, 'candidate_sentence_truncated_count': 3, 'selected_pair_count': 7231, 'selected_sentence_truncated_count': 3}

Prepared-input inference cost: {'evaluation_seconds': 559.688434000127, 'seconds_per_claim': 0.3867922833449392, 'peak_cuda_allocated_mib': 1318.99072265625, 'peak_cuda_reserved_mib': 1340.0, 'prepared_input_inference_seconds': 554.3605176999699, 'prepared_input_seconds_per_claim': 0.38311024029023494}

## Training resource measurements

Training times exclude validation and checkpoint writes. Shared preparation executes once; phase costs below must not be double-counted across model configurations.

| Job | Phase | Training minutes | Peak allocated MiB | Peak reserved MiB |
|---|---|---:|---:|---:|
| tuning\lambda_0.1 | m2 | 341.5 | 2999.4 | 3208.0 |
| tuning\lambda_0.5 | m2 | 392.5 | 3000.5 | 3206.0 |
| tuning\lambda_1.0 | m2 | 366.6 | 3001.9 | 3206.0 |
| tuning\lr_1e-05 | m1 | 125.4 | 2346.4 | 2524.0 |
| tuning\lr_1e-05 | reranker | 136.4 | 2446.2 | 2628.0 |
| tuning\lr_1e-05 | warmup | 207.0 | 2346.4 | 2628.0 |
| tuning\lr_2e-05 | m1 | 113.8 | 2345.0 | 2520.0 |
| tuning\lr_2e-05 | reranker | 120.9 | 2446.2 | 2628.0 |
| tuning\lr_2e-05 | warmup | 182.5 | 2345.0 | 2628.0 |
| seeds\seed13-branches | a1 | 293.0 | 2344.4 | 2524.0 |
| seeds\seed13-branches | a2 | 222.5 | 2999.7 | 3206.0 |
| seeds\seed13-branches | m2 | 334.0 | 3001.8 | 3204.0 |
| seeds\seed13-m0 | m0 | 176.2 | 1814.4 | 1954.0 |
| seeds\seed13-parent | m1 | 163.1 | 2344.4 | 2524.0 |
| seeds\seed13-parent | reranker | 154.1 | 2446.2 | 2626.0 |
| seeds\seed13-parent | warmup | 223.7 | 2344.4 | 2626.0 |
| seeds\seed2026-branches | a1 | 305.9 | 2346.4 | 2522.0 |
| seeds\seed2026-branches | a2 | 323.9 | 2999.4 | 3204.0 |
| seeds\seed2026-branches | m2 | 1044.8 | 2999.1 | 3206.0 |
| seeds\seed2026-m0 | m0 | 196.8 | 1814.4 | 1954.0 |
| seeds\seed2026-parent | m1 | 122.3 | 2346.4 | 2522.0 |
| seeds\seed2026-parent | reranker | 119.9 | 2446.2 | 2626.0 |
| seeds\seed2026-parent | warmup | 181.2 | 2346.4 | 2626.0 |
| seeds\seed42-ablations | a1 | 401.2 | 2346.4 | 2524.0 |
| seeds\seed42-ablations | a2 | 275.4 | 2998.4 | 3204.0 |
| seeds\seed42-m0 | m0 | 250.3 | 1814.4 | 1954.0 |

GPU memory figures use the PyTorch allocator and exclude driver/display allocations. Inference timings exclude VnCoreNLP segmentation.

## Limitations and reproduction

Official splits contain overlapping source articles; this study does not measure generalization to unseen articles. Three training seeds provide a small estimate of training variability. A1 and A2 isolate weighting and verification-feedback paths; these controls do not establish general causal claims about other architectures. D0 and manual contradiction annotation were not run.

This full experiment follows the pilot and uses the same public official test split. The pilot previously exposed test scores; full-study selection nevertheless uses validation alone. It is not a new blind test set.

Pinned data checksums: {'train': '0dcceb37266f303af12648932f42f95f112fd5d5db7c4db4556c6a20bdc7fa87', 'dev': '8df8d83dde0b61a226ddb68cf2d1b898adee9796f8311431fa80f1678a6b9621', 'test': '7d0317bf5ec1c425e7010086c743f69a36b665258b64d76b61a8c9292ebb1ce6'}

Exact configurations, source hashes and tuning protocol: `full_plan.json`. Hyperparameter trial scores: `selected_hyperparameters.json`. Per-claim predictions, checkpoints, metrics and behavior checks: `seeds/seed*/`. Parent training sources remain in `tuning/`. Runtime progress and completion times: `status.json` and `completion.json`.
