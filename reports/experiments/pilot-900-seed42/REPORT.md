# M0/M1/M2: reduced-data pilot study

## Experimental setup

We trained on 900 official training claims (300 per label), selected by stratified random sampling with seed 42. Sampling did not filter by evidence alignment. Validation used all 723 claims; final evaluation used all 1,447 test claims. This is a single-seed pilot, not the full proposal experiment.

The training subset contains 331 claims eligible for balanced evidence supervision. Unaligned claims and NEI remain in verification training. Hyperparameters were fixed before training: learning rate 2e-5 and evidence lambda 0.5; no hyperparameter search was performed.

Both components use separate PhoBERT-base parameters. Inputs keep the proposal's 256-token pair limit and 96-token claim limit. BM25 retrieves up to 20 candidates; the verifier uses up to five sentences. Short articles retain their actual sentence count. All configurations use Adafactor, eight-claim accumulation, mixed precision and activation checkpointing.

M0 uniformly averages sentence label probabilities from BM25 top five. M1 reranks BM25 top 20 and uses softmax-weighted probabilities with a frozen reranker. M2 uses the same aggregation while verification gradients reach reranker scores, alongside eligible evidence supervision. Claim verification loss is applied after aggregation; gold evidence is never inserted into the candidate set.

The reranker receives three epochs and is selected by validation MRR. The weighted verifier receives three initialization epochs; M1 and M2 branch from the same selected checkpoint for two continuation epochs. M0 receives five epochs from the same initial verifier weights. Final branch checkpoints are selected by validation Macro-F1; checkpoint ties favor the earlier epoch.

## Classification results

| Model | Validation Macro-F1 | Test Macro-F1 | Test accuracy |
|---|---:|---:|---:|
| M0 | 0.5546 | 0.5101 | 0.5169 |
| M1 | 0.5200 | 0.5248 | 0.5411 |
| M2 | 0.5287 | 0.5484 | 0.5646 |

| Model | Label | Precision | Recall | F1 | Test support |
|---|---|---:|---:|---:|---:|
| M0 | SUPPORTED | 0.5353 | 0.6417 | 0.5837 | 508 |
| M0 | REFUTED | 0.5000 | 0.3974 | 0.4429 | 468 |
| M0 | NOT_ENOUGH_INFO | 0.5064 | 0.5011 | 0.5037 | 471 |
| M1 | SUPPORTED | 0.5037 | 0.7972 | 0.6174 | 508 |
| M1 | REFUTED | 0.6400 | 0.4103 | 0.5000 | 468 |
| M1 | NOT_ENOUGH_INFO | 0.5423 | 0.3949 | 0.4570 | 471 |
| M2 | SUPPORTED | 0.5590 | 0.7461 | 0.6391 | 508 |
| M2 | REFUTED | 0.6964 | 0.3333 | 0.4509 | 468 |
| M2 | NOT_ENOUGH_INFO | 0.5174 | 0.5987 | 0.5551 | 471 |

## Evidence retrieval

Metrics use aligned SUPPORTED/REFUTED claims only. NEI and unsuccessful/ambiguous alignment are excluded. Hit counts any retrieved gold sentence; recall and complete-gold coverage account for all annotated sentences.

| Model | Eligible claims | Hit@5 | Hit@20 | Recall@5 | Recall@20 | MRR | Complete gold@5 | Complete gold@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 539 | 0.9777 | 1.0000 | 0.9460 | 0.9974 | 0.9433 | 0.9072 | 0.9907 |
| M1 | 539 | 0.9759 | 1.0000 | 0.9401 | 0.9974 | 0.9450 | 0.8998 | 0.9907 |
| M2 | 539 | 0.9666 | 1.0000 | 0.9301 | 0.9974 | 0.9477 | 0.8887 | 0.9907 |

## Paired differences and uncertainty

Intervals use 2,000 paired article-cluster bootstrap replicates with bootstrap seed 2026. They quantify test-sample uncertainty conditional on this training sample and seed; they do not measure variability across training seeds.

| Comparison | Macro-F1 difference [95% CI] | REFUTED F1 difference [95% CI] |
|---|---:|---:|
| M1-M0 | +0.0147 [-0.0136, +0.0422] | +0.0571 [+0.0121, +0.1027] |
| M2-M1 | +0.0236 [+0.0035, +0.0443] | -0.0491 [-0.0833, -0.0161] |

## Runtime and GPU memory

Training times exclude validation and checkpoint serialization. M1/M2 each include their shared preparation cost for comparison; that preparation executed once in the workflow. Inference latency uses prepared segmented input and excludes VnCoreNLP segmentation. Memory values are PyTorch allocator measurements, excluding driver/display allocations.

| Model | Training minutes | Peak allocated MiB | Peak reserved MiB | Inference seconds/claim |
|---|---:|---:|---:|---:|
| M0 | 45.4 | 1811.9 | 1948.0 | 0.1192 |
| M1 | 69.0 | 2376.2 | 2548.0 | 0.3573 |
| M2 | 98.3 | 2939.7 | 3146.0 | 0.4153 |

## Confusion matrices and diagnostic errors

Rows are gold labels; columns are predicted labels, in SUPPORTED / REFUTED / NOT_ENOUGH_INFO order.

### M0

| Gold / predicted | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
|---|---:|---:|---:|
| SUPPORTED | 326 | 84 | 98 |
| REFUTED | 150 | 186 | 132 |
| NOT_ENOUGH_INFO | 133 | 102 | 236 |

Error-stage counts: alignment_unavailable=243; bm25_incomplete_gold=2; correct=748; nei_verification_error=235; selection_incomplete_gold=23; verification_with_complete_gold=196.

Truncation: 3/1447 claims, 3/22962 candidate sentences, and 3/7231 selected sentences.

### M1

| Gold / predicted | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
|---|---:|---:|---:|
| SUPPORTED | 405 | 35 | 68 |
| REFUTED | 187 | 192 | 89 |
| NOT_ENOUGH_INFO | 212 | 73 | 186 |

Error-stage counts: alignment_unavailable=192; bm25_incomplete_gold=4; correct=783; nei_verification_error=285; selection_incomplete_gold=28; verification_with_complete_gold=155.

Truncation: 3/1447 claims, 3/22962 candidate sentences, and 3/7231 selected sentences.

### M2

| Gold / predicted | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
|---|---:|---:|---:|
| SUPPORTED | 379 | 25 | 104 |
| REFUTED | 153 | 156 | 159 |
| NOT_ENOUGH_INFO | 146 | 43 | 282 |

Error-stage counts: alignment_unavailable=230; bm25_incomplete_gold=4; correct=817; nei_verification_error=189; selection_incomplete_gold=39; verification_with_complete_gold=168.

Truncation: 3/1447 claims, 3/22962 candidate sentences, and 3/7231 selected sentences.

## Evidence examples

Examples below are the first correct and first incorrect test prediction, if available, in official test order for each model. They are illustrative examples and are not a manually annotated contradiction sample.

### M0: correct example (test:19639539:559)

Claim: Vì lượng mưa khổng lồ do cơn bão tạo ra đã dẫn đến nhiều vụ sạt lở và ngập lụt, miền nam Mozambique nhận lượng mưa cao hơn gấp đôi hàng năm, và dự kiến nếu bão cứ tiếp tục độ bổ thì lượng mưa tại đây sẽ cao nhất trong lịch sử từ trước đến giờ.

Gold: NOT_ENOUGH_INFO; predicted: NOT_ENOUGH_INFO.
- Weight 0.2000: Miền nam Mozambique nhận lượng mưa cao hơn gấp đôi hàng năm trong thời gian bão Freddy đổ bộ.
- Weight 0.2000: Cơn bão trút lượng mưa khổng lồ xuống đất liền, dẫn tới nhiều vụ sạt lở và ngập lụt, khiến hàng chục nghìn người mất nhà ở và dịch tả ở Malawi càng thêm nghiêm trọng.
- Weight 0.2000: Malawi cũng nhận khoảng 0,5 m mưa chỉ trong 72 giờ, theo WMA.
- Weight 0.2000: Nếu được xác nhận, nó sẽ là bão nhiệt đới có năng lượng cao nhất trên Trái Đất.
- Weight 0.2000: Cơn bão hiện nay đã rút ra biển và dự kiến sẽ tiêu tan.

### M0: incorrect example (test:19050703:1383)

Claim: Bộ xương bản sao được đưa đến Lodon bằng cách nghép thành con hoàn chỉnh rồi vận chuyển đến.

Gold: REFUTED; predicted: NOT_ENOUGH_INFO.
- Weight 0.2000: Để chuẩn bị cho cuộc trưng bày đặc biệt này, bà Sinead Marron, đại diện Bảo tàng Lịch sử tự nhiên London cho biết bộ xương bản sao đã được đưa đến London trong 32 thùng riêng biệt, nghĩa là "mỗi mảnh phải được ghép lại với nhau như một trò chơi ghép hình khổng lồ".
- Weight 0.2000: Bản sao của Patagotitan mayorum được trưng bày tại Bảo tàng Lịch sử tự nhiên London là bản sao bộ xương hóa thạch của một trong 6 con titanosaur được tìm thấy sau khi một nông dân ở vùng Argentina phát hiện ra một xương đùi khổng lồ nhô ra khỏi mặt đất vào năm 2010, dẫn đến quá trình khai quật kéo dài nhiều năm.
- Weight 0.2000: Trước đó, bảo tàng này cũng từng trưng bày bộ xương hóa thạch của khủng long Diplodocus "Dippy" nổi tiếng cho đến năm 2017.
- Weight 0.2000: Du khách đến với triển lãm sẽ có cơ hội chạm vào mô hình trưng bày cũng như tham gia các trò chơi tương tác.
- Weight 0.2000: Cách đây nhiều năm, các nhà khoa học ở Argentina đã phát hiện một trong những loài khủng long lớn nhất từng tồn tại trên Trái Đất là Patagotitan mayorum.

### M1: correct example (test:18855494:3399)

Claim: Đối tượng cố gắng bỏ chạy sau khi lực lượng công an đón dừng xe để kiểm tra.

Gold: SUPPORTED; predicted: SUPPORTED.
- Weight 0.9276: (PLO)- Khi lực lượng công an đón dừng xe để kiểm tra, Thành đã tăng tốc độ để bỏ chạy và va chạm với một xe ô tô khác đi ngược chiều.
- Weight 0.0357: Cụ thể, vào khoảng 17 giờ ngày 22-3, Tổ công tác tuần tra của Công an huyện Quảng Trạch phối hợp Công an xã Quảng Tùng đón dừng và kiểm tra xe ô tô mang BKS 73A-110.31 do Bùi Văn Thành điều khiển để kiểm tra.
- Weight 0.0332: Lực lượng công an sau đó đã đón chặn, vây bắt được Thành.
- Weight 0.0023: Kiểm tra phương tiện và người, lực lượng công an phát hiện và thu giữ 17 viên nén hình tròn và năm túi ni lông chứa chất tinh thể màu trắng (khoảng 2,5gram).
- Weight 0.0013: Khi thấy lực lượng công an, Thành đã tăng tốc độ gây va chạm với xe ô tô đi ngược chiều và tiếp tục phóng xe tẩu thoát.

### M1: incorrect example (test:19050703:1383)

Claim: Bộ xương bản sao được đưa đến Lodon bằng cách nghép thành con hoàn chỉnh rồi vận chuyển đến.

Gold: REFUTED; predicted: NOT_ENOUGH_INFO.
- Weight 0.9357: Để chuẩn bị cho cuộc trưng bày đặc biệt này, bà Sinead Marron, đại diện Bảo tàng Lịch sử tự nhiên London cho biết bộ xương bản sao đã được đưa đến London trong 32 thùng riêng biệt, nghĩa là "mỗi mảnh phải được ghép lại với nhau như một trò chơi ghép hình khổng lồ".
- Weight 0.0184: Nhóm này bao gồm nhiều loài khác có cấu trúc cơ thể tương tự như Brontosaurus và Diplodocus sống ở Bắc Mỹ trong Kỷ Jura, trước Kỷ Phấn trắng.
- Weight 0.0176: Trước đó, bảo tàng này cũng từng trưng bày bộ xương hóa thạch của khủng long Diplodocus "Dippy" nổi tiếng cho đến năm 2017.
- Weight 0.0145: Khủng long titanosaur là một phân nhóm của nhóm khủng long lớn hơn được gọi là sauropod.
- Weight 0.0138: Bà Marron cho biết mục đích của triển lãm là kể câu chuyện về hành trình của Patagotitan mayorum từ khi nở ra từ một quả trứng nhỏ hơn quả bóng đá tới khi trở thành một con khủng long khổng lồ nặng 57 tấn đáng kinh ngạc.

### M2: correct example (test:18655554:7117)

Claim: Phần vỏ bánh được làm rất công phu, từ giai đoạn xay gạo đến cuốn bánh hết tổng 4 tiếng đồng hồ.

Gold: NOT_ENOUGH_INFO; predicted: NOT_ENOUGH_INFO.
- Weight 0.2353: Ông Chính dùng que tre dẹt cuốn lấy vỏ bánh đặt lên đĩa, cho phần nhân đã được xào chín vào rồi cẩn thận cuộn tròn sao cho vỏ không bị rách, cắt đôi và rắc thêm chút hành khô.
- Weight 0.2024: Mỗi ngày hai vợ chồng ông Chính làm khoảng 5-7 kg gạo, cho ra hơn 200 đĩa bánh.
- Weight 0.1879: Nhưng ông vẫn duy trì quán một phần vì muốn giữ nghề gia truyền, phục vụ những khách hàng lâu năm.
- Weight 0.1872: Duy Anh cũng cho biết anh đã ăn thử nhiều hàng có tiếng khác ở Hà Nội nhưng với anh bánh cuốn ở đây vẫn ngon nhất.
- Weight 0.1872: Hấp khoảng một phút vỏ bánh vừa chín tới, có độ dai nhẹ.

### M2: incorrect example (test:19050703:1383)

Claim: Bộ xương bản sao được đưa đến Lodon bằng cách nghép thành con hoàn chỉnh rồi vận chuyển đến.

Gold: REFUTED; predicted: NOT_ENOUGH_INFO.
- Weight 0.7491: Để chuẩn bị cho cuộc trưng bày đặc biệt này, bà Sinead Marron, đại diện Bảo tàng Lịch sử tự nhiên London cho biết bộ xương bản sao đã được đưa đến London trong 32 thùng riêng biệt, nghĩa là "mỗi mảnh phải được ghép lại với nhau như một trò chơi ghép hình khổng lồ".
- Weight 0.0687: Khủng long titanosaur là một phân nhóm của nhóm khủng long lớn hơn được gọi là sauropod.
- Weight 0.0634: Sau phát hiện ở Argentina, các chuyên gia về khủng long đã quét 3D từng mảnh xương riêng lẻ để tạo ra các bản sao làm từ nhựa polyester và sợi thủy tinh, chúng được gắn trên một cấu trúc thép.
- Weight 0.0595: Một công ty ở Canada đã mất hơn 6 tháng để thực hiện công việc đúc khuôn khổng lồ, dựa trên số lượng xương hóa thạch được khai quật.
- Weight 0.0593: Ông Paul Barrett, trưởng nhóm khoa học của triển lãm, cho biết: "Họ đã phát hiện ra một nghĩa địa khủng long titanosaur với 6 con khủng long khác nhau trong lòng đất".

## Limitations and interpretation

Only 900 balanced training claims and one training seed were used. Hyperparameters were untuned; no between-seed standard deviation is available. These results are preliminary and cannot establish the behavior of full-data, three-seed training. Official splits share source articles, so results do not establish generalization to previously unseen articles.

M1 versus M0 combines changed sentence selection and evidence weighting. M2 versus M1 measures the overall change from joint training; A1 and A2 are needed to isolate the individual mechanisms. Diagnostic retrieval/selection/verification categories are not causal proof. A1, A2, D0 and manual contradiction annotation were not run.

Evidence alignment was conservative. Test alignment status counts: {'valid': 857, 'unmatched': 579, 'ambiguous': 11}. No fuzzy alignment or manual repair was used for this pilot. PyTorch may use a nondeterministic attention backward kernel despite seeded execution; independent GPU training is not claimed to be bitwise reproducible.

Pilot source-article overlaps: {'train-dev': 270, 'train-test': 410, 'dev-test': 360}. The full official-split audit remains in the data manifest.

## Reproduction records

GPU: NVIDIA GeForce RTX 3050 Laptop GPU. Dataset revision: `2df623632299a3459ad95feeff9ac67d7205a42a`. PhoBERT revision: `01daacda68afe13d83023d16ec647239e344a1e6`.

Exact IDs, settings, epoch histories, behavioral checks, per-claim test predictions and statistical comparisons are in the [archived result package](README.md). The compact `results.csv` remains beside this report.
