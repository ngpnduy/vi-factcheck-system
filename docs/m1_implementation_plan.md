# Kế hoạch triển khai baseline M1 trên ViFactCheck

Ngày đối chiếu: 01/10/2026. Căn cứ: Proposal Word hiện tại, mục 3–7; pipeline M0 đã triển khai; dữ liệu train/validation tại thời điểm đối chiếu. Đây là kế hoạch, chưa triển khai hoặc huấn luyện M1.

## 1. Định nghĩa M1

```text
claim + context đi kèm
    → cùng bộ tách câu và BM25 của M0
    → tối đa 20 câu ứng viên
    → reranker PhoBERT chấm điểm từng cặp claim–câu
    → chọn tối đa 5 câu điểm cao nhất
    → softmax điểm reranker trên các câu đã chọn → trọng số
    → verifier PhoBERT dự đoán 3 nhãn cho từng cặp
    → tổng có trọng số của các xác suất → verdict
```

Reranker và verifier là hai encoder PhoBERT có tham số riêng, cùng phiên bản pretrained/tokenizer. Reranker học trước từ chú thích evidence của train. Sau khi chọn checkpoint bằng validation, reranker được cố định trong toàn bộ quá trình học verifier M1.

Nếu các câu được chọn có điểm `s_i` và phân phối verifier `p_i(y)`, dùng:

```text
w_i = softmax(s)_i                  # chỉ trên n câu được chọn, n <= 5
P(y | claim) = Σ_i w_i * p_i(y)
L_verdict = -log P(gold_label | claim)
```

Cần chuẩn hóa lại trên top-5, không lấy trọng số từ softmax top-20 rồi bỏ 15 câu mà không chuẩn hóa. Không lấy softmax của trung bình logits; không dùng sigmoid score thay cho raw ranking logits trong softmax. Cố định temperature=1 cho bản đầu, không thêm một chiều tuning ngoài Proposal.

M0 dùng trọng số đều và chọn câu bằng BM25; M1 đổi cả tập câu lẫn trọng số. Vì vậy M1–M0 đo thay đổi tổng thể. A1, triển khai ở bước đối chứng sau, mới tách được tác động riêng của chọn câu và gán trọng số. M1 không phải huấn luyện phối hợp; đó là M2.

## 2. Những phần đã có và khoảng trống

| Thành phần hiện có | Cách sử dụng cho M1 | Việc cần bổ sung |
|---|---|---|
| `data/sentences.py`, `data/prepare_m0.py` | Cùng câu, ID, offset, split và checksum | Hoàn tất pilot trước khi khóa dữ liệu |
| `retrieval/context_bm25.py` | Dùng nguyên tập BM25 top-20 trong từng context | Không tạo một corpus toàn cục mới |
| `data/evidence_alignment.py` | Gold sentence IDs và trạng thái alignment | Tạo mask giám sát ranking, báo mẫu đủ điều kiện |
| `data/word_segmentation.py` | VnCoreNLP đã pin và bridge đã kiểm tra | Dùng cùng quy trình cho cả hai encoder |
| `training/m0_dataset.py` | Tham khảo/tách phần tokenization cặp 256/96 | Dataset reranker top-20 và dataset verifier có trọng số |
| `models/verifier.py` | Pair classifier ba lớp, NLL theo claim | Weighted pooling ổn định trong log-space |
| `training/train_m0.py` | Lịch học, chọn checkpoint, logging, accumulation | Trainer reranker riêng; đóng băng/cache; snapshot đầy đủ epoch 3 |
| `evaluation/evaluate_m0.py` | Công thức verdict/retrieval metrics | Đọc xếp hạng mới, báo trước/sau reranking trên cùng mẫu |
| `m0_config.py`, cấu hình M0 | Backbone/revision, seed, token budget | Cấu hình M1 riêng; không sửa M0 thành weighted pooling |

Các đường dẫn module trong bảng tính từ `src/factcheck/`.

**Điểm cần tránh khi tái sử dụng:** `M0Dataset` kiểm tra schema ứng viên và thứ tự BM25 rất chặt, nên không đưa danh sách reranked vào rồi giả làm BM25. `m0_runtime.verify_prepared()` cũng chỉ kiểm tra giao thức M0. Cần contract M1 hoặc adapter có kiểm tra rõ ràng; chỉ tách helper dùng chung khi đã có regression test bảo vệ M0.

M0 mới hoàn tất code/smoke, chưa có run đầy đủ. Không cần chờ huấn luyện M0 đủ ba seed mới viết M1; M1 là nơi chọn learning rate chung trước các run so sánh cuối.

## 3. Audit giám sát reranker từ dữ liệu hiện có

Đếm trực tiếp từ input top-20 và targets hiện tại, không đọc test:

| Nhóm | Train | Validation |
|---|---:|---:|
| Tổng claim | 5.062 | 723 |
| Supported + Refuted | 3.409 | 500 |
| S/R có gold căn chỉnh hợp lệ | 1.873 | 271 |
| Có ít nhất 1 câu gold và 1 câu ngoài gold trong top-20 | 1.871 | 271 |
| S/R aligned nhưng top-20 không có câu gold | 2 | 0 |
| S/R chưa căn chỉnh đầy đủ/chắc chắn | 1.536 | 229 |
| NEI | 1.653 | 223 |

Trong 1.871 claim train đủ điều kiện: 2.627 cặp dương, 26.189 cặp âm theo chú thích. Các số này phải được tính lại nếu thay bộ tách câu hoặc alignment. Các con số validation dùng để tính metric/lựa chọn checkpoint, không dùng để cập nhật gradient.

Quy tắc giám sát:

- Câu thuộc gold của SUPPORTED hoặc REFUTED đều là **positive cho chọn bằng chứng**. Một câu bác bỏ claim REFUTED không phải negative của reranker chỉ vì nó bác bỏ claim.
- Các câu khác trong top-20 là negative theo chú thích, không khẳng định chúng hoàn toàn vô ích.
- Chỉ tính ranking loss cho claim có alignment hợp lệ và có cả positive lẫn negative trong ứng viên.
- NEI, unmatched, ambiguous và partial không bị biến thành toàn bộ negative.
- BM25 không lấy được gold thì không chèn gold vào top-20. Mẫu đó vẫn được dùng học verdict.
- Verifier vẫn học trên **đủ 5.062 claim**, bao gồm mọi nhãn và mọi trạng thái alignment.

Coverage S/R hiện khoảng 55%, là giới hạn quan trọng của cả giám sát và evaluation. Trước chạy chính thức, hoàn tất rà soát 100 context pilot, xem lỗi alignment điển hình. Chỉ sửa quy tắc dựa trên train/validation, có version và căn cứ; không ép fuzzy match thành gold. Nếu thay preprocessing, chuẩn bị lại cả M0/M1 và dùng cùng mẫu số retrieval. Không yêu cầu đạt một coverage tùy ý bằng cách nới tiêu chuẩn.

## 4. Các bước triển khai

### Bước 1 — Khóa đầu vào và tạo dữ liệu ranking

**Việc làm**

1. Kiểm tra manifest, checksum, ID và split; dùng cùng BM25 top-20 với M0.
2. Tạo targets ranking riêng với danh sách candidate IDs, mask positive/negative, `rank_loss_eligible` và lý do bỏ ranking loss.
3. Giữ gold ngoài dữ liệu inference. Xác định nhãn positive bằng sentence ID, không bằng thứ hạng BM25.
4. Định nghĩa schema M1 rõ ràng cho `bm25_score`, `reranker_score`, `reranker_rank`, `selected_weight`; không ghi đè điểm BM25.
5. Lưu audit theo nhãn, độ dài context, alignment và số positive/negative thực dùng.

**Nghiệm thu:** số mẫu khớp audit mục 3; mọi câu thuộc context tương ứng; thay gold/label không thay input của reranker lúc suy luận. Mẫu thiếu gold vẫn có đường đi đến verifier.

### Bước 2 — Xây và huấn luyện reranker

**Kiến trúc**

- PhoBERT-base-v2 cùng commit với M0 + head một logit cho mỗi cặp claim–câu.
- VnCoreNLP word segmentation; cặp tối đa 256 subword token gồm special tokens; claim tối đa 96.
- Chấm điểm mọi câu BM25 top-20. Batch theo claim; chia microbatch cặp nếu cần bộ nhớ, nhưng không đổi định nghĩa loss.

**Loss đề xuất cụ thể** — Proposal yêu cầu cân bằng dương/âm nhưng chưa ấn định công thức. Chọn balanced binary cross-entropy theo claim để có quy tắc triển khai rõ ràng:

```text
L_rank(c) = 0.5 * mean_{i thuộc positive} softplus(-s_i)
          + 0.5 * mean_{j thuộc negative} softplus( s_j)
L_rank(batch) = mean trên các claim đủ điều kiện
```

Dùng BCE-with-logits hoặc softplus để ổn định số học. Không cộng mọi cặp rồi chia theo số cặp toàn batch vì context dài sẽ có trọng số lớn hơn. Batch không có claim đủ điều kiện không tạo optimizer/scheduler step cho reranker; trong M2 sau này vẫn có thể có verdict step.

**Huấn luyện/lựa chọn**

- Tối đa 3 epoch trên train hợp lệ.
- Quy định rõ **MRR@20 validation** để chọn reranker, tie chọn epoch sớm hơn. Đây là cách cụ thể hóa “MRR validation” trong Proposal cho danh sách 20 ứng viên.
- Mẫu số là mọi S/R validation căn chỉnh hợp lệ, kể cả trường hợp BM25 miss; không chỉ đánh giá các mẫu đủ ranking loss. Không tìm được gold trong top-20 thì reciprocal rank bằng 0.
- Kèm Hit@5, Recall@5, MRR@5, complete-gold@5 và coverage. Nếu không có validation gold hợp lệ thì báo lỗi, không tự chọn bằng train loss.

**Nghiệm thu:** giảm loss trên một tập nhỏ; score/gradient hữu hạn; positive REFUTED xử lý đúng; lưu/nạp model cho cùng score ở eval. Có một run reranker seed 42 hoàn chỉnh trước khi ghép verifier.

### Bước 3 — Đóng băng reranker và tạo cache M1

**Việc làm**

- Nạp checkpoint reranker tốt nhất, gọi `eval()` và `requires_grad_(False)`; chấm điểm với `no_grad()`/`inference_mode()`.
- Sắp xếp toàn bộ top-20 theo score giảm dần; hòa điểm theo index câu tăng dần, rồi lấy tối đa 5.
- Softmax raw scores của các câu được chọn; context chỉ có n<5 câu thì chuẩn hóa trên n câu thật.
- Lưu toàn bộ danh sách reranked top-20 để đánh giá, cùng top-5 và trọng số cho verifier. Gắn cache với hash checkpoint, inputs, tokenizer/preprocessing và cấu hình; cache cũ không hợp lệ nếu một thành phần thay đổi.
- Chạy inference/cache cho mọi claim train/validation, không chỉ những claim có giám sát ranking.

**Nghiệm thu:** trọng số không âm, tổng bằng 1; 1 câu có trọng số 1; padding không nhận trọng số. Thay gold không làm đổi cache. Tập sentence IDs top-20 trước/sau reranking phải giống nhau; do đó Hit@20/Recall@20/complete-gold@20 không đổi, còn MRR@20 có thể đổi. Không đo improvement Hit@20 do chỉ hoán vị cùng 20 câu.

### Bước 4 — Weighted verifier và lịch học M1

**Việc làm**

- Tạo `M1Dataset`/collator đọc top-5 + trọng số đúng thứ tự; claim labels từ targets riêng.
- Dùng pair verifier ba lớp như M0, nhưng pooling:

```text
log P(y) = logsumexp_i(log(w_i) + log_softmax(pair_logits_i)[y])
```

- Tính NLL một lần mỗi claim, trung bình theo claim. Có test weights đều phải khớp M0 trên cùng cặp; tính tay ví dụ có trọng số khác nhau.
- Khởi tạo verifier từ cùng pretrained và cùng snapshot head theo seed như M0/A1, **không dùng checkpoint smoke hoặc verifier M0 đã fine-tune**.
- Reranker không nằm trong optimizer verifier; test hash tham số trước/sau train chứng minh nó không đổi. Cố định eval mode còn tránh dropout làm thay cache dù không có gradient.
- Verifier M1 học 3 epoch khởi tạo + 2 epoch tiếp tục, dùng cùng lịch optimizer/scheduler liên tục với M0. Chọn best Macro-F1 validation trên các checkpoint đã đánh giá, tie chọn sớm hơn.

**Chuẩn bị đúng điểm xuất phát cho M2/A2**

Ngoài checkpoint best, lưu riêng **trạng thái cuối epoch 3**: reranker đã chọn, verifier, optimizer/scheduler verifier, RNG Python/NumPy/Torch/CUDA, DataLoader generator/order, config và hash dữ liệu. Từ snapshot này tạo ba nhánh M1/M2/A2 học thêm 2 epoch. Không thay bằng best cuối epoch 5 của M1. M1 tiếp tục chỉ cập nhật verifier; M2/A2 sẽ có trainer riêng sau.

M0 trainer hiện chỉ lưu best model/tokenizer, chưa lưu đầy đủ trạng thái fork. Cần bổ sung snapshot cho trainer M1 và tách helper dùng chung khi có kiểm tra. Cùng seed không tự bảo đảm cùng head verifier nếu đã tiêu thụ RNG trong lúc train reranker; dùng snapshot khởi tạo verifier riêng theo seed và DataLoader generator độc lập.

**Nghiệm thu:** forward/backward, probability sums, pooling, gradient isolation, save/reload đều đúng; nhánh M1 chạy liên tục 5 epoch và resume từ snapshot epoch 3 cho kết quả tương đương trong môi trường kiểm tra xác định.

### Bước 5 — Chạy thử và chọn learning rate chung

1. Smoke test offline bằng mô hình nhỏ để kiểm tra contract; smoke thật 16–32 claim bằng PhoBERT/VnCoreNLP để đo tài nguyên.
2. Tại seed 42, thử **hai cấu hình LR chung** `1e-5` và `2e-5`. Đề xuất mỗi cấu hình dùng cùng LR cho giai đoạn reranker và verifier, không mở rộng thành grid 2×2 ngoài Proposal.
3. Với mỗi LR: train reranker tối đa 3 epoch → chọn bằng MRR@20 → freeze/cache → train verifier 3+2 epoch → chọn bằng Macro-F1 validation.
4. Chọn LR theo Macro-F1 validation tốt nhất của M1; nếu hai LR bằng điểm, ưu tiên `1e-5` theo quy tắc định trước. Tie giữa epoch vẫn chọn mốc sớm hơn.
5. Ghi kết quả của cả hai LR và chi phí; khóa cấu hình chung rồi chạy M0 bằng LR đã chọn. Không dùng kết quả M0 smoke để so với M1 đã học đủ.

Đây là đề xuất làm rõ những chi tiết Proposal chưa quy định, cần được ghi vào protocol trước thực nghiệm. Không có lambda trong M1: ranking loss chỉ dùng khi train reranker riêng; verifier phase chỉ dùng verdict loss. Lambda dành cho các cấu hình phối hợp sau.

**Nghiệm thu:** hai run chọn LR seed 42, config đã khóa và đường dẫn checkpoint/cache truy vết được. Cấu hình sau này phải dùng cùng đơn vị batch hiệu dụng theo claim, token budget, dữ liệu và thời gian verifier.

### Bước 6 — Ba seed, đánh giá và phân tích đối chứng

- Chạy LR đã chọn với seed 13, 42, 2026; mỗi seed có reranker riêng được train trên train và chọn checkpoint bằng validation. Có thể dùng lại run seed 42 nếu đúng hoàn toàn protocol cuối, có đủ artefact và không thay preprocessing.
- Hoàn tất M0 cùng thiết lập chung để so sánh có ý nghĩa. Báo rõ M1 có thêm chi phí train/inference reranker; không tuyên bố hai mô hình có cùng tổng compute chỉ vì cùng số epoch verifier.
- Test chỉ chạy sau khi khóa giao thức, chuẩn bị ở thư mục riêng và không dùng để chọn checkpoint.
- Verdict metrics trên toàn bộ tập hợp lệ: Macro-F1, accuracy, precision/recall/F1 từng lớp, confusion matrix, mean/std qua seed.
- Retrieval metrics trước/sau trên cùng tập S/R aligned; không đổi mẫu số theo model. Lưu câu/score/weight để truy vết BM25 miss, top-5 miss và verifier error.
- Báo cả chi phí reranker + verifier khi inference online. Số đo chỉ chạy verifier với cache phải có nhãn riêng, không dùng làm tổng độ trễ M1 so với M0.
- Paired bootstrap theo cụm context/bài báo 2.000 lần cho chênh lệch Macro-F1 và F1 REFUTED khi có đủ predictions M0/M1; mean/std seed báo riêng.
- A1 là bước mở rộng sau M1: cùng cache câu, thay trọng số đều và huấn luyện verifier riêng để phân tích tác động của weighting. Không đưa A1 vào điều kiện chạy được M1.

**Nghiệm thu:** ba run chính thức và tái tính được bảng từ prediction files; số liệu không cần chứng minh M1 tốt hơn M0 mới được coi là thí nghiệm hoàn thành.

## 5. Module dự kiến

Đây là các file đề xuất chưa tồn tại, không phải các lệnh đã chạy được.

| File dự kiến | Trách nhiệm |
|---|---|
| `src/factcheck/m1_config.py` | Giao thức M1, hai giai đoạn, seed/LR, checkpoint và cache |
| `configs/experiments/m1.yaml` | Cấu hình thực nghiệm |
| `src/factcheck/models/reranker.py` | PhoBERT + scalar head, balanced ranking loss |
| `src/factcheck/training/reranker_dataset.py` | Top-20, mask ranking, lọc giám sát hợp lệ |
| `src/factcheck/training/train_reranker.py` | Train và chọn reranker theo MRR@20 validation |
| `src/factcheck/retrieval/rerank.py` | Inference reranker, top-5, weights, cache top-20 |
| `src/factcheck/training/m1_dataset.py` | Input verifier có trọng số, tách gold |
| `src/factcheck/models/weighted_verifier.py` | Weighted probability pooling và verdict NLL |
| `src/factcheck/training/train_m1.py` | Freeze reranker, train verifier, snapshot epoch 3 |
| `src/factcheck/inference/predict_m1.py` | Claim/context → BM25 → rerank → weighted verdict, không mở gold |
| `src/factcheck/evaluation/evaluate_m1.py` | Adapter metrics và báo trước/sau reranking |
| `scripts/run_m1.sh` | Các lệnh prepare-ranking/train-reranker/rerank/train/predict/evaluate |

Ưu tiên tái sử dụng công thức metrics và tokenizer cặp; không chép nguyên evaluator/trainer M0 thành nhiều bản khó đồng bộ. Kiểm tra riêng ranking loss, cache provenance, weighting, freeze, resume và integration. Regression tests M0 phải tiếp tục pass sau khi tách helper chung.

## 6. Thứ tự làm việc đề nghị

1. **Đầu tiên:** dataset ranking và audit; giải thích một claim REFUTED với positive/negative cụ thể.
2. **Tiếp theo:** reranker chạy một batch rồi train thật seed 42.
3. **Sau đó:** xem top-20/top-5 trước và sau rerank, kiểm tra weights và đóng băng model.
4. **Kế tiếp:** weighted verifier, snapshot epoch 3 và smoke end-to-end.
5. **Cuối cùng:** chọn LR, chạy M0/M1 đầy đủ, ba seed, đánh giá test theo protocol.

Dự kiến tổ chức khoảng 8–12 buổi cho code và kiểm tra kỹ thuật nếu tận dụng được M0; thời gian huấn luyện đầy đủ tính riêng sau benchmark. Môi trường đã kiểm tra mới nhận CPU; GPU 16–24 GB là tài nguyên dự kiến trong Proposal, chưa được xác nhận có sẵn. Không hứa mốc hoàn thành thực nghiệm dựa vào thời gian smoke.

## 7. Checklist nghiệm thu

- [ ] Cùng câu/BM25 top-20, preprocessing và nguồn dữ liệu với M0.
- [ ] Pilot và alignment có audit, coverage; không đưa gold vào retrieval lúc dự đoán.
- [ ] Ranking targets đúng cho REFUTED; NEI/unmatched không thành toàn-negative.
- [ ] Balanced ranking loss chỉ tính khi có cả positive và negative.
- [ ] Reranker chọn bằng MRR@20 validation, không loại BM25 misses khỏi metric.
- [ ] M1 cố định reranker và cache có checksum; score/weights truy vết được.
- [ ] Softmax weights trên top-5 thực, weighted mean probabilities, NLL theo claim.
- [ ] Verifier học toàn bộ train; cùng khởi tạo và thứ tự mẫu theo seed.
- [ ] Snapshot epoch 3 đầy đủ, resume/fork có kiểm tra cho M1/M2/A2 sau này.
- [ ] Chọn LR chung bằng M1 seed 42, rồi M0/M1 cùng protocol cuối.
- [ ] Ba seed, test chỉ cuối quy trình; metric và chi phí không trộn smoke với run thật.
- [ ] Dự đoán lưu lại đủ để tái đánh giá và phân tích từng giai đoạn.
