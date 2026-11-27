# ViFactCheck — phân tích test mBERT

**M2 tăng +10.41 điểm phần trăm Macro-F1 so với M0 ở seed 42**, từ 58.20% lên 68.61%. Cặp này dùng cùng 1.447 claim. Khoảng tin cậy 95% cho chênh lệch là **7.46 đến 13.34 điểm** theo bootstrap ghép cặp 2.000 lần trên 758 bài báo. Khoảng này mô tả độ bất định trên tập test đối với checkpoint của seed 42.

## Kiểm tra đầu vào

- Đã giải nén ZIP. Cả bốn file predictions có 1.447 ID duy nhất và khớp claim, nhãn gold, bài báo, alignment cùng tập ứng viên BM25 theo từng ID. Tính lại Macro-F1, accuracy, precision/recall/F1 theo nhãn, confusion matrix và các chỉ số retrieval bên dưới. Tất cả khớp với metrics lưu trong ZIP.
- SHA-256 của dữ liệu test do server ghi khớp với manifest mBERT. Lượt inference dùng commit `dc5879042b89508ae1c9bc58bf14fcce684630b0`. M0 của cả hai seed và M2 seed 42 khớp file chọn checkpoint. M2 seed 42 được chọn ở continuation epoch 4.
- **M2 seed 13 cần giữ riêng.** Checkpoint dùng trong inference có SHA-256 bắt đầu bằng `356be646`. Bản warmup epoch 3 trên Mac bắt đầu bằng `80261b1d`, nên chưa xác nhận được epoch hoặc stage của checkpoint inference. Hồ sơ training cũ còn ghi lỗi `Verification loss failed to reach the reranker`. Điểm test seed 13 có thể dùng như kết quả bổ sung kèm tình trạng này. Cần metadata chọn checkpoint khớp SHA-256 và trạng thái hậu kiểm của lượt tương ứng để xác nhận thêm. [Checksum lượt inference](backbone-mbert-test-two-seeds/checkpoints.sha256), [hồ sơ provenance và lỗi training đã ghi nhận](backbone-mbert-test-two-seeds/PROVENANCE.json)

| Seed / model | SHA-256 ghi khi inference | Stage và epoch đã xác nhận |
|---|---|---|
| 42 / M0 | `f9caf17f0d77a9f9bcb9e6ebb5bb45ee950317e096e0e8ae12fd21b7401ae5ae` | m0, epoch 5 |
| 42 / M2 | `1a6b0c7dbf3f5270a2307f8e665b572c1abe9b22cde8797e631ea7d8e6dd4799` | continuation, epoch 4 |
| 13 / M0 | `8f1882ab92a1d135f59cdd76daa0ca7651e33947f187643d0ae53b89dc8dd0f8` | m0, epoch 3 |
| 13 / M2 | `356be646d775177415d270763af3b48b3859ccebba68f4a58c7ee6446404b4c4` | Cần metadata khớp SHA-256 |

## Verdict trên 1.447 claim

| Seed | M0 Macro-F1 (%) | M2 Macro-F1 (%) | Δ Macro-F1 (điểm %) | M0 accuracy (%) | M2 accuracy (%) | Phạm vi báo cáo |
|---|---:|---:|---:|---:|---:|---|
| 42 | 58.20 | 68.61 | +10.41 | 58.12 | 68.76 | Kết quả chính |
| 13 | 57.78 | 67.97 | +10.19 | 58.19 | 68.00 | Bổ sung, checkpoint M2 cần đối chiếu |

Ở seed 42, M2 sửa **305** ca M0 đoán sai và làm sai **151** ca M0 đoán đúng, tức thêm ròng **154** dự đoán đúng. Trên seed 13, số ca tương ứng là **311**, **169** và **142**.

### F1 theo nhãn

| Seed | Nhãn gold | Số claim | M0 F1 (%) | M2 F1 (%) | Δ F1 (điểm %) |
|---|---|---:|---:|---:|---:|
| 42 | SUPPORTED | 508 | 60.48 | 69.74 | +9.27 |
| 42 | REFUTED | 468 | 55.16 | 66.43 | +11.27 |
| 42 | NOT_ENOUGH_INFO | 471 | 58.96 | 69.66 | +10.70 |
| 13 | SUPPORTED | 508 | 62.65 | 66.39 | +3.74 |
| 13 | REFUTED | 468 | 53.66 | 67.66 | +14.00 |
| 13 | NOT_ENOUGH_INFO | 471 | 57.05 | 69.86 | +12.81 |

Ở seed 42, `REFUTED` có mức tăng F1 lớn nhất. **Precision tăng từ 51,48% lên 76,03%** vì số claim `SUPPORTED` hoặc `NOT_ENOUGH_INFO` bị gán nhầm thành `REFUTED` giảm từ **262 xuống 87**. **Recall gần như giữ nguyên** ở **59,40% → 58,97%**, tương ứng **278 → 276** claim `REFUTED` đoán đúng. Hai nhãn `SUPPORTED` và `NOT_ENOUGH_INFO` lần lượt có thêm **98** và **58** claim đúng.

### Confusion matrix

Các hàng là nhãn gold, ba cột số là nhãn mà mô hình dự đoán.

| Seed | Model | Gold | Dự đoán SUPPORTED | Dự đoán REFUTED | Dự đoán NOT_ENOUGH_INFO |
|---|---|---|---:|---:|---:|
| 42 | M0 | SUPPORTED | 280 | 127 | 101 |
| 42 | M0 | REFUTED | 85 | 278 | 105 |
| 42 | M0 | NOT_ENOUGH_INFO | 53 | 135 | 283 |
| 42 | M2 | SUPPORTED | 378 | 48 | 82 |
| 42 | M2 | REFUTED | 107 | 276 | 85 |
| 42 | M2 | NOT_ENOUGH_INFO | 91 | 39 | 341 |
| 13 | M0 | SUPPORTED | 348 | 69 | 91 |
| 13 | M0 | REFUTED | 140 | 231 | 97 |
| 13 | M0 | NOT_ENOUGH_INFO | 115 | 93 | 263 |
| 13 | M2 | SUPPORTED | 317 | 90 | 101 |
| 13 | M2 | REFUTED | 71 | 317 | 80 |
| 13 | M2 | NOT_ENOUGH_INFO | 59 | 62 | 350 |

## Retrieval trên 539 claim đủ điều kiện

Tập này gồm claim `SUPPORTED` hoặc `REFUTED` có evidence gold được align hợp lệ. Hit@5 là tỷ lệ có ít nhất một câu gold trong top 5. Recall@5 là tỷ lệ câu gold được thu hồi, lấy trung bình theo claim. Đủ gold@5 yêu cầu toàn bộ câu gold có trong top 5. MRR là trung bình nghịch đảo thứ hạng của câu gold đầu tiên. Bảng biểu diễn mọi chỉ số trên thang 0–100.

| Seed | Chỉ số | M0 | M2 | Δ (điểm trên thang 0–100) |
|---|---|---:|---:|---:|
| 42 | Hit@5 | 97.77 | 98.70 | +0.93 |
| 42 | Recall@5 | 94.60 | 94.71 | +0.10 |
| 42 | Đủ gold@5 | 90.72 | 90.17 | -0.56 |
| 42 | MRR | 94.33 | 94.70 | +0.37 |
| 13 | Hit@5 | 97.77 | 98.14 | +0.37 |
| 13 | Recall@5 | 94.60 | 94.75 | +0.15 |
| 13 | Đủ gold@5 | 90.72 | 91.28 | +0.56 |
| 13 | MRR | 94.33 | 95.30 | +0.97 |

Cả bốn lượt có **Hit@20 = 100%** và **Recall@20 = 99,74%**. Ở seed 42, Recall@5 chỉ tăng **0,10 điểm** và tỷ lệ đủ gold@5 giảm **0,56 điểm**. Verdict tăng trong khi retrieval thay đổi ít. So sánh M0–M2 gộp cả tác động của reranking, pooling theo trọng số và huấn luyện, nên đây là kết quả ở cấp hệ thống.

## Hai claim minh họa ở seed 42

### M0 sai, M2 đúng — `test:18813035:6689`

Claim nói nghệ sĩ Tấn Beo **đã ngừng diễn ở tuổi 58** và từng theo cha đi diễn từ nhỏ. Gold là **`REFUTED`**.

- **M0** dự đoán `NOT_ENOUGH_INFO` với **40,20%**. Top 5 có 2/3 câu gold, bao gồm câu #27 nói ông vẫn lưu diễn ở tuổi 58. Mỗi câu có trọng số **20%**.
- **M2** dự đoán `REFUTED` với **57,72%**. Top 5 có 3/3 câu gold, câu #27 được gán **35,96%** trọng số và câu #29 về việc đi diễn cùng cha được gán **63,11%**. Ca này cho thấy cả danh sách được chọn lẫn trọng số đã thay đổi.

### M0 đúng, M2 sai — `test:19025391:1231`

Claim nói các kiến trúc sư và chuyên gia **nhất trí ủng hộ** đề xuất lắp mái tôn dọc đường Lê Lợi. Gold là **`REFUTED`**.

- **M0** dự đoán `REFUTED` với **75,41%**. Câu gold #1 ghi nhận ý kiến của TS Võ Kim Cương rằng phương án dùng tôn không phù hợp. Câu này đứng hạng 3, có trọng số **20%**.
- **M2** dự đoán `NOT_ENOUGH_INFO` với **44,88%**. Reranker đẩy câu gold #1 xuống hạng 8, ngoài top 5, và gán **61,29%** trọng số cho câu #0. Câu #0 cũng nhận xét đề xuất thiếu thuyết phục, nên ca này vừa cho thấy mất câu gold được gán nhãn, vừa để ngỏ lý do verifier quyết định `NOT_ENOUGH_INFO`.

Tìm hai ID trên trong `seed-42/m0_test_predictions.jsonl` và `seed-42/m2_test_predictions.jsonl` bên trong ZIP inference lưu cùng hồ sơ benchmark. [SHA-256 của ZIP và từng file predictions](backbone-mbert-test-two-seeds/PROVENANCE.json)

## Bảng cho paper — Test Macro-F1, seed 42

| Backbone | M0 (%) | M2 (%) | Δ M2 − M0 (điểm %) |
|---|---:|---:|---:|
| PhoBERT-base | 56.92 | 70.16 | +13.24 |
| mBERT | 58.20 | 68.61 | +10.41 |

Hai run dùng cùng revision ViFactCheck và cùng SHA-256 của **test Parquet gốc**, với cùng phân bố nhãn trong 1.447 claim. SHA-256 của file test **đã prepare** khác nhau. Bảng thể hiện mức tăng M2 so với M0 trong từng run. Điểm M2 của PhoBERT cao hơn mBERT **1,55 điểm** ở seed 42. Đối chiếu này mang tính mô tả vì ngoài backbone, quy trình xử lý input cũng khác. Kết quả XLM-R sẽ được thêm khi có test artifacts. [Metrics PhoBERT M0](listwise-best5-v2-three-seeds-20261112/seeds/seed42/m0_test_metrics.json), [metrics PhoBERT M2](listwise-best5-v2-three-seeds-20261112/seeds/seed42/m2_test_metrics.json)

## Results — đoạn tiếng Anh cho paper

On the official ViFactCheck test split of 1,447 claims, mBERT M2 achieved 68.61% Macro-F1 and 68.76% accuracy at seed 42, compared with 58.20% and 58.12% for M0. The Macro-F1 gain was 10.41 percentage points, with a 95% paired article-cluster bootstrap interval of 7.46 to 13.34 points. REFUTED showed the largest absolute F1 gain, with precision increasing from 51.48% to 76.03% and recall remaining near 59%. On the 539 claims eligible for retrieval evaluation, Hit@5 increased from 97.77% to 98.70%, while sentence Recall@5 increased from 94.60% to 94.71%. The corresponding within-backbone Macro-F1 gain for the PhoBERT reference at seed 42 was 13.24 points. Both runs used the same pinned raw test split, with backbone-specific input processing.
