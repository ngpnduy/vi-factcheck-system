# Giải thích nhánh `Rush-version`

> Đây là tóm tắt bằng tiếng Việt về toàn bộ những gì bạn mình đã làm trên nhánh này. Đọc từ trên xuống dưới là ổn nhất.

---

## 1. Bức tranh tổng thể — Bạn mình làm gì?

Bạn mình **xây một hệ thống fact-checking hoàn chỉnh từ đầu đến cuối**, chạy được trên laptop GPU cá nhân (RTX 3050 4GB VRAM). Toàn bộ code được đặt trong thư mục `rush/` — tách biệt hoàn toàn với code gốc của nhóm ở `src/`.

Hệ thống gồm **3 mô hình** (M0, M1, M2) — mỗi cái thông minh hơn cái trước một chút — và đã **chạy thực nghiệm pilot thật sự** với kết quả cụ thể.

---

## 2. Ba mô hình M0 / M1 / M2

Cả 3 đều dùng **PhoBERT** (mô hình ngôn ngữ tiếng Việt của VinAI) làm backbone.

Pipeline chung của cả 3:
```
Claim (câu cần kiểm chứng)
    → BM25 lấy top-20 câu ứng viên từ bài báo
    → Chọn top-5 câu liên quan nhất
    → PhoBERT mã hóa từng cặp (claim, câu)
    → Gộp xác suất → ra nhãn: SUPPORTED / REFUTED / NOT_ENOUGH_INFO
```

### M0 — Baseline đơn giản nhất
- Lấy **top-5 câu từ BM25** (không có reranker).
- **Gộp đều** (uniform average) xác suất từ 5 câu đó.
- Chỉ có 1 encoder PhoBERT cho verifier.

### M1 — Thêm Reranker
- Huấn luyện thêm **1 encoder PhoBERT riêng** làm reranker (xếp hạng lại 20 câu → chọn top-5 tốt hơn).
- Khi infer: reranker **đóng băng** (không cập nhật thêm), chỉ verifier học tiếp.
- **Gộp theo softmax weight** thay vì gộp đều — câu nào reranker chấm cao thì được cộng nhiều hơn.

### M2 — Joint Training (mạnh nhất)
- Giống M1 nhưng **cả 2 encoder đều học cùng nhau** (joint training).
- Thêm **evidence loss** (BCE loss) để ép reranker học phân biệt câu bằng chứng đúng/sai.
- Gradient từ verification loss **chảy ngược qua** reranker scores.

> **Tóm lại**: M0 < M1 < M2 về độ phức tạp và (thường) về hiệu suất.

---

## 3. Kết quả thực nghiệm Pilot (quan trọng!)

**Setup thực nghiệm:**
- Dữ liệu: 900 claim train (300 mỗi nhãn), toàn bộ 723 dev + 1,447 test.
- GPU: RTX 3050 Laptop (4GB VRAM).
- Chỉ 1 seed (seed=42) → **chưa phải kết quả cuối cùng**, chỉ là pilot.
- Hyperparameter chưa tune: LR=2e-5, lambda=0.5.

### Kết quả phân loại (Test set — 1,447 claim)

| Model | Test Macro-F1 | Test Accuracy |
|-------|:---:|:---:|
| M0    | 0.5101 | 51.7% |
| M1    | 0.5248 | 54.1% |
| **M2**    | **0.5484** | **56.5%** |

→ **M2 tốt nhất**, Macro-F1 tăng ~4% so với M0.

### Kết quả retrieval (lấy bằng chứng)

| Model | Hit@5 | MRR |
|-------|:---:|:---:|
| M0 | 97.8% | 0.943 |
| M1 | 97.6% | 0.945 |
| M2 | 96.7% | 0.948 |

→ Cả 3 retrieval đều rất tốt (~97-98%); BM25 cơ bản đã đủ lấy bằng chứng.

### Chi tiết từng nhãn (Test set)

| Model | Nhãn | F1 |
|-------|------|:---:|
| M0 | SUPPORTED | 0.584 |
| M0 | REFUTED | 0.443 |
| M0 | NOT_ENOUGH_INFO | 0.504 |
| M1 | SUPPORTED | **0.617** |
| M1 | REFUTED | 0.500 |
| M1 | NOT_ENOUGH_INFO | 0.457 |
| M2 | SUPPORTED | 0.639 |
| M2 | REFUTED | 0.451 |
| M2 | NOT_ENOUGH_INFO | **0.555** |

> ⚠️ **Nhãn REFUTED khó nhất** — cả 3 model đều dưới 0.51 F1. Đây là điểm cần cải thiện.

### Thời gian training & bộ nhớ

| Model | Thời gian train | Peak GPU RAM |
|-------|:---:|:---:|
| M0 | 45.4 phút | 1,812 MiB |
| M1 | 69.0 phút | 2,376 MiB |
| M2 | 98.3 phút | 2,940 MiB |

→ M2 tốn gần **2x thời gian** và RAM so với M0, nhưng vừa với RTX 3050 4GB.

---

## 4. Kỹ thuật đáng chú ý (đọc thêm nếu muốn)

Bạn mình có áp dụng nhiều trick để chạy được trên laptop GPU nhỏ:

| Kỹ thuật | Tác dụng |
|---|---|
| **Mixed precision (FP16)** | Giảm RAM ~50% khi forward pass |
| **Activation checkpointing** | Giảm RAM khi backward bằng cách recompute |
| **Gradient accumulation (8 claims)** | Simulate batch_size=8 dù mỗi lần chỉ chạy 1 |
| **Adafactor optimizer** | Dùng ít RAM hơn Adam/AdamW |
| **LRU cache tokenizer** | Cache kết quả encode → không encode lại câu đã encode |

---

## 5. Sự khác biệt với code gốc của nhóm (`src/`)

Bạn mình dùng **cấu hình khác** so với code nhóm đang có:

| Thứ | Code nhóm (`src/`) | Rush (`rush/`) |
|---|---|---|
| Backbone | `phobert-base-v2` | `phobert-base` |
| Tokenizer | `conservative_vi_v1` | VnCoreNLP |
| BM25 | BM25S Lucene | BM25Okapi |
| Optimizer | AdamW | Adafactor |
| Precision | Float32 | Mixed (FP16/FP32) |
| Checkpointing | Chỉ best model | Model + optimizer + RNG state (resume được) |

> ⚠️ **Hai bên không thể so sánh điểm số trực tiếp** vì dùng khác nhau quá nhiều thứ. Muốn merge cần thống nhất lại config.

---

## 6. Cấu trúc code quan trọng trong `rush/`

```
rush/
├── vifactcheck/          ← Code chính
│   ├── model.py          ← PairEncoder, VerificationSystem (M0/M1/M2)
│   ├── training.py       ← Vòng lặp train, checkpoint, resume
│   ├── data.py           ← Load & chuẩn bị dataset
│   ├── text.py           ← Segmentation, BM25, evidence matching
│   ├── metrics.py        ← Macro-F1, MRR, Hit@k
│   ├── pilot.py          ← Chạy full pipeline M0+M1+M2 một lần
│   ├── cli.py            ← CLI: train / evaluate / predict / compare
│   └── config.py         ← Hyperparameters mặc định
├── results/pilot-900-seed42/   ← Kết quả đã chạy xong
│   ├── REPORT.md         ← Báo cáo kết quả chi tiết ← ĐỌC CÁI NÀY
│   ├── results.csv       ← Bảng tóm tắt
│   ├── comparison.json   ← So sánh M0/M1/M2
│   └── experiment.json   ← Toàn bộ settings đã dùng
├── tests/                ← 29 unit tests (28 passed, 1 skip do thiếu VnCoreNLP)
├── VALIDATION.md         ← Smoke test results (24-train chạy thử)
└── README.md             ← Hướng dẫn chạy đầy đủ
```

---

## 7. Những hạn chế bạn mình tự ghi nhận

Bạn mình **tự note rõ ràng** trong báo cáo:
1. Chỉ 900/5,062 claim train → **chưa đủ dữ liệu**.
2. Chỉ 1 seed → **không có độ lệch chuẩn**, không biết kết quả ổn định không.
3. Hyperparameter **chưa tune** đúng cách.
4. Official splits **share source articles** → train/dev/test có bài báo trùng nhau (không phải lỗi của bạn mình, đây là vấn đề của dataset gốc).
5. REFUTED F1 thấp → model khó phân biệt "bác bỏ".

---

## 8. Việc cần làm tiếp theo (nếu muốn merge vào main)

Theo chính bạn mình ghi trong `BRANCH_NOTES.md`:

- [ ] Thống nhất backbone: `phobert-base` hay `phobert-base-v2`?
- [ ] Thống nhất BM25 và tokenizer
- [ ] Thống nhất optimizer (AdamW vs Adafactor)
- [ ] Chạy full 5,062 claim với 3 seeds
- [ ] Tune hyperparameter đúng cách
- [ ] Tách A1/A2 ablation (để hiểu reranker vs evidence loss đóng góp bao nhiêu)

---

## Tóm lại một câu

> Bạn mình đã **implement và chạy xong** một hệ thống fact-checking 3 model (M0/M1/M2) dựa trên PhoBERT, tối ưu để chạy trên laptop, đạt Macro-F1 ~55% trên test set với dữ liệu nhỏ. Code sạch, có test, có kết quả thực nghiệm đầy đủ — nhưng **chưa sẵn sàng merge** vì dùng config khác với code nhóm.
