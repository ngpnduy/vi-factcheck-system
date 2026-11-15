# 📄 Quy Trình Ra Một Bài Báo Khoa Học
### Dành cho sinh viên năm 3 bắt đầu làm research — từ A đến Z

> [!NOTE]
> Tài liệu này được viết theo ngữ cảnh cụ thể: **3 người, 3 tháng, hướng NLP/AI, hội nghị trong nước (KSE/RIVF/FAIR).**
> Tuy nhiên quy trình tổng thể áp dụng được cho hầu hết các bài báo Applied AI.

---

## 🗺️ Bức Tranh Tổng Quan

```
[1. Chọn đề tài & Đọc paper]
        ↓
[2. Xác định Research Gap]
        ↓
[3. Viết Proposal (Đề cương)]
        ↓
[4. Thu thập & Xử lý dữ liệu]
        ↓
[5. Xây dựng & Chạy thử nghiệm]
        ↓
[6. Phân tích kết quả]
        ↓
[7. Viết paper]
        ↓
[8. Review nội bộ & Sửa]
        ↓
[9. Submit]
        ↓
[10. Phản hồi reviewers (Rebuttal)]
```

---

## GIAI ĐOẠN 1 — Chuẩn Bị (Tuần 1–2)

### Bước 1: Chọn đề tài & Xác định phạm vi

**Câu hỏi cần trả lời được:**
- Bài toán mình muốn giải là gì? (Fact-checking? Sentiment? QA?)
- Input/Output của bài toán là gì?
- Tại sao bài toán này quan trọng trong thực tế?

**Lỗi hay gặp:** Chọn đề tài quá rộng ("làm AI cho tiếng Việt") hoặc quá hẹp ("cải thiện F1 thêm 0.5%"). Cần một bài toán **cụ thể, đo được, và có ý nghĩa**.

---

### Bước 2: Đọc và tổng hợp tài liệu (Literature Review)

Đây là bước **quan trọng nhất và hay bị bỏ qua nhất**.

#### Công cụ tìm paper:
| Công cụ | Dùng khi nào |
|---|---|
| [Google Scholar](https://scholar.google.com) | Tìm kiếm tổng quát |
| [ACL Anthology](https://aclanthology.org) | Paper về NLP/Linguistics |
| [Semantic Scholar](https://www.semanticscholar.org) | Tìm citation graph |
| [arXiv](https://arxiv.org) | Paper mới nhất (chưa published) |
| [Papers With Code](https://paperswithcode.com) | Tìm SOTA + code |

#### Cách đọc paper hiệu quả (không đọc từ đầu đến cuối):
1. **Abstract** → Bài này làm gì? Kết quả gì?
2. **Introduction** → Tại sao họ làm? Gap là gì?
3. **Related Work** → Họ so sánh với ai?
4. **Table kết quả** → Số liệu của họ là bao nhiêu?
5. **Conclusion / Limitation** → Họ tự nhận họ làm được gì, chưa làm được gì?

> [!TIP]
> Dùng **Zotero** (miễn phí) để quản lý reference. Mỗi paper đọc xong ghi lại 3 dòng: (1) Họ làm gì, (2) Phương pháp chính, (3) Điểm yếu/hạn chế. Sau 2 tuần bạn sẽ có kho tư liệu sẵn để viết Related Work.

#### Số lượng paper cần đọc:
- **Tối thiểu:** 15–20 paper liên quan trực tiếp
- **Bao gồm:** 5 paper baseline chính + 10 paper liên quan + 5 paper phương pháp bạn định dùng

---

### Bước 3: Xác định Research Gap

Research Gap là **khoảng trống** giữa những gì đã được làm và những gì chưa được giải quyết.

**Cách tìm Gap:**
- Đọc phần **"Limitation"** và **"Future Work"** của các bài báo gần nhất → họ tự nói điểm yếu của họ
- Hỏi: *"Bài báo này fail ở đâu?"* (Domain khác? Ngôn ngữ khác? Loại dữ liệu khác?)
- Hỏi: *"Có ai test mô hình X trên dữ liệu Y chưa?"*

**Viết Gap thành 1-2 câu rõ ràng:**
> *"Các nghiên cứu trước (ViFactCheck, AAAI 2025) chỉ đánh giá mô hình trên dữ liệu cùng phân phối. Chưa có nghiên cứu nào đánh giá khả năng tổng quát hóa (generalization) của mô hình khi áp dụng lên nguồn báo mới, khác phong cách viết."*

> [!IMPORTANT]
> Gap phải **chứng minh được bằng trích dẫn**. Nếu bạn nói "chưa ai làm X", bạn phải có ít nhất 3–5 bài báo dẫn chứng rằng họ không làm X.

---

## GIAI ĐOẠN 2 — Đề Cương (Tuần 2–3)

### Bước 4: Viết Proposal (Đề cương nghiên cứu)

Một Proposal tốt gồm các phần:

```
1. Tên đề tài (Working title)
2. Tóm tắt vấn đề (Problem Statement) — 1 đoạn
3. Research Questions (RQ) — 2–3 câu hỏi cụ thể, đo được
4. Đóng góp dự kiến (Contributions) — liệt kê dạng bullet
5. Phương pháp dự kiến (Methodology) — mô tả sơ đồ
6. Dataset sẽ dùng
7. Metrics đánh giá
8. Phân công và Timeline
```

**Ví dụ Contributions viết đúng cách:**
```
Bài báo này đóng góp:
(1) Một tập dataset mới gồm 800 mẫu fact-checking từ [tên báo],
    được gán nhãn thủ công theo schema ViFactCheck.
(2) Một benchmark đánh giá cross-domain đầu tiên cho bài toán
    fact-checking tiếng Việt, bao gồm 4 mô hình PLM và 3 LLM.
(3) Phân tích lỗi chi tiết giúp xác định loại claim nào
    khó kiểm chứng nhất.
```

> [!TIP]
> **Mentor / Thầy hướng dẫn:** Gửi Proposal này cho thầy/cô xem TRƯỚC khi làm. Thầy sẽ cho biết hướng có khả thi không và Gap có thật sự mới không. Đừng code rồi mới hỏi — sẽ mất thời gian.

---

## GIAI ĐOẠN 3 — Dữ Liệu (Tuần 3–6)

### Bước 5: Thu thập dữ liệu

#### Nếu dùng dataset có sẵn:
- Download từ HuggingFace, GitHub của paper gốc
- Đọc kỹ **License** — một số dataset cấm dùng thương mại hoặc yêu cầu xin phép
- Kiểm tra **train/val/test split** — không được dùng test set để tune hyperparameter

#### Nếu tự xây dựng dataset (như trường hợp của nhóm bạn):
Đây là quá trình tốn thời gian nhất. Cần làm đúng quy trình:

**Bước 5a: Xác định Annotation Schema**
- Nhãn là gì? Bao nhiêu nhãn? Định nghĩa từng nhãn rõ ràng.
- Viết **Annotation Guideline** (hướng dẫn gán nhãn) — tài liệu giải thích cho người gán nhãn biết cách làm.

**Bước 5b: Pilot Annotation (Thử nghiệm nhỏ)**
- Lấy ~50 mẫu, mỗi người gán nhãn độc lập
- Tính **Inter-Annotator Agreement (IAA)** — dùng **Cohen's Kappa (κ)**
  - κ > 0.6: Chấp nhận được
  - κ > 0.8: Tốt
  - κ < 0.4: Phải xem lại định nghĩa nhãn

```python
# Tính Cohen's Kappa đơn giản
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(annotator_1_labels, annotator_2_labels)
```

**Bước 5c: Annotate toàn bộ**
- Chia đều công việc
- Mỗi mẫu nên có ít nhất 2 người gán nhãn → lấy majority vote
- Những mẫu bất đồng → họp lại thảo luận

> [!WARNING]
> Không bao giờ dùng kết quả của LLM (ChatGPT, Gemini) làm nhãn ground-truth cho dataset. Điều này vi phạm nguyên tắc nghiên cứu và paper sẽ bị reject nếu reviewer phát hiện.

**Bước 5d: Thống kê dataset**

Bắt buộc phải có trong paper:
- Tổng số mẫu, phân bố theo nhãn (class distribution)
- Số mẫu train/val/test
- Độ dài trung bình của claim, evidence (số token)
- So sánh với dataset gốc (ViFactCheck) nếu có

---

### Bước 6: Tiền xử lý dữ liệu (Preprocessing)

- Loại bỏ HTML tags, ký tự đặc biệt
- Kiểm tra duplicate (trùng lặp)
- Kiểm tra **data leakage** — train và test không được share mẫu tương tự

---

## GIAI ĐOẠN 4 — Thử Nghiệm (Tuần 5–10)

### Bước 7: Thiết kế thử nghiệm (Experimental Design)

Trước khi code, phải trả lời được:
- **Baseline là gì?** — Mô hình đơn giản nhất để so sánh (ví dụ: TF-IDF + Logistic Regression)
- **Mô hình đề xuất của bạn là gì?** — Cái gì mới hơn baseline?
- **Metrics đánh giá là gì?** — Macro F1, Accuracy, Precision, Recall
- **Hyperparameters?** — Learning rate, batch size, số epochs

#### Thiết kế thí nghiệm điển hình cho NLP:

```
Experiment 1: Baseline — TF-IDF + Logistic Regression
Experiment 2: PLM — PhoBERT-base fine-tuned
Experiment 3: PLM — XLM-R-large fine-tuned
Experiment 4: LLM — Gemma-7B zero-shot
Experiment 5: LLM — Gemma-7B few-shot (3-shot)
Experiment 6: LLM — Gemma-7B fine-tuned (LoRA)
[Nếu có phương pháp đề xuất]
Experiment 7: Phương pháp của nhóm
```

> [!TIP]
> Dùng **Weights & Biases (wandb)** hoặc ít nhất là **MLflow** để log kết quả tự động. Không bao giờ ghi kết quả vào Excel tay — dễ nhầm và không reproduce được.

---

### Bước 8: Cài đặt & Chạy thử nghiệm

**Reproduce baseline trước:**
- Chạy lại kết quả của bài báo gốc trên dataset gốc của họ
- Nếu kết quả gần với paper gốc (±1%) → pipeline của bạn đúng
- Nếu lệch nhiều → debug trước khi tiếp tục

**Quy tắc vàng:**
```
Code → Chạy trên 10% data → Kết quả có vẻ đúng → Chạy full
```

Không bao giờ chạy full ngay vì sẽ tốn GPU hours và không phát hiện được bug.

**Reproducibility:**
- Set random seed ở tất cả mọi nơi (`random`, `numpy`, `torch`)
- Lưu lại toàn bộ hyperparameters vào file config (YAML/JSON)
- Lưu checkpoint tốt nhất (best model)

```python
# Bắt buộc có trong tất cả experiment scripts
import random, numpy as np, torch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

---

### Bước 9: Phân tích kết quả (Analysis)

**Ngoài bảng số, paper tốt cần có:**

1. **Error Analysis (Phân tích lỗi):**
   - Lấy ~50 mẫu bị dự đoán sai
   - Phân loại lỗi: Model sai vì lý do gì? (Thiếu bằng chứng? Claim có số liệu? Claim mỉa mai?)
   - Đây là phần giúp bài báo có chiều sâu và thuyết phục reviewer nhất

2. **Ablation Study (Thử nghiệm cắt bỏ):**
   - Nếu bạn propose phương pháp mới có nhiều component, hãy thử bỏ từng phần đi để chứng minh mỗi phần đều cần thiết
   - Ví dụ: "Nếu bỏ bước retrieval, F1 giảm từ 85% xuống 72%"

3. **Statistical Significance:**
   - Chạy mỗi experiment ít nhất 3 lần với seed khác nhau
   - Report mean ± std
   - Dùng **paired t-test** để chứng minh kết quả tốt hơn không phải do may mắn

---

## GIAI ĐOẠN 5 — Viết Paper (Tuần 9–12)

### Bước 10: Cấu trúc một bài báo NLP chuẩn

```
Title (Tiêu đề)
Abstract (Tóm tắt — 150–250 từ)
1. Introduction (Giới thiệu)
2. Related Work (Công trình liên quan)
3. Dataset (Mô tả dữ liệu — nếu có contribution dataset)
4. Methodology (Phương pháp)
5. Experiments (Thử nghiệm)
6. Results & Analysis (Kết quả & Phân tích)
7. Conclusion (Kết luận)
References (Tài liệu tham khảo)
```

---

### Bước 11: Hướng dẫn viết từng phần

#### Abstract (Tóm tắt)
Viết SAU CÙNG, nhưng phải có đủ 4 thứ:
1. **Bối cảnh** (1 câu): Vì sao bài toán này quan trọng?
2. **Gap** (1 câu): Các nghiên cứu trước chưa làm được gì?
3. **Đóng góp** (1–2 câu): Chúng tôi đề xuất gì?
4. **Kết quả** (1 câu): Kết quả đạt được là bao nhiêu?

#### Introduction (Giới thiệu)
Cấu trúc phổ biến nhất (CARS model):
```
Đoạn 1: Bài toán quan trọng vì sao trong thực tế?
Đoạn 2: Các nghiên cứu trước đã làm gì?
Đoạn 3: Họ còn hạn chế gì? (Gap)
Đoạn 4: Chúng tôi đề xuất gì để giải quyết gap đó?
Đoạn 5: Liệt kê Contributions
Đoạn 6: Cấu trúc phần còn lại của paper
```

#### Related Work (Công trình liên quan)
- **Không** chỉ tóm tắt từng paper một cách riêng lẻ
- **Phải** so sánh, đối chiếu: "Khác với [A], chúng tôi..."
- Chia thành các nhóm chủ đề (ví dụ: 2.1 Vietnamese NLP, 2.2 Fact-Checking, 2.3 Evidence Retrieval)

#### Methodology (Phương pháp)
- Phải có **sơ đồ kiến trúc** (Figure/Diagram) — bắt buộc
- Mô tả đủ chi tiết để người khác có thể reproduce
- Mỗi ký hiệu toán học phải được định nghĩa

#### Experiments (Thử nghiệm)
- Mô tả rõ **Dataset**: kích thước, phân phối nhãn, split ratio
- Mô tả **Implementation Details**: model, hyperparameters, hardware (GPU gì, bao nhiêu giờ)
- Mô tả **Baselines**: tại sao chọn các baseline này?
- Mô tả **Metrics**: tại sao dùng Macro F1 thay vì Accuracy?

#### Results & Analysis (Kết quả)
- Bảng kết quả phải có **bold** số tốt nhất
- Viết **nhận xét** bên dưới mỗi bảng — không để bảng "nói chuyện một mình"
- Có phần **Error Analysis** với ví dụ cụ thể (case study)

#### Conclusion (Kết luận)
```
Đoạn 1: Tóm tắt lại đóng góp chính (không copy abstract)
Đoạn 2: Limitation (Hạn chế) — phải thành thật
Đoạn 3: Future Work (Hướng tiếp theo)
```

> [!IMPORTANT]
> **Limitation** không phải điểm yếu của bài báo — nó thể hiện sự trưởng thành học thuật. Reviewer sẽ tôn trọng bạn hơn nếu bạn thành thật về giới hạn của nghiên cứu.

---

### Bước 12: Công cụ viết paper

| Công cụ | Mục đích |
|---|---|
| **Overleaf** | Soạn thảo LaTeX (chuẩn nhất, miễn phí) |
| **Google Docs** | Draft ban đầu, cộng tác |
| **Grammarly** | Kiểm tra ngữ pháp tiếng Anh |
| **DeepL** | Dịch nhanh để tham khảo |
| **draw.io** | Vẽ sơ đồ kiến trúc (miễn phí) |
| **Matplotlib/Seaborn** | Vẽ đồ thị kết quả |

> [!TIP]
> Dùng **ACL Anthology style template** hoặc template của conference bạn target ngay từ đầu. Đừng viết xong rồi mới format lại — rất mất công.

---

## GIAI ĐOẠN 6 — Hoàn Thiện & Submit (Tuần 11–12)

### Bước 13: Review nội bộ

Trước khi submit, cần qua các vòng:

**Vòng 1 — Tự review:**
- [ ] Mọi claim đều có citation?
- [ ] Mọi số liệu trong bảng đều chạy được reproduce?
- [ ] Sơ đồ rõ ràng, có caption?
- [ ] Tài liệu tham khảo đúng format?
- [ ] Đủ số trang theo quy định?

**Vòng 2 — Đọc chéo trong nhóm:**
- Mỗi người đọc phần người khác viết
- Đặt câu hỏi: *"Đọc phần này tôi không hiểu chỗ nào?"*

**Vòng 3 — Thầy hướng dẫn đọc:**
- Gửi bản draft ít nhất **2 tuần trước deadline**
- Không gửi 2 ngày trước và kỳ vọng thầy sửa kịp

---

### Bước 14: Submit

- Đọc kỹ **Call for Papers (CfP)** của conference — chú ý deadline, page limit, format
- Chuẩn bị:
  - PDF bài báo (anonymized nếu double-blind review)
  - Abstract (copy-paste)
  - Keywords (3–5 từ khóa)
  - Author information
- Submit sớm ít nhất **24 giờ** trước deadline — hệ thống hay bị quá tải vào phút chót

---

### Bước 15: Phản hồi Reviewers (Rebuttal)

Nếu conference có vòng rebuttal (phản hồi):
- Đọc kỹ tất cả nhận xét của reviewers
- **Không phản ứng cảm tính** — reviewers anonymous nhưng thường là các chuyên gia có kinh nghiệm
- Với mỗi điểm reviewer raise:
  - Đồng ý → Thừa nhận và giải thích sẽ sửa thế nào
  - Không đồng ý → Lịch sự phản bác bằng evidence cụ thể
- Nếu có thể, **chạy thêm experiment** để trả lời câu hỏi của reviewer

---

## 📅 Timeline Mẫu: 3 Người, 3 Tháng

```
Tuần 1–2  : Đọc paper, xác định Gap, viết Proposal
Tuần 3    : Thầy approve Proposal. Setup môi trường, download data
Tuần 4–5  : Thu thập & annotate dataset mới (nếu có)
Tuần 5–6  : Reproduce baseline của paper gốc
Tuần 6–8  : Chạy các experiments chính
Tuần 8–9  : Phân tích kết quả, Error Analysis
Tuần 9–10 : Viết draft paper (mỗi người viết 1-2 section)
Tuần 11   : Review nội bộ, gửi thầy
Tuần 12   : Sửa theo ý kiến thầy, submit
```

---

## ⚠️ Những Lỗi Phổ Biến Của Sinh Viên Lần Đầu Làm Research

| Lỗi | Hậu quả | Cách tránh |
|---|---|---|
| Code trước, đọc paper sau | Làm lại từ đầu | Đọc đủ 15 paper trước khi code |
| Không log experiment | Không biết số nào đúng | Dùng wandb/MLflow từ đầu |
| Chỉ report kết quả tốt nhất | Paper bị reject vì thiếu tin cậy | Report mean ± std, chạy 3 lần |
| Không có baseline rõ ràng | Không biết mình tốt hơn gì | Luôn có ít nhất 2 baseline |
| Gửi thầy 2 ngày trước deadline | Thầy không đọc kịp | Gửi trước 2 tuần |
| Quên kiểm tra data leakage | Kết quả inflate, bị phát hiện sau | Kiểm tra kỹ split trước khi train |
| Abstract viết quá chung chung | Người đọc không hiểu bài làm gì | Phải có con số cụ thể trong abstract |

---

## 📚 Tài Liệu Bổ Sung Nên Đọc

- **"How to write a great research paper"** — Simon Peyton Jones (Microsoft Research) — YouTube, rất ngắn và súc tích
- **"A Survival Guide to a PhD"** — Andrej Karpathy — blog, thực tế và thẳng thắn
- **ACL Rolling Review Guidelines** — học cách reviewers đánh giá paper
- **"Writing in the Sciences"** — Coursera (Stanford) — miễn phí, dạy viết khoa học

---

*Tài liệu này được tạo cho nhóm nghiên cứu ViFactCheck — Đồ án Tổng hợp 2026.*
