# 📖 Cẩm Nang Thuật Ngữ ViFactCheck & Fact-Checking trong NLP
### Dành cho sinh viên bắt đầu nghiên cứu bài toán Kiểm chứng thông tin tiếng Việt

---

## 1. Bức Tranh Tổng Quan: Bài Toán Fact-Checking Là Gì?

Trong Xử lý ngôn ngữ tự nhiên (NLP), **Fact-Checking (Automated Fact-Checking - AFC)** là bài toán tự động xác minh tính đúng/sai của một phát biểu dựa trên các bằng chứng đáng tin cậy.

Quy trình chuẩn thường phỏng theo khung nghiên cứu quốc tế **FEVER (Fact Extraction and VERification)** gồm 3 giai đoạn:

```
[Claim: Câu nhận định cần kiểm]
             │
             ▼
┌────────────────────────────────────────┐
│ Giai đoạn 1: Evidence Retrieval        │
│ (Truy xuất câu/đoạn làm bằng chứng)    │
│  - Bước 1.1: Document Retrieval (BM25) │
│  - Bước 1.2: Sentence Reranking (SBERT)│
└────────────────────────────────────────┘
             │
             ▼ [Top-K Evidences tìm được]
┌────────────────────────────────────────┐
│ Giai đoạn 2: Claim Verification        │
│ (Xác minh nhận định / Phân loại nhãn)  │
│  - Input: Pair (Claim, Evidence)       │
│  - Model: PLM (PhoBERT) hoặc LLM (LoRA)│
│  - Output: SUPPORTED / REFUTED / NEI   │
└────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│ Giai đoạn 3: Justification/Explanation │
│ (Sinh lời giải thích - Tùy chọn)       │
└────────────────────────────────────────┘
```

---

## 2. Nhóm Thuật Ngữ BẮT BUỘC PHẢI BIẾT (Level 1 - Nền tảng)

Đây là những từ xuất hiện trong **tất cả** bài báo, dataset và file code của hệ thống. Không nắm được nhóm này sẽ không đọc hiểu được paper hay code schema.

### 2.1. Về Dữ liệu (Data Entities)
* **Claim (Nhận định / Phát biểu):**
  * *Định nghĩa:* Câu văn mang tính khẳng định một sự kiện, thông tin nào đó mà ta cần kiểm tra xem có đúng sự thật hay không.
  * *Ví dụ:* `"Việt Nam xuất khẩu 8 triệu tấn gạo trong năm 2023."`
* **Context (Ngữ cảnh / Bài báo gốc):**
  * *Định nghĩa:* Toàn bộ bài báo hoặc văn bản dài có liên quan đến chủ đề của claim, chứa cả thông tin đúng lẫn thông tin nhiễu.
* **Evidence (Bằng chứng):**
  * *Định nghĩa:* Một hoặc vài câu văn cụ thể trích từ Context, cung cấp dữ kiện trực tiếp chứng minh hoặc bác bỏ Claim.
  * *Ví dụ:* `"Theo Bộ Nông nghiệp, tổng lượng gạo xuất khẩu của cả nước năm 2023 đạt hơn 8,1 triệu tấn."`
* **Gold Evidence (Bằng chứng chuẩn):**
  * *Định nghĩa:* Bằng chứng do chuyên gia/người gán nhãn thủ công xác nhận là căn cứ chuẩn mực nhất (Ground Truth).
* **Retrieved Evidence (Bằng chứng truy xuất được):**
  * *Định nghĩa:* Bằng chứng do thuật toán máy học (như BM25, SBERT) tự động đi tìm từ Context. Có thể đúng hoặc tìm lệch câu.

### 2.2. Về Nhãn Kết Luận (Verdict Labels)
Theo chuẩn ViFactCheck, nhãn phân loại chỉ gồm đúng 3 lớp:
* **SUPPORTED (Được hỗ trợ / Đúng sự thật):** Bằng chứng khẳng định rõ ràng rằng Claim là đúng.
* **REFUTED (Bị bác bỏ / Sai sự thật):** Bằng chứng chỉ ra Claim có thông tin sai lệch, mâu thuẫn trực tiếp với dữ kiện thực tế.
* **NOT_ENOUGH_INFO (Viết tắt: NEI - Không đủ thông tin):** Bằng chứng/Ngữ cảnh không chứa đủ dữ kiện để khẳng định Claim đúng hay sai (không có nghĩa là Claim bịa đặt, chỉ là chưa đủ bằng chứng trong bài báo để kết luận).

### 2.3. Về Bản chất Tác vụ (Task Nature)
* **NLI (Natural Language Inference - Suy luận ngôn ngữ tự nhiên):**
  * Bản chất kỹ thuật của bước Claim Verification là một bài toán NLI.
  * Trong NLI, ta có cặp: **Premise (Tiền đề)** chính là Evidence, và **Hypothesis (Giả thuyết)** chính là Claim.
  * Mối quan hệ tương đương:
    * *Entailment* (Kéo theo) $\rightarrow$ **SUPPORTED**
    * *Contradiction* (Mâu thuẫn) $\rightarrow$ **REFUTED**
    * *Neutral* (Trung lập) $\rightarrow$ **NOT_ENOUGH_INFO**

---

## 3. Nhóm Thuật Ngữ CẦN BIẾT (Level 2 - Kỹ thuật & Pipeline)

Nhóm này giúp bạn hiểu cấu trúc source code trong repo `vi-factcheck-system` và cách thiết lập thí nghiệm.

### 3.1. Các Phương Pháp Truy Xuất Bằng Chứng (Retrieval)
* **Lexical / Sparse Retrieval (Truy xuất từ vựng thưa):**
  * Dựa vào tần suất xuất hiện trùng khớp của từ khóa (Keyword matching).
  * **BM25 (Best Matching 25 / BM25Okapi):** Thuật toán xếp hạng văn bản cải tiến từ TF-IDF. Ưu điểm: Chạy siêu nhanh, chính xác khi claim chứa đúng các từ/tên riêng có trong bài báo. Nhược điểm: Bất lực trước từ đồng nghĩa hoặc diễn đạt biến thể.
* **Dense Semantic Retrieval (Truy xuất ngữ nghĩa dày đặc):**
  * Chuyển câu văn thành vector số thực nhiều chiều (embedding) thông qua mô hình Transformer, sau đó so sánh độ tương đồng góc (Cosine Similarity).
  * **SBERT (Sentence-BERT):** Kiến trúc dùng mạng Siamese để sinh embedding cho toàn bộ câu, giúp tìm được bằng chứng dù câu bằng chứng dùng từ đồng nghĩa khác với claim.
  * **Mean Pooling:** Kỹ thuật lấy trung bình cộng các vector token ở tầng cuối của Transformer để tạo thành 1 vector duy nhất đại diện cho cả câu.
* **Hybrid Retrieval (Truy xuất lai):**
  * Kết hợp cả BM25 và SBERT.
  * *Quy trình trong ViFactCheck:*
    1. Dùng BM25 lọc ra top-10 câu ứng viên sáng giá nhất (kèm điểm BM25).
    2. Dùng SBERT tính Cosine Similarity giữa Claim và 10 câu đó.
    3. Nhân điểm: $\text{Final Score} = \text{Scaled SBERT Cosine} \times \text{BM25 Weight}$.
    4. Sắp xếp lại để lấy top-5 câu tốt nhất.

### 3.2. Các Dòng Mô Hình Phân Loại (Models)
* **PLM (Pre-trained Language Model - Mô hình ngôn ngữ tiền huấn luyện):**
  * Mô hình dạng Encoder (BERT-style) chuyên cho việc hiểu văn bản.
  * *Ví dụ tiêu biểu:* **PhoBERT** (VinAI - chuyên tiếng Việt), **XLM-RoBERTa** (đa ngôn ngữ).
  * *Cơ chế:* Ghép `[CLS] Claim [SEP] Evidence [SEP]` đưa qua PLM $\rightarrow$ Lấy vector tại token `[CLS]` $\rightarrow$ Đưa qua tầng Linear Classifier để ra 3 nhãn.
* **LLM (Large Language Model - Mô hình ngôn ngữ lớn):**
  * Mô hình dạng Decoder-only với số lượng tham số lớn (7B, 8B, 13B...).
  * *Ví dụ:* Gemma-7B, Llama-3-8B, Qwen-2.5-7B.
* **LoRA (Low-Rank Adaptation):**
  * Kỹ thuật tinh chỉnh tham số hiệu quả (PEFT - Parameter-Efficient Fine-Tuning).
  * Thay vì cập nhật toàn bộ hàng tỷ trọng số của LLM (cực kỳ tốn RAM/VRAM), LoRA đóng băng (freeze) mô hình gốc và chỉ huấn luyện 2 ma trận phụ kích thước nhỏ (Rank $r$).
* **QLoRA (Quantized LoRA):**
  * Nén trọng số gốc của LLM xuống định dạng 4-bit (sử dụng thư viện `bitsandbytes` hoặc `unsloth`) trước khi gắn adapter LoRA. Nhờ đó có thể fine-tune model 7B/8B trên GPU cá nhân (16GB - 24GB VRAM).

### 3.3. Các Chế Độ Đầu Vào (Input Modes)
Trong code ViFactCheck có 2 chế độ thử nghiệm chính:
* **`gold_evidence` mode:** Đưa cho mô hình cặp `(Claim, Gold Evidence)`. Dùng để đo trần năng lực suy luận của mô hình phân loại khi bằng chứng đã được tìm đúng 100%.
* **`full_context` mode:** Đưa cho mô hình `(Claim, Toàn bộ bài báo)`. Mô hình phải tự "vật lộn" đọc cả bài báo dài để tìm dữ kiện rồi mới phân loại.

### 3.4. Thước Đo Đánh Giá (Evaluation Metrics)
* **Tác vụ Retrieval:**
  * **Recall@K:** Trong số các câu bằng chứng chuẩn (gold), có bao nhiêu câu lọt được vào Top-K câu máy tìm ra?
  * **MAP@K (Mean Average Precision at K):** Đo chất lượng xếp hạng; câu bằng chứng đúng xuất hiện càng ở thứ tự cao (hạng 1, 2) thì điểm càng cao.
* **Tác vụ Classification:**
  * **Macro F1-Score:** Trung bình cộng F1 của cả 3 lớp (Supported, Refuted, NEI). **Đây là metric chính của ViFactCheck** vì dữ liệu thường bị lệch lớp (imbalance). Nếu chỉ đo Accuracy thông thường sẽ bị ảo tưởng độ chính xác.
* **Tác vụ End-to-End (FEVER Score / Strict Accuracy):**
  * Một dự đoán chỉ được tính là **ĐÚNG** nếu và chỉ nếu:
    1. Máy tìm được đúng câu Gold Evidence, **VÀ**
    2. Máy dự đoán đúng Verdict Label.

---

## 4. Nhóm Thuật Ngữ NÊN BIẾT (Level 3 - Nâng cao & Hướng Nghiên Cứu)

Nhóm thuật ngữ này xuất hiện khi bạn muốn mở rộng đề tài, tìm Research Gap để viết bài báo khoa học.

* **Check-worthiness (Độ đáng kiểm chứng):**
  * Không phải câu văn nào cũng cần kiểm chứng (ví dụ câu cảm thán, câu hỏi, lời chào). Check-worthiness là bộ lọc bước 0 để phát hiện xem một phát biểu có chứa dữ kiện khách quan cần kiểm chứng hay không.
* **Multi-hop Reasoning (Suy luận đa bước / bắc cầu):**
  * Tình huống một câu bằng chứng đơn lẻ không đủ kết luận, mà phải kết hợp nhiều câu bằng chứng nằm ở các đoạn khác nhau hoặc các bài báo khác nhau để suy ra chân tướng.
* **Numerical Reasoning (Suy luận số học - Trọng tâm ViNumFCR):**
  * Khả năng mô hình đọc hiểu và thao tác với các con số: tính phần trăm tăng trưởng, đổi tỷ giá, so sánh mốc thời gian, tính tổng/hiệu. Đây là điểm yếu chí mạng của các mô hình chỉ dựa vào khớp ngữ nghĩa.
* **Hard Negatives (Mẫu âm tính gây nhiễu mạnh):**
  * Những câu văn trong bài báo có chứa đầy đủ các từ khóa của Claim (làm BM25 và SBERT chấm điểm tương đồng rất cao) nhưng thực chất lại nói về một chủ đề khác hoặc không chứng minh được claim. Huấn luyện model phân biệt được Hard Negatives là chìa khóa nâng cao SOTA.
* **Out-of-Distribution (OOD) / Cross-Domain Generalization:**
  * Khả năng mô hình huấn luyện trên 1 nguồn báo (ví dụ VnExpress mảng Chính trị) có chạy tốt khi đem sang kiểm chứng một nguồn báo hoàn toàn mới (ví dụ báo Thể thao, Diễn đàn tài chính) hay không.
* **Hallucination (Ảo giác của LLM):**
  * Tình trạng LLM tự suy diễn hoặc bịa ra dữ kiện không hề có trong đoạn bằng chứng được cung cấp.
* **Faithfulness (Độ trung thực của lời giải thích):**
  * Khi bắt LLM giải thích lý do tại sao claim Sai, lời giải thích đó có thực sự bám sát dữ kiện trong bằng chứng hay không, hay mô hình dùng kiến thức nội tại tự bịa ra để giải thích.
* **Chain-of-Thought (CoT - Chuỗi suy luận):**
  * Kỹ thuật nhắc (prompting) yêu cầu mô hình giải thích từng bước logic trung gian trước khi chốt nhãn phân loại cuối cùng.

---

## 5. Bảng Tra Cứu Nhanh (Glossary Cheat Sheet)

| Thuật ngữ tiếng Anh | Thuật ngữ tiếng Việt | Định nghĩa ngắn gọn |
|---|---|---|
| **Claim** | Nhận định / Lời tuyên bố | Câu cần kiểm tra đúng hay sai |
| **Context** | Ngữ cảnh | Bài báo gốc chứa thông tin liên quan |
| **Evidence** | Bằng chứng | Câu/đoạn trích dùng để đối chiếu claim |
| **Verdict** | Nhãn kết luận | SUPPORTED / REFUTED / NOT_ENOUGH_INFO |
| **Gold Standard** | Chuẩn đối sánh | Dữ liệu do người gán nhãn chuẩn xác 100% |
| **Retrieval** | Truy xuất thông tin | Bước tìm kiếm các câu bằng chứng liên quan |
| **Verification** | Xác minh / Kiểm chứng | Bước đối chiếu và ra quyết định nhãn |
| **Sparse Retrieval** | Truy xuất thưa | Tìm kiếm theo từ khóa (BM25, TF-IDF) |
| **Dense Retrieval** | Truy xuất dày đặc | Tìm kiếm theo vector ngữ nghĩa (SBERT) |
| **Reranking** | Tái xếp hạng | Chấm điểm lại danh sách kết quả tìm kiếm |
| **Sentence-pair** | Cặp câu | Đầu vào chuẩn của NLI: `(Text A, Text B)` |
| **Macro F1** | F1 trung bình các lớp | Thước đo chuẩn tránh thiên lệch dữ liệu |
| **Strict Accuracy** | Độ chính xác tuyệt đối | Đúng cả câu bằng chứng lẫn đúng nhãn |
| **Zero-shot** | Không có mẫu học | Bắt mô hình dự đoán chỉ bằng câu lệnh |
| **Few-shot** | Học với vài mẫu | Cho mô hình xem trước 2–5 ví dụ mẫu |
| **Fine-tuning** | Huấn luyện tinh chỉnh | Cập nhật trọng số mô hình trên tập dữ liệu đích |
| **PEFT / LoRA** | Tinh chỉnh tham số hiệu quả | Chỉ gắn thêm ma trận nhỏ để tiết kiệm tài nguyên |

---

## 6. Những Nhầm Lẫn Phổ Biến Cần Tránh

1. **Nhầm lẫn giữa "Fake News Detection" và "Fact-Checking":**
   * *Fake News Detection:* Thường chỉ đọc văn bản rồi phán đoán dựa trên phong cách viết, tiêu đề giật gân, cảm xúc (thường không cần chỉ ra bằng chứng).
   * *Fact-Checking:* Bắt buộc phải có **bằng chứng cụ thể (Evidence)** trích từ nguồn tin uy tín để đối chiếu từng câu chữ.
2. **Nhầm lẫn nhãn `NOT_ENOUGH_INFO` là "Tin Giả":**
   * `REFUTED` mới là tin giả / sai sự thật.
   * `NOT_ENOUGH_INFO` chỉ mang nghĩa: Trong bài báo cung cấp, không tìm thấy câu nào nhắc đến dữ kiện này, nên hệ thống không dám khẳng định.
3. **Nhầm lẫn giữa `gold_evidence` và `retrieved_evidence` khi báo cáo kết quả:**
   * Khi viết báo cáo, nếu chạy trên `gold_evidence`, F1 có thể lên đến 85–90% (vì bằng chứng người đưa sẵn đã quá chuẩn).
   * Nhưng khi chạy full pipeline (hệ thống tự retrieve bằng chứng rồi mới phân loại), điểm F1 thực tế thường rớt xuống 60–75%. Phải phân biệt rõ 2 kịch bản này trong bảng kết quả.
