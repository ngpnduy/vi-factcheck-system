<SemEval-2025 Task 7: Multilingual and Crosslingual Fact-Checked Claim Retrieval>
## Thông tin cơ bản
- **Tác giả:** Qiwei Peng, Róbert Móro, Michal Gregor, Ivan Srba, et al.
- **Hội nghị/Tạp chí:** Proceedings of the 19th International Workshop on Semantic Evaluation (SemEval-2025)
- **Năm:** 2025
- **Link:** https://aclanthology.org/2025.semeval-1.323/

## Tóm tắt (Abstract)
SemEval-2025 Task 7 là một shared task tập trung vào bài toán "Multilingual and Cross-lingual Fact-Checked Claim Retrieval" (Truy xuất các claim đã được kiểm chứng ở môi trường đa ngữ và chéo ngôn ngữ). Nhiệm vụ này nhằm giải quyết thách thức trong việc xác định các bài đăng trên mạng xã hội có chứa thông tin đã được kiểm chứng từ trước, giúp thu hẹp khoảng cách cho các ngôn ngữ low-resource. Shared task có hai track: Monolingual (đơn ngữ) và Cross-lingual (chéo ngôn ngữ). Cuộc thi thu hút 179 người tham gia và 31 đội nộp kết quả, sử dụng nhiều kỹ thuật tiên tiến như LLMs và Contrastive Learning.

## Bài toán & Động lực
Sự lây lan của tin giả diễn ra trên phạm vi toàn cầu, nhưng hầu hết dữ liệu fact-check lại tập trung ở các ngôn ngữ lớn. Khi một tin giả (ví dụ bằng tiếng Anh) được lan truyền sang ngôn ngữ khác (như tiếng Việt) thông qua dịch thuật hoặc biên soạn lại, các hệ thống cần có khả năng truy xuất lại bài fact-check gốc. Động lực của task này là xây dựng các hệ thống Information Retrieval (IR) chéo ngôn ngữ mạnh mẽ.

## Phương pháp chính
Các đội tham gia đã sử dụng nhiều phương pháp khác nhau, chủ yếu tập trung vào:
1. **Multilingual Embeddings:** Sử dụng các mô hình nhúng mạnh mẽ (như E5-large, Jina, MPNet, LaBSE) để ánh xạ văn bản từ các ngôn ngữ khác nhau vào chung một không gian vector.
2. **Large Language Models (LLMs):** Tinh chỉnh các LLMs như LLaMA để sinh ra các truy vấn hoặc trực tiếp đánh giá độ tương đồng (re-ranking).
3. **Contrastive Learning:** Huấn luyện mô hình để kéo gần các biểu diễn của bài đăng mạng xã hội và claim tương ứng (dù ở hai ngôn ngữ khác nhau), đồng thời đẩy xa các cặp không liên quan.

## Dataset sử dụng
Một tập dữ liệu benchmark hoàn toàn mới bao gồm các bài đăng mạng xã hội và bộ sưu tập các claim đã được fact-check, bao phủ nhiều ngôn ngữ. Dữ liệu chia thành hai phần đánh giá: Monolingual và Cross-lingual.

## Kết quả chính
- Các mô hình dựa trên Multilingual Embeddings kết hợp với chiến lược Re-ranking bằng LLM mang lại kết quả tốt nhất (SOTA).
- Xếp hạng (ranking) cho các ngôn ngữ có nguồn tài nguyên trung bình (mid-resource) đôi khi còn vượt qua các ngôn ngữ high-resource nhờ vào đặc thù của tập dữ liệu và tính chất đa dạng của các ngôn ngữ.

## Điểm mạnh
- Cung cấp một framework đánh giá (benchmark) tiêu chuẩn, quy mô lớn cho bài toán Cross-lingual Fact-checking.
- Khuyến khích nghiên cứu vào việc ánh xạ ngữ nghĩa (semantic alignment) giữa các ngôn ngữ.

## Hạn chế / Limitations
- Hiệu năng phân giải cho các ngôn ngữ "bất ngờ" (surprise/unseen languages) vẫn chưa đạt mức lý tưởng.
- Sự mất cân bằng dữ liệu giữa các ngôn ngữ vẫn làm thiên lệch kết quả của các mô hình đa ngữ.

## Liên quan đến ViFactCheck
- **Ứng dụng:** ViFactCheck có thể tích hợp bài toán **Claim Retrieval** như một module tiền xử lý (như hệ thống RAG). Nếu người dùng nhập một claim tiếng Việt, hệ thống có thể search trong database các claim tiếng Anh/Việt đã được fact-check (từ Snopes, AFP Fact Check, VietnamNet).
- **Phương pháp:** Áp dụng các multilingual embedding models (ví dụ: `multilingual-e5-large`) được đề cập trong task này vào hệ thống Retrieval của ViFactCheck để tối ưu hóa quá trình so khớp claim.
