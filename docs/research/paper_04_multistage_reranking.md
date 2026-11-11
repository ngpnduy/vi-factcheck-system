<Evidence Retrieval for Fact Verification using Multi-stage Reranking>
## Thông tin cơ bản
- **Tác giả:** Shrikant Malviya, Stamos Katsigiannis
- **Hội nghị/Tạp chí:** Findings of the Association for Computational Linguistics: EMNLP 2024
- **Năm:** 2024
- **Link:** https://aclantholbâyogy.org/2024.findings-emnlp.428/

## Tóm tắt (Abstract)
Nghiên cứu này đề xuất một mô hình Multi-stage ReRanking (M-ReRank) nhằm khắc phục những hạn chế của các phương pháp trích xuất bằng chứng một giai đoạn (single-stage) trong bài toán fact verification. Bằng cách kết hợp các mô hình dense retrieval và list-aware rerankers, mô hình tối ưu hóa việc truy xuất cả dữ liệu có cấu trúc (bảng biểu, ô dữ liệu) lẫn phi cấu trúc (câu văn).

## Bài toán & Động lực
Các phương pháp truy xuất bằng chứng truyền thống thường chỉ sử dụng một bước (như BM25 hoặc Dense Retrieval cơ bản), dẫn đến bỏ sót nhiều thông tin quan trọng hoặc xếp hạng sai khi đối mặt với dữ liệu phức tạp. Việc thiếu khả năng xử lý dữ liệu đa dạng (ví dụ: các bảng biểu so với văn bản thuần túy) làm giảm hiệu suất của toàn bộ hệ thống fact-checking.

## Phương pháp chính
- **Multi-stage ReRanking (M-ReRank):** Sử dụng một quy trình gồm nhiều bước. Bước đầu tiên thực hiện retrieval để lọc ra các ứng viên tiềm năng. Bước sau áp dụng reranking để xếp hạng lại bằng chứng một cách chính xác hơn.
- Cải thiện việc trích xuất bằng chứng dạng bảng (tables/cells) và dạng câu (sentences) đồng thời thông qua các list-aware rerankers.

## Dataset sử dụng
- **FEVEROUS** (Fact Extraction and VERification Over Unstructured and Structured information).

## Kết quả chính
- Mô hình đạt **93.63% recall rate** cho các trang Wikipedia.
- Đem lại mức tăng đáng kể so với SOTA trước đó trên tập dev: tăng 7.85% recall cho câu văn, 8.29% cho bảng biểu, và 3% cho các ô dữ liệu (cells).

## Điểm mạnh
- Xử lý tốt cả dữ liệu có cấu trúc (bảng biểu) và phi cấu trúc.
- Cải thiện recall một cách rõ rệt nhờ quy trình reranking hiệu quả.

## Hạn chế / Limitations
- Cần nhiều tài nguyên tính toán hơn do phải chạy qua nhiều mô hình/stage khác nhau (retrieval sau đó là reranking).
- Tập trung chủ yếu vào Wikipedia, có thể cần tinh chỉnh cho các nguồn dữ liệu chuyên biệt khác.

## Liên quan đến ViFactCheck
- **Ứng dụng tiềm năng:** ViFactCheck có thể áp dụng kiến trúc truy xuất đa giai đoạn (multi-stage) thay vì chỉ dùng một bước Vector Database. Một pipeline gồm BM25 + Dense Retrieval (bước 1) và Cross-Encoder Reranker (bước 2) sẽ giúp tăng recall và độ chính xác cho hệ thống.
</Evidence Retrieval for Fact Verification using Multi-stage Reranking>
