# Face the Facts! Evaluating RAG-based Pipelines for Professional Fact-Checking

## Thông tin cơ bản
- **Tác giả:** Daniel Russo, Stefano Menini, Jacopo Staiano, Marco Guerini
- **Hội nghị/Tạp chí:** INLG 2025 (arXiv)
- **Năm:** 2024 / 2025
- **Link:** https://arxiv.org/abs/2412.15189

## Tóm tắt (Abstract)
Nghiên cứu loại bỏ một số ràng buộc của các pipeline RAG hiện tại trong fact-checking tự động để đánh giá chúng theo tiêu chuẩn fact-checking chuyên nghiệp. Nghiên cứu benchmark các phương pháp RAG trong việc sinh ra các verdicts (văn bản ngắn thảo luận tính đúng đắn của claim) đối với các claim phức tạp về mặt văn phong và dựa trên nguồn tri thức (knowledge bases) hỗn tạp nhưng đáng tin cậy.

## Bài toán & Động lực
Mặc dù hệ thống NLP/NLG cho thấy tiềm năng hỗ trợ fact-checker chuyên nghiệp, các benchmark truyền thống thường đơn giản hoá bài toán. Bài báo hướng tới đánh giá thực tế hơn: xử lý các claim phức tạp (về mặt văn phong) và đối mặt với knowledge base heterogeneous (không đồng nhất, đến từ nhiều nguồn) giống như công việc thực tế của fact-checkers.

## Phương pháp chính
Nghiên cứu tập trung benchmark (đánh giá chuẩn) toàn diện các module của hệ thống RAG:
- Đánh giá các phương pháp retriever (LLM-based so với các phương pháp truyền thống).
- Đánh giá khả năng generation của các mô hình có kích thước khác nhau (lớn vs nhỏ).
- Đánh giá các thiết lập prompt (zero-shot, one-shot, fine-tuning).

## Dataset sử dụng
Sử dụng các claim mang tính phức tạp về văn phong và các Knowledge Base đa dạng (chưa công bố chi tiết tên dataset trong abstract, data có sẵn trên github face-the-facts).

## Kết quả chính
- **Retriever:** LLM-based retrievers vượt trội hơn các kỹ thuật retrieval khác, nhưng vẫn gặp khó khăn với các knowledge base quá dị thể (heterogeneous).
- **Generation (Faithfulness vs Context adherence):** Các mô hình lớn xuất sắc trong việc tạo ra verdict trung thực (faithfulness), trong khi các mô hình nhỏ hơn lại bám sát context tốt hơn.
- **Con người đánh giá:** Phương pháp zero-shot và one-shot được đánh giá cao về tính thông tin (informativeness), trong khi các mô hình fine-tune phù hợp hơn về mặt cảm xúc (emotional alignment).

## Điểm mạnh
- Cách tiếp cận rất thực tế khi so sánh với chuẩn mực của fact-checker chuyên nghiệp thay vì chỉ dùng các metric NLP truyền thống.
- Chỉ ra trade-off thú vị giữa mô hình lớn (faithfulness tốt) và mô hình nhỏ (context adherence tốt).

## Hạn chế / Limitations
- Kết quả cho thấy retriever vẫn là một nút thắt cổ chai lớn khi đối mặt với dữ liệu nhiễu và đa dạng.
- Phụ thuộc nhiều vào khả năng của LLM để truy xuất, có thể gây chi phí cao (LLM-based retriever).

## Liên quan đến ViFactCheck
- **Xây dựng benchmark:** ViFactCheck cần có bộ benchmark mang tính chuyên nghiệp, không chỉ đo accuracy mà còn đo độ trung thực (faithfulness) và khả năng cung cấp thông tin hữu ích cho người dùng.
- **Lựa chọn mô hình:** Nếu ViFactCheck ưu tiên bám sát văn bản (context adherence), có thể dùng các LLM nhỏ/nhẹ, nhưng nếu cần lập luận trung thực tổng quát, phải dùng LLM lớn. LLM-based retriever có thể là hướng đi cần thiết để xử lý nguồn tin tiếng Việt đa dạng.
