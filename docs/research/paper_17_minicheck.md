# MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents

## Thông tin cơ bản
- **Tác giả:** Liyan Tang, et al.
- **Hội nghị/Tạp chí:** arXiv:2404.10774
- **Năm:** 2024
- **Link:** https://arxiv.org/abs/2404.10774

## Tóm tắt (Abstract)
Kiểm tra xem văn bản sinh ra từ LLM có dựa trên các tài liệu nền tảng (grounding documents) hay không là một bài toán cốt lõi trong nhiều ứng dụng như RAG (Retrieval-Augmented Generation), tóm tắt văn bản. Việc sử dụng các LLM khổng lồ để kiểm chứng (fact-check) từng phần văn bản rất tốn kém. Nghiên cứu này giới thiệu cách xây dựng các mô hình fact-checking nhỏ gọn (Small Language Models - SLMs) nhưng đạt hiệu năng ngang ngửa GPT-4, với chi phí thấp hơn 400 lần. Họ đạt được điều này bằng cách tạo ra dữ liệu huấn luyện tổng hợp (synthetic training data) thông qua GPT-4 và huấn luyện các mô hình nhỏ gọn.

## Bài toán & Động lực
Việc sử dụng trực tiếp GPT-4 hoặc Llama-3-70B cho bước đánh giá fact-checking trong quy trình RAG thực tế là không khả thi về mặt chi phí và độ trễ (latency) khi phải xử lý số lượng lớn yêu cầu. Cần có những mô hình chuyên biệt (specialized models) đủ nhỏ để chạy cục bộ nhanh chóng nhưng phải có khả năng suy luận và đối chiếu thông tin chính xác.

## Phương pháp chính
- **Data Synthesis (Tạo dữ liệu tổng hợp):** Sử dụng GPT-4 để tạo ra các lỗi sai sự thật thực tế nhưng khó phát hiện (challenging instances of factual errors) bằng một quy trình sinh văn bản có cấu trúc.
- **Model Fine-tuning:** Huấn luyện các SLM (Small Language Models) trên tập dữ liệu tổng hợp này. Việc huấn luyện dạy mô hình cách kiểm tra từng "fact" (sự việc) trong câu claim và nhận biết cách tổng hợp thông tin qua nhiều câu.
- **Unified Benchmark:** Giới thiệu benchmark mới là LLM-AggreFact để đánh giá khả năng fact-checking và grounding.

## Dataset sử dụng
- **Synthetic Data:** Dữ liệu tự động sinh ra bởi GPT-4 đóng vai trò "Teacher".
- **LLM-AggreFact (Evaluation):** Benchmark tổng hợp từ các nghiên cứu trước đây về fact-checking và đánh giá grounding của LLM.

## Kết quả chính
- Mô hình **MiniCheck-FT5** (với chỉ 770 triệu tham số) vượt trội hơn tất cả các hệ thống cùng kích cỡ.
- Đạt được độ chính xác tương đương với GPT-4 trong việc đánh giá fact-checking.
- Chi phí vận hành giảm khoảng 400 lần so với việc gọi API của GPT-4 cho mỗi lần kiểm tra.

## Điểm mạnh
- Cung cấp giải pháp cực kỳ hiệu quả về mặt chi phí và tốc độ (latency giảm từ 10-30x).
- Phương pháp sinh dữ liệu tổng hợp rất thông minh, giúp chuyển giao (distill) khả năng suy luận của mô hình lớn sang mô hình nhỏ gọn.
- Open-source toàn bộ code, data và model (MiniCheck-FT5).

## Hạn chế / Limitations
- Hiệu năng vẫn có thể bị hạn chế đối với những suy luận logic nhiều bước (multi-hop reasoning) cực kỳ phức tạp vốn đòi hỏi lượng kiến thức thế giới (world knowledge) khổng lồ mà SLM không có sẵn.

## Liên quan đến ViFactCheck
MiniCheck chứng minh rằng "nhỏ hơn là tốt hơn" khi chúng ta đã xác định rõ bài toán (fact-checking) và có dữ liệu huấn luyện tốt. Trong dự án ViFactCheck, việc sử dụng kiến trúc modular với các SLM chuyên biệt (có thể được fine-tune thông qua LoRA trên các mô hình tiếng Việt như Qwen2.5-7B hoặc Vistral) để thực hiện các bước như Evidence Retrieval, Claim Detection và Verdict Evaluation sẽ là hướng đi tối ưu. Thay vì gọi API đắt đỏ, ta có thể xây dựng pipeline tự động sinh dữ liệu tiếng Việt từ GPT-4/Claude 3.5 và distill xuống các mô hình SLM tiếng Việt cục bộ.
