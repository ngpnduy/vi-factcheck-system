# OpenFactCheck: Building, Benchmarking Customized Fact-Checking Systems and Evaluating the Factuality of Claims and LLMs

## Thông tin cơ bản
- **Tác giả:** Yuxia Wang et al.
- **Hội nghị/Tạp chí:** arXiv
- **Năm:** 2024
- **Link:** https://arxiv.org/abs/2405.05583

## Tóm tắt (Abstract)
Sự gia tăng sử dụng Large Language Models (LLMs) trong các ứng dụng thực tế đòi hỏi cơ chế xác minh tính chính xác thực tế (factual accuracy) của các đầu ra từ LLM. Khó khăn nằm ở việc đánh giá tính xác thực của các câu trả lời tự do trong các miền mở (open domains), và sự thiếu đồng nhất trong các benchmark. Bài báo đề xuất OpenFactCheck, một framework thống nhất giúp xây dựng các hệ thống fact-checking tự động tùy chỉnh, đánh giá độ chính xác, đánh giá tính xác thực của LLMs và xác minh claims.

## Bài toán & Động lực
- LLMs thường sinh ra thông tin sai lệch (hallucination) trong câu trả lời tự do.
- Việc đánh giá tính xác thực của LLMs hiện tại gặp khó khăn do thiếu công cụ đánh giá thống nhất, benchmark bị phân mảnh khiến việc so sánh giữa các model trở nên khó khăn.

## Phương pháp chính
OpenFactCheck bao gồm 3 module chính:
1. **CUSTCHECKER:** Cho phép người dùng dễ dàng tùy chỉnh một hệ thống fact-checker tự động để xác minh tính đúng đắn của tài liệu và claims.
2. **LLMEVAL:** Framework đánh giá thống nhất nhằm đo lường khả năng factuality của LLM từ nhiều góc độ.
3. **CHECKEREVAL:** Giải pháp mở rộng để đo lường độ tin cậy của kết quả xác minh từ các fact-checker tự động thông qua các tập dữ liệu được con người gán nhãn.

## Dataset sử dụng
Sử dụng các human-annotated datasets tích hợp sẵn trong module CHECKEREVAL để benchmark các hệ thống kiểm chứng.

## Kết quả chính
Cung cấp một framework toàn diện và mã nguồn mở (https://github.com/yuxiaw/openfactcheck) giúp tiêu chuẩn hóa việc xây dựng và đánh giá hệ thống fact-checking, tạo tiền đề cho các nghiên cứu và ứng dụng thực tế dễ dàng đo lường và so sánh.

## Điểm mạnh
- Cung cấp một quy trình (pipeline) end-to-end từ việc xây dựng công cụ kiểm tra đến việc đánh giá LLM và đánh giá lại chính công cụ kiểm tra.
- Tính tùy biến cao, cung cấp framework linh hoạt cho nhiều bài toán.

## Hạn chế / Limitations
- Phụ thuộc vào chất lượng của các bộ dataset đánh giá và có thể gặp khó khăn với các ngôn ngữ low-resource do thiếu dữ liệu gán nhãn.

## Liên quan đến ViFactCheck
- **Kiến trúc hệ thống:** ViFactCheck có thể tham khảo kiến trúc module CUSTCHECKER để xây dựng pipeline tự động kiểm chứng cho tiếng Việt.
- **Đánh giá tự động:** Có thể ứng dụng ý tưởng của CHECKEREVAL để xây dựng tập benchmark đánh giá chất lượng của chính hệ thống ViFactCheck.
