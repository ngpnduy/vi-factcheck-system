# Use of Retrieval-Augmented Large Language Model Agent for Long-Form COVID-19 Fact-Checking

## Thông tin cơ bản
- **Tác giả:** Jingyi Huang, Yuyi Yang, Mengmeng Ji, Charles Alba, Sheng Zhang, Ruopeng An
- **Hội nghị/Tạp chí:** arXiv (cs.IR, cs.AI, cs.CL)
- **Năm:** 2025
- **Link:** https://arxiv.org/abs/2512.00007

## Tóm tắt (Abstract)
Nghiên cứu trình bày SAFE (System for Accurate Fact Extraction and Evaluation), một hệ thống multi-agent kết hợp LLM với RAG để cải thiện fact-checking tự động đối với các thông tin sai lệch dài (long-form misinformation) về COVID-19. Hệ thống sử dụng LOTR-RAG trên một kho dữ liệu gồm 130,000 tài liệu nghiên cứu và đánh giá sự hiệu quả so với mô hình cơ sở, cho thấy sự cải thiện mạnh mẽ về độ nhất quán và khả năng giải thích (explainability).

## Bài toán & Động lực
Thông tin sai lệch về COVID-19 (infodemic) đòi hỏi các giải pháp fact-checking có khả năng mở rộng để xử lý văn bản dài với độ chính xác và độ tin cậy cao. Các LLM thông thường gặp giới hạn về tính nhất quán và khả năng giải thích khi đối mặt với văn bản dài và phức tạp.

## Phương pháp chính
Hệ thống thiết kế theo hướng Agentic RAG:
- Agent 1: Trích xuất các claim (Claim extraction).
- Agent 2: Xác minh claim bằng LOTR-RAG (Lord of the Retrievers).
Hệ thống cũng thử nghiệm thêm một biến thể kết hợp Self-RAG (SAFE + SRAG) để tinh chỉnh quá trình truy xuất thông qua query rewriting.

## Dataset sử dụng
Kho tài liệu nghiên cứu gồm 130,000 document về COVID-19.
Đánh giá trên 50 bài báo tin giả (fake news articles) dài 2-17 trang, chứa 246 claim được dán nhãn bởi các chuyên gia y tế công cộng (true, partly true, false, partly false, misleading).

## Kết quả chính
- Các hệ thống SAFE vượt trội so với baseline LLMs trên mọi độ đo (p < 0.001).
- Về độ nhất quán (thang 0-1), SAFE (LOTR-RAG) đạt 0.629, cao hơn SAFE (+SRAG) (0.577) và baseline (0.279).
- Trong đánh giá chủ quan (thang Likert 0-4), SAFE (LOTR-RAG) đạt điểm cao nhất về tính hữu ích (3.640), độ rõ ràng (3.800) và tính xác thực (3.526).
- Thêm SRAG làm giảm nhẹ hiệu năng tổng thể, ngoại trừ cải thiện nhỏ về độ rõ ràng.

## Điểm mạnh
- Thiết kế hệ thống multi-agent rõ ràng cho phép chia tách việc trích xuất và xác minh claim.
- Giải quyết trực tiếp các văn bản misinformation dài (long-form), điều mà đa số các hệ thống trước đây bỏ qua.
- Thử nghiệm đánh giá chủ quan bằng chuyên gia (public health professionals).

## Hạn chế / Limitations
- Biến thể dùng Self-RAG với Query Rewriting phức tạp nhưng lại làm giảm hiệu năng tổng quát, cho thấy chưa tối ưu trong ghép nối pipeline.
- Giới hạn trong miền dữ liệu hẹp (COVID-19).

## Liên quan đến ViFactCheck
- **Kiến trúc hệ thống:** Có thể ứng dụng cách thiết kế Multi-agent (Agent trích xuất claim riêng, Agent xác minh riêng) vào ViFactCheck để chia nhỏ bài toán.
- **Rút ra kinh nghiệm:** Việc áp dụng Self-RAG (query rewriting) cần được thử nghiệm cẩn thận, bởi nghiên cứu này chỉ ra không phải lúc nào query rewriting cũng mang lại kết quả tốt hơn cho fact-checking (thậm chí làm giảm tính nhất quán).
