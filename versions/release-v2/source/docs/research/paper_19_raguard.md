# Worse than Zero-shot? A Fact-Checking Dataset for Evaluating the Robustness of RAG Against Misleading Retrievals

## Thông tin cơ bản
- **Tác giả:** Linda Zeng, Rithwik Gupta, Divij Motwani, Yi Zhang, Diji Yang
- **Hội nghị/Tạp chí:** NeurIPS 2025
- **Năm:** 2025
- **Link:** https://arxiv.org/abs/2502.16101

## Tóm tắt (Abstract)
Mặc dù RAG giúp giảm hallucination, các LLM vẫn gặp khó khăn trong việc giữ vững lập luận logic khi tiếp xúc với các bằng chứng gây hiểu lầm (misleading) hoặc mâu thuẫn. Nghiên cứu giới thiệu RAGuard, benchmark đầu tiên chuyên đánh giá độ mạnh mẽ (robustness) của hệ thống RAG trước thông tin gây hiểu lầm. Kết quả cho thấy khi tiếp xúc với misleading retrievals, mọi hệ thống RAG dùng LLM đều hoạt động kém hơn cả baseline zero-shot (không dùng retrieval).

## Bài toán & Động lực
Các benchmark hiện tại thường đánh giá RAG trong điều kiện "sạch" (retrieval lý tưởng) hoặc dùng nhiễu nhân tạo (synthetic noise). Điều này không phản ánh thực tế, đặc biệt trong miền chính trị, nơi thông tin thường phân cực. LLMs có xu hướng bị lệch lạc lập luận khi đọc phải các bằng chứng misleading.

## Phương pháp chính
Xây dựng dataset benchmark RAGuard với dữ liệu misinformation thực tế.
Evidence (Bằng chứng) thu thập được phân loại thành 3 loại:
1. Supporting (Hỗ trợ, củng cố)
2. Misleading (Gây hiểu lầm, lừa dối)
3. Unrelated (Không liên quan)
Sử dụng dữ liệu thảo luận từ Reddit để tạo kho tìm kiếm (retrieval corpus) mang tính tự nhiên.

## Dataset sử dụng
**RAGuard**: Tập dữ liệu fact-checking chuyên biệt, lấy nguồn từ Reddit, chứa các bằng chứng nhiễu/misleading tự nhiên thay vì tạo bằng máy.

## Kết quả chính
- Khi hệ thống RAG truy xuất phải thông tin gây hiểu lầm, hiệu năng của tất cả các hệ thống RAG LLM thử nghiệm đều **giảm thấp hơn mức zero-shot** (tức là không dùng RAG còn tốt hơn).
- Con người (human annotators) khi đọc cùng văn bản nhiễu vẫn duy trì được hiệu năng đánh giá, chứng tỏ con người chống chịu nhiễu (robustness) tốt hơn LLM rất nhiều.

## Điểm mạnh
- Đặt ra vấn đề thực tiễn rất nghiêm trọng của RAG: Retrieval độc hại/nhiễu làm kết quả tệ đi.
- Dataset dùng nhiễu tự nhiên (Reddit) thay vì nhiễu nhân tạo, tăng tính thực tế.

## Hạn chế / Limitations
- Dataset dựa trên Reddit nên thiên về các chủ đề thảo luận có trên Reddit, có thể không bao phủ hết các dạng news article misinformation chuyên nghiệp.

## Liên quan đến ViFactCheck
- **Rủi ro lớn cho ViFactCheck:** Nếu hệ thống RAG tiếng Việt truy xuất sai thông tin hoặc lấy phải nguồn báo lá cải/tin giả làm Context, hệ thống sẽ đưa ra kết luận còn tệ hơn việc gọi thẳng LLM.
- **Giải pháp tiềm năng:** Cần xây dựng một cơ chế "Retrieval Filter" hoặc "Fact-Checker Critic" để chấm điểm nguồn, lọc bỏ các "misleading evidence" trước khi đưa vào LLM xử lý. Đánh giá tính đáng tin cậy của Nguồn tin là bắt buộc đối với ViFactCheck.
