# DEFAME: Dynamic Evidence-based Framework for Multimodal Explainable fact-checking

## Thông tin cơ bản
- **Tác giả:** Nhóm nghiên cứu Multimodal AI Lab
- **Hội nghị/Tạp chí:** 42nd International Conference on Machine Learning (ICML)
- **Năm:** 2025
- **Link:** https://arxiv.org/abs/2412.10510

## Tóm tắt (Abstract)
DEFAME là một framework fact-checking đa phương thức (multimodal) sử dụng MLLMs (Multimodal Large Language Models) dạng zero-shot, có khả năng kết hợp linh hoạt các công cụ (dynamic tools). Thay vì chỉ xử lý văn bản, DEFAME đánh giá tính xác thực của claim dựa trên cả hình ảnh và văn bản. Hệ thống bao gồm một pipeline 6 bước linh hoạt để chọn công cụ, truy xuất bằng chứng (cả text và image) và sinh ra một báo cáo fact-checking chi tiết, có cấu trúc, dễ hiểu cho người dùng.

## Bài toán & Động lực
Fact-checking hiện đại thường gặp các claim chứa cả chữ và ảnh (ví dụ: meme, tin tức kèm ảnh giả mạo). Các hệ thống text-only hoặc chỉ dựa vào kiến thức nội tại (parametric knowledge) của LLM bị giới hạn bởi thời gian cắt dữ liệu (knowledge cutoff) và không thể xử lý tốt thông tin đa phương thức.

## Phương pháp chính
- **Pipeline 6 bước:** Hệ thống tự động phân tích claim, quyết định độ sâu tìm kiếm, linh hoạt gọi các công cụ truy xuất ngoài (như web search, reverse image search) để tìm kiếm các bằng chứng văn bản và hình ảnh.
- **Báo cáo đa phương thức (Multimodal report):** Thay vì chỉ xuất ra nhãn Đúng/Sai, DEFAME tạo ra một bản báo cáo kiểm chứng chi tiết giải thích quá trình thu thập thông tin và kết luận dựa trên cả văn bản lẫn hình ảnh.

## Dataset sử dụng
- Benchmark trên các tập dữ liệu lớn: **VERITE, AVeriTeC, MOCHEG**.
- Bộ dataset mới được giới thiệu: **ClaimReview2024+** (chứa các claim rất gần đây nhằm kiểm tra tính tổng quát hóa theo thời gian và tránh rò rỉ dữ liệu của các LLM hiện tại).

## Kết quả chính
DEFAME đạt được State-of-the-Art trên các benchmark uni-modal và multimodal. Vượt trội hơn đáng kể so với các baseline như GPT-4o trên bộ dữ liệu các sự kiện mới (ClaimReview2024+).

## Điểm mạnh
- Giải quyết trực diện vấn đề tin giả đa phương thức.
- Zero-shot, không yêu cầu huấn luyện lại hoặc fine-tune toàn bộ với dữ liệu mới.
- Khả năng sử dụng công cụ động (dynamic tool use) giúp cập nhật liên tục các sự kiện mới, phá vỡ rào cản knowledge cutoff.

## Hạn chế / Limitations
- Quá trình pipeline 6 bước có thể tốn nhiều chi phí tính toán (API calls) và thời gian thực thi (latency) hơn các hệ thống end-to-end đơn giản.
- Phụ thuộc vào chất lượng của các công cụ bên ngoài (ví dụ: Google Reverse Image Search).

## Liên quan đến ViFactCheck
- **Mở rộng tính năng:** Hiện tại ViFactCheck tập trung vào văn bản, nhưng trong tương lai có thể học hỏi DEFAME để tích hợp kiểm chứng hình ảnh bằng MLLM và reverse image search.
- **Động lực thiết kế báo cáo:** Báo cáo Fact-checking có cấu trúc (thay vì chỉ là một đoạn text dài) giúp tăng trải nghiệm người dùng, làm rõ các bước mà hệ thống đã thực hiện để tìm ra bằng chứng.
