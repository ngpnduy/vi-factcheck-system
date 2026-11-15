# DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts

## Thông tin cơ bản
- **Tác giả:** TU Darmstadt & hessian.AI researchers
- **Hội nghị/Tạp chí:** ICML (2024/2025)
- **Năm:** 2024
- **Link:** [GitHub / arXiv](https://github.com/multimodal-ai-lab/DEFAME)

## Tóm tắt (Abstract)
DEFAME là một pipeline dạng modular, zero-shot dựa trên Multimodal Large Language Model (MLLM) được thiết kế cho bài toán xác minh thông tin đa phương thức (text-image) trong miền mở (open-domain). Thay vì chỉ dựa vào văn bản hay kiến thức nội tại hạn chế, DEFAME tương tác động với các công cụ bên ngoài để thu thập bằng chứng hình ảnh và văn bản nhằm đưa ra kết luận và báo cáo giải thích chi tiết.

## Bài toán & Động lực
- Sự gia tăng của thông tin sai lệch đa phương thức (ví dụ: hình ảnh thật nhưng bị đặt sai ngữ cảnh - Out-Of-Context images, deepfakes).
- Các hệ thống kiểm chứng sự thật truyền thống thường chỉ xử lý văn bản, thiếu khả năng kiểm chứng hình ảnh kết hợp văn bản.
- Thiếu các hệ thống có tính minh bạch (explainability) cao để người dùng có thể hiểu tại sao một thông tin bị đánh giá là sai.

## Phương pháp chính
Hệ thống hoạt động qua một pipeline gồm 6 bước chính (zero-shot):
1. **Plan Actions:** MLLM đánh giá claim và lập kế hoạch các hành động cần thiết.
2. **Execute Actions:** Sử dụng các công cụ như Web Search, Image Search, Reverse Image Search (RIS), Geolocation.
3. **Summarize Results:** Tổng hợp các kết quả thu được từ các công cụ.
4. **Reasoning & Verdict:** Thực hiện suy luận đa phương thức để đưa ra nhãn kiểm chứng (verdict).
5. **Report Generation:** Tạo một báo cáo kiểm chứng chi tiết, dễ đọc cho con người kèm theo trích dẫn nguồn.

## Dataset sử dụng
- Các benchmark tiêu chuẩn: **VERITE**, **AVeriTeC**, **MOCHEG**.
- Dataset mới được giới thiệu: **ClaimReview2024+** (chứa các claim sau knowledge cutoff của GPT-4o để tránh rò rỉ dữ liệu và đánh giá khả năng tổng quát hóa theo thời gian).

## Kết quả chính
- Vượt qua các phương pháp State-of-the-Art (SOTA) trước đó trên các tập dữ liệu VERITE, AVeriTeC, và MOCHEG.
- Xử lý hiệu quả các trường hợp phức tạp như Out-Of-Context (OOC) images.

## Điểm mạnh
- Xử lý tự nhiên các claim chứa cả văn bản và hình ảnh.
- **Tính minh bạch (Explainability):** Không hoạt động như một "hộp đen", hệ thống sinh ra báo cáo chi tiết mô phỏng quá trình của người kiểm chứng thực tế.
- Khả năng sử dụng công cụ động (Dynamic tool-use) giúp truy xuất thông tin cập nhật, vượt qua giới hạn về knowledge cutoff.

## Hạn chế / Limitations
- Phụ thuộc vào chất lượng và tốc độ của các công cụ tìm kiếm bên ngoài (Web/Image Search APIs).
- Độ trễ (latency) của toàn bộ pipeline 6 bước có thể cao, chưa phù hợp cho kiểm chứng real-time tức thời ở quy mô quá lớn.

## Liên quan đến ViFactCheck
- **ViFactCheck** có thể tham khảo kiến trúc agent-based và pipeline modular của DEFAME để mở rộng từ kiểm chứng văn bản đơn thuần sang đa phương thức (Multimodal).
- Tích hợp công cụ **Reverse Image Search** cho các hình ảnh lan truyền trên mạng xã hội Việt Nam.
- Học hỏi cách sinh báo cáo minh bạch (explainable report generation) để tăng độ tin cậy cho người dùng ViFactCheck.
