<ViFactCheck: A New Benchmark Dataset and Methods for Multi-domain News Fact-Checking in Vietnamese>
## Thông tin cơ bản
- **Tác giả:** Tran Thai Hoa, Tran Quang Duy, Tran Khanh Quoc, and Kiet Van Nguyen
- **Hội nghị/Tạp chí:** AAAI Conference on Artificial Intelligence 2025
- **Năm:** 2025
- **Link:** https://arxiv.org/abs/2412.15308

## Tóm tắt (Abstract)
Bài báo giới thiệu ViFactCheck, bộ dữ liệu benchmark đầu tiên được thiết kế dành riêng cho bài toán fact-checking tin tức đa lĩnh vực (multi-domain) trong tiếng Việt. Với hơn 7,000 cặp claim-evidence được gán nhãn thủ công từ nhiều nguồn tin tức uy tín, bài báo cung cấp một cơ sở vững chắc để phát triển và đánh giá các hệ thống phát hiện tin giả. Bài báo cũng thử nghiệm và đánh giá các phương pháp khác nhau, đặc biệt là ứng dụng của các mô hình ngôn ngữ lớn (LLMs).

## Bài toán & Động lực
Bài toán kiểm chứng thông tin (fact-checking) trong tiếng Việt vẫn gặp nhiều khó khăn do thiếu các bộ dữ liệu chất lượng cao, đặc biệt là trong bối cảnh tin tức đa dạng chủ đề. Động lực chính là xây dựng một nền tảng dữ liệu chuẩn xác để thúc đẩy nghiên cứu fact-checking cho các ngôn ngữ ít tài nguyên (low-resource languages) như tiếng Việt, giúp tự động hóa quá trình xác minh thông tin.

## Phương pháp chính
Nghiên cứu tập trung vào việc tạo lập dữ liệu chuẩn và thử nghiệm các mô hình SOTA (XLM-R, PhoBERT) cùng các LLMs như Gemma. Quá trình giải quyết thường chia làm hai giai đoạn (pipeline): trích xuất bằng chứng (evidence retrieval) để tìm ra những câu liên quan nhất, và dự đoán tính xác thực (veracity prediction) để phân loại claim.

## Dataset sử dụng
**ViFactCheck**: Gồm 7,232 cặp claim-evidence được gán nhãn bởi con người với 3 nhãn (Supported, Refuted, Not Enough Information/NEI). Dữ liệu được thu thập từ 9 trang báo điện tử uy tín của Việt Nam, bao phủ 12 lĩnh vực khác nhau (Chính trị, Kinh tế, Giáo dục, Công nghệ, Y tế, Thể thao...). Hệ số Fleiss Kappa đạt 0.83 cho thấy mức độ đồng thuận và tin cậy cao của dữ liệu.

## Kết quả chính
Các mô hình LLM như Gemma thể hiện hiệu suất rất ấn tượng trên bộ dữ liệu này, đạt macro F1 score lên đến 89.90% trong tập kiểm tra. Các mô hình ngôn ngữ huấn luyện trước như XLM-R cũng cung cấp baseline tốt. Dữ liệu và mô hình được công khai hoàn toàn trên HuggingFace và Github để phục vụ cộng đồng.

## Điểm mạnh
- Bộ dữ liệu có quy mô lớn đầu tiên cho fact-checking tin tức tiếng Việt đa lĩnh vực.
- Chất lượng gán nhãn chuyên gia cao (kappa 0.83) và đa dạng chủ đề, sát với thực tế tin tức.
- Có đánh giá baseline chi tiết, toàn diện trên nhiều mô hình mạnh hiện nay.

## Hạn chế / Limitations
- Chưa đề cập sâu tới cách xử lý thông tin đa phương tiện (ảnh, video giả mạo) mà chủ yếu giải quyết ở cấp độ văn bản.

## Liên quan đến ViFactCheck
Bài báo này có liên quan trực tiếp và có thể coi là tài nguyên cốt lõi cho dự án ViFactCheck. Bộ dữ liệu này có thể được sử dụng trực tiếp để huấn luyện, tinh chỉnh (fine-tune) và đánh giá các module phát hiện tin giả trong hệ thống. Cấu trúc pipeline (Evidence Retrieval -> Veracity Prediction) được mô tả trong bài báo cũng chính là kiến trúc tham khảo chuẩn cho hệ thống của chúng ta.
</ViFactCheck: A New Benchmark Dataset and Methods for Multi-domain News Fact-Checking in Vietnamese>
