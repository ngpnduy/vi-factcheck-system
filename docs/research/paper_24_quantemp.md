# QuanTemp: A real-world open-domain benchmark for fact-checking numerical claims

## Thông tin cơ bản
- **Tác giả:** Venktesh V, Abhijit Anand, Avishek Anand, và Vinay Setty
- **Hội nghị/Tạp chí:** SIGIR 2024
- **Năm:** 2024
- **Link:** [arXiv](https://arxiv.org/abs/2403.00000) (Placeholder link to arxiv)

## Tóm tắt (Abstract)
QuanTemp là một bộ dữ liệu và benchmark quy mô lớn, đa lĩnh vực (multi-domain) được thiết kế đặc biệt cho bài toán kiểm chứng các tuyên bố chứa yếu tố định lượng và thời gian (numerical and temporal claims). Khác với các bộ dữ liệu trước đây thường tập trung vào dữ liệu tổng hợp (synthetic data) từ Wikipedia, QuanTemp sử dụng các tuyên bố số học phức tạp từ thế giới thực.

## Bài toán & Động lực
- Thông tin sai lệch thường lạm dụng các con số, số liệu thống kê để tạo ra "ảo giác về tính xác thực" (mirage of veracity).
- Các hệ thống fact-checking hiện tại gặp khó khăn trong việc suy luận toán học, thống kê, so sánh (tăng/giảm, lớn hơn/nhỏ hơn) và xử lý khoảng thời gian (intervals).
- Các bộ dữ liệu hiện có chưa phản ánh được độ khó của các claim chứa con số trong thực tế.

## Phương pháp chính
- **Xây dựng Dataset:** Thu thập các claim thực tế, kèm theo metadata chi tiết và quy trình thu thập bằng chứng nhằm tránh rò rỉ dữ liệu theo thời gian (temporal leakage).
- **Claim Decomposition:** Nhóm tác giả thử nghiệm và đề xuất phương pháp phân rã claim (claim decomposition) thành các bài toán con (sub-tasks) nhỏ hơn để mô phỏng lại quy trình kiểm chứng của con người (đặc biệt phát triển sâu hơn ở dự án QuanTemp++).
- Đánh giá các mô hình dựa trên khả năng hiểu toán học và logic (Numerical-understanding-based models - NLI).

## Dataset sử dụng
- **QuanTemp Dataset:** Bao gồm các bài toán liên quan đến con số, thời gian, và thống kê trong nhiều lĩnh vực khác nhau.

## Kết quả chính
- Các mô hình baseline tốt nhất chỉ đạt mức macro-F1 khoảng 58.32, cho thấy kiểm chứng con số là một bài toán còn rất nhiều thách thức đối với các mô hình ngôn ngữ lớn (LLMs).
- Việc phân rã câu (decomposition) giúp tăng cường khả năng theo dõi và giải thích quá trình tính toán, suy luận.

## Điểm mạnh
- Cung cấp một bộ dữ liệu thực tế, lấp đầy khoảng trống lớn trong mảng Numerical Reasoning Fact Verification.
- Cấu trúc đánh giá chặt chẽ, ngăn chặn rò rỉ thông tin từ tương lai vào quá khứ (temporal leakage).

## Hạn chế / Limitations
- Giới hạn trong các thao tác toán học và thống kê cơ bản, có thể chưa xử lý được các suy luận quá sâu về xác suất hoặc các mô hình tài chính phức tạp.
- Đòi hỏi các nguồn bằng chứng (evidence) chứa chính xác dữ liệu gốc để đối chiếu, điều này không phải lúc nào cũng khả thi trên Internet.

## Liên quan đến ViFactCheck
- **ViFactCheck** có thể bổ sung một module riêng biệt hoặc luồng xử lý riêng (agent tool) chuyên giải quyết **Numerical Reasoning**, chẳng hạn như trích xuất số liệu từ văn bản, thực hiện các phép toán so sánh, hoặc tra cứu các biểu đồ thống kê.
- Việc áp dụng **Claim Decomposition** cho các câu tiếng Việt dài, chứa nhiều số liệu (ví dụ: các bài báo cáo kinh tế, thông số y tế) sẽ giúp ViFactCheck tăng độ chính xác thay vì để mô hình đoán mò.
