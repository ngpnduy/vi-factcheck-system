<ViNumFCR: A Novel Vietnamese Benchmark for Numerical Reasoning Fact Checking on Social Media News>
## Thông tin cơ bản
- **Tác giả:** (Các tác giả công bố tại INLG)
- **Hội nghị/Tạp chí:** Proceedings of the 18th International Natural Language Generation Conference (INLG)
- **Năm:** 2025
- **Link:** https://aclanthology.org/2025.inlg-main.9/

## Tóm tắt (Abstract)
Bài báo giới thiệu ViNumFCR, một bộ dữ liệu và benchmark độc đáo được thiết kế đặc biệt để giải quyết bài toán fact-checking yêu cầu khả năng suy luận số liệu (numerical reasoning) trên mạng xã hội và tin tức tiếng Việt. Nó cung cấp tập dữ liệu để đánh giá khả năng của các mô hình AI trong việc đối chiếu, tính toán và xác minh sự chính xác của các con số được đưa ra trong văn bản.

## Bài toán & Động lực
Một phương thức phổ biến của tin giả và thông tin sai lệch là lạm dụng, bóp méo hoặc đưa ra các số liệu sai lệch về kinh tế, y tế, xã hội nhằm tăng tính thuyết phục. Tuy nhiên, các mô hình ngôn ngữ thông thường vốn giỏi xử lý văn bản ngữ nghĩa nhưng lại thường gặp khó khăn nghiêm trọng trong việc suy luận các phép toán học hoặc logic số lượng từ văn bản. Tiếng Việt thiếu vắng hoàn toàn các bộ dữ liệu chuyên biệt để đánh giá khía cạnh này.

## Phương pháp chính
Nghiên cứu đề xuất một benchmark chuyên sâu, trong đó các claim không chỉ cần được đánh giá dựa trên câu chữ mà đòi hỏi mô hình phải thực hiện được các phép so sánh (lớn hơn, nhỏ hơn), nhận diện đại lượng tương đương, hoặc tính toán đơn giản để xem claim có khớp với số liệu trong bài báo gốc (evidence) hay không.

## Dataset sử dụng
**ViNumFCR**: Tập trung đặc thù vào các claim chứa thông tin số liệu được trích xuất từ tin tức mạng xã hội tiếng Việt. Đòi hỏi khả năng xử lý numerical reasoning để phân loại veracity.

## Kết quả chính
Bài báo cung cấp điểm chuẩn (baseline) cho các mô hình ngôn ngữ hiện tại về khả năng xử lý bài toán suy luận số liệu bằng tiếng Việt, vạch ra những điểm yếu của AI hiện tại khi đối diện với dữ liệu số và mở ra hướng nghiên cứu mới về tính chính xác số học trong fact-checking tiếng Việt.

## Điểm mạnh
- Đi sâu vào một khía cạnh cực kỳ quan trọng, khó và mang tính thời sự cao của fact-checking: suy luận số liệu.
- Phản ánh đúng thực trạng tin giả chứa dữ liệu thống kê sai lệch trên mạng xã hội hiện nay.

## Hạn chế / Limitations
- Phạm vi tương đối hẹp (chỉ giải quyết các claim có chứa số liệu). Cần được sử dụng kết hợp với các mô hình suy luận văn bản tổng quát khác để tạo ra một hệ thống hoàn chỉnh.

## Liên quan đến ViFactCheck
Paper này rất hữu ích để tham khảo nhằm xây dựng hoặc nâng cấp dự án ViFactCheck. Chúng ta có thể coi đây là cơ sở để phát triển một "module/plugin" chuyên biệt trong hệ thống dùng để xử lý các tin tức có chứa nhiều số liệu (như báo cáo tài chính, thống kê y tế, kết quả bầu cử). Nó giúp nâng cao tính toàn diện và khả năng chống lại các loại tin giả tinh vi của hệ thống.
</ViNumFCR: A Novel Vietnamese Benchmark for Numerical Reasoning Fact Checking on Social Media News>
