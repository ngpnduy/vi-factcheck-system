# FIRE: Fact-checking with Iterative Retrieval and Verification

## Thông tin cơ bản
- **Tác giả:** Nhóm nghiên cứu từ MBZUAI (MBZUAI-NLP)
- **Hội nghị/Tạp chí:** arXiv:2411.00784
- **Năm:** 2024
- **Link:** https://arxiv.org/abs/2411.00784

## Tóm tắt (Abstract)
Fact-checking các văn bản dài là một bài toán đầy thách thức, thường yêu cầu phải chia nhỏ văn bản thành nhiều "atomic claims" (nhận định cốt lõi). Phương pháp truyền thống thường lấy một lượng bằng chứng (evidence) cố định ban đầu rồi tiến hành xác minh (verification). Tuy nhiên, cách này không hiệu quả về chi phí và không phản ánh đúng quá trình tìm kiếm thông tin lặp đi lặp lại của con người. Khung làm việc (framework) **FIRE** ra đời để giải quyết vấn đề này. FIRE tích hợp việc truy xuất bằng chứng và xác minh claim theo một quá trình lặp (iterative). Dựa trên độ tin cậy (confidence) của đánh giá hiện tại, FIRE sẽ quyết định việc đưa ra kết luận cuối cùng hay tạo thêm các truy vấn tìm kiếm mới.

## Bài toán & Động lực
Cách tiếp cận truyền thống (Single-turn RAG) trong fact-checking thường gặp lỗi thiếu bằng chứng đối với các claim phức tạp, hoặc lãng phí tài nguyên tính toán đối với các claim đơn giản. Nếu chia quá nhiều truy vấn cũng làm tăng chi phí gọi API tìm kiếm và chi phí xử lý LLM. Động lực của FIRE là mô phỏng cách con người kiểm chứng: tìm kiếm, đọc, nếu chưa đủ tự tin thì tìm kiếm tiếp (multi-turn/iterative), giúp cân bằng giữa độ chính xác và chi phí.

## Phương pháp chính
- **Cơ chế Agent-based Iterative Framework:** FIRE hoạt động như một agent. Nó có khả năng đánh giá bằng chứng hiện tại.
- **Cơ chế tự đánh giá (Confidence-based Decision):** Sau mỗi lần truy xuất, FIRE sử dụng một cơ chế thống nhất để quyết định: (1) Kết thúc và đưa ra phán quyết (Verdict) nếu đã đủ tự tin, hoặc (2) Sinh ra một truy vấn tìm kiếm (search query) tiếp theo để đào sâu thêm thông tin.
- **Tích hợp chặt chẽ Retrieval và Verification:** Quá trình tìm kiếm và xác minh không bị tách rời hoàn toàn mà bổ trợ cho nhau qua mỗi vòng lặp.

## Dataset sử dụng
- Đánh giá trên các bộ dữ liệu Fact-checking tiêu chuẩn (thường liên quan đến văn bản dài và cần truy xuất từ web/Wikipedia).

## Kết quả chính
- FIRE đạt được hiệu năng (accuracy) nhỉnh hơn một chút so với các framework fact-checking mạnh mẽ hiện tại.
- Điểm đột phá là **giảm chi phí sử dụng LLM trung bình tới 7.6 lần** và **giảm chi phí tìm kiếm (search costs) tới 16.5 lần**.
- Cho thấy tiềm năng rất lớn khi ứng dụng vào các hệ thống fact-checking quy mô lớn (large-scale operations).

## Điểm mạnh
- Giải quyết bài toán tối ưu chi phí (cost-effectiveness) cực kỳ xuất sắc.
- Cơ chế lặp lại phản ánh tính logic tự nhiên của con người.
- Hỗ trợ tốt cho các tài liệu dài cần xác minh nhiều bước.

## Hạn chế / Limitations
- Việc lập trình và điều khiển Agent có thể gặp khó khăn nếu mô hình LLM cơ sở (base LLM) không tuân theo các định dạng đầu ra (như JSON) một cách nghiêm ngặt.
- Độ trễ (latency) của một phiên truy vấn có thể bị kéo dài nếu hệ thống rơi vào vòng lặp tìm kiếm nhiều lần đối với các claim khó hoặc không có thông tin trên mạng.

## Liên quan đến ViFactCheck
FIRE cung cấp một kiến trúc Agent tuyệt vời mà hệ thống ViFactCheck có thể học hỏi. Thay vì chỉ truy xuất tài liệu một lần rồi đưa cho mô hình đánh giá, ViFactCheck có thể áp dụng luồng (pipeline) lặp lại: LLM tiếng Việt (như Qwen2.5/Vistral) nhận claim -> sinh truy vấn Google/Bing Search -> đánh giá nhanh các bài báo tiếng Việt -> nếu chưa chắc chắn, sinh truy vấn tìm kiếm ngách hơn. Điều này vừa giúp tiết kiệm token cho các claim hiển nhiên, vừa đảm bảo tính chính xác cao cho các tin giả tinh vi trên mạng xã hội Việt Nam.
