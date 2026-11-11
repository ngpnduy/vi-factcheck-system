<ViWikiFC: Fact-Checking for Vietnamese Wikipedia-Based Textual Knowledge Source>
## Thông tin cơ bản
- **Tác giả:** (arXiv preprint)
- **Hội nghị/Tạp chí:** arXiv
- **Năm:** 2024
- **Link:** https://arxiv.org/abs/2405.07615

## Tóm tắt (Abstract)
ViWikiFC là một bộ ngữ liệu (corpus) mã nguồn mở quy mô lớn đầu tiên dành cho fact-checking mở (open-domain) tiếng Việt dựa trên bách khoa toàn thư Wikipedia. Nó chứa hơn 20,000 claim được tự động khởi tạo và gán nhãn thủ công (SUPPORTS, REFUTES, NEI) tương thích chặt chẽ với format của bộ dữ liệu FEVER nổi tiếng.

## Bài toán & Động lực
Nghiên cứu về fact-checking tự động đã phát triển mạnh mẽ cho tiếng Anh, nhưng còn rất hạn chế với tiếng Việt do sự thiếu hụt trầm trọng của dữ liệu huấn luyện quy mô lớn. Việc sử dụng Wikipedia làm nguồn tri thức là một bước đi hợp lý vì độ tin cậy và tính bao quát thông tin cao. Động lực là lấp đầy khoảng trống dữ liệu này để thúc đẩy cộng đồng NLP tiếng Việt.

## Phương pháp chính
Tương tự phương pháp tạo tập FEVER, bộ dữ liệu được xây dựng thông qua việc trích xuất các câu bằng chứng (evidence) từ Wikipedia tiếng Việt. Sau đó, thông qua các bước biến đổi ngôn ngữ (đảo ngữ, thay đổi thực thể, phủ định) để tạo ra các claim tương ứng và tiến hành gán nhãn thủ công bởi người thật. Bài báo thiết lập benchmark bằng cách dùng BM25 để truy xuất bằng chứng và InfoXLM/PhoBERT để dự đoán tính xác thực.

## Dataset sử dụng
**ViWikiFC**: Hơn 20,916 claims, được gán 3 nhãn (SUPPORTS, REFUTES, NEI). Nguồn dữ liệu thuần túy từ Wikipedia tiếng Việt.

## Kết quả chính
Các phương pháp pipeline hiện hành (BM25 kết hợp với InfoXLM) chỉ đạt độ chính xác khắt khe (strict accuracy) khoảng 67.00%. Điều này cho thấy bộ dữ liệu ViWikiFC là một thử thách khó nhằn và bài toán fact-checking tiếng Việt dựa trên tri thức vẫn còn rất nhiều dư địa để cải tiến đối với các mô hình hiện tại.

## Điểm mạnh
- Quy mô rất lớn (hơn 20K claims), là một trong những bộ dữ liệu fact-checking tiếng Việt lớn nhất hiện nay.
- Tuân thủ chặt chẽ tiêu chuẩn của bộ dữ liệu FEVER, giúp dễ dàng chuyển giao (transfer learning) các mô hình đã thành công trên tiếng Anh sang tiếng Việt.

## Hạn chế / Limitations
- Chỉ tập trung vào ngôn ngữ mang tính hàn lâm, trung lập của Wikipedia. Nó không phản ánh đầy đủ sự phức tạp, ngôn từ lóng, hay bối cảnh cập nhật liên tục của tin tức thời sự và mạng xã hội.

## Liên quan đến ViFactCheck
Bộ dữ liệu ViWikiFC cung cấp một nguồn tài nguyên phụ trợ vô cùng quý giá cho dự án. Mặc dù không phải là văn bản tin tức, chúng ta có thể dùng ViWikiFC để pre-train hoặc fine-tune các mô hình hiểu ngôn ngữ (NLU) cho hệ thống thực tế nhằm nâng cao năng lực đọc hiểu ngữ cảnh và suy luận logic (reading comprehension and logic reasoning) trước khi áp dụng vào miền dữ liệu tin tức.
</ViWikiFC: Fact-Checking for Vietnamese Wikipedia-Based Textual Knowledge Source>
