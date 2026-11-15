<EX-Claim: Entity-aware Cross-lingual Claim Detection for Automated Fact-checking>
## Thông tin cơ bản
- **Tác giả:** Rrubaa Panchendrarajan and Arkaitz Zubiaga
- **Hội nghị/Tạp chí:** Findings of the Association for Computational Linguistics: EACL 2026 (also on arXiv)
- **Năm:** 2025 (arXiv ID: 2503.15220)
- **Link:** https://arxiv.org/abs/2503.15220

## Tóm tắt (Abstract)
Mô hình EX-Claim (Entity-aware Cross-lingual Claim Detection) được thiết kế nhằm cải thiện bài toán fact-checking tự động, đặc biệt đối với các tin giả lan truyền trên mạng xã hội. Bài báo giải quyết vấn đề nhận diện các claim cần kiểm chứng ở các ngôn ngữ khác nhau. Khác với các mô hình đa ngữ truyền thống thường gặp hạn chế khi chuyển giao tri thức sang các ngôn ngữ chưa được học (unseen languages), EX-Claim tích hợp thông tin thực thể (entity knowledge) thông qua Named Entity Recognition (NER) và Entity Linking để tăng cường hiệu suất và tính mạnh mẽ của mô hình trên 27 ngôn ngữ khác nhau.

## Bài toán & Động lực
Các mô hình ngôn ngữ đa ngữ (multilingual language models) hiện tại có thể xử lý nhiều ngôn ngữ, nhưng khả năng chuyển giao tri thức (cross-lingual transfer) để phát hiện claim giữa các ngôn ngữ high-resource và low-resource/unseen vẫn còn rất hạn chế. Động lực chính là lợi dụng thông tin từ các thực thể (entities - vốn thường giữ tính nhất quán cao qua các ngôn ngữ) để kết nối ngữ nghĩa và cải thiện khả năng phát hiện claim đa ngữ.

## Phương pháp chính
Mô hình EX-Claim sử dụng cơ chế "Entity-Aware" (nhận thức thực thể). Phương pháp kết hợp:
1. **Named Entity Recognition (NER):** Nhận diện các thực thể có trong câu.
2. **Entity Linking:** Liên kết các thực thể này với một Knowledge Base chung để lấy thông tin ngữ nghĩa độc lập với ngôn ngữ.
3. Tích hợp các đặc trưng thực thể này vào kiến trúc Transformer đa ngữ, giúp mô hình bám sát vào nội dung thực thể cốt lõi khi quyết định xem câu có phải là một claim cần kiểm chứng hay không.

## Dataset sử dụng
Đánh giá trên tập dữ liệu đa ngữ chứa 27 ngôn ngữ khác nhau (bao gồm cả các ngôn ngữ được huấn luyện và ngôn ngữ chưa từng thấy - unseen languages).

## Kết quả chính
- Mô hình đạt được mức tăng hiệu suất nhất quán trên 27 ngôn ngữ.
- Cải thiện đáng kể khả năng cross-lingual knowledge transfer, đặc biệt là khi áp dụng zero-shot cross-lingual transfer từ ngôn ngữ giàu tài nguyên (như tiếng Anh) sang các ngôn ngữ ít tài nguyên.

## Điểm mạnh
- Sử dụng NER và Entity Linking một cách sáng tạo để thu hẹp khoảng cách ngôn ngữ.
- Độ bao phủ ngôn ngữ rộng (27 ngôn ngữ), tính tổng quát cao.
- Có khả năng hoạt động tốt trên các ngôn ngữ chưa từng xuất hiện trong tập huấn luyện.

## Hạn chế / Limitations
- Phụ thuộc vào chất lượng của công cụ NER và Entity Linking cho từng ngôn ngữ; nếu NER hoạt động kém ở ngôn ngữ đích, hiệu suất chung có thể bị ảnh hưởng.
- Quá trình linking có thể làm tăng chi phí tính toán (computational overhead) so với các mô hình end-to-end thông thường.

## Liên quan đến ViFactCheck
- **Ứng dụng:** ViFactCheck có thể mượn ý tưởng "Entity-Aware" để tăng cường khả năng phát hiện claim trong tiếng Việt. Bằng cách nhận diện các thực thể (Tên người, Tổ chức, Địa điểm) thông qua PhoNER_COVID19 hoặc các bộ NER tiếng Việt, ViFactCheck có thể cải thiện việc so khớp claim-evidence.
- **Cross-lingual:** Ý tưởng này mở ra hướng kết hợp các claim tiếng Việt với các nguồn evidence tiếng Anh (hoặc ngược lại) thông qua việc gióng hàng các thực thể chung (ví dụ: "WHO", "Donald Trump").
