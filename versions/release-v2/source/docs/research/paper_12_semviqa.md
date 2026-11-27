<SemViQA: A Semantic Question Answering System for Vietnamese Information Fact-Checking>
## Thông tin cơ bản
- **Tác giả:** Nhóm nghiên cứu (Preprint on arXiv / ACL 2026 industry track)
- **Hội nghị/Tạp chí:** arXiv / GitHub release
- **Năm:** 2025 (arXiv ID: 2503.00955)
- **Link:** https://arxiv.org/abs/2503.00955 / https://github.com/DAVID-NGUYEN-S16/SemViQA

## Tóm tắt (Abstract)
SemViQA (Semantic Question Answering System for Vietnamese Information Fact-Checking) là một framework nghiên cứu hướng tới việc giải quyết vấn nạn tin giả trong tiếng Việt. Khác với các mô hình chung chung, SemViQA được tối ưu hóa đặc biệt cho các thách thức ngôn ngữ của tiếng Việt (như từ đồng âm, cấu trúc phức tạp, sự khan hiếm dữ liệu). Hệ thống thiết kế một pipeline kết hợp Semantic-based Evidence Retrieval (SER) và Two-step Verdict Classification (TVC) để phân loại tính đúng đắn của claim.

## Bài toán & Động lực
Fact-checking tiếng Việt (low-resource language) gặp nhiều khó khăn do giới hạn về các tập dữ liệu chất lượng cao, các từ vựng đặc thù và hiện tượng đa nghĩa/đồng âm. Các hệ thống dịch máy (translate-then-verify) thường làm mất đi sắc thái ngữ nghĩa (nuances). Động lực của bài báo là tạo ra một end-to-end framework gốc bằng tiếng Việt với tốc độ suy luận nhanh và độ chính xác cao.

## Phương pháp chính
Hệ thống SemViQA gồm các component:
1. **Data Processing:** Quản lý ngữ cảnh dài (long-context inputs), sử dụng thư viện `Underthesea` để word-segmentation (tách từ tiếng Việt).
2. **Semantic-based Evidence Retrieval (SER):** Kết hợp TF-IDF truyền thống với mô hình Question Answering Token Classifier (QATC) để truy xuất các bằng chứng (evidence) có độ chính xác cao, đồng thời giảm thời gian suy luận.
3. **Two-step Verdict Classification (TVC):** Sử dụng chiến lược phân loại phân cấp (hierarchical classification), được tối ưu hóa bởi Cross-Entropy và Focal Loss, để đưa ra nhãn cuối cùng: **Supported**, **Refuted**, hoặc **Not Enough Information (NEI)**.
4. **SemViQA Faster:** Một phiên bản được nén/tối ưu hóa giúp tăng tốc độ suy luận gấp 7 lần.

## Dataset sử dụng
Mô hình được đánh giá trên các tập dữ liệu fact-checking tiếng Việt lớn:
- **ISE-DSC01**
- **ViWikiFC**

## Kết quả chính
- Đạt độ chính xác (accuracy) **78.97%** trên tập ISE-DSC01.
- Đạt độ chính xác **80.82%** trên tập ViWikiFC.
- Đóng góp mã nguồn mở và pre-trained checkpoints (trên HuggingFace).

## Điểm mạnh
- Được thiết kế tối ưu riêng cho tiếng Việt, tận dụng tốt các thư viện tokenizer của tiếng Việt (Underthesea).
- Mô hình chạy nhanh (với phiên bản Faster), thích hợp cho triển khai hệ thống thời gian thực (real-time).
- Cấu trúc module rõ ràng (Retrieval + Classification).

## Hạn chế / Limitations
- Chưa tận dụng tối đa khả năng của các mô hình sinh văn bản lớn (Generative LLMs) để cung cấp phần "giải thích" (explanation) thân thiện với người dùng cuối, mà mới tập trung vào bài toán Classification.
- Chủ yếu dựa vào nguồn dữ liệu Wikipedia (ViWikiFC) hoặc cấu trúc câu hỏi có sẵn, có thể gặp khó khăn với tin giả trên mạng xã hội (văn phong lóng, không chính thức).

## Liên quan đến ViFactCheck
- **Ứng dụng Trực tiếp:** Đây là công trình nghiên cứu cực kỳ sát với ViFactCheck. ViFactCheck có thể tham khảo trực tiếp pipeline Two-step Verdict Classification (TVC) và kiến trúc SER của SemViQA.
- **Tài nguyên:** Có thể sử dụng lại tập dữ liệu **ISE-DSC01**, **ViWikiFC** và các pre-trained models của SemViQA (có trên HuggingFace) để fine-tune hoặc làm baseline cho hệ thống ViFactCheck.
- **Tokenizer:** Bài báo nhấn mạnh vai trò của `Underthesea` trong word-segmentation, ViFactCheck cũng nên tích hợp bước tiền xử lý này để tăng hiệu quả embedding tiếng Việt.
