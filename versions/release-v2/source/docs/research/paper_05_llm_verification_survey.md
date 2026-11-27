<Claim Verification in the Age of Large Language Models: A Survey>
## Thông tin cơ bản
- **Tác giả:** Alphaeus Dmonte, Roland Oruche, Marcos Zampieri, Prasad Calyam, Isabelle Augenstein
- **Hội nghị/Tạp chí:** arXiv (Preprint)
- **Năm:** 2024 (cập nhật 2025)
- **Link:** https://arxiv.org/abs/2408.14317

## Tóm tắt (Abstract)
Khảo sát này cung cấp một cái nhìn toàn diện về các hệ thống kiểm chứng thông tin tự động (automated claim verification) sử dụng Large Language Models (LLMs). Bài viết tổng hợp sự dịch chuyển từ các mô hình Deep Learning truyền thống sang các hệ thống dựa trên LLM, đặc biệt là kiến trúc Retrieval-Augmented Generation (RAG), đồng thời vạch ra các thách thức lớn như xung đột tri thức và tính đa ngôn ngữ.

## Bài toán & Động lực
Sự bùng nổ của thông tin sai lệch đòi hỏi các hệ thống fact-checking phải hoạt động tự động ở quy mô lớn. LLMs có khả năng sinh ngôn ngữ và suy luận tốt, nhưng thường bị "ảo giác" (hallucination). RAG được sử dụng như một giải pháp thiết yếu để "neo" (ground) LLMs vào các bằng chứng thực tế, tuy nhiên cần có một tài liệu tổng hợp hệ thống hóa các phương pháp và đường ống (pipeline) tốt nhất hiện nay.

## Phương pháp chính
Bài báo phân tích các thành phần của một pipeline RAG cho Fact-checking:
1. **Claim Detection & Identification:** Nhận diện các câu chứa thông tin cần kiểm chứng.
2. **Evidence Retrieval:** Lấy tài liệu từ web hoặc cơ sở dữ liệu để làm bằng chứng.
3. **Rationale Selection & Veracity Prediction:** Chọn các câu quan trọng làm lý do (rationale) và dự đoán nhãn đúng/sai.
4. **Explanation Generation:** LLMs sinh ra lời giải thích chi tiết cho kết luận.
Bài viết phân tích các kỹ thuật Prompting, In-context learning và Fine-tuning áp dụng cho từng bước.

## Dataset sử dụng
Khảo sát phân tích các benchmark phổ biến hiện có như FEVER, HOVER, LIAR, v.v.

## Kết quả chính
- Xác định RAG (Retrieval-Augmented Generation) là phương pháp tối ưu và "de-facto standard" cho LLM fact-checking.
- Nêu bật sức mạnh của các workflow agentic (VD: CRAG, Self-RAG) trong việc tự động đánh giá và tự sửa lỗi bằng chứng thu thập được.

## Điểm mạnh
- Tổng hợp toàn diện, cập nhật nhất (đến đầu năm 2025) về xu hướng sử dụng LLMs cho Fact-checking.
- Trình bày cấu trúc rõ ràng các thành phần của pipeline, thuận tiện cho việc thiết kế hệ thống.

## Hạn chế / Limitations
- Vì là một paper khảo sát (survey), không đề xuất một mô hình kỹ thuật cụ thể mới mà chỉ tổng hợp các nghiên cứu đã có.

## Liên quan đến ViFactCheck
- **Ứng dụng tiềm năng:** Định hướng ViFactCheck phát triển theo chuẩn RAG Pipeline được phân tích trong bài. Cung cấp cơ sở lý thuyết để thiết kế các prompt hiệu quả cho bước Verification và Generation giải thích trong tiếng Việt. Nó cũng cảnh báo hệ thống cần phải xử lý vấn đề context không liên quan hoặc mâu thuẫn từ nguồn.
</Claim Verification in the Age of Large Language Models: A Survey>
