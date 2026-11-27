# ClaimVer: Explainable Claim-Level Verification and Evidence Attribution of Text Through Knowledge Graphs

## Thông tin cơ bản
- **Tác giả:** Preetam Prabhu Srikar Dammu, Himanshu Naidu, Mouly Dewan, YoungMin Kim, Tanya Roosta, Aman Chadha, Chirag Shah
- **Hội nghị/Tạp chí:** Findings of the Association for Computational Linguistics (EMNLP)
- **Năm:** 2024
- **Link:** https://arxiv.org/abs/2403.09724

## Tóm tắt (Abstract)
ClaimVer là một framework giải thích (explainable) và hướng đến người dùng (human-centric) được thiết kế cho việc xác minh ở cấp độ claim (claim-level verification) và phân bổ bằng chứng (evidence attribution) trong văn bản. Hệ thống xác minh các tuyên bố độc lập dựa trên các đồ thị tri thức (Knowledge Graphs) tin cậy và gán điểm số phân bổ (attribution score) để định lượng độ tin cậy của bằng chứng.

## Bài toán & Động lực
Sự bùng nổ của thông tin sai lệch và ảo giác (hallucinations) từ AI đòi hỏi các phương pháp đánh giá minh bạch. Đa số các hệ thống hiện tại chỉ cung cấp nhãn xác minh ở mức văn bản hoặc mức độ chung chung, làm tăng tải nhận thức (cognitive load) cho người dùng khi họ phải tự tìm hiểu xem phần nào đúng/sai và tại sao.

## Phương pháp chính
- **Xác minh ở cấp độ Claim (Claim-level):** Trích xuất và cô lập từng claim cụ thể trong văn bản để đánh giá chéo với Knowledge Graph (ví dụ: Wikidata).
- **Phân bổ bằng chứng (Evidence Attribution):** Hệ thống không chỉ xác minh mà còn đưa ra bằng chứng hỗ trợ hoặc phản bác, kết hợp với một "điểm phân bổ" (attribution score) để đánh giá mức độ tin cậy của bằng chứng đó.
- Cung cấp các lời giải thích ngắn gọn, rõ ràng để xây dựng lòng tin.

## Dataset sử dụng
(Bài báo có công bố dataset đi kèm trên Hugging Face nhằm hỗ trợ việc đánh giá các tác vụ claim-level verification và evidence attribution).

## Kết quả chính
Cung cấp một pipeline hoàn chỉnh giúp phân rã văn bản thành các claim, kiểm tra với knowledge graph và đưa ra điểm số kèm giải thích. Cách tiếp cận này giúp giảm tải nhận thức và tăng tính tin cậy cho quá trình đọc hiểu và kiểm chứng thông tin của người dùng.

## Điểm mạnh
- Focus vào yếu tố con người (human-centric), làm cho kết quả fact-checking trở nên dễ hiểu, actionable.
- Sử dụng Knowledge Graph giúp kết quả có độ chuẩn xác và tính cấu trúc cao.
- Giải quyết được bài toán ảo giác sinh văn bản của LLM.

## Hạn chế / Limitations
- Phụ thuộc vào độ phủ và mức độ cập nhật của Knowledge Graph (ví dụ Wikidata có thể thiếu các sự kiện "breaking news").
- Xác minh bằng KG đôi khi hạn chế đối với các nhận định mang tính ngữ nghĩa phức tạp hoặc không có dạng quan hệ cấu trúc rõ ràng.

## Liên quan đến ViFactCheck
- **Ứng dụng:** ViFactCheck có thể áp dụng ý tưởng "Attribution Score" để đánh giá độ uy tín của nguồn tin/bằng chứng thu thập được. Đồng thời, việc chia nhỏ bài viết thành các claim nhỏ (claim-level) cũng là chiến lược tốt để người dùng dễ theo dõi phần nào trong tin tức là sai lệch.
- **Định hướng:** Việc sinh giải thích nên ngắn gọn và đi kèm các trích dẫn (citations) rõ ràng để người dùng có thể đối chiếu.
