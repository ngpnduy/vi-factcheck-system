<Evidence Retrieval is almost All You Need for Fact Verification>
## Thông tin cơ bản
- **Tác giả:** Liwen Zheng, Chaozhuo Li, Xi Zhang, Yu-Ming Shang, Feiran Huang, Haoran Jia
- **Hội nghị/Tạp chí:** Findings of the Association for Computational Linguistics: ACL 2024
- **Năm:** 2024
- **Link:** https://aclanthology.org/2024.findings-acl.551/

## Tóm tắt (Abstract)
Nghiên cứu này chỉ ra rằng hiệu suất của các hệ thống fact verification phụ thuộc rất nhiều vào chất lượng của khâu trích xuất bằng chứng (evidence retrieval), vốn thường bị xem nhẹ so với khâu xác thực (claim verification). Nhóm tác giả đề xuất RAV (Retrieval-Augmented Verification), một framework sử dụng phương pháp trích xuất kết hợp (hybrid retrieval) và huấn luyện end-to-end kết hợp tín hiệu phản hồi từ module xác thực.

## Bài toán & Động lực
Hầu hết các nghiên cứu hiện tại thường tập trung thiết kế các mô hình suy luận (reasoning) phức tạp cho bước claim verification, nhưng lại dùng các phương pháp truy xuất bằng chứng cơ bản dựa trên độ tương đồng ngữ nghĩa. Điều này dẫn đến việc lấy phải các bằng chứng không liên quan, làm giảm hiệu năng toàn bộ hệ thống. Do đó, cần một cách tiếp cận tập trung mạnh mẽ vào chất lượng truy xuất.

## Phương pháp chính
- **Hybrid Evidence Retrieval:** Áp dụng phương pháp hai bước: (1) dùng một retriever nhanh để cắt tỉa (pruning) và chọn lọc danh sách ứng viên ban đầu, (2) dùng một ranker chính xác hơn để sắp xếp lại (reranking) danh sách đó.
- **Joint Fact Verification (End-to-End):** Cho phép lan truyền ngược (back-propagate) gradient từ module kiểm chứng (verification) về module trích xuất bằng chứng, giúp quá trình chọn lọc bằng chứng được tối ưu trực tiếp cho mục tiêu cuối cùng (veracity prediction).

## Dataset sử dụng
- **FEVER** (Fact Extraction and VERification).

## Kết quả chính
- Mô hình RAV cho thấy việc tối ưu khâu retrieval đóng vai trò quyết định đến hiệu năng tổng thể.
- Đạt được kết quả tốt và vượt trội trên tập dữ liệu FEVER so với nhiều mô hình có module suy luận phức tạp.

## Điểm mạnh
- Chứng minh rõ ràng bằng thực nghiệm luận điểm "truy xuất bằng chứng là yếu tố quan trọng nhất".
- Phương pháp học end-to-end giúp hệ thống tự điều chỉnh và tìm ra bằng chứng thực sự "hữu ích" cho việc xác thực thay vì chỉ có ngữ nghĩa tương đồng.

## Hạn chế / Limitations
- Việc huấn luyện end-to-end toàn bộ pipeline đòi hỏi kỹ thuật phức tạp và tốn kém tài nguyên tính toán hơn so với việc tách rời (pipeline độc lập).

## Liên quan đến ViFactCheck
- **Ứng dụng tiềm năng:** Nhắc nhở dự án ViFactCheck nên ưu tiên nguồn lực và tối ưu hóa hệ thống Retrieval (vector search + BM25, reranking) thay vì chỉ chăm chăm tối ưu Prompt cho LLM. Áp dụng cơ chế Hybrid Retrieval (kết hợp keyword-based và semantic-based) như đề xuất của bài báo để cải thiện khả năng tìm kiếm bằng chứng tiếng Việt.
</Evidence Retrieval is almost All You Need for Fact Verification>
