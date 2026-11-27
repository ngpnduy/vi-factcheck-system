# Improving Explainable Fact-Checking with Claim-Evidence Correlations

## Thông tin cơ bản
- **Tác giả:** Xin Tan, Bowei Zou, Ai Ti Aw
- **Hội nghị/Tạp chí:** Proceedings of the 31st International Conference on Computational Linguistics (COLING)
- **Năm:** 2025
- **Link:** https://aclanthology.org/2025.coling-main.108/

## Tóm tắt (Abstract)
Các hệ thống fact-checking tự động sử dụng Large Language Models (LLMs) đã đạt được hiệu suất ngang bằng con người trong việc chống lại thông tin sai lệch. Tuy nhiên, các hệ thống dựa trên LLM hiện tại thất bại trong việc tiết lộ các nguyên tắc suy luận đằng sau việc ra quyết định cho nhãn tính xác thực của claim. Nghiên cứu này đề xuất CorXFact (Correlation-Enhanced Explainable Fact-Checking), một hệ thống mô phỏng nguyên tắc suy luận của con người: đánh giá và đo lường sự tương quan giữa claim và từng phần evidence. CorXFact cho phép xác minh claim hiệu quả và tạo ra các giải thích minh bạch. Ngoài ra, nhóm nghiên cứu cũng đóng góp bộ dữ liệu CorFEVER để đánh giá toàn diện khả năng của hệ thống.

## Bài toán & Động lực
Mặc dù LLM giúp tự động hóa việc fact-checking rất tốt, nhưng chúng thường hoạt động như những "hộp đen" (black-box), không giải thích được lý do tại sao lại đưa ra quyết định đó. Điều này làm giảm sự tin cậy của người dùng. Việc mô phỏng cách con người cân nhắc các luồng thông tin và bằng chứng là cần thiết để tạo ra lý do/phân tích (explanation).

## Phương pháp chính
- **CorXFact framework:** Đánh giá độ tương quan (correlation) một cách tường minh giữa một claim và các bằng chứng (evidence) được truy xuất.
- Mô phỏng quá trình suy luận của con người bằng cách tính trọng số cho độ liên quan của từng bằng chứng, thay vì chỉ dựa vào xác suất nội tại của LLM.
- **Tạo sinh giải thích (Explanation generation):** Dựa trên mức độ tương quan để tổng hợp thành lời giải thích minh bạch, trực quan cho người dùng.

## Dataset sử dụng
- **CorFEVER:** Một tập test set mới được giới thiệu trong bài báo, dùng để đánh giá hệ thống ở hai tác vụ: nhận dạng sự tương quan giữa claim-evidence và xác minh claim trong các kịch bản thực tế cũng như closed-domain.

## Kết quả chính
CorXFact vượt trội hơn so với 4 baseline mạnh về fact-checking trong cả việc dự đoán tính xác thực của claim (authenticity prediction) và tạo sinh giải thích cho kết quả (verdict explanation).

## Điểm mạnh
- Cung cấp cách tiếp cận có khả năng giải thích (explainable) cao, không chỉ đưa ra nhãn Đúng/Sai mà còn chỉ ra lý do dựa trên bằng chứng nào.
- Dễ dàng tích hợp với các API LLM lớn (như GPT-4) hay các mô hình vừa (như Llama-2).

## Hạn chế / Limitations
- Phụ thuộc vào chất lượng của hệ thống truy xuất bằng chứng (evidence retrieval).
- Có thể chưa xử lý tốt các claim phức tạp yêu cầu suy luận đa bước dài hoặc bằng chứng đa phương tiện (multimodal).

## Liên quan đến ViFactCheck
- **Ứng dụng:** ViFactCheck có thể học hỏi cách tính toán và biểu diễn độ tương quan (correlation) giữa các bằng chứng (tiếng Việt) và claim để từ đó sinh ra các đoạn giải thích (justification) dễ hiểu và thuyết phục cho người dùng, thay vì chỉ đưa ra nhãn đính kèm.
- **Động lực:** Hướng tới tính minh bạch cho hệ thống ViFactCheck, giúp nhà báo hoặc người dùng thông thường dễ dàng kiểm chứng lại kết quả của AI.
