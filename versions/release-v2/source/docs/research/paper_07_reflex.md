# REFLEX: REason-guided Fact-checking with Latent EXplanations

## Thông tin cơ bản
- **Tác giả:** Kong et al.
- **Hội nghị/Tạp chí:** ACL
- **Năm:** 2024/2026
- **Link:** N/A (Tìm kiếm qua thông tin xuất bản mới)

## Tóm tắt (Abstract)
REFLEX là một framework nhằm cải thiện độ chính xác và tính trung thực (faithfulness) của các hệ thống fact-checking có khả năng giải thích (explainable fact-checking). Bằng cách chuyển bài toán fact-checking thành quy trình hỏi-đáp đối thoại (dialogue-style question-answering), REFLEX huấn luyện LLM để đồng thời đưa ra phán quyết (verdict) và lời giải thích. Framework gồm 3 giai đoạn: huấn luyện theo phong cách đối thoại, chọn lọc mẫu thích ứng (adaptive sample selection), và điều hướng dựa trên tự giải thích (Self-Explanation Guided Steering).

## Bài toán & Động lực
- Các hệ thống hiện tại thường phụ thuộc nhiều vào truy xuất tri thức bên ngoài, có thể gây ra độ trễ hoặc bị nhiễu thông tin.
- LLM có xu hướng gặp hiện tượng "faithfulness hallucinations" - sinh ra lời giải thích nghe có vẻ hợp lý nhưng không thực sự phản ánh quá trình suy luận dẫn đến phán quyết cuối cùng của mô hình.

## Phương pháp chính
REFLEX sử dụng cách tiếp cận fine-tuning / instruction-tuning tinh vi gồm 3 bước:
1. **Dialogue-style Fact-Checker Training:** Instruction-tune LLM để sinh ra phán quyết và giải thích theo định dạng hội thoại, kích hoạt tri thức bên trong của mô hình thay vì chỉ phụ thuộc external retrieval.
2. **Adaptive Sample Selection:** Xác định các mẫu huấn luyện mà mô hình backbone và mô hình fine-tuned không đồng tình (self-disagreement) để cô lập các đường dẫn suy luận chất lượng cao.
3. **Self-Explanation Guided Steering (S-EGS):** Trích xuất steering vectors từ các mẫu đã chọn để tách biệt phần "sự thật" khỏi "phong cách". Ở quá trình suy luận (inference), các vector này được dùng để triệt tiêu các yếu tố văn phong đánh lừa và khuếch đại lập luận trung thực.

## Dataset sử dụng
Sử dụng các bộ dữ liệu fact-checking tiêu chuẩn để fine-tune (instruction tuning) và sinh dữ liệu dạng đối thoại.

## Kết quả chính
REFLEX giảm thiểu đáng kể hiện tượng "faithfulness hallucinations", đảm bảo lời giải thích bám sát vào lý do thực sự đằng sau phán quyết của mô hình, đồng thời tăng tốc độ do giảm bớt sự phụ thuộc vào truy xuất bên ngoài.

## Điểm mạnh
- Định dạng task thành đối thoại giúp tự nhiên hóa quá trình giải thích.
- Kỹ thuật Steering vectors ở thời gian chạy (inference) là một cách tiếp cận tiên tiến để kiểm soát tính trung thực của mô hình.

## Hạn chế / Limitations
- Quá trình huấn luyện 3 bước phức tạp và đòi hỏi kỹ thuật cao.
- Có thể bị giới hạn bởi tri thức nội tại của LLM (nếu không kết hợp RAG, có thể không kiểm chứng được các sự kiện quá mới).

## Liên quan đến ViFactCheck
- **Ứng dụng:** ViFactCheck có thể áp dụng Prompt Engineering hoặc Instruction Tuning theo phong cách đối thoại (Dialogue-style) để mô hình sinh ra lời giải thích tự nhiên và logic hơn cho người dùng Việt Nam.
- Phương pháp điều hướng sinh văn bản có thể được tham khảo để giảm thiểu rủi ro LLM "bịa" ra lời giải thích sai lệch so với phán quyết fact-check ban đầu.
