# Fact-Checking the Output of Large Language Models via Token-Level Uncertainty Quantification

## Thông tin cơ bản
- **Tác giả:** Không đề cập chi tiết trong abstract (Từ arXiv:2403.04696)
- **Hội nghị/Tạp chí:** Findings of ACL 2024
- **Năm:** 2024
- **Link:** https://arxiv.org/abs/2403.04696

## Tóm tắt (Abstract)
Large language models (LLMs) nổi tiếng với hiện tượng hallucination (sinh ra thông tin sai lệch). Các thông tin sai lệch nhỏ thường bị che lấp bởi phần lớn văn bản đúng, khiến người dùng rất khó phát hiện. Bài báo đề xuất một pipeline phát hiện hallucination và fact-checking dựa trên việc định lượng độ không chắc chắn (uncertainty) ở mức độ token. Phương pháp "Claim Conditioned Probability" (CCP) được giới thiệu để loại bỏ nhiễu và chỉ đo lường độ không chắc chắn của một claim cụ thể.

## Bài toán & Động lực
- Các hệ thống LLM hiện tại không cung cấp cơ chế để người dùng nhận biết các đoạn văn bản được sinh ra thiếu độ tin cậy.
- Rất khó để phát hiện các lỗi sai thực tế (factual inaccuracies) nằm rải rác trong một câu trả lời dài có vẻ hợp lý.

## Phương pháp chính
- **Token-Level Uncertainty Quantification:** Sử dụng thông tin từ output của neural network hoặc các layers của nó để tính toán điểm không chắc chắn (uncertainty scores) nhằm phát hiện các dự đoán không đáng tin cậy.
- **Claim Conditioned Probability (CCP):** Một phương pháp định lượng độ không chắc chắn ở mức token cải tiến, loại bỏ ảnh hưởng của sự không chắc chắn về "nên sinh ra claim nào" và "sử dụng hình thức bề mặt nào" (surface form). CCP chỉ tập trung đo lường độ không chắc chắn của giá trị claim thực sự được mô hình biểu đạt.

## Dataset sử dụng
Thử nghiệm trên bài toán sinh tiểu sử (biography generation) với 7 mô hình LLMs khác nhau trên 4 ngôn ngữ.

## Kết quả chính
- Phương pháp CCP mang lại sự cải thiện mạnh mẽ so với các baseline.
- Đánh giá của con người (Human evaluation) cho thấy pipeline fact-checking dựa trên định lượng uncertainty có hiệu suất cạnh tranh với các công cụ fact-checking sử dụng external knowledge (tri thức bên ngoài).

## Điểm mạnh
- Cung cấp phương pháp đánh giá độ tin cậy trực tiếp từ mô hình (white-box / gray-box signals) thay vì chỉ phụ thuộc vào truy xuất bên ngoài.
- Giải quyết được bài toán khó là hallucination tiềm ẩn trong văn bản dài.

## Hạn chế / Limitations
- Việc tính toán uncertainty ở mức token có thể đòi hỏi quyền truy cập vào logprobs của mô hình (không phải API nào cũng hỗ trợ hoàn chỉnh).
- Có thể gặp khó khăn khi áp dụng cho các mô hình closed-source API.

## Liên quan đến ViFactCheck
- **Ứng dụng:** ViFactCheck có thể tích hợp kỹ thuật định lượng uncertainty (như lấy logprobs từ API) để highlight các token/claim có nguy cơ sai lệch cao trong câu trả lời, từ đó tập trung quá trình fact-checking vào các vùng đó thay vì kiểm tra toàn bộ câu.
