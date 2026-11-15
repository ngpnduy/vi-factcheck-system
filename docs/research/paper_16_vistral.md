# Vistral-7B-Chat: Towards a State-of-the-Art Large Language Model for Vietnamese

## Thông tin cơ bản
- **Tác giả:** Nhóm phát triển Vistral (Bui et al.)
- **Hội nghị/Tạp chí:** Technical Report / Hugging Face Release
- **Năm:** 2024
- **Link:** https://huggingface.co/Viet-Mistral/Vistral-7B-Chat

## Tóm tắt (Abstract)
Vistral-7B-Chat là một mô hình ngôn ngữ lớn (LLM) được thiết kế đặc biệt và tối ưu hóa cho tiếng Việt, dựa trên kiến trúc của Mistral-7B. Mô hình được huấn luyện thông qua quá trình continual pre-training và supervised fine-tuning (SFT) trên một lượng lớn dữ liệu tiếng Việt chất lượng cao. Mục tiêu là tạo ra một mô hình mã nguồn mở có khả năng xử lý ngôn ngữ tự nhiên tiếng Việt đạt mức state-of-the-art (SOTA), vượt qua các rào cản về mặt ngôn ngữ và văn hóa mà các mô hình quốc tế thường gặp phải khi xử lý tiếng Việt.

## Bài toán & Động lực
Mặc dù các mô hình LLM lớn như GPT-4 hay Llama 3 có khả năng tiếng Việt, chúng thường không tối ưu về mặt tokenization (chia từ) cho tiếng Việt, dẫn đến chi phí suy luận cao và đôi khi không nắm bắt tốt các sắc thái văn hóa, ngôn ngữ địa phương. Các mô hình tiếng Việt trước đây (như PhoGPT) đã đạt được những tiến bộ nhất định, nhưng cộng đồng vẫn cần các mô hình mã nguồn mở dựa trên kiến trúc hiện đại (như Mistral) để dễ dàng triển khai, fine-tune và sử dụng cho các bài toán đặc thù với chi phí phần cứng thấp.

## Phương pháp chính
- **Continual Pre-training:** Sử dụng kiến trúc Mistral 7B và tiếp tục huấn luyện (pre-train) trên kho ngữ liệu tiếng Việt lớn để mô hình học được ngữ pháp và từ vựng tiếng Việt một cách tự nhiên.
- **Mở rộng Tokenizer:** Cải tiến tokenizer để xử lý tiếng Việt hiệu quả hơn, giảm thiểu hiện tượng fragmented tokens (chia cắt từ tiếng Việt quá nhỏ), giúp tăng tốc độ inference và giảm dung lượng context window cần thiết.
- **Supervised Fine-tuning (SFT) & Alignment:** Fine-tune mô hình trên các tập dữ liệu hội thoại (chat) và chỉ thị (instruction) tiếng Việt chất lượng cao, giúp mô hình trả lời tự nhiên, an toàn và tuân thủ các chuẩn mực văn hóa Việt Nam.

## Dataset sử dụng
- Dữ liệu pre-train gồm các văn bản tiếng Việt thu thập từ tin tức, Wikipedia, và các nguồn web tiếng Việt chất lượng cao.
- Dữ liệu SFT là các tập hội thoại và instruction-following đa dạng đã được dịch hoặc tạo ra bằng tiếng Việt.

## Kết quả chính
- Vistral-7B-Chat thể hiện hiệu năng xuất sắc trên các benchmark xử lý ngôn ngữ tự nhiên tiếng Việt (như VMLU).
- Cung cấp khả năng sinh văn bản tiếng Việt trôi chảy, tự nhiên và chính xác hơn nhiều so với các mô hình cùng cỡ không được tối ưu cho tiếng Việt.
- Lọt top các mô hình mã nguồn mở được tải và sử dụng nhiều nhất trong cộng đồng AI Việt Nam (trên Hugging Face).

## Điểm mạnh
- **Tokenization hiệu quả:** Tokenizer tối ưu cho tiếng Việt giúp tiết kiệm chi phí tính toán.
- **Hiệu năng cao với kích thước nhỏ:** Chỉ với 7B tham số, mô hình có thể chạy dễ dàng trên các GPU phổ thông hoặc thậm chí consumer hardware, rất phù hợp cho nghiên cứu và triển khai thực tế.
- **Mã nguồn mở:** Đóng góp to lớn cho cộng đồng, thúc đẩy nghiên cứu NLP tiếng Việt.

## Hạn chế / Limitations
- Kích thước 7B tham số giới hạn khả năng suy luận logic (reasoning) phức tạp so với các mô hình khổng lồ như GPT-4.
- Có thể vẫn gặp hiện tượng hallucination trong một số tác vụ đòi hỏi kiến thức chuyên sâu hoặc xác minh tính sự thật (fact-checking) nếu không có hệ thống RAG đi kèm.

## Liên quan đến ViFactCheck
Vistral-7B-Chat là một ứng viên sáng giá làm model nền tảng (backbone) cho dự án ViFactCheck. Thay vì phụ thuộc vào API của các mô hình lớn (đắt đỏ và chậm), chúng ta có thể sử dụng các kỹ thuật PEFT (như LoRA/QLoRA) để fine-tune Vistral-7B-Chat trực tiếp trên tập dữ liệu kiểm chứng thông tin tiếng Việt. Điều này giúp tối ưu hóa hệ thống cả về mặt chi phí vận hành lẫn độ chính xác khi phân tích ngữ nghĩa, nhận diện claim và đánh giá evidence bằng tiếng Việt.
