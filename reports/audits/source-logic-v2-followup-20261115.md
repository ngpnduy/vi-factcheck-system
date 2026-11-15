# Đối chiếu audit nguồn với kết quả Listwise v2

Audit [11/11/2026](source-logic-v2.md) được viết trước khi run ba seed hoàn tất. Bản này cập nhật **trạng thái diễn giải**, không thay đổi số liệu hoặc nội dung audit gốc. Nguồn kết quả là [REPORT.md](../experiments/listwise-best5-v2-three-seeds-20261112/REPORT.md), nguồn code là [`src/vifactcheck/`](../../src/vifactcheck/).

| Phát hiện | Trạng thái sau run ba seed |
|---|---|
| F1 — alignment hạn chế mẫu retrieval | Vẫn áp dụng. Test classification tính 1.447 claim, retrieval chỉ tính 539 claim SUPPORTED/REFUTED có alignment hợp lệ. Không diễn giải Hit@k như kết quả cho toàn test. |
| F2 — D0 và phân tích loại mâu thuẫn | Chưa có trong study năm cấu hình ba seed. Đây là phần việc nghiên cứu còn mở, không được tính vào kết luận RQ3. |
| F3 — mẫu số `alignment_coverage` | Vẫn áp dụng. Giá trị `0.5923` là valid trên 1.447 claim; retrieval count là 539. |
| F4 — warmup checkpoint có thể thắng | Protocol vẫn cho phép. Tuy nhiên cả sáu checkpoint M2/A2 được chọn của ba seed trong report đều ở giai đoạn `continuation`, nên trường hợp lo ngại không xảy ra trong kết quả cuối này. |
| F5 — `zip(rows, rankings)` có thể cắt dữ liệu | Còn mở trong hàm metrics hiện tại. Không có bằng chứng report 15 test bị ảnh hưởng; cần xử lý riêng khi sửa source. |
| F6 — top 5 không luôn chứa đủ gold | Vẫn là giới hạn thiết kế; `complete_gold@5` cần đọc cùng Hit@5 và số gold sentences. |
| F7 — nguồn tham khảo | Ghi chú paper mở rộng đã được đưa vào `docs/research/`. Cần tiếp tục phân biệt nguồn cảm hứng với việc tái hiện paper. |

Các checkpoint weights không có trong repo. Metadata `selected_stage` và `selected_global_epoch` ghi cách chọn checkpoint, không thay thế file weights.
