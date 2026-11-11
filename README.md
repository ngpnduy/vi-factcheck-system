# ViFactCheck — pipeline thí nghiệm v2

Repo này có **một pipeline chính** tại [`src/vifactcheck/`](src/vifactcheck/): BM25 chọn câu trong bài báo, PhoBERT rerank tối đa 20 câu thành 5 câu, rồi verifier dự đoán `SUPPORTED`, `REFUTED` hoặc `NOT_ENOUGH_INFO`. Các cấu hình M0, M1, A1, M2, A2 và protocol v2 được giải thích trong [hướng dẫn thí nghiệm](docs/protocol/v2.md).

## Kết quả hiện có

M2 listwise đạt **0.7046 test Macro-F1** trong thí nghiệm full-data **một seed (42)** ngày 30/10/2026; đây là điểm cao nhất trong năm cấu hình của thí nghiệm đó. [Báo cáo và bảng điểm](reports/experiments/listwise-seed42/) ghi rõ giao thức và giới hạn. [Pilot 900 claim](reports/experiments/pilot-900-seed42/) là một thí nghiệm trước đó, không dùng điểm pilot thay cho điểm full-data.

V2 chạy lại năm cấu hình trên các seed 42, 13, 2026, chọn learning rate/lambda bằng validation và chọn checkpoint tốt nhất trong năm epoch verifier. **Chưa có báo cáo kết quả full v2 được đưa vào nhánh này.** Không lấy điểm smoke hoặc điểm tháng 10 làm kết quả v2. Checkpoint weights không nằm trong Git; [trạng thái checkpoint](reports/MODELS.md) ghi rõ khả năng chạy inference.

## Cấu trúc

| Nơi | Nội dung |
|---|---|
| `src/vifactcheck/` | Source chính của pipeline v2 |
| `scripts/` | Setup và launcher Windows cho v2 |
| `tests/` | Kiểm tra cho code chính |
| `docs/protocol/` | Giao thức, cách chạy và bằng chứng validation |
| `docs/research/` | Ghi chú paper phục vụ báo cáo |
| `reports/` | Proposal, slide và báo cáo kết quả |

Source tái hiện paper, M0 độc lập, pilot/slides scripts và dữ liệu thí nghiệm chi tiết được giữ tại commit [`baa0bdc`](https://github.com/ngpnduy/vi-factcheck-system/tree/baa0bdc80379d26e661bd1dcf5e37b3b48ba14a8) trên nhánh `archive/pre-release-v2`. Lịch sử này được giữ nguyên để truy vết; nhánh An dùng để train v2 vẫn ở commit gốc.

## Chạy trên Windows GPU

Yêu cầu Python 3.10+, Java, GPU CUDA phù hợp và đủ dung lượng checkpoint. Các lệnh chạy từ **gốc repo**:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
.\.venv\Scripts\python.exe -m vifactcheck prepare
.\.venv\Scripts\python.exe -m vifactcheck smoke --models M0 M1 A1 M2 A2 --reranker-architecture listwise --output runs\smoke-listwise-v2
powershell -ExecutionPolicy Bypass -File scripts\start_listwise_v2.ps1 -CheckOnly
powershell -ExecutionPolicy Bypass -File scripts\start_listwise_v2.ps1
```

Launcher giữ checkpoint ở `%LOCALAPPDATA%\ViFactCheck\full-five-listwise-best5-v2` và ghi trạng thái vào `runs\full-five-listwise-best5-v2`. Chạy lại **cùng lệnh, cùng source và dữ liệu** để resume. Có thể truyền `-DataDirectory`, `-ModelDirectory`, `-SegmenterDirectory`, `-OutputDirectory` và `-StatusDirectory` khi dùng vị trí khác. [Chi tiết protocol và resume](docs/protocol/v2.md).

Để dự đoán bằng checkpoint đã có:

```powershell
.\.venv\Scripts\python.exe -m vifactcheck predict --checkpoint <duong-dan-checkpoint> --input examples\claim.json --output runs\prediction.json
```

Run An đang huấn luyện cần tiếp tục trên checkout và launcher cũ, với checkpoint/data/config/output path đã ghi nhận. Các file Python được chuyển **nguyên byte**, nhưng đường dẫn package và launcher ở nhánh release khác; dùng nhánh này cho lần chạy mới sau khi run cũ hoàn tất.
