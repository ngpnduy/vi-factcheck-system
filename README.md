# ViFactCheck — pipeline thí nghiệm v2

Repo này có **một pipeline chính** tại [`src/vifactcheck/`](src/vifactcheck/): BM25 chọn câu trong bài báo, PhoBERT rerank tối đa 20 câu thành 5 câu, rồi verifier dự đoán `SUPPORTED`, `REFUTED` hoặc `NOT_ENOUGH_INFO`. Các cấu hình M0, M1, A1, M2, A2 và protocol v2 được giải thích trong [hướng dẫn thí nghiệm](docs/protocol/v2.md).

## Kết quả hiện có

Thí nghiệm **Listwise v2 ba seed** đã hoàn tất trên 5.062 train, 723 dev và 1.447 test claim. M2 đạt **0.7066 ± 0.0057 test Macro-F1** (trung bình ± sample SD trên seeds 42, 13, 2026). Learning rate `1e-5` và evidence lambda `0.1` được chọn trên seed 42/dev trước khi đánh giá test. Đọc [báo cáo đầy đủ](reports/experiments/listwise-best5-v2-three-seeds-20261112/REPORT.md), [bảng kết quả](reports/experiments/listwise-best5-v2-three-seeds-20261112/results.csv) và [trạng thái checkpoint](reports/MODELS.md).

[Pilot 900 claim](reports/experiments/pilot-900-seed42/) và [full-data một seed tháng 10](reports/experiments/listwise-seed42/) là kết quả lịch sử, dùng protocol khác. Để bắt đầu từ khái niệm verdict, retrieval, loss, seed và metrics, đọc [hướng dẫn tiếng Việt](docs/guides/learning_guide_vi.md).

## Cấu trúc

| Nơi | Nội dung |
|---|---|
| `src/vifactcheck/` | Source chính của pipeline v2 |
| `scripts/` | Setup và launcher Windows cho v2 |
| `tests/` | Kiểm tra cho code chính |
| `docs/protocol/` | Giao thức, cách chạy và bằng chứng validation |
| `docs/research/` | Ghi chú paper phục vụ báo cáo |
| `docs/guides/` | Hướng dẫn đọc dự án, thuật ngữ và quy trình viết báo |
| `docs/validation/` | Bằng chứng kiểm tra và provenance của v2 |
| `reports/` | Proposal, slide và báo cáo kết quả |

Source của run ba seed được ghi ở commit `baa0bdc`; 15 file Python của run giống byte-for-byte với `src/vifactcheck/` trong nhánh này. Bản công bố kết quả gốc là commit `0af01fd` và tag `listwise-v2-three-seeds-20261112`. Report và metadata nằm trong Git; ZIP phát hành có thêm predictions và source snapshot. Checkpoint weights và dataset không nằm trong Git/ZIP.

## Chạy trên Windows GPU

Yêu cầu Python 3.10+, Java, GPU CUDA phù hợp và đủ dung lượng checkpoint. Các lệnh chạy từ **gốc repo**:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
.\.venv\Scripts\python.exe -m vifactcheck prepare
.\.venv\Scripts\python.exe -m vifactcheck smoke --models M0 M1 A1 M2 A2 --reranker-architecture listwise --output runs\smoke-listwise-v2
powershell -ExecutionPolicy Bypass -File scripts\start_listwise_v2.ps1 -CheckOnly
powershell -ExecutionPolicy Bypass -File scripts\start_listwise_v2.ps1
```

Launcher của nhánh release dùng `%LOCALAPPDATA%\ViFactCheck\full-five-listwise-best5-v2` và `runs\full-five-listwise-best5-v2` cho **run mới**. Run đã công bố dùng đường dẫn ghi trong metadata trên máy thí nghiệm. Có thể truyền `-DataDirectory`, `-ModelDirectory`, `-SegmenterDirectory`, `-OutputDirectory` và `-StatusDirectory`. [Chi tiết protocol và resume](docs/protocol/v2.md).

Để dự đoán bằng checkpoint đã có:

```powershell
.\.venv\Scripts\python.exe -m vifactcheck predict --checkpoint <duong-dan-checkpoint> --input examples\claim.json --output runs\prediction.json
```

Run An đang huấn luyện cần tiếp tục trên checkout và launcher cũ, với checkpoint/data/config/output path đã ghi nhận. Các file Python được chuyển **nguyên byte**, nhưng đường dẫn package và launcher ở nhánh release khác; dùng nhánh này cho lần chạy mới sau khi run cũ hoàn tất.
