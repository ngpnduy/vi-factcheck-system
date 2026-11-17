# ViFactCheck — pipeline thí nghiệm v2

Repo này có **một pipeline chính** tại [`src/vifactcheck/`](src/vifactcheck/): BM25 chọn câu trong bài báo, PhoBERT rerank tối đa 20 câu thành 5 câu, rồi verifier dự đoán `SUPPORTED`, `REFUTED` hoặc `NOT_ENOUGH_INFO`. Các cấu hình M0, M1, A1, M2, A2 và protocol v2 được giải thích trong [hướng dẫn thí nghiệm](docs/protocol/v2.md).

## Kết quả hiện có

Thí nghiệm **Listwise v2 ba seed** đã hoàn tất trên 5.062 train, 723 dev và 1.447 test claim. M2 đạt **0.7066 ± 0.0057 test Macro-F1** (trung bình ± sample SD trên seeds 42, 13, 2026). Learning rate `1e-5` và evidence lambda `0.1` được chọn trên seed 42/dev trước khi đánh giá test. Đọc [báo cáo đầy đủ](reports/experiments/listwise-best5-v2-three-seeds-20261112/REPORT.md), [bảng kết quả](reports/experiments/listwise-best5-v2-three-seeds-20261112/results.csv) và [trạng thái checkpoint](reports/MODELS.md).

[Pilot 900 claim](reports/experiments/pilot-900-seed42/) và [full-data một seed tháng 10](reports/experiments/listwise-seed42/) là kết quả lịch sử, dùng protocol khác. Để bắt đầu từ khái niệm verdict, retrieval, loss, seed và metrics, đọc [hướng dẫn tiếng Việt](docs/guides/learning_guide_vi.md).

## Công việc tiếp theo: hội tụ và nhiều backbone

- [Audit loss/dev hiện có](reports/diagnostics/convergence-20261117/REPORT.md): train objective còn giảm; chưa đủ bằng chứng hội tụ sau năm epoch.
- [Kế hoạch hội tụ 15 epoch trên laptop hoặc P100](docs/protocol/convergence.md): lệnh Windows/Linux, điều kiện khởi chạy, train/dev loss tách thành phần, giữ best/last và resume; không đọc test.
- [Hướng dẫn teammate so sánh baseline với M2 theo backbone](docs/guides/backbone-benchmark.md) và [setup Conda/P100](docs/guides/server-conda.md).

Các công cụ chuẩn bị này dựa trên `release/v2`. Cập nhật 17/11/2026: run 15 epoch chưa khởi chạy; laptop đang chờ ổ output đạt ít nhất 35 GiB trống và P100 vẫn cần smoke trên server. Giữ toàn bộ checkpoint cũ. Classifier baseline độc lập và adapter tổng quát cho các backbone mới là bước triển khai kế tiếp, chưa phải chức năng đã hoàn tất của CLI. Giữ study năm epoch đã công bố riêng với pilot dài hơn và khóa protocol benchmark sau pilot.

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

Source của run ba seed được ghi ở commit `baa0bdc`; bản `release/v2` tại `2045e64` giữ nguyên 15 file Python của run. Nhánh công việc mới bổ sung observer/lịch LR tùy chọn trong `training.py` và module hội tụ riêng; source hiện tại không còn là snapshot nguyên byte của run cũ. Snapshot thực thi được giữ trong ZIP công bố, không sửa các báo cáo gốc. Bản công bố kết quả gốc là commit `0af01fd` và tag `listwise-v2-three-seeds-20261112`. Report và metadata nằm trong Git; ZIP phát hành có thêm predictions và source snapshot. Checkpoint weights và dataset không nằm trong Git/ZIP; M2 seed 13 được chia sẻ riêng tại [Release](https://github.com/ngpnduy/vi-factcheck-system/releases/tag/listwise-v2-three-seeds-20261112), cần quyền đọc repo private.

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
.\.venv\Scripts\python.exe -m vifactcheck predict --checkpoint '<duong-dan-checkpoint>' --input examples\claim.json --output runs\prediction.json
```

Run An đang huấn luyện cần tiếp tục trên checkout và launcher cũ, với checkpoint/data/config/output path đã ghi nhận. Các file Python được chuyển **nguyên byte**, nhưng đường dẫn package và launcher ở nhánh release khác; dùng nhánh này cho lần chạy mới sau khi run cũ hoàn tất.
