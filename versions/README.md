# Các phiên bản thực thi

Code đang phát triển nằm ở `../src/vifactcheck/`. Thư mục này giữ các snapshot
code, cấu hình môi trường, tests và hướng dẫn cũ để không phải đổi nhánh khi
đối chiếu. Mỗi snapshot có commit gốc và SHA-256 từng file trong `MANIFEST.json`.
Các file nguồn được giữ nguyên byte; không sửa thuật toán hay chạy lại training.

| Phiên bản | Code | Kết quả / trạng thái |
|---|---|---|
| Rush M0/M1/M2 | [rush](rush/README.md) | [Pilot 900 claim](../reports/experiments/pilot-900-seed42/); runner full data cũng có trong snapshot |
| List-aware, năm cấu hình, seed 42 | [listwise-seed42](listwise-seed42/README.md) | [Báo cáo và artifacts tháng 10](../reports/experiments/listwise-seed42/) |
| Listwise v2, năm cấu hình, ba seed | [listwise-v2](listwise-v2/README.md) | [Bộ kết quả tháng 11](../reports/experiments/listwise-best5-v2-three-seeds-20261112/) |
| Release v2 và chuẩn bị hội tụ | [release-v2](release-v2/README.md) | [Phân tích hội tụ từ run cũ](../reports/diagnostics/convergence-20261117/REPORT.md); chưa có kết quả run 15 epoch trong Git |
| Benchmark backbone | [Code hiện tại](../src/vifactcheck/) | [mBERT: hai seed đã đánh giá test](../reports/experiments/backbone-mbert-test-two-seeds/); provenance M2 seed 13 còn thiếu metadata huấn luyện |

## Nhánh nào tương ứng phiên bản nào?

- `Rush-version` → `rush`.
- `slides-listwise-seed42` → `listwise-seed42`.
- `listwise-best5-three-seeds` là bản công bố; code thực thi được ghim ở commit
  `baa0bdc`, lưu trong `listwise-v2`. Việc đổi `rush/` thành `research/` diễn ra
  sau run và không tạo thêm kết quả thí nghiệm.
- `research/convergence-and-backbones` đã được merge vào `release/v2`; snapshot
  `release-v2` chứa phần mở rộng đó.
- `feat/p100-backbone-benchmark` là nền của code hiện tại ở gốc dự án.

[registry.json](registry.json) ghi đầy đủ commit của sáu nhánh thuộc dòng Rush tại thời điểm
thu thập. Tên nhánh có thể thay đổi; commit và manifest xác định snapshot.

## Chạy một phiên bản

1. Chọn snapshot và mở hướng dẫn đi kèm. Thư mục `source/` là project root của
   snapshot; không chạy nó bằng môi trường của một phiên bản khác.
2. Tạo môi trường riêng theo `pyproject.toml`, lockfile hoặc requirements của
   phiên bản đó. Chuẩn bị dataset, pretrained model và bộ tách câu theo revision
   trong cấu hình/provenance gốc.
3. Chỉ định **đường dẫn tuyệt đối riêng** cho data, cache, output và status khi
   launcher có các tham số này. Một số launcher cũ tự tìm thư mục của máy chạy
   trước hoặc dùng `%LOCALAPPDATA%`; không dùng đường dẫn mặc định để chạy lại.
4. Nếu cần giữ nguyên cả cấu trúc thư mục và hành vi tìm đường dẫn của launcher,
   tạo checkout riêng tại commit trong manifest và làm theo hướng dẫn gốc.

Không có dataset, environment, downloaded model hay trained checkpoint trong
snapshot. ZIP/predictions không có trong Git được giữ bằng liên kết và checksum
gốc, không được coi là đã có sẵn trên máy. Xem [trạng thái weights](../reports/MODELS.md).
Các liên kết trong tài liệu snapshot được giữ nguyên lịch sử; dùng bảng phía trên
để tìm kết quả tại vị trí mới.

Kiểm tra tính toàn vẹn từ gốc repo, không cần GPU hoặc thư viện ML:

```bash
python3 scripts/verify_versions.py
```

Code mới chỉ sửa ở `src/`; snapshot dùng để đối chiếu và tái lập. Khi thêm một đợt
chạy, tạo thư mục kết quả riêng, ghi commit thực thi, config, seed, dataset/model
revision, checkpoint hash và trạng thái hoàn tất. Không ghi đè kết quả đã công bố.
