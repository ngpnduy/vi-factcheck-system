# Trạng thái mô hình và checkpoint

| Thí nghiệm | Mô hình | Kết quả test | Checkpoint weights |
|---|---|---|---|
| Listwise v2, ba seed (12/11/2026) | M2; seeds 42, 13, 2026 | Macro-F1 `0.7066 ± 0.0057` | Được tạo trên máy training; không có trong Git hoặc ZIP công bố |
| Listwise full-data, một seed (30/10/2026) | M2; seed 42 | Macro-F1 `0.7046` | Không có trong Git |
| Pilot 900 claim | M0/M1/M2; seed 42 | Xem báo cáo pilot | Không có trong Git |

[Báo cáo v2](experiments/listwise-best5-v2-three-seeds-20261112/REPORT.md) chứa hash checkpoint, selected epoch/stage và kết quả của cả năm cấu hình. Hash hoặc đường dẫn lịch sử trong JSON không phải file weights có thể nạp. Muốn chạy inference cần checkpoint tương thích lấy từ máy đã train.
