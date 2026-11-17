# Trạng thái mô hình và checkpoint

| Thí nghiệm | Mô hình | Kết quả test | Checkpoint weights |
|---|---|---|---|
| Listwise v2, ba seed (12/11/2026) | M2; seeds 42, 13, 2026 | Macro-F1 `0.7066 ± 0.0057` | Seed 13 được chia sẻ thành asset riêng trên Release ngày 17/11; weights khác ở máy training |
| Listwise full-data, một seed (30/10/2026) | M2; seed 42 | Macro-F1 `0.7046` | Không có trong Git |
| Pilot 900 claim | M0/M1/M2; seed 42 | Xem báo cáo pilot | Không có trong Git |

[Báo cáo v2](experiments/listwise-best5-v2-three-seeds-20261112/REPORT.md) chứa hash checkpoint, selected epoch/stage và kết quả của cả năm cấu hình. Hash hoặc đường dẫn lịch sử trong JSON không phải file weights có thể nạp. Muốn chạy inference cần checkpoint tương thích lấy từ máy đã train.

M2 seed 13, epoch 4 được chọn để bàn giao bằng **dev Macro-F1 0.709033**, không chọn bằng test. Tải `m2_best.pt`, handoff ZIP và checksum từ [GitHub Release](https://github.com/ngpnduy/vi-factcheck-system/releases/tag/listwise-v2-three-seeds-20261112). Repo private cần tài khoản có quyền đọc. SHA-256: `49033381616dee546090efbbb711ef1ea81b4eb2a1bed36390cee4566afad924`. Một file chứa đủ reranker, verifier, hai heads và candidate attention. Handoff có loader đổi đường dẫn assets trong bộ nhớ cho Linux, không sửa bytes checkpoint.

Pilot [15 epoch hội tụ](../docs/protocol/convergence.md) được chuẩn bị để chạy mới trên P100. Checkpoint inference đã bàn giao không phải terminal state để tiếp tục tùy ý một thí nghiệm dài hơn. Pilot chưa có kết quả training trên server và không thay đổi các điểm năm epoch ở trên.
