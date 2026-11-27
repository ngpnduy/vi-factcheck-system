# release-v2

Bố cục src/vifactcheck và công cụ hội tụ; có code chuẩn bị chạy 15 epoch, chưa có kết quả 15 epoch trong Git.

- Commit: `4a314fabd4f19eea28f1bb33e6a0712bc5305a3b`; nguồn: `origin/release/v2`.
- [Code và môi trường gốc](source/); [hướng dẫn gốc](source/docs/protocol/v2.md).
- [Manifest checksum](MANIFEST.json) ghi từng file được chép nguyên byte và các file được loại khỏi snapshot.
- Chạy từ `versions/release-v2/source/`, dùng môi trường riêng theo hướng dẫn gốc. Cần chuẩn bị dataset, model và Java/CUDA nếu hướng dẫn yêu cầu. Không dùng chung checkpoint/output giữa các protocol.
- Snapshot chỉ chứa code, cấu hình và tài liệu; kết quả đã được tập trung ở [reports/experiments](../../reports/experiments/). Liên kết kết quả trong tài liệu gốc có thể vẫn chỉ đường dẫn lịch sử.
- Để tái lập cả cấu trúc Git gốc, dùng commit đã ghim ở trên trong một checkout riêng. Việc lưu snapshot không có nghĩa đã chạy lại training.
