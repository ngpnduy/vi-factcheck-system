# listwise-seed42

M0/M1/A1/M2/A2, kết quả full data seed 42. Commit chạy c07b0b1 được ghi trong EXPORT_MANIFEST của kết quả.

- Commit: `a7ee344c5ac861b8c02edcecda789075eec24840`; nguồn: `origin/slides-listwise-seed42`.
- [Code và môi trường gốc](source/); [hướng dẫn gốc](source/docs/SLIDES_0410_IMPLEMENTATION.md).
- [Manifest checksum](MANIFEST.json) ghi từng file được chép nguyên byte và các file được loại khỏi snapshot.
- Chạy từ `versions/listwise-seed42/source/`, dùng môi trường riêng theo hướng dẫn gốc. Cần chuẩn bị dataset, model và Java/CUDA nếu hướng dẫn yêu cầu. Không dùng chung checkpoint/output giữa các protocol.
- Snapshot chỉ chứa code, cấu hình và tài liệu; kết quả đã được tập trung ở [reports/experiments](../../reports/experiments/). Liên kết kết quả trong tài liệu gốc có thể vẫn chỉ đường dẫn lịch sử.
- Để tái lập cả cấu trúc Git gốc, dùng commit đã ghim ở trên trong một checkout riêng. Việc lưu snapshot không có nghĩa đã chạy lại training.
