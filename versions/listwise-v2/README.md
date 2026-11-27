# listwise-v2

Snapshot commit thực thi của thí nghiệm năm cấu hình × ba seed; publication ở 0af01fd.

- Commit: `baa0bdc80379d26e661bd1dcf5e37b3b48ba14a8`; nguồn: `baa0bdc`.
- [Code và môi trường gốc](source/); [hướng dẫn gốc](source/LISTWISE_V2.md).
- [Manifest checksum](MANIFEST.json) ghi từng file được chép nguyên byte và các file được loại khỏi snapshot.
- Chạy từ `versions/listwise-v2/source/`, dùng môi trường riêng theo hướng dẫn gốc. Cần chuẩn bị dataset, model và Java/CUDA nếu hướng dẫn yêu cầu. Không dùng chung checkpoint/output giữa các protocol.
- Snapshot chỉ chứa code, cấu hình và tài liệu; kết quả đã được tập trung ở [reports/experiments](../../reports/experiments/). Liên kết kết quả trong tài liệu gốc có thể vẫn chỉ đường dẫn lịch sử.
- Để tái lập cả cấu trúc Git gốc, dùng commit đã ghim ở trên trong một checkout riêng. Việc lưu snapshot không có nghĩa đã chạy lại training.
