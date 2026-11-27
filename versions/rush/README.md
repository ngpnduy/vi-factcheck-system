# rush

M0/M1/M2; pilot 900 claim và runner full data. Snapshot đầu nhánh; provenance pilot lưu riêng trong PROVENANCE.json.

- Commit: `cec939d87a5bb137b556b2125c0be3ee23690150`; nguồn: `origin/Rush-version`.
- [Code và môi trường gốc](source/); [hướng dẫn gốc](source/FULL_PIPELINE.md).
- [Manifest checksum](MANIFEST.json) ghi từng file được chép nguyên byte và các file được loại khỏi snapshot.
- Chạy từ `versions/rush/source/`, dùng môi trường riêng theo hướng dẫn gốc. Cần chuẩn bị dataset, model và Java/CUDA nếu hướng dẫn yêu cầu. Không dùng chung checkpoint/output giữa các protocol.
- Snapshot chỉ chứa code, cấu hình và tài liệu; kết quả đã được tập trung ở [reports/experiments](../../reports/experiments/). Liên kết kết quả trong tài liệu gốc có thể vẫn chỉ đường dẫn lịch sử.
- Để tái lập cả cấu trúc Git gốc, dùng commit đã ghim ở trên trong một checkout riêng. Việc lưu snapshot không có nghĩa đã chạy lại training.
