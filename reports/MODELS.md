# Checkpoint và mô hình được báo cáo

| Trạng thái | Mô hình | Kết quả | Checkpoint |
|---|---|---|---|
| Tốt nhất đã công bố trong thí nghiệm listwise ngày 30/10/2026 | M2, seed 42 | Test Macro-F1 0.7046 | Weights ở máy thí nghiệm; chưa phát hành trong repo |
| Protocol v2, ba seed | M0/M1/A1/M2/A2 | Chưa có báo cáo cuối trong nhánh release | Không có checkpoint công khai |

Code trong `src/vifactcheck/` có thể train và dự đoán khi được cấp checkpoint tương thích. Không gọi một checkpoint là “best v2” trước khi báo cáo ba seed hoàn tất. Khi nhóm phát hành weights, bổ sung vị trí tải, SHA-256, seed, protocol, cấu hình và commit source tại đây.
