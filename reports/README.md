# Báo cáo và tài liệu nộp

- [Proposal v2](Proposal_v2.docx) và [các bản trình bày](presentations/README.md) ngày 04/10, 02/11.
- [Pilot 900 claim, seed 42](experiments/pilot-900-seed42/REPORT.md): M0/M1/M2, training rút gọn.
- [Full-data listwise, seed 42](experiments/listwise-seed42/REPORT.md): M0/M1/A1/M2/A2, kết quả tháng 10.
- [Listwise v2, năm cấu hình × ba seed](experiments/listwise-best5-v2-three-seeds-20261112/REPORT.md): kết quả chính hiện tại, tuning trên dev và 15 lần đánh giá test.
- [Trạng thái checkpoint](MODELS.md): kết quả đã công bố và giới hạn inference.

Mỗi thí nghiệm có `results.csv` cạnh báo cáo. Thư mục v2 giữ metric, lịch sử train, provenance và manifest; [GitHub Release](experiments/listwise-best5-v2-three-seeds-20261112/README.md) cung cấp ZIP với toàn bộ predictions. Các báo cáo trước v2 giữ nguyên để đối chiếu lịch sử.
