# Phân công chạy benchmark backbone

Bản kế hoạch có **ba backbone mới cho cả nhóm**. Thành phụ trách hai backbone trên P100. An phụ trách PhoBERT-v2 trên laptop RTX 3050. Mỗi backbone chạy **Baseline + M2**, seeds **42, 13, 2026**.

## Thành — Tesla P100, Conda và Slurm

1. **mBERT** (`google-bert/bert-base-multilingual-cased`) chạy trước. Dùng dữ liệu đã prepare trên `ood-master`, smoke cả Baseline và M2, rồi chạy benchmark. Bàn giao báo cáo, số liệu từng seed, log và ZIP kết quả.
2. **XLM-R** (`FacebookAI/xlm-roberta-base`) chạy sau nếu thời gian đo từ mBERT cho thấy còn kịp mốc khóa kết quả. Smoke riêng trên P100 trước khi chạy full.

Lệnh và vị trí output nằm trong [guide chạy benchmark P100](backbone-benchmark.md). Conda, CUDA FP16 và dữ liệu đã được kiểm tra trên server. Pilot hội tụ 15 epoch đang tạm gác.

## An — laptop RTX 3050

- **PhoBERT-v2** (`vinai/phobert-base-v2`) là backbone mới thứ ba của nhóm. Đầu việc theo kế hoạch là Baseline + M2, sau khi có đường chạy phù hợp và smoke đạt về bộ nhớ. Nhánh triển khai hiện có runner cho mBERT và XLM-R. PhoBERT-v2 vẫn là phần triển khai riêng cần hoàn tất trước khi chạy trên laptop.
- Rà bảng kết quả **PhoBERT-base** đã có và chuẩn bị số liệu đối chiếu. M2 ba seed của PhoBERT-base là mốc từ study cũ. Nếu đưa PhoBERT-base vào bảng so sánh Baseline–M2 theo protocol mới, cần Baseline claim–article tương ứng và kiểm tra protocol trước khi dùng lại điểm M2.

## Chốt bảng và bàn giao

- Ưu tiên hoàn thành một cặp Baseline–M2 đủ ba seed rồi mới mở thêm backbone. Đo thời gian sau trial đầu và tính cả dev, lưu checkpoint và lượt test.
- Một backbone hoàn chỉnh có **6 kết quả test**. P100 có tối đa **12 kết quả** cho hai backbone. Cả nhóm có tối đa **18 kết quả** cho ba backbone mới nếu hoàn thành đầy đủ. Một hàng chỉ có seed 42 được ghi rõ là kết quả sơ bộ.
- Mốc trong kế hoạch gốc là khóa số liệu ngày 25/11, rà và viết ngày 26–28/11, nộp nội bộ 29/11 và hạn cuối 30/11. Nhóm dùng ETA đo được để chốt lại phạm vi nếu mốc khóa số liệu cần thay đổi.
- Thành tải `download/results.zip` và file `.sha256` của từng run hoàn tất qua Open OnDemand. Giữ file `.pt` trên server và tra đường dẫn trong `download/checkpoint_locations.json` khi cần tải weights.

Nguồn phân công và mốc gốc nằm trong [kế hoạch ngày 17/11, mục 3 và 10](https://github.com/ngpnduy/vi-factcheck-system/blob/4a314fa/docs/guides/backbone-benchmark.md).
