# Thăm dò hội tụ trên P100: M2/PhoBERT, seed 42

Chuẩn bị ngày 17/11/2026 từ `release/v2` (`2045e64`), trên nhánh làm việc `research/convergence-and-backbones`. Người dùng đã chọn **chuẩn bị để teammate chạy trên P100**, không chạy training dài trên laptop. [Setup Conda](../guides/server-conda.md) và [kế hoạch nhiều backbone](../guides/backbone-benchmark.md) là hai hướng dẫn đi kèm.

## 1. Hiện có thể kết luận gì?

[Audit log ba seed](../../reports/diagnostics/convergence-20261117/REPORT.md) chỉ đọc lịch sử train/dev của đợt đã hoàn tất:

| Seed | Objective train joint, epoch 4 → 5 | Dev Macro-F1, epoch 4 → 5 | Epoch được chọn |
|---|---|---|---:|
| 42 | 0.492155 → 0.334965 | 0.678650 → 0.706201 | 5 |
| 13 | 0.447933 → 0.281225 | 0.709033 → 0.708027 | 4 |
| 2026 | 0.507966 → 0.341371 | 0.687712 → 0.681083 | 4 |

Loss train vẫn giảm, nhưng chỉ có hai điểm joint training. Một seed tốt nhất ở ranh giới cuối; hai seed có dev giảm nhẹ. **Chưa chứng minh hội tụ, cũng chưa chứng minh mô hình tốt nhất có thể.** Train loss thấp nhất không đồng nghĩa verdict tốt nhất. Các log cũ không lưu riêng dev NLL hay từng thành phần loss, nên không thể dựng ngược chúng từ Macro-F1.

Warmup tối ưu verification, continuation thêm evidence BCE và mở gradient reranker. Không nối hai objective thành một đường loss rồi kết luận về hội tụ. Reranker pretraining lại là objective riêng.

## 2. Thiết kế pilot đã chuẩn bị

- Chỉ dùng official train 5.062 và dev 723; **không đọc test**, không báo test score từ pilot.
- PhoBERT-base pin gốc, seed **42**, LR **1e-5**, lambda **0.1**, listwise một layer.
- Reranker: **3 epoch**, chọn checkpoint bằng dev MRR.
- Verifier: **15 epoch tổng = 3 warmup + 12 continuation**, không phải thêm 15 vào checkpoint cũ.
- Giữ token **256/96**, BM25 **20**, evidence **5**, một claim, accumulation **8**, Adafactor, AMP FP16 và activation checkpointing. M2 tiếp tục train cả hai encoder.
- Giữ **LR horizon = 5**: lịch ramp LR vẫn tính theo năm verifier epochs của protocol gốc. Nếu tự đổi horizon thành 15, những update đầu cũng thay đổi; đó là một thí nghiệm khác. Reranker vẫn có lịch riêng ba epoch.
- Chọn checkpoint dev Macro-F1 tốt nhất trong toàn bộ 15 epoch, kể cả warmup; hòa chọn epoch sớm. Giữ last state độc lập với best. Đây là **exploratory extended-epoch diagnostic**, không được gắn nhãn kết quả best-of-five đã công bố.
- Chạy đủ horizon đã chốt để quan sát; không có early stopping tự động. OOM/nonfinite loss/dung lượng không đủ dừng có thông báo, không tự đổi mô hình.

`scripts/run_convergence.py` dùng các phase hiện có, thêm observer riêng và lựa chọn LR horizon có lưu trong checkpoint. Khi không bật observer/horizon, đường chạy cũ giữ nguyên mặc định. Không resume checkpoint năm epoch cũ bằng cách sửa trường epoch hoặc lấy `m2_best.pt` làm terminal state. Pilot chạy mới từ pretrained weights; trên P100 khác phần cứng không cam kết tái hiện từng bit của năm epoch đầu laptop.

## 3. Chuẩn bị và preflight

Làm trong checkout đã chứa runner mới, sau khi môi trường Conda, CUDA, Java, pinned assets và prepared data đã được kiểm tra. Lệnh `prepare` mặc định tạo đúng các thư mục dưới đây; nếu thay chỗ, truyền đúng đường dẫn. `assets.json` phải ở thư mục cha của `phobert/` và giữ layout `phobert/<file>` trong manifest.

```bash
python scripts/run_convergence.py --help
python scripts/run_convergence.py \
  --data data/prepared \
  --model-path .cache/phobert \
  --output runs/convergence-phobert-seed42-e15 \
  --total-epochs 15 --warmup-epochs 3 --reranker-epochs 3 \
  --lr-horizon-epochs 5 --seed 42 --lr 1e-5 --evidence-lambda 0.1 \
  --check-only
```

Preflight kiểm tra train/dev counts, nhãn, ID, revision/hash dữ liệu, PhoBERT file hashes, CUDA/P100 `sm_60`, ít nhất **35 GiB** trống trên ổ output và identity của thư mục nếu resume. Nó không tạo thư mục run, không khởi tạo/train encoder và không ghi checkpoint. Trước actual run còn một probe forward/backward trên tám workload dài nhất, bỏ weights probe trước khi bắt đầu train. P100/Conda vẫn cần được kiểm tra trực tiếp trên server; test CPU nhỏ không thay cho full-model smoke.

Smoke runner trên server dùng thư mục riêng:

```bash
python scripts/run_convergence.py \
  --data data/prepared --model-path .cache/phobert \
  --output runs/convergence-smoke \
  --smoke --total-epochs 5 --warmup-epochs 3 --reranker-epochs 1 \
  --lr-horizon-epochs 5
```

Smoke dùng 24 train/12 dev. Điểm smoke chỉ xác nhận pipeline. Giữ riêng nó với full-data pilot.

## 4. Chạy, theo dõi và resume

Bỏ `--check-only` từ lệnh full-data đã kiểm tra. Giữ log terminal bằng công cụ của server, ví dụ `tmux` nếu máy đã có; không cần môi trường desktop Windows.

```bash
python -u scripts/run_convergence.py \
  --data data/prepared --model-path .cache/phobert \
  --output runs/convergence-phobert-seed42-e15 \
  --total-epochs 15 --warmup-epochs 3 --reranker-epochs 3 \
  --lr-horizon-epochs 5 --seed 42 --lr 1e-5 --evidence-lambda 0.1 \
  > convergence-seed42.log 2>&1
```

Chạy lại **đúng lệnh, source, environment và đường dẫn** để resume từ checkpoint cuối epoch. Epoch chưa lưu được sẽ chạy lại. Nếu ghi checkpoint xong nhưng journal JSON chưa ghi kịp, runner dựng lại journal từ state có thẩm quyền. Reranker/warmup đã hoàn tất được reload thay vì train lại; warmup end được bảo vệ hash. Các phase trước được dựng model lại và reload nên việc resume vẫn cần đủ RAM/VRAM.

Tạo file `STOP` trong thư mục run để yêu cầu dừng ở progress boundary. Xóa riêng file STOP đó trước khi resume. Thao tác dừng giữa epoch mất phần chưa checkpoint của epoch, không phải resume chính xác từng claim. Đọc `failure.json` khi lỗi; kiểm tra dung lượng trước khi lặp lại. Không tự xóa checkpoint cũ.

## 5. Log để quan sát

Sau mỗi epoch đã checkpoint, `diagnostics.csv` và `diagnostics.jsonl` tổng hợp:

- **Online train:** verification NLL, balanced evidence BCE trung bình trên claim eligible, số claim eligible, objective tổng, và phần lambda × evidence đóng góp trung bình trên tất cả claim.
- **Dev:** verification NLL tính trực tiếp từ log-probability ổn định, BCE và mẫu số tương ứng, objective, Macro-F1/MRR.
- **Train probe:** tối đa 96 train claims cố định, cân bằng nhãn, đánh giá ở eval mode giống dev. So NLL probe–dev để nhìn khoảng cách generalization; probe không đại diện chính xác toàn train.
- Stage/global epoch, training time, diagnostic evaluation time và peak GPU memory trong history.

NEI không có evidence loss. Balanced BCE chỉ tính trên S/R aligned có cả candidate dương và âm. Trung bình BCE trên eligible **khác** contribution trong objective toàn bộ claim: `lambda × sum(BCE_eligible) / N_claims`. Warmup không tối ưu BCE, nên BCE ở các trường tương ứng để trống. Reranker chỉ có BCE/MRR, không có verdict NLL.

Các forward diagnostic dùng eval mode, không bật dropout, không cập nhật weights và phục hồi RNG. Chúng bổ sung các lượt eval nên runtime dài hơn pipeline cũ; đo riêng thay vì dùng ETA laptop như số đo P100.

Artifacts: `convergence_plan.json`, hardware, source snapshot/hashes, resource probe, status/failure, diagnostics CSV/JSONL, histories, `reranker_{best,last}.pt`, `warmup_{best,last,end}.pt`, `m2_{best,last}.pt`, completion. Chỉ giữ best/last cần thiết, không tạo 15 bản weights theo từng epoch. Giữ nguồn warmup cho resume. Checkpoint best có cả reranker và verifier; không chỉ classifier head.

## 6. Phân tích và quyết định tiếp theo

```bash
python scripts/analyze_convergence.py \
  --run runs/convergence-phobert-seed42-e15 \
  --output runs/convergence-phobert-seed42-e15-analysis
python scripts/plot_convergence.py \
  --csv runs/convergence-phobert-seed42-e15-analysis/convergence.csv \
  --output runs/convergence-phobert-seed42-e15-analysis/curves
```

Plot cần matplotlib, có thể chạy trong môi trường phân tích riêng; nó không train hoặc nạp weights. Xem riêng objective từng stage, NLL train-probe/dev và dev F1. Plateau chỉ là quan sát trong một cửa sổ hữu hạn; dev xấu đi trong khi train NLL giảm là dấu hiệu cần xem overfit, không tự động chứng minh hội tụ.

Nếu môi trường training không có matplotlib, tạo môi trường Conda riêng thay vì đổi lock đang dùng cho resume. Bản plot lịch sử ở repo được kiểm tra với matplotlib 3.9.3:

```bash
conda create -n vifactcheck-analysis python=3.12 pip -y
conda run -n vifactcheck-analysis python -m pip install matplotlib==3.9.3
conda run -n vifactcheck-analysis python scripts/plot_convergence.py \
  --csv runs/convergence-phobert-seed42-e15-analysis/convergence.csv \
  --output runs/convergence-phobert-seed42-e15-analysis/curves
```

Chọn prefix mới nếu đã có hình cùng tên; script không ghi đè hình cũ.

Trước khi xem đường mới, quy định checkpoint luôn chọn bằng dev F1. Sau pilot, ghi lại **epoch cap, LR schedule và stopping policy** cho benchmark. Nếu best vẫn ở epoch 15 hoặc dev còn tăng, báo **chưa thấy plateau trong 15 epoch**; không tự khẳng định epoch 15 tối ưu. Nếu tốt nhất ở sớm rồi dev NLL tăng, giữ checkpoint dev-best và báo diễn biến. Các quyết định đổi ngân sách có thể cần pilot mới, không chỉnh run identity đang dở.

Chỉ sau khi chốt protocol mới mới chạy baseline và M2 trên các backbone/seed. Giữ ngân sách verifier và selection criterion đã công bố cho cả hai, báo thêm chi phí pretrain reranker của M2. Pilot một seed không thay kết quả ba seed hoặc một test set độc lập.
