# Benchmark Baseline và M2 trên P100

P100 chạy [mBERT](https://huggingface.co/google-bert/bert-base-multilingual-cased/commit/3f076fdb1ab68d5b2880cb87a0886f315b8146f8) trước. Chạy [XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-base/commit/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089) nếu thời gian đo trên mBERT phù hợp với hạn nhóm đã chốt. Mỗi backbone chạy Baseline và M2 với seeds 42, 13, 2026. Thiết kế này giữ trần **5 epoch verifier** của v2. Pilot hội tụ 15 epoch đang tạm gác.

Xem [phân công ba backbone của nhóm](backbone-work-plan.md) để biết phần của Thành trên P100 và phần PhoBERT-v2 của An trên RTX 3050.

## 1. Điều kiện và dữ liệu dùng chung

- Hoàn tất bước 1–4 trong [guide Conda/P100](server-conda.md). Dùng cùng môi trường đã kiểm tra CUDA FP16 và chạy từ gốc repo. `python -m vifactcheck benchmark --help` cần hiện các lệnh con `prepare`, `smoke`, `run`, `export`, `compare`.
- Tái sử dụng `data/prepared/` từ `python -m vifactcheck prepare`, gồm 5.062 train, 723 dev, 1.447 test. Cả hai dùng chung tách câu, căn chỉnh evidence và danh sách BM25. Runner đối chiếu hash của các split khi đọc.
- `benchmark prepare` tải weights safetensors và tokenizer pretrained theo revision đã ghim vào `.cache/backbones/` và ghi hash từng file. Dùng riêng cache mỗi backbone. Checkpoint PhoBERT đã train chỉ hữu ích cho việc tái lập inference của PhoBERT.
- Với full run mới, cần khoảng trống ban đầu 35 GiB cho mBERT hoặc 60 GiB cho XLM-R trên ổ output. Kiểm tra cả quota cá nhân nếu cụm có áp quota.

## 2. Chạy mBERT từ terminal trên server

Nếu đã có `data/prepared/`, bắt đầu ở đây trong terminal Open OnDemand. Thứ tự là **checkout → tải mBERT → smoke → full mBERT → quyết định XLM-R → tải kết quả**.

### 2.1 Kiểm tra checkout, Conda và dữ liệu

Nếu checkout vẫn ở `release/v2`, chuyển sang nhánh benchmark trước lần chạy đầu tiên:

```bash
cd ~/vi-factcheck-system
git fetch origin
git switch --track origin/feat/p100-backbone-benchmark
```

Nếu đã ở nhánh benchmark, tiếp tục từ đoạn bên dưới. Nếu đã bắt đầu job, giữ đúng commit của lần chạy đó khi resume.

```bash
cd ~/vi-factcheck-system
conda activate vifactcheck
export PYTHONNOUSERSITE=1
git rev-parse --short HEAD
ls -lh data/prepared/{manifest.json,train.jsonl,dev.jsonl,test.jsonl}
df -h .
python -m vifactcheck benchmark --help
```

Cần thấy bốn file dữ liệu và các lệnh con `prepare`, `smoke`, `run`, `export`, `compare`. Full run dùng `runs/` trong repo theo lệnh dưới đây, nên kiểm tra dung lượng của ổ đó nếu bạn đặt `runs/` ở ổ khác.

### 2.2 Tải pretrained assets mBERT

```bash
python -m vifactcheck benchmark prepare --backbone mbert
```

Lệnh chạy trong terminal thường và ghi model, tokenizer cùng checksum vào `.cache/backbones/mbert/`. Runner lấy dữ liệu chung từ `data/prepared/`.

### 2.3 Submit smoke mBERT trên một P100

Batch launcher lấy Python và Java từ Conda env bạn truyền vào, ghi log vào thư mục output và tiếp tục chạy khi đóng tab terminal. Ghi lại job ID mà `sbatch` trả về.

```bash
sbatch --partition=p100 --gres=gpu:tesla_p100-pcie-16gb:1 \
  --time=00:30:00 scripts/benchmark-p100.sbatch \
  smoke mbert runs/backbone-mbert-smoke "$CONDA_PREFIX"
```

Smoke chạy probe M2 với tám claim có workload token lớn nhất, sau đó train Baseline và M2 trên 24 train/12 dev. Nó kiểm tra loss hữu hạn, cập nhật weights, gradient verdict tới reranker và prediction sau khi reload. `completion.json` phải ghi `"status": "passed"` trước khi submit full. Điểm smoke lấy từ dev để kiểm tra pipeline.

```bash
squeue -u "$USER"
cat runs/backbone-mbert-smoke/status.json
cat runs/backbone-mbert-smoke/completion.json
```

Nếu chưa có `completion.json`, kiểm tra hàng đợi và log trước khi submit job mới. Nếu job lỗi, đọc `failure.json` và `slurm-<job-id>.log` trong thư mục smoke. Trước khi Python khởi động, lỗi của launcher có thể nằm ở `slurm-<job-id>.out` tại gốc repo.

### 2.4 Chốt chính sách rồi submit full mBERT

Lệnh mặc định tune trên dev ở seed 42, sau đó dùng tham số đã chọn để chạy seeds 13 và 2026. Nếu cả nhóm đã chốt dùng cấu hình cố định để tiết kiệm thời gian, xem cờ `--fixed-hyperparameters` ở mục 3 trước khi submit. `--time=08:00:00` là ví dụ thời lượng cấp cho **một job Slurm**. Chọn thời lượng mà partition cho phép.


```bash
sbatch --partition=p100 --gres=gpu:tesla_p100-pcie-16gb:1 \
  --time=08:00:00 scripts/benchmark-p100.sbatch \
  run mbert runs/backbone-mbert "$CONDA_PREFIX"
```

### 2.5 Theo dõi, resume và xác nhận mBERT đã xong

```bash
squeue -u "$USER"
cat runs/backbone-mbert/status.json
cat runs/backbone-mbert/job_timings.json
```

`status.json` cho biết seed, phase, epoch và số claim đã xử lý. `job_timings.json` có thời gian quan sát theo job đã xong và train seconds tính từ lịch sử epoch. Một attempt được resume có thể chỉ đo phần còn lại, nên đối chiếu history và thời gian Slurm khi ước lượng XLM-R. Dùng `sacct -j JOB_ID` nếu cụm bật tính năng accounting.

- Nếu Slurm hết giờ, đợi job cũ kết thúc rồi submit lại cùng thư mục output, mã nguồn, dữ liệu, Conda env và tùy chọn benchmark. Có thể đổi thời lượng cấp của Slurm. Runner tiếp tục từ epoch đã checkpoint gần nhất và bỏ qua các job con đã hoàn thành sau khi kiểm tra hash. Epoch chưa kịp lưu sẽ chạy lại.
- Nếu Python báo lỗi, đọc `failure.json` và log trước khi chạy lại. Có thể đóng gói diagnostics sau khi job đã dừng theo mục 4.
- Khi hoàn tất, `runs/backbone-mbert/completion.json` ghi `"status": "complete"`, `"split": "test"` và `"evaluation_count": 6`. Kiểm tra có cả Baseline và M2 trên ba seed trong `per_seed.csv`.

### 2.6 Chạy XLM-R nếu còn kịp hạn

Dùng thời gian đo từ mBERT, thời gian chờ Slurm và mốc chốt số liệu để quyết định. XLM-R dùng thư mục riêng và cần smoke riêng:

```bash
python -m vifactcheck benchmark prepare --backbone xlmr

sbatch --partition=p100 --gres=gpu:tesla_p100-pcie-16gb:1 \
  --time=00:30:00 scripts/benchmark-p100.sbatch \
  smoke xlmr runs/backbone-xlmr-smoke "$CONDA_PREFIX"
```

Khi `runs/backbone-xlmr-smoke/completion.json` ghi `"status": "passed"`, submit full với cùng chính sách hyperparameter đã chọn cho mBERT:

```bash
sbatch --partition=p100 --gres=gpu:tesla_p100-pcie-16gb:1 \
  --time=08:00:00 scripts/benchmark-p100.sbatch \
  run xlmr runs/backbone-xlmr "$CONDA_PREFIX"
```

Theo dõi và resume như mBERT, dùng đường dẫn `runs/backbone-xlmr/`. Khi full hoàn tất, lấy ZIP cho từng backbone rồi gộp bảng theo mục 4.

## 3. Protocol ghi trong báo cáo

- **Baseline** dùng một encoder và classifier ba nhãn. Nó nhận một cặp claim–bài báo mỗi claim, cắt claim tối đa 96 token và cả cặp tối đa 256 token gồm special tokens. Baseline huấn luyện 5 epoch.
- **M2** dùng hai encoder được khởi tạo độc lập từ cùng pretrained backbone. BM25 lấy tối đa 20 câu, listwise reranker lấy tối đa 5. Reranker pretrain 3 epoch, chọn bằng dev MRR. Verifier warmup 3 epoch rồi joint training 2 epoch. M2 chọn checkpoint bằng dev Macro-F1 trên cả 5 epoch.
- mBERT và XLM-R dùng văn bản gốc đã chuẩn hóa và đúng tokenizer của mình. Cặp mBERT có `token_type_ids` dành cho claim và context. BM25 vẫn dùng đầu vào đã tách từ để giữ cùng danh sách ứng viên.
- Mặc định tune seed 42 trên dev. Baseline thử LR `1e-5, 2e-5`. M2 chọn LR qua M1 với cùng hai giá trị, rồi thử lambda `0.1, 0.5, 1.0` từ cùng warmup. Chọn điểm dev cao nhất, hòa chọn epoch hoặc trial xuất hiện sớm hơn. Giữ lại checkpoint thắng cho seed 42 và dùng bộ tham số thắng cho 13 và 2026.
- Nếu nhóm chốt trước một cấu hình cố định để rút thời gian, truyền `--fixed-hyperparameters` **sau** `"$CONDA_PREFIX"` ở cả hai lệnh full. Cờ này dùng LR `1e-5`, lambda M2 `0.1` cho cả ba seed. Dùng output mới nếu đổi chính sách. Bảng tổng hợp kiểm tra hai run dùng cùng chính sách.
- Runner ghi checksum toàn bộ checkpoint được chọn trước khi đánh giá test. Bảng chính dùng đủ 1.447 claim mỗi checkpoint, mean ± sample SD theo ba seed, cùng chênh lệch M2–Baseline. Bootstrap ghép theo bài báo và tính theo từng seed. Retrieval chỉ có ở M2 và có mẫu số riêng. Các split có bài báo trùng nhau, nên benchmark này không đo khả năng tổng quát hóa trên toàn bộ bài báo chưa từng thấy.
- Ghi số tham số, lượng token encoder xử lý, mức cắt context, thời gian và VRAM. Baseline và M2 dùng cùng giới hạn token **mỗi cặp**, còn tổng compute khác vì M2 encode nhiều cặp và có thêm pretraining reranker.

## 4. Output để tải qua Open OnDemand

Trong `runs/backbone-mbert/` hoặc `runs/backbone-xlmr/`:

- **`status.json` và `slurm-<job-id>.log`** cho biết job đang ở đâu. Khi Python báo lỗi, xem thêm `failure.json`. Nếu scheduler hết giờ hoặc hủy job, xem trạng thái Slurm và dòng cuối log.
- **`plan.json`**, **`tuning_trials.json`** (khi bật tuning), **`selected_hyperparameters.json`** và **`job_timings.json`** ghi cấu hình, revision, hash, môi trường, dev scores và thời gian. `jobs/` có history và metadata từng phase. Runner giữ best và state cần cho resume; last dư được dọn sau khi job con đã ghi ledger.
- **`REPORT.md`**, **`summary.csv`**, **`per_seed.csv`** là bảng để đọc và đưa vào báo cáo. **`comparison.json`** thêm chênh lệch và khoảng tin cậy ghép theo seed. **`evaluation/`** chứa metrics chi tiết và predictions từng claim. `completion.json` phân biệt full benchmark đã xong với smoke.
- **`download/results.zip`** được tạo tự động khi hoàn tất. Tải nó cùng **`download/results.zip.sha256`** qua Open OnDemand → Files. ZIP gồm bảng, metrics, log, cấu hình, histories và prediction gọn với ID, xác suất, ranking và vị trí evidence. ZIP giữ checkpoint SHA-256, còn weights nằm trên server. Trên Mac, lưu cặp ZIP và checksum của mỗi backbone vào một thư mục riêng vì hai run dùng cùng tên file.
- **`selected_checkpoints.json`** và **`download/checkpoint_locations.json`** liệt kê đúng đường dẫn, kích thước và hash của các `*_best.pt`. Muốn giữ weights, tải file `.pt` tương ứng qua Files và đối chiếu với **`checkpoints.sha256`**. Với full run có sáu checkpoint được chọn cho mỗi backbone.

Nếu job đã dừng và cần gửi log hoặc kết quả đang dở, tạo `download/diagnostics.zip`. Lệnh tương tự cũng tạo lại ZIP kết quả đã hoàn tất với log mới nhất.

```bash
python -m vifactcheck benchmark export --run runs/backbone-mbert
```

Để kiểm tra file trên Mac sau khi tải, chạy `shasum -a 256 -c results.zip.sha256` trong thư mục chứa cả ZIP lẫn checksum.

Khi cả hai backbone đã hoàn tất, tạo bảng chung:

```bash
python -m vifactcheck benchmark compare \
  --runs runs/backbone-mbert runs/backbone-xlmr \
  --output runs/backbone-comparison
```

Đọc `runs/backbone-comparison/REPORT.md` hoặc tải `summary.csv` và `per_seed.csv`. Khi chỉ có mBERT hoàn tất, bảng của mBERT đã dùng được và ghi rõ phạm vi backbone.
