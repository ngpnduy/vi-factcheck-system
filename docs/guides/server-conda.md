# Setup Conda cho teammate chạy Tesla P100

Cập nhật 17/11/2026. Hướng dẫn dưới đây dùng **Bash trên Linux** và bố cục `release/v2`. GPU đã được xác nhận là Tesla P100; VRAM, driver và OS vẫn phải lấy từ máy thật. Các lệnh chưa được chạy trên server của teammate; smoke trên server mới là bằng chứng tương thích.

Đọc [kế hoạch baseline–M2 theo backbone](backbone-benchmark.md) trước. CLI hiện tại sẵn sàng cho pipeline **PhoBERT**, chưa có classifier baseline độc lập hoặc adapter tổng quát cho mBERT/XLM-R. Setup thành công không đồng nghĩa benchmark đa backbone đã được triển khai.

## Bước 1 — Ghi cấu hình máy và chọn checkout

Chạy trong terminal của server:

```bash
nvidia-smi
uname -a
conda --version
df -h .
```

Gửi tên GPU, VRAM, driver, OS và dung lượng trống cho team lead, không gửi token hoặc mật khẩu. `CUDA Version` trong `nvidia-smi` là khả năng driver hỗ trợ, không phải bằng chứng Python đang dùng đúng CUDA wheel.

Repo là private; dùng Git credential helper hoặc GitHub CLI đã đăng nhập với tài khoản có quyền đọc:

```bash
git clone --branch release/v2 https://github.com/ngpnduy/vi-factcheck-system.git
cd vi-factcheck-system
git status --short --branch
git rev-parse HEAD
```

Team lead cung cấp **approved commit** sau review. Checkout commit đó trên cả hai máy trước khi train, ghi SHA vào từng run và không cập nhật source giữa một job. Base release đang được dùng khi viết hướng dẫn là `2045e642bda9be40347f27a8c9ea319904f48077`; nếu cần các script mới thì phải dùng commit đã chứa chúng. Không tạo lại cây `rush/` hoặc chạy từ thư mục nghiên cứu cũ.

## Bước 2 — Tạo môi trường riêng

```bash
conda create -n vifactcheck-v2 python=3.12 pip -y
conda activate vifactcheck-v2
conda install -c conda-forge openjdk=17 -y
which python
which java
java -version
```

Không sửa packages toàn server. Nếu tên environment đã tồn tại, kiểm tra nó trước; không xóa/ghi đè môi trường đang được công việc khác sử dụng.

P100 là compute capability **6.0**. Đề xuất ban đầu là PyTorch **2.6.0, CUDA 11.8**, sau khi driver và OS được kiểm tra. [NVIDIA xác nhận compute capability của P100](https://developer.nvidia.com/cuda/gpus/legacy); [PyTorch cung cấp wheel 2.6 cho CUDA 11.8](https://docs.pytorch.org/get-started/previous-versions/).

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

PyTorch được cài bằng pip **bên trong Conda environment**. Không cài nguyên `requirements-tested.txt` ngay sau lệnh trên vì file ghi Torch `2.6.0+cu124` của laptop. Cài các top-level pin còn lại từ đúng checkout:

```bash
python - <<'PY'
from pathlib import Path
import subprocess
import sys

pins = []
for raw in Path("requirements-tested.txt").read_text().splitlines():
    line = raw.strip()
    if line and not line.startswith("#") and not line.startswith("torch=="):
        pins.append(line)
subprocess.run([sys.executable, "-m", "pip", "install", *pins], check=True)
PY
python -m pip install --no-deps -e .
python -m pip check
```

Đây là pin cấp package của study trước, không phải lock Linux đã xác nhận. Nếu dependency resolution thất bại, lưu log và xử lý ở môi trường riêng; không dùng `pip install -U` hàng loạt để đổi phiên bản âm thầm. `scripts/check_environment.py` hiện so sánh Torch với chính xác chuỗi `+cu124`, nên lỗi khác CUDA build trên server `+cu118` phải được diễn giải đúng; không ép server cài build laptop chỉ để checker xanh.

## Bước 3 — Kiểm tra CUDA bằng tính toán thật

```bash
python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA unavailable"
arch = torch.cuda.get_arch_list()
assert "sm_60" in arch, f"Wheel does not list P100 sm_60: {arch}"
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
print("architectures:", arch)
x = torch.randn(64, 64, device="cuda", requires_grad=True)
with torch.autocast("cuda", dtype=torch.float16):
    loss = (x @ x).square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
print("CUDA FP16 forward/backward passed")
PY
python -m vifactcheck --help
python -m vifactcheck train --help
```

Giữ **FP16 AMP**, Adafactor với explicit LR, accumulation 8 và activation checkpointing. Không bật BF16 hoặc yêu cầu FlashAttention chỉ vì một hướng dẫn GPU khác dùng chúng. Probe nhỏ trên không thay full-model smoke; không dự báo P100 nhanh hơn laptop khi chưa đo.

## Bước 4 — Storage, data và assets

Chọn ổ output thật, có đủ dung lượng. Ví dụ kiểm tra filesystem chứa gốc repo:

```bash
python - <<'PY'
from pathlib import Path
import shutil

path = Path.cwd().resolve()
free = shutil.disk_usage(path).free / 1024**3
print(path, "free GiB:", round(free, 2))
assert free >= 35, "Need storage before full training; no checkpoint is deleted"
PY
python -m vifactcheck prepare --cache .cache --output data/prepared
```

Nếu output nằm trên ổ khác, kiểm tra đúng ổ đó. 35 GiB là mốc khởi động của full v2 worker; model lớn có thể cần hơn. Mọi lần lưu cần reserve cho file tạm; không gộp cache của các backbone vào `.cache/phobert` hoặc sửa manifest PhoBERT để giả làm model khác.

Lệnh prepare hiện tải **PhoBERT-base** được pin cùng VnCoreNLP và official splits. Nó không tải mBERT/XLM-R/PhoBERT-v2. Những assets mới phải qua adapter/manifest đã được triển khai theo kế hoạch benchmark.

Xác nhận dữ liệu và import đúng source:

```bash
python - <<'PY'
from pathlib import Path
import json
import vifactcheck
from vifactcheck.data import load_split

package = Path(vifactcheck.__file__).resolve()
assert package.is_relative_to(Path("src/vifactcheck").resolve()), package
print("package:", package)
manifest = json.loads(Path("data/prepared/manifest.json").read_text())
assert manifest["assets"]["dataset_revision"] == "2df623632299a3459ad95feeff9ac67d7205a42a"
for split, expected in {"train": 5062, "dev": 723, "test": 1447}.items():
    rows = load_split("data/prepared", split)
    assert len(rows) == expected, (split, len(rows))
    assert {r["label"] for r in rows} == {0, 1, 2}
    print(split, len(rows), "hash verified")
PY
```

Chuẩn bị file test không phải chọn model bằng test. Không chạy final test inference hoặc xem điểm test trong lúc chọn LR, lambda, epoch cap và checkpoint.

## Bước 5 — Test và PhoBERT smoke có thật trong CLI

```bash
python -m pytest -q
python -u -m vifactcheck smoke \
  --models M0 M1 A1 M2 A2 \
  --reranker-architecture listwise \
  --data data/prepared \
  --model-path .cache/phobert \
  --segmenter-path .cache/vncorenlp \
  --output runs/server-phobert-smoke
```

Smoke dùng 24 train / 12 dev, một reranker epoch, một warmup và một continuation; token/evidence caps vẫn 256/96 và 20/5. Lưu `smoke_report.json`, failure log nếu có, peak allocated/reserved GPU memory và thời gian. Scores này chỉ kiểm tra đường chạy.

Nếu thư mục output có run khác, dùng tên mới; nếu smoke đã hoàn tất và không đổi source/config/data, kiểm tra artifacts trước khi chạy lại. Không có kết quả server được xác nhận chỉ vì guide chứa các lệnh trên. Một lỗi CUDA/OOM phải dừng để chẩn đoán; không tự freeze encoder hoặc đổi kiến trúc.

Sau khi baseline/adapter mới được triển khai, chạy smoke riêng cho **cả baseline và M2 của từng backbone**. M2 phải qua workload claim dài/nhiều candidates và cả hai encoder vẫn trainable. Team lead cấp lệnh thực tế từ `--help`; không thay đường dẫn PhoBERT bằng mBERT rồi coi smoke trên đã đủ.

### Job được chuẩn bị trước: pilot hội tụ M2 trên P100

P100 là một lựa chọn chạy diagnostic dài sau khi xác nhận môi trường server. Cập nhật 17/11/2026: laptop cũng đã được cho phép chạy cùng protocol khi ổ output đạt ít nhất 35 GiB trống; run 15 epoch hiện chưa khởi chạy. Đọc [kế hoạch hội tụ và lệnh Windows/Linux](../protocol/convergence.md), chọn máy/output cho run mới, rồi kiểm tra:

```bash
python scripts/run_convergence.py \
  --data data/prepared \
  --model-path .cache/phobert \
  --output runs/convergence-phobert-seed42-e15 \
  --lr-horizon-epochs 5 \
  --check-only
```

Preflight kiểm tra assets/device/disk mà không ghi checkpoint. Khi đạt, bỏ `--check-only` để chạy. Lệnh mặc định dùng seed 42, LR `1e-5`, lambda `0.1`, ba reranker epoch và 15 verifier epochs (ba warmup + 12 continuation); `--lr-horizon-epochs 5` giữ lịch LR năm epoch đầu. Chạy lại chính lệnh bình thường để resume cùng run identity; không sửa số epoch giữa chừng. `--smoke` hoặc `--total-epochs 5` chỉ dùng cho một output mới khi kiểm tra đường chạy. Run test nhỏ và pilot hội tụ có mục đích khác nhau, không trộn scores.

## Bước 6 — Checkpoint tham chiếu từ laptop

Checkpoint [M2 seed 13 đã chia sẻ](https://github.com/ngpnduy/vi-factcheck-system/releases/tag/listwise-v2-three-seeds-20261112) đi kèm `ViFactCheck-M2-seed13-handoff.zip` và `ViFactCheck-M2-seed13-SHA256SUMS.txt`. Download bằng tài khoản GitHub có quyền đọc; repo private trả 404 khi chưa có access.

Kiểm tra checksum rồi extract helper vào thư mục `handoff/`:

```bash
sha256sum -c ViFactCheck-M2-seed13-SHA256SUMS.txt
python -m zipfile -e ViFactCheck-M2-seed13-handoff.zip handoff
python handoff/predict_checkpoint.py \
  --checkpoint m2_best.pt \
  --model-path .cache/phobert \
  --segmenter-path .cache/vncorenlp \
  --input examples/claim.json \
  --output runs/server-reference-prediction.json \
  --device cuda
```

Helper kiểm tra checkpoint hash rồi rebase asset paths trong bộ nhớ. Saved Windows paths khiến ordinary loader của release có thể không dùng được trực tiếp trên Linux; CLI `predict/evaluate` hiện không có override `--model-path`. Không sửa và lưu đè file gốc để đổi đường dẫn. Checkpoint M2 chứa đủ reranker + verifier, không phải chỉ một trong hai. Dùng nó kiểm tra inference, không dùng nó để khởi tạo backbone khác hoặc coi nó thay thế terminal resume state của một run dài.

## Bước 7 — Khóa môi trường, chạy job và resume

```bash
mkdir -p runs/server-environment
git rev-parse HEAD > runs/server-environment/source-commit.txt
conda env export --no-builds > runs/server-environment/environment-resolved.yml
conda list --explicit > runs/server-environment/conda-explicit.txt
python -m pip freeze > runs/server-environment/requirements-resolved.txt
nvidia-smi > runs/server-environment/nvidia-smi.txt
```

Lưu thêm lệnh cài CUDA wheel và tên OS/driver; `pip freeze` không tự ghi lại index URL đã dùng. Trước khi share lock/log, kiểm tra và bỏ URL/token của private package index nếu có. Đây là môi trường Linux của server, không được ghi đè lock laptop. Nếu một backbone đòi phiên bản khác, tạo environment riêng và chạy lại acceptance checks.

Với job PhoBERT dùng CLI train có thật, resume một phase từ **`*_last.pt`**, giữ cùng checkout/data/assets và output. Ví dụ sau đây là mẫu đường dẫn phải thay bằng file last thật của chính job đó:

```bash
python -u -m vifactcheck train \
  --resume runs/my-phobert-job/m2_last.pt
```

Không dùng `m2_best.pt` để giả làm trạng thái cuối epoch; không đổi các epoch args và mong `--resume` tự nhận ngân sách mới: nó đọc cấu hình đã lưu. Nhánh/tổng epoch mới cần một run identity mới và cơ chế continuation được chẩn đoán, giữ original checkpoints. Full launcher v2 điều phối M0/M1/M2 hoặc cả năm model, nên không dùng nó thay cho runner baseline–M2 tương lai.

Cho mỗi job dùng một output directory và một writer. Dùng scheduler của server hoặc session được duy trì theo quy định máy; dừng có checkpoint trước maintenance. Không tùy ý tắt process của người khác. Bàn giao lịch sử, selection, revisions/hashes và compact predictions theo [artifact contract](backbone-benchmark.md).

## Khi nào teammate được bắt đầu benchmark thật?

Sau khi có đủ: approved implementation commit; baseline + adapter đã qua test; epoch/tuning policy đã khóa bằng train/dev; môi trường và storage đạt gate; smoke cho đúng backbone đạt; thư mục run mới và task ownership rõ. Khi chưa đủ, chỉ setup/prepare/kiểm tra compatibility, không chạy một thử nghiệm dài rồi đặt tên như benchmark hoàn chỉnh.
