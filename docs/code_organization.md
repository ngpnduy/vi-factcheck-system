# Tổ chức code sau khi hợp nhất M0 và ViFactCheck

Hai luồng dùng chung dữ liệu chuẩn `DatasetRecord` trong
`data/processed/vifactcheck/`, nhưng có giao thức thí nghiệm và định dạng dự đoán
riêng. Không dùng điểm của luồng này thay cho luồng kia.

## Các thành phần và điểm vào

| Luồng | Module | Vai trò |
|---|---|---|
| Tái hiện ViFactCheck | `classification/plm/` | Phân loại cặp claim–gold evidence hoặc claim–full context |
| Tái hiện ViFactCheck | `classification/llm/` | Prompt, fine-tune Unsloth/LoRA và inference |
| Tái hiện ViFactCheck | `retrieval/bm25.py` | BM25Okapi và các metric được port từ notebook |
| Tái hiện ViFactCheck | `retrieval/sbert.py`, `retrieval/hybrid.py` | SBERT và BM25 kết hợp SBERT |
| Tái hiện ViFactCheck | `retrieval/eval_bm25.py`, `retrieval/eval_sbert.py` | Đánh giá retrieval theo giao thức notebook, tách câu bằng NLTK |
| M0 | `data/prepare_m0.py`, `retrieval/context_bm25.py` | Tách câu có offset, BM25 Lucene trong context riêng, inputs/targets tách biệt |
| M0 | `training/`, `models/verifier.py`, `inference/predict_m0.py` | PhoBERT cho từng cặp claim–câu, trung bình xác suất theo claim |
| M0 | `evaluation/evaluate_m0.py` | Verdict và sentence retrieval metrics, kiểm tra ID và alignment |
| Corpus BM25 | `retrieval/build_corpus.py`, `retrieval/corpus_bm25.py` | Index BM25S trên corpus chunk dùng chung |
| Corpus BM25 | `retrieval/evaluate_bm25.py` | Đánh giá chunk retrieval |
| Rush M0/M1/M2 | `rush/vifactcheck/` (từ gốc repo) | Subproject riêng, pilot 900 claim/seed 42, test/report tự động |

Các đường dẫn module trong bảng thuộc `src/factcheck/`.

Riêng `rush/` nằm tại gốc repository và có `pyproject.toml`, môi trường,
preprocessing, checkpoint và test suite riêng. Xem [hướng dẫn Rush](../rush/README.md).
Giao thức Rush dùng PhoBERT-base, VnCoreNLP sentence boundaries, BM25Okapi và
Adafactor; kết quả không thay thế trực tiếp cho M0 PhoBERT-base-v2 trên `main`.

Trước merge, cả hai nhánh đều có `retrieval/bm25.py` với API khác nhau. Bản
BM25Okapi trên `main` giữ tên này, nên các import của SBERT/hybrid vẫn hoạt động.
Bản corpus BM25S từ `feat/m0-baseline` chuyển sang `retrieval/corpus_bm25.py`.
Nếu từng gọi trực tiếp `python -m factcheck.retrieval.bm25 --corpus ...`, hãy đổi
sang `python -m factcheck.retrieval.corpus_bm25 --corpus ...`. Các script
`run_vifactcheck_bm25.sh` và `evaluate_vifactcheck_bm25.sh` vẫn giữ tên cũ.
M0 tiếp tục dùng `scripts/run_m0.sh`; cấu hình và thuật toán M0 không đổi.

## Môi trường chung

```bash
bash scripts/setup.sh
bash scripts/download_vifactcheck.sh
bash scripts/preprocess_vifactcheck.sh
```

`setup.sh` đồng bộ các nhóm dependency trong `uv.lock`. Nhóm `baseline` bao gồm
Torch, Transformers và NLTK. Cài thêm dữ liệu tokenizer khi dùng retrieval của
bản tái hiện paper:

```bash
.venv/bin/python -m nltk.downloader punkt_tab
```

Các module không tải NLTK resource khi import hoặc xem `--help`.
M0 dùng VnCoreNLP thay vì NLTK và cần Java 11+:

```bash
bash scripts/setup_m0.sh
bash scripts/run_m0.sh prepare
bash scripts/run_m0.sh train --smoke --seed 42 --output-dir outputs/m0_check
```

Chạy các lệnh từ thư mục gốc repo. Vì project đặt `package = false`, khi gọi
module trực tiếp phải đặt `PYTHONPATH=src` như các ví dụ bên dưới; các shell script
đã tự đặt biến này.

## Train, predict và evaluate PLM

```bash
bash scripts/train_plm.sh \
  --model-name vinai/phobert-base \
  --input-mode gold_evidence \
  --output-dir outputs/plm

PYTHONPATH=src .venv/bin/python -m factcheck.classification.plm.inference \
  --model-name vinai/phobert-base \
  --input-mode gold_evidence \
  --checkpoint outputs/plm/best_model.pt \
  --split validation \
  --output outputs/plm/validation_predictions.jsonl

bash scripts/eval_plm.sh \
  --predictions outputs/plm/validation_predictions.jsonl \
  --gold data/processed/vifactcheck/validation.jsonl
```

Model và input mode khi inference phải khớp run train. Đầu ra PLM/LLM dùng
`{"id": "...", "prediction": "SUPPORTED"}` và evaluator chung
`factcheck.evaluation.evaluator`. M0 dùng `label`, `probabilities`, `sentence_ids`
và evaluator riêng qua `scripts/run_m0.sh evaluate`; hai định dạng không thay thế
cho nhau. Các metric retrieval của paper cũng khác metric sentence/coverage M0.

## Retrieval của bản tái hiện paper

```bash
PYTHONPATH=src .venv/bin/python -m factcheck.retrieval.eval_bm25 \
  --gold data/processed/vifactcheck/validation.jsonl --top-k 5

PYTHONPATH=src .venv/bin/python -m factcheck.retrieval.eval_sbert \
  --method hybrid --bm25-candidates 20 \
  --gold data/processed/vifactcheck/validation.jsonl --top-k 5
```

SBERT tải model khi thực sự chạy đánh giá. CLI hybrid truyền số ứng viên vào
`HybridRetriever(..., bm25_top_n=...)`. Các lệnh trên dùng validation; chỉ đánh
giá test sau khi chốt giao thức.

## LLM và giới hạn kiểm chứng

Các module `classification/llm/train_unsloth.py` và `inference.py` giữ cách nạp
Unsloth/TRL khi thực thi. Unsloth/TRL chưa nằm trong môi trường khóa chung; cần
môi trường GPU tương thích riêng trước khi chạy LLM. Không tự động cài chúng vào
môi trường M0 trên macOS.

Chế độ LLM `retrieved_evidence` hiện đọc `record.evidence`. Phải tạo dữ liệu đầu
vào riêng chứa bằng chứng đã retrieve trước khi dùng chế độ này; dữ liệu processed
gốc chứa gold evidence. Chưa có bộ chuyển tự động từ outputs M0/corpus sang định
dạng đó. Không ghi đè gold của bộ processed để tránh nhầm dữ liệu đánh giá.

Kiểm tra tích hợp dùng mô hình nhỏ cục bộ và fixture, không tải model hoặc chạy
đánh giá trên test thật. Smoke PhoBERT kiểm tra pipeline, không xác nhận chất lượng
baseline sau huấn luyện đầy đủ. Tham khảo [giao thức M0](m0.md) và
[giải thích luồng M0](m0_workflow_explained.md).
