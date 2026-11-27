# Hướng dẫn nhóm: một baseline và M2 trên nhiều backbone

Cập nhật ngày 17/11/2026. Tài liệu dùng cấu trúc `release/v2`: chạy từ gốc repo, package ở `src/vifactcheck/`, script ở `scripts/`, kết quả ở `reports/`. Đây là **kế hoạch triển khai và vận hành**, không phải thông báo các backbone mới đã chạy xong.

## 1. Chốt câu hỏi và hai hệ thống

Mỗi backbone được dùng trong cùng hai thiết kế:

| Hệ thống | Input và cách dự đoán | Tham số học |
|---|---|---|
| **Baseline classifier** | Claim + article, truncate theo ngân sách đã công bố, phân loại ba nhãn | Một encoder và classification head |
| **M2** | BM25 lấy tối đa 20 câu; reranker chọn tối đa 5; verifier gộp xác suất có trọng số; verification feedback đi tới reranker | Hai encoder riêng biệt cùng loại backbone, các head và list-aware attention |

Mục tiêu là đo `Macro-F1(M2) − Macro-F1(baseline)` **trong từng backbone**. Baseline này là claim–article classifier, không phải M0. M0/M1/A1/A2 trong study PhoBERT đã hoàn tất vẫn là các phép đối chứng để giải thích thành phần. So sánh baseline–M2 đánh giá cả hệ thống; nó không tự tách riêng đóng góp của attention hoặc gradient feedback.

Giới hạn token của baseline phải ghi rõ là **article đã truncate**, không gọi đó là toàn bộ bài báo được encoder nhìn thấy. M2 có nhiều lượt encode, hai encoder và evidence supervision; công bố thêm tổng token được xử lý, số tham số, runtime và memory. Cùng giới hạn 256 token mỗi pair không đồng nghĩa cùng tổng compute hoặc cùng số token bài báo được đọc.

## 2. Cái gì chạy được trong code hiện tại?

| Tác vụ | Trạng thái trên nền release/v2 |
|---|---|
| Prepare, smoke, train, tune, evaluate, predict cho pipeline PhoBERT M0/M1/A1/M2/A2 | Đã có CLI và protocol |
| Classifier baseline claim–article độc lập | **Chưa có trong CLI; cần triển khai trước benchmark** |
| Chuyển đồng loạt giữa PhoBERT/mBERT/XLM-R với preprocessing và provenance đúng | **Chưa có registry/adapter tổng quát; cần triển khai và kiểm thử** |
| Đổi `--model-path` thành mBERT rồi coi là hỗ trợ hoàn chỉnh | Chưa đủ: code hiện đưa text đã word-segment cho mọi encoder và không gửi BERT `token_type_ids` |

Các lệnh CLI có thật được trình bày trong [protocol v2](../protocol/v2.md) và [hướng dẫn Conda/P100](server-conda.md). Tài liệu này không đưa lệnh giả như `train-baseline` hoặc `--backbone`: chúng chưa tồn tại. Team lead bổ sung lệnh benchmark sau khi implementation vượt qua các gate ở bước 4–5.

## 3. Ma trận, người phụ trách và điều kiện tái sử dụng

| Backbone pretrained | Người/máy chính | Kế hoạch |
|---|---|---|
| `vinai/phobert-base` | Laptop / kết quả hiện có | Baseline mới; M2 ba seed đã có làm mốc lịch sử |
| `vinai/phobert-base-v2` | Laptop RTX 3050 4 GB | Baseline + M2 sau khi đủ storage và smoke memory |
| `google-bert/bert-base-multilingual-cased` | Teammate / Tesla P100 + Conda | Baseline + M2; backbone mới đầu tiên trên server |
| `FacebookAI/xlm-roberta-base` | Teammate / P100 | Baseline + M2 sau khi ETA cho phép hoàn tất trước cutoff |

Mỗi hàng hoàn chỉnh có hai hệ thống × seeds **42, 13, 2026** = sáu kết quả test. Ba backbone mới tạo 18 kết quả, không phải chạy lại cả năm cấu hình cho từng backbone. Reranker pretraining, warmup và tuning vẫn là công việc bổ sung của M2; sáu kết quả không có nghĩa chỉ sáu job tính toán.

M2 PhoBERT cũ chỉ được tái sử dụng trong bảng chính khi protocol mới giữ nguyên chính sách train/chọn checkpoint phù hợp. Nếu sau pilot hội tụ nhóm tăng trần epoch hoặc thay preprocessing/objective, ghi một protocol mới và chạy đối chiếu cùng điều kiện. Khi đó điểm năm epoch cũ là **mốc lịch sử**, không gắn nhãn kết quả của protocol mới.

## 4. Bước 1 — Chốt epoch bằng một pilot hội tụ trước

1. Chạy chẩn đoán trên **seed 42, train/dev**, giữ nguyên LR/lambda và kiến trúc trong một run mới. Người dùng đã chọn **chuẩn bị run này cho P100 của teammate**; không khởi chạy training dài trên laptop trong bước chuẩn bị. Không dùng test để xác định khi nào dừng.
2. Vẽ riêng train verification NLL, evidence BCE, objective tổng, dev NLL và dev Macro-F1 theo epoch. Đánh dấu reranker, warmup và continuation. Warmup và continuation dùng objective khác nhau nên không kết luận từ một đường `average_loss` nối thẳng qua hai stage.
3. Trần lớn hơn năm epoch giúp quan sát plateau/overfit; nó không chứng minh nghiệm tốt nhất toàn cục. Chọn checkpoint theo dev metric đã định, không chọn epoch chỉ vì train loss nhỏ nhất.
4. Trước benchmark, lưu quyết định: tổng verifier epochs tối đa `E`, warmup epochs, lịch LR và horizon, chính sách best/early-stop, tie-break, metric và lý do. Nếu dùng early stopping, định trước patience/min-delta và áp dụng cùng quy tắc cho cả hai hệ thống. Phân biệt **một run pilot** với kết quả ba seed.
5. Baseline được cùng trần `E` verifier epochs; M2 giữ ba warmup epoch và `E−3` continuation nếu đó là lịch được chốt. Reranker pretraining được báo riêng. Giữ bảng v2 năm epoch cũ nguyên vẹn.

Runner chẩn đoán mới là `scripts/run_convergence.py`, độc lập với full worker của study đã công bố. Mặc định seed 42, LR `1e-5`, lambda `0.1`, ba reranker epoch và **15 verifier epochs = 3 warmup + 12 continuation**, listwise, token 256/96 và evidence 20/5. Đây là trần pilot để quan sát, chưa phải ngân sách benchmark đã được lựa chọn. Giữ `--lr-horizon-epochs 5` để tăng số epoch không âm thầm đổi lịch LR của năm epoch đầu so với protocol gốc.

Đọc [protocol hội tụ](../protocol/convergence.md) và chuẩn bị môi trường P100 trước. Preflight không train và không ghi checkpoint:

```bash
python scripts/run_convergence.py \
  --data data/prepared \
  --model-path .cache/phobert \
  --output runs/convergence-phobert-seed42-e15 \
  --lr-horizon-epochs 5 \
  --check-only
```

Khi mọi gate đạt, teammate bỏ `--check-only` để bắt đầu. Chạy lại chính lệnh đó để resume ở ranh giới epoch, không sửa `*_best.pt` hoặc ghép checkpoint nguồn khác. `--smoke` là kiểm tra 24 train / 12 dev; nó không đủ để kết luận hội tụ trên full data. Runner chẩn đoán này không tạo baseline hoặc làm các backbone mới tự được hỗ trợ.

## 5. Bước 2 — Cố định source, dữ liệu và cấu hình

Hai máy checkout cùng **commit đã review**, xuất phát từ `release/v2`. Ghi `git rev-parse HEAD` và diff; không để mỗi người tự sửa model trong lúc các run chính đang chạy.

- Dataset: `tranthaihoa/vifactcheck`, revision `2df623632299a3459ad95feeff9ac67d7205a42a`.
- Số claim: train **5.062**, dev **723**, test **1.447**; `0=SUPPORTED`, `1=REFUTED`, `2=NOT_ENOUGH_INFO`.
- Lưu hash `manifest.json`, JSONL và tập ID; dùng đúng official split, không tự chia lại bằng random split.
- Cố định sentence boundaries, original offsets, conservative evidence alignment và BM25 20→5 của M2. Không thêm gold vào candidates.
- Claim/context gốc phải còn nguyên để dùng native tokenizer. PhoBERT/PhoBERT-v2 nhận word-segmented text; mBERT/XLM-R nhận normalized raw text. BM25 vẫn dùng cùng segmentation để giữ candidate IDs giữa các backbone.
- Không bắt đầu từ encoder ViFactCheck của tác giả khác nếu chưa biết nó được train trên những ID nào. Bắt đầu từ generic pretrained backbone, lưu model **và tokenizer** revision bất biến cùng hash asset.

## 6. Bước 3 — Hoàn thiện implementation một lần, rồi dùng chung

Team lead tạo baseline và backbone adapter; teammate review trước khi chạy:

1. Một baseline head ba nhãn, một pair claim–article mỗi claim, claim cap 96 và tổng pair cap 256 gồm special tokens. Ghi số article token còn lại và tỷ lệ truncate. Khởi tạo classification head được seed hóa.
2. Adapter model/tokenizer ghi rõ native preprocessing, pair special tokens, hidden size, pooling, dropout, model constructor và loại forward arguments. mBERT phải có segment IDs hợp lệ cho claim/article.
3. M2 dùng hai bản sao pretrained độc lập, không chia sẻ storage/trọng số. Đảm bảo gradients đi qua score weights của các câu được chọn; top-k indices vẫn discrete. NEI không có evidence loss.
4. Reranker pretraining/chọn dev MRR và verifier chọn dev Macro-F1 được giữ theo protocol đã khóa. Warmup best và warmup end là hai trạng thái khác nhau. Resume phải giữ đúng optimizer, AMP scaler và RNG.
5. Run identity gồm backbone/revision, architecture, source, dữ liệu, preprocessing, epoch policy và seed; không resume giữa hai identity khác nhau. Mở rộng tên artifact nhưng không đổi nghĩa của kết quả lịch sử.
6. Bộ đánh giá kiểm tra số prediction/ID, xác suất, thứ tự nhãn và số ranking bằng số claim; không cho `zip` âm thầm cắt dữ liệu. Ghi rõ mẫu số retrieval khác classification.

Sau khi có code, team lead thêm các lệnh CLI/launcher thực tế vào phần này bằng chính output `--help`, rồi mới giao lệnh train cho từng máy. **Đây là gate bắt buộc hiện chưa được tuyên bố hoàn tất trong hướng dẫn này.**

## 7. Bước 4 — Setup và smoke từng backbone

Teammate theo [server-conda.md](server-conda.md); laptop theo [README](../../README.md). Trước download/run, kiểm tra storage. Mốc hiện có là ít nhất **35 GiB** cho full worker; với model lớn phải dự trù từ checkpoint thực tế và giữ reserve mỗi lần ghi. Không tự xóa weights cũ để vượt gate.

Với mỗi cặp backbone–hệ thống, chạy smoke thật **24 train / 12 dev** và giữ cap 256/96, M2 20/5:

- Baseline và M2 đều forward/backward được; loss hữu hạn; optimizer cập nhật tham số dự kiến.
- Reranker/verifier của M2 đều cập nhật, gold không được chèn vào selection; NEI không có evidence BCE.
- M2 qua được những claim dài/nhiều candidates nhất, không chỉ mẫu ngắn.
- Reload và resume qua một điểm gián đoạn; raw/prepared prediction dùng cùng preprocessing.
- Lưu peak GPU allocated/reserved, giây/claim, thời gian dev và kích thước checkpoint. Cross-GPU có thể có sai khác số học nhỏ; lưu số đo thực tế.

OOM hoặc lỗi CUDA phải dừng và lưu diagnostics. Không tự freeze encoder, giảm token/evidence limits, đổi sang LoRA hoặc CPU rồi gộp với hàng chạy cấu hình ban đầu. Smoke score chỉ là pipeline check.

## 8. Bước 5 — Tuning dev, khóa cấu hình, chạy ba seed

Chốt một search plan trước khi đọc test:

| Hệ thống | Search seed 42/dev đề xuất | Checkpoint criterion |
|---|---|---|
| Baseline | LR `{1e-5, 2e-5}` với cùng epoch policy; chọn best dev Macro-F1 | Hòa điểm chọn epoch sớm hơn, rồi thứ tự search |
| M2 | Giữ search hai bước v2: hai LR qua M1, rồi lambda `{0.1,0.5,1.0}` cho M2 tại LR thắng; mỗi lambda dùng cùng warmup end | Best dev Macro-F1 trong trần epoch mới đã khóa; giữ warmup candidates theo protocol |

Hai hệ thống có cùng search cho LR nhưng **không cùng số trial hoặc GPU hours**: M2 có thêm search lambda và phase chuẩn bị. Phải công bố cả hai ngân sách; không gọi đây là so sánh compute-matched. Nếu nhóm muốn giới hạn khác vì deadline, chốt và ghi lại trước các run, không quyết định sau khi xem test.

Tái sử dụng checkpoint thắng tuning làm seed 42 của chính study mới. Chạy seeds 13/2026 với hyperparameters đã khóa cho **từng backbone**. Không retune mỗi seed. Lưu tất cả trial, kể cả trial kém; không bỏ backbone chỉ vì dev/test của nó không thuận lợi.

Các lệnh `python -m vifactcheck tune` hiện chỉ là tuning cho pipeline PhoBERT; chúng không train baseline mới và không tự chuẩn bị registry cho các backbone khác. [Protocol v2](../protocol/v2.md) mô tả lệnh lịch sử. Không chạy `full` để thực hiện thiết kế hai-hệ-thống này: full worker hiện điều phối M0/M1/M2, thêm A1/A2 khi yêu cầu.

## 9. Bước 6 — Đánh giá và bàn giao

Hoàn tất tuning và các run retained, seal checkpoint/config/hash trước final test. Mỗi checkpoint được chọn đánh giá trên đúng **1.447 test IDs**. Báo cáo:

- Macro-F1, accuracy, per-label precision/recall/F1 và confusion matrix; mean ± **sample SD** trên ba seeds, không phải ensemble.
- Chênh lệch M2−baseline trong từng backbone/seed, paired article-bootstrap CI; không coi ba bản test lặp qua seeds là ba tập test độc lập.
- M2 retrieval trên các claim S/R có aligned gold hợp lệ, kèm count/coverage; test hiện có 539 eligible S/R trong 976 S/R, trong khi classification dùng đủ 1.447. Baseline classifier không có sentence ranking nên ghi retrieval là N/A.
- Context/truncation, selected epoch/stage, tham số, tổng compute, latency và GPU memory. So sánh timing trên cùng hardware/workload hoặc ghi khác hardware, không suy hiệu năng P100 từ laptop.

Mỗi job bàn giao một thư mục gồm `config`, model/tokenizer/data/source revisions và hashes, environment lock, history/log, selection metadata, checkpoint SHA-256, dev metrics, final per-claim test predictions và metrics. Baseline checkpoint chứa encoder + trained classifier; M2 chứa đủ hai encoder + heads + attention. Checkpoint chỉ có encoder không đủ để dự đoán verdict: các PLM artifacts ViFactCheck đã kiểm tra trước đây thiếu classifier head.

Chỉ gửi compact reports/configs/predictions và checksum qua repo/Release; lưu weights thành asset riêng khi được yêu cầu. Không đưa token, mật khẩu, raw datasets, model cache hay môi trường vào ZIP. Repo private nên người nhận cần quyền đọc.

## 10. Phân công và mốc trước deadline

| Mốc | Laptop / người dùng | Server P100 / teammate |
|---|---|---|
| 17–19/11 | Audit loss, chuẩn bị/review runner và khóa protocol; không launch long run local | Ghi driver/VRAM/OS, setup Conda, smoke và chạy pilot hội tụ sau preflight |
| 19–23/11 | PhoBERT-v2 baseline + M2 khi adapter đã sẵn sàng; chuẩn bị method/analysis | mBERT baseline + M2; XLM-R chỉ khi ETA qua gate |
| 24–25/11 | Kiểm tra manifests, tổng hợp controls đã có | Hoàn tất retained seeds, seal artifacts, đánh giá test |
| 26–28/11 | Cùng hoàn thành draft, per-label/errors và review | Cùng kiểm tra số liệu, citations, reproducibility |
| 29/11 | Mục tiêu nộp nội bộ | Mục tiêu nộp nội bộ |
| 30/11 | Buffer cho deadline thật và múi giờ của venue | Buffer |

Sau epoch/trial đầu, tính ETA riêng trên từng máy, gồm dev và save overhead. Ưu tiên một cặp baseline–M2 đủ ba seed trên mỗi máy trước khi thêm backbone. Dừng thêm job tùy chọn sau 23/11; khóa kết quả chính ngày 25/11. Nếu một hàng chỉ có seed 42, ghi rõ exploratory và để riêng bảng ba seed.

## Nguồn và tài liệu liên quan

- [Protocol và checkpoint selection v2](../protocol/v2.md); [báo cáo study đã hoàn tất](../../reports/experiments/listwise-best5-v2-three-seeds-20261112/REPORT.md).
- [Hướng dẫn Conda/P100](server-conda.md).
- Model cards: [PhoBERT/PhoBERT-v2](https://huggingface.co/vinai/phobert-base-v2), [mBERT](https://huggingface.co/google-bert/bert-base-multilingual-cased), [XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-base). Pin revision khi chuẩn bị run.
