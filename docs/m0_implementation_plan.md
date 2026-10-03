# Kế hoạch hoàn thành baseline M0 trên ViFactCheck

Ngày lập kế hoạch: 30/09/2026. Căn cứ: mã nguồn tại thời điểm lập và Proposal Word đã chỉnh sửa, mục 4–7.

Cập nhật 01/10/2026: các module dữ liệu, retrieval, verifier, trainer, predict và evaluator đã được triển khai; có smoke test PhoBERT/VnCoreNLP thật và kiểm tra lưu/nạp checkpoint. Xem [hướng dẫn M0](m0.md) cho trạng thái và lệnh chạy hiện tại. Các bảng “hiện trạng” và “dự kiến” dưới đây được giữ làm mốc kế hoạch ban đầu. Chưa có run đầy đủ 5 epoch/3 seed; LR còn phụ thuộc M1 và pilot tách câu chưa rà soát thủ công đủ 100 context.

## 1 Định nghĩa M0 cần tuân thủ

Luồng dự đoán: claim và context đi kèm → tách context thành câu → BM25 trong context đó → tối đa 5 câu → PhoBERT dự đoán ba nhãn cho từng cặp claim–câu → trung bình xác suất các câu → nhãn có xác suất cao nhất.

- BM25 không được huấn luyện bằng nhãn; M0 chỉ huấn luyện verifier.
- Mỗi câu được chọn có trọng số 1/n, với n là số câu thực sử dụng, không luôn chia cho 5.
- Tính softmax riêng trên logits của từng cặp rồi mới lấy trung bình xác suất. Trung bình logits trước softmax và nối 5 câu thành một đầu vào là các mô hình khác.
- Loss là negative log-likelihood của nhãn claim trên phân phối đã tổng hợp. Không dùng nhãn claim để tính loss độc lập trên từng câu không được gán nhãn.
- Tất cả ba nhãn đều tham gia học và đánh giá: SUPPORTED, REFUTED, NOT_ENOUGH_INFO.
- Gold evidence chỉ dùng cho đánh giá chất lượng chọn bằng chứng trong M0, không dùng để chọn câu huấn luyện hoặc dự đoán.
- Có thể lưu BM25 top-20 để đánh giá Hit@20 và chuẩn bị ứng viên cho M1; verifier M0 chỉ nhận top-5 theo BM25.

## 2 Hiện trạng đã kiểm tra

| Thành phần hiện có | Có thể tận dụng | Khoảng cách đến M0 |
|---|---|---|
| `src/factcheck/data/schemas.py` | Nhãn chuẩn, cấu trúc claim/context/evidence và kiểm tra dữ liệu | Cần cấu trúc riêng cho câu và đầu vào dự đoán không chứa gold |
| `src/factcheck/data/vifactcheck.py` | Download, normalize, validate, publish và manifest | Không cần tải lại chỉ để bắt đầu M0; cần đối chiếu manifest và audit dữ liệu |
| `src/factcheck/retrieval/build_corpus.py` | Cách ghi dữ liệu, checksum, truy vết nguồn | Đang chia cửa sổ 200 từ, overlap 40, trên corpus gộp; không phải tách câu trong context |
| `src/factcheck/retrieval/corpus_bm25.py` | Thư viện bm25s và cơ chế lưu thông tin retrieval | Đang nhận CorpusChunk và tìm trong index chung; cần retriever theo context |
| `src/factcheck/retrieval/tokenizer.py` | Có tokenizer Unicode và chuẩn hóa matching | Chưa phải tách từ tiếng Việt cho PhoBERT; matching hiện tại bỏ dấu câu nên không dùng nguyên trạng làm gold alignment chính thức |
| `src/factcheck/retrieval/evaluate_bm25.py` | Cách đối chiếu query, kết quả và báo mẫu chưa khớp | Đang đánh giá chunk; `recall_at_k` hiện mang nghĩa hit theo query, không phải sentence Recall@k |
| `pyproject.toml` | Đã khai báo nhóm baseline gồm torch, transformers, accelerate và sentencepiece | Cần xác minh môi trường thực chạy, cấu hình GPU và bộ tách câu/từ |
| Verifier, trainer, dự đoán verdict | Chưa có module trong checkout đã đọc | Cần xây dựng |

Snapshot dữ liệu đã đọc từ ba tập processed:

| Tập | Số claim | Thiếu context | Context phân biệt |
|---|---:|---:|---:|
| Train | 5062 | 0 | 1035 |
| Validation | 723 | 0 | 496 |
| Test | 1447 | 0 | 758 |

Theo hash nội dung context nguyên trạng: 495 context phân biệt ở validation cũng xuất hiện trong train; 753 context phân biệt ở test cũng xuất hiện trong train. Đây là số context, không phải số claim, và chưa phải audit gần trùng hoặc URL. Không tự suy ra rò rỉ nhãn; tuy nhiên, đánh giá trên split chính thức không chứng minh khả năng xử lý bài báo hoàn toàn chưa thấy. Báo cáo hiện tượng này và sử dụng split chính thức cho kết quả chính theo Proposal.

Kết quả BM25 validation đang có ở `outputs/rag/vifactcheck_bm25s_top5/validation_retrieval_metrics.json` thuộc thiết lập chunk/corpus chung. Không sử dụng các điểm đó như kết quả M0 hoặc như điểm sentence retrieval mới.

## 3 Thứ tự triển khai

### Bước 1 Chốt giao thức và kiểm tra dữ liệu

**Việc làm**

- Đối chiếu các tập processed với checksum trong manifest; kiểm tra ID, nhãn, context và trùng nguồn.
- Khai báo label mapping cố định, chẳng hạn SUPPORTED=0, REFUTED=1, NOT_ENOUGH_INFO=2; dùng cùng mapping lúc train, dự đoán và đánh giá.
- Cấu hình M0 ghi rõ phạm vi `within_context`, k=5, trung bình xác suất, chiều dài cặp 256 token, claim tối đa 96 token và ba seed 13/42/2026.
- Chốt một phiên bản PhoBERT và revision cụ thể. Đề xuất khởi đầu PhoBERT-base-v2; các cấu hình so sánh sau đó phải dùng cùng phiên bản.
- Xác minh môi trường dependency đã khai báo và tài nguyên thực tế; GPU 16–24 GB trong Proposal là dự kiến, không phải phần cứng đã xác nhận.

**Hoàn thành khi:** có một cấu hình rõ ràng và báo cáo dữ liệu; chưa cần huấn luyện mô hình.

### Bước 2 Tạo đơn vị câu và căn chỉnh gold

**Việc làm**

- Tách câu trước khi tách từ; lựa chọn bộ tách câu/từ tương thích PhoBERT qua pilot train/validation. VnCoreNLP là phương án cần thử đầu tiên.
- Mỗi câu có ID ổn định, context ID, thứ tự câu, raw text, khoảng ký tự trong context và text đã tiền xử lý cho mô hình.
- Cache theo nội dung context để không xử lý lặp các claim chung bài; không gộp các context khác nhau chỉ vì cùng URL.
- Kiểm tra ít nhất 100 context train/validation, chú ý chữ viết tắt, ngày tháng, số thập phân, dấu ba chấm, phủ định và đơn vị.
- Ánh xạ span gold vào một hoặc nhiều câu bằng chuẩn hóa nhẹ có truy vết offset. Không loại dấu thập phân hay thay giá trị ngày tháng để ép matching thành công. Khớp nhiều vị trí phải được đánh dấu/rà soát.
- Dữ liệu câu dùng cho inference tách khỏi nhãn và gold sentence IDs dùng cho học/đánh giá. Mẫu gold không khớp vẫn tham gia đánh giá verdict.

**Hoàn thành khi:** câu có thể truy ngược về context, số claim được bảo toàn và có báo cáo độ bao phủ gold alignment.

### Bước 3 BM25 trong context và chọn top-5

**Việc làm**

- Xây BM25 trên các câu của từng context; thống kê IDF/độ dài cũng tính trên context đó. Không lấy top-k từ index toàn corpus rồi lọc theo bài.
- Dùng cùng tokenizer cho claim và câu trong BM25; cấu hình tokenizer riêng với bộ subword tokenizer của PhoBERT. Có thể bắt đầu bằng tokenizer Unicode hiện có, nhưng phải cố định và ghi rõ cho tất cả cấu hình.
- Khai báo rõ method Lucene, k1 và b trong cấu hình thay vì dựa ngầm vào mặc định thư viện; đề xuất k1=1.5, b=0.75.
- Lưu thứ hạng, sentence IDs, điểm BM25 và số câu thực chọn; xử lý hòa điểm theo thứ tự câu ổn định trên toàn bộ ứng viên trước khi cắt top-k.
- Ít hơn 5 câu thì dùng toàn bộ. Điểm bằng 0 vẫn được xếp theo quy tắc hòa điểm. Nếu không tạo được câu hoặc không có token hợp lệ, báo rõ mẫu lỗi để xử lý trước khi chạy chính thức; không âm thầm bỏ mẫu hoặc lấy gold thay thế.
- Chuẩn bị top-20 và top-5 cho train/validation. Cùng phép biến đổi sẽ áp dụng test khi giao thức được khóa.

**Kiểm tra có ý nghĩa:** thay nhãn/gold phải không làm đổi retrieval; mọi câu trả về phải thuộc context của claim; context ngắn và trường hợp hòa điểm phải ổn định.

**Hoàn thành khi:** mỗi claim có danh sách top-5 hợp lệ và có điểm Hit@5, sentence Recall@5, MRR trên tập validation căn chỉnh được.

### Bước 4 Dữ liệu đầu vào và verifier M0

**Việc làm**

- Tạo các cặp `(claim, sentence)` từ top-5 BM25. Không nối các câu thành một đoạn duy nhất.
- Tách từ đúng yêu cầu PhoBERT, sau đó subword tokenization. Tổng chiều dài tối đa 256 bao gồm special tokens; claim tối đa 96, ngân sách còn lại dành cho câu. Ghi tỷ lệ cắt.
- Batch tổ chức theo claim: giữ quan hệ giữa từng cặp và claim nguồn. Dùng mask để câu padding không tham gia trung bình.
- PhoBERT và head phân loại sinh logits ba lớp cho từng cặp. Softmax theo chiều nhãn rồi lấy trung bình các câu thật của cùng claim.
- Tính một loss trên mỗi claim, lấy trung bình theo claim. Ưu tiên tính log-probability ổn định bằng log-softmax và logsumexp tương đương trung bình xác suất.
- Tại inference, chỉ cần claim và các câu đã chọn; không truyền gold evidence hoặc label vào model.

**Kiểm tra có ý nghĩa:** tổng xác suất bằng một; một câu thì kết quả giống dự đoán cặp đó; padding không đổi kết quả; hoán đổi thứ tự các câu ở chế độ eval không đổi trung bình; loss có gradient về verifier.

**Hoàn thành khi:** chạy forward/backward được trên một batch nhỏ và khớp kết quả pooling tính tay.

### Bước 5 Huấn luyện thử và chạy validation

**Việc làm**

- Dùng một tập nhỏ, ví dụ 16–32 claim train, để kiểm tra mô hình có thể học giảm loss. Đây là kiểm tra kỹ thuật, không phải kết quả nghiên cứu.
- Chạy thử ngắn để đo bộ nhớ/thời gian và kiểm tra lưu, nạp lại mô hình.
- Chạy M0 một seed trên train, theo dõi Macro-F1 validation; bảo toàn NEI và các mẫu thiếu gold alignment.
- Xây trainer dùng lại được cho các cấu hình so sánh về sau: seed, thứ tự dữ liệu, batch hiệu dụng, lịch học, optimizer, cách chọn checkpoint và log chi phí đều được ghi nhận.
- Ngân sách verifier theo Proposal là 3 epoch khởi tạo + 2 epoch tiếp tục, tối đa 5 epoch. Lịch learning rate/optimizer qua ranh giới hai giai đoạn phải được quy định nhất quán với M1/A1.

**Điểm phụ thuộc Proposal:** learning rate chung phải được lựa chọn theo M1 trên validation. Vì M1 chưa có, M0 hiện tại có thể chạy với learning rate thử nghiệm (ví dụ 2e-5) để hoàn thiện pipeline. Không gọi cấu hình tạm này là cấu hình đã khóa của bảng so sánh cuối. Khi M1 chốt learning rate, chạy lại M0 nếu cần.

**Hoàn thành khi:** có một run M0 train→validation đầy đủ, checkpoint nạp lại được và dự đoán tái đánh giá ra cùng metric.

### Bước 6 Đánh giá chính thức và đóng gói kết quả

**Việc làm**

- Chốt thiết lập chung với M1 theo Proposal trước khi đánh giá test chính thức.
- Huấn luyện ba seed 13/42/2026; chọn checkpoint theo Macro-F1 validation, hòa điểm chọn sớm hơn. Không chọn checkpoint theo test.
- Lưu kết quả theo claim: ID, sentence IDs, xác suất ba lớp, verdict, model/seed/config. Gold label được ghép vào ở evaluator, không phải đầu vào predict.
- Báo Macro-F1, accuracy, precision/recall/F1 từng nhãn, confusion matrix, trung bình/độ lệch chuẩn giữa seed, chi phí và độ trễ.
- Báo sentence retrieval trên S/R căn chỉnh hợp lệ, kèm số mẫu và coverage; NEI không nhận một recall bằng chứng giả định.
- Lưu dấu vết phù hợp cho D0 và phân tích lỗi sau này. Các so sánh paired bootstrap M0–M1/M2 thực hiện khi có các mô hình đối chứng, không phải điều kiện để riêng pipeline M0 chạy được.
- Ghi rõ overlap context trong báo cáo. Không tự thay split chính thức trong quá trình hoàn thiện baseline.

**Hoàn thành khi:** đủ ba run cuối, kết quả test theo giao thức đã khóa và người khác tái tính được các bảng đánh giá từ dự đoán đã lưu.

## 4 Phân chia module dự kiến

Các đường dẫn dưới đây tính từ gốc repository và là đề xuất, chưa phải các lệnh hay module có sẵn.

| Module dự kiến | Trách nhiệm |
|---|---|
| `src/factcheck/data/sentences.py` | Cấu trúc câu, tách câu, offset và cache |
| `src/factcheck/data/evidence_alignment.py` | Căn chỉnh gold phục vụ đánh giá; không được gọi từ inference |
| `src/factcheck/retrieval/context_bm25.py` | Chọn câu trong context bằng BM25 |
| `src/factcheck/models/verifier.py` | PhoBERT, phân phối từng cặp và pooling đồng đều |
| `src/factcheck/training/m0_dataset.py` | Tạo cặp, token budget, batch và mask |
| `src/factcheck/training/train_m0.py` | Train, validation và chọn checkpoint |
| `src/factcheck/inference/predict_m0.py` | Dự đoán không đọc nhãn/gold |
| `src/factcheck/evaluation/evaluate_m0.py` | Verdict metrics và sentence retrieval metrics |
| `configs/experiments/m0.yaml` | Cấu hình khai báo giao thức và tham số thực nghiệm |

Ưu tiên bổ sung các module có trách nhiệm rõ ràng; không đổi tên CorpusChunk thành câu hoặc ghi đè corpus cũ để làm cho thiết lập khác trông giống M0. Chỉ tách helper chung khi thực sự cần cho hai luồng.

## 5 Mốc triển khai cho người đang học code

1. Hiểu `DatasetRecord`, `normalize_row`, `normalize_dataset`, `publish_splits`: biết một claim đã chuẩn hóa trông như thế nào.
2. Hoàn thành tách câu và xem thủ công một context cùng các sentence IDs.
3. Hoàn thành BM25 theo context, giải thích được vì sao 5 câu được chọn.
4. Hoàn thành một lượt dự đoán PhoBERT và tính tay trung bình xác suất.
5. Hoàn thành một batch học, rồi một run train/validation.
6. Sau khi khóa thiết lập chung, chạy ba seed và tổng hợp kết quả test.

Ước lượng tổ chức công việc: khoảng 10–15 buổi làm việc tập trung để có pipeline M0 và run validation đầu tiên, chưa tính thời gian học nền tảng, chờ GPU, khắc phục lỗi dữ liệu hoặc huấn luyện dài. Thời gian GPU chỉ ước lượng sau lần đo thử. Mốc kết quả so sánh cuối còn phụ thuộc việc lựa chọn learning rate chung của M1.

## 6 Checklist nghiệm thu

- [ ] Đầu vào, top-k, label mapping và backbone thống nhất với Proposal.
- [ ] Có audit split và context overlap.
- [ ] Tách câu có offset, bảo toàn dấu hiệu số lượng/thời gian/phủ định.
- [ ] BM25 chỉ tìm trong context tương ứng, không đọc gold/label.
- [ ] Verifier tổng hợp xác suất với trọng số 1/n, có mask câu padding.
- [ ] Loss ở mức claim, không gán nhãn claim cho từng câu riêng lẻ.
- [ ] Có kiểm tra batch, gradient, lưu/nạp và học trên mẫu nhỏ.
- [ ] Có một run validation hoàn chỉnh trước khi chạy nhiều seed.
- [ ] Tham số cuối đã khóa theo giao thức chung; test không dùng để tuning.
- [ ] Có ba seed, metric verdict, metric bằng chứng với coverage và chi phí.
- [ ] Tái tính được kết quả từ dự đoán lưu lại.
