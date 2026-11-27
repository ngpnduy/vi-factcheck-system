# Hướng dẫn đọc sâu dự án ViFactCheck

> Tài liệu này giải thích phiên bản nghiên cứu **Listwise v2, best-of-five, three seeds** đang được báo cáo trong repo. Mục tiêu là giúp bạn đi từ câu hỏi “mô hình đang làm gì?” đến “các con số chứng minh được điều gì?”. Đây là hướng dẫn đọc và trình bày, không thay thế protocol và report gốc.

## Đọc theo thứ tự nào?

1. Đọc **Verdict là gì?** và **Bài toán** để hiểu mô hình nhận đầu vào nào và phải dự đoán gì.
2. Đọc **Dataset** và **Flow inference** để hình dung dữ liệu chạy qua các thành phần.
3. Đọc **M0–M2, A1–A2** để hiểu năm phép so sánh.
4. Đọc **Loss và training** để biết mô hình học bằng tín hiệu nào.
5. Đọc **Metrics** trước khi xem bảng kết quả; mỗi metric trả lời một câu hỏi khác nhau.
6. Đọc **Kết quả v2** và **Giới hạn** để tập nói kết luận chính xác, không nói quá điều thí nghiệm chứng minh.

Các nguồn chính trong repo: [protocol v2](../protocol/v2.md), [report ba seeds](../../reports/experiments/listwise-best5-v2-three-seeds-20261112/REPORT.md), [README](../../README.md), mã [metrics](../../src/vifactcheck/metrics.py), [model/loss](../../src/vifactcheck/model.py) và [proposal](../../reports/Proposal_v2.docx).

---

## 1. Verdict là gì?

**Verdict** là kết luận của hệ thống về một claim (một phát biểu cần kiểm chứng), dựa trên nội dung bài báo/ngữ cảnh được đưa vào. Dataset dùng ba verdict:

| Verdict | Ý nghĩa dễ hiểu | Cách đọc |
|---|---|---|
| `SUPPORTED` | Ngữ cảnh cung cấp thông tin ủng hộ claim. | Claim được hỗ trợ bởi nội dung đã cho. |
| `REFUTED` | Ngữ cảnh có thông tin mâu thuẫn với claim. | Claim bị bác bỏ bởi nội dung đã cho. |
| `NOT_ENOUGH_INFO` (`NEI`) | Ngữ cảnh không đủ thông tin để kết luận ủng hộ hay bác bỏ. | Chưa thể xác minh từ ngữ cảnh hiện có. |

Ví dụ: bài viết ghi “sự kiện xảy ra tháng 4 năm 1936”, claim ghi “sự kiện xảy ra tháng 4 năm 1930”. Nếu hai mốc nói về cùng sự kiện, bằng chứng mâu thuẫn với claim nên verdict là `REFUTED`.

**Bằng chứng (evidence) không phải verdict.** Evidence là câu hoặc các câu trong bài viết được dùng để suy luận; verdict là nhãn kết luận cuối. Một hệ thống có thể tìm đúng câu bằng chứng nhưng vẫn suy luận nhầm nhãn, hoặc dự đoán nhãn đúng dù câu bằng chứng được chọn chưa đầy đủ. Vì thế dự án đo cả phân loại verdict lẫn việc tìm evidence.

`NOT_ENOUGH_INFO` không có nghĩa claim chắc chắn sai, cũng không có nghĩa “không tìm thấy câu vì hệ thống yếu”. Nhãn này nói rằng, theo thông tin trong ngữ cảnh được xét, không đủ cơ sở để kết luận. Thực tế, mô hình vẫn có thể dự đoán sai NEI; cần đọc chung với confusion matrix và recall/F1 từng lớp.

### Câu trả lời mô hình tạo ra

Về khái niệm, mô hình cho ba xác suất, chẳng hạn:

```text
P(SUPPORTED) = 0.72
P(REFUTED) = 0.18
P(NOT_ENOUGH_INFO) = 0.10
```

Nhãn có xác suất cao nhất trở thành verdict dự đoán, ở ví dụ này là `SUPPORTED`. Xác suất là độ tin của mô hình theo cách nó được huấn luyện; không nên tự động diễn giải `0.72` là “đúng 72%” nếu chưa đánh giá calibration.

---

## 2. Bài toán nghiên cứu

Hệ thống phải giải quyết hai việc liên tiếp nhưng liên quan nhau:

1. **Tìm câu hữu ích:** từ bài viết dài, chọn những câu có khả năng làm bằng chứng cho claim.
2. **Xác minh claim:** đọc claim cùng các câu đã chọn và dự đoán một trong ba verdict.

Nếu bước đầu bỏ sót bằng chứng quan trọng, verifier có thể không bao giờ nhìn thấy nó. Nếu đã có bằng chứng phù hợp nhưng verifier hiểu sai quan hệ giữa claim và evidence, verdict vẫn sai. Đây là lý do dự án không chỉ báo cáo một điểm accuracy.

Proposal đặt các câu hỏi nghiên cứu quanh việc reranking có giúp chọn evidence/phân loại hay không, huấn luyện chung có hơn huấn luyện tách rời hay không, và lỗi xuất hiện ở đâu/loại mâu thuẫn nào. Các kết quả hiện tại trả lời khá trực tiếp phần so sánh mô hình và pipeline; việc phân tích sâu theo từng kiểu mâu thuẫn còn cần hoàn thiện riêng.

### Từ vựng cốt lõi

| Từ | Nghĩa trong dự án |
|---|---|
| Claim / statement | Phát biểu cần kiểm chứng. |
| Context / article | Nội dung bài viết được dùng làm nguồn để kiểm chứng. |
| Sentence | Một câu đã được tách từ context. |
| Evidence | Một hoặc nhiều câu được gán/đối chiếu là hỗ trợ hoặc bác bỏ claim. |
| Candidate | Câu được bộ truy hồi đưa vào danh sách có thể hữu ích. |
| Ranker / reranker | Thành phần chấm điểm và sắp xếp candidate theo độ liên quan. |
| Verifier | Thành phần suy luận nhãn verdict từ claim và các câu được chọn. |
| Pooling / aggregation | Cách gộp dự đoán từ nhiều câu thành một dự đoán cho claim. |
| Ablation | Thí nghiệm bỏ/thay một cơ chế để xem cơ chế đó đóng góp gì. |
| Seed | Giá trị khởi tạo các nguồn ngẫu nhiên để đo độ nhạy với randomness. |

---

## 3. Dataset và các nhãn

Nghiên cứu v2 dùng ba split chính thức của ViFactCheck:

| Split | Số claim | Vai trò |
|---|---:|---|
| Train | 5,062 | Cập nhật trọng số mô hình. |
| Dev / validation | 723 | Chọn checkpoint và tuning tham số. |
| Test | 1,447 | Đánh giá cuối; không dùng để chọn tham số/checkpoint trong protocol v2. |

Mỗi record về cơ bản có statement/claim, context, label và thông tin evidence. Topic, author, URL và các trường tương tự là metadata; chúng không đồng nghĩa với đầu vào mà verifier đọc. Hãy xem schema prepared data để biết chính xác trường nào được dùng ở mỗi bước.

Trong mã, nhãn số được ánh xạ theo thứ tự:

| Mã | Tên |
|---:|---|
| 0 | `SUPPORTED` |
| 1 | `REFUTED` |
| 2 | `NOT_ENOUGH_INFO` |

### Vì sao số claim của retrieval khác số claim classification?

Classification được tính trên toàn bộ **1,447** test claim. Retrieval metrics chỉ tính trên **539 claim đủ điều kiện** trong báo cáo: claim thuộc `SUPPORTED`/`REFUTED`, alignment evidence hợp lệ và có gold sentence IDs. NEI không có yêu cầu truy hồi một bằng chứng ủng hộ/bác bỏ cụ thể theo cách đo này; claim không align được hoặc alignment mơ hồ cũng không đưa vào mẫu số retrieval.

`alignment_coverage = 0.5923` trong report là tỷ lệ record test có alignment status hợp lệ trên toàn test, gồm cả NEI. Nó **không** có nghĩa ranker chỉ tìm được bằng chứng cho 59.23% claim, cũng không phải Hit@5. Khi đọc từng metric retrieval, luôn nhìn cột `Eligible claims`/`count` cùng với metric đó.

---

## 4. Flow của hệ thống: từ văn bản đến verdict

Ở mức khái quát, một claim đi qua các bước sau:

```text
Claim + bài viết
       ↓
Tách bài viết thành các câu và giữ offset câu gốc
       ↓
BM25 lấy danh sách tối đa 20 câu ứng viên (top 20)
       ↓
Ranker sắp xếp lại và chọn tối đa 5 câu (top 5)
       ↓
Verifier đọc claim với từng câu được chọn
       ↓
Gộp xác suất của các câu thành xác suất cho cả claim
       ↓
Chọn lớp có xác suất cao nhất: verdict
```

### Bước 1 — Tách câu

Bài viết được chia thành câu để hệ thống có thể chọn evidence ở mức câu. Offset giúp kết quả inference chỉ ra câu nào trong văn bản gốc đã được chọn. Chất lượng sentence splitting quan trọng: nếu câu bị tách sai, evidence có thể bị cắt hoặc nối không đúng.

### Bước 2 — BM25 truy hồi ban đầu

BM25 là phương pháp truy hồi dựa trên độ khớp từ/ngữ liệu. Nó tạo danh sách ứng viên nhanh, nhưng không phải bộ suy luận verdict và không “hiểu” mâu thuẫn như con người. Trong M1/M2/A1/A2, ranker nhận tối đa 20 BM25 candidates. Trong M0, hệ thống dùng tối đa 5 câu BM25 trực tiếp.

### Bước 3 — Reranker chọn top 5

Reranker đọc thông tin claim–candidate rồi cho điểm để sắp xếp các câu có vẻ liên quan nhất. Đây là bước học được; nó khác BM25 ở chỗ có thể dùng biểu diễn ngôn ngữ để xếp hạng. “Listwise” chỉ cách kiến trúc/huấn luyện xem xét một danh sách ứng viên; tên này không tự nói rằng verifier chắc chắn đúng.

### Bước 4 — Verifier phân loại từng cặp claim–câu

Verifier dự đoán ba xác suất cho claim khi ghép với từng câu được chọn. Một câu riêng lẻ có thể chưa đủ kết luận. Vì vậy dự án tổng hợp phân phối xác suất qua các câu, thay vì chỉ lấy nhãn của một câu.

### Bước 5 — Gộp thành verdict claim

Gọi `p_i(y)` là xác suất lớp `y` khi xét câu thứ `i`, `w_i` là trọng số câu đó. Dự đoán cho claim được gộp theo dạng:

```text
P(y | claim, context) = Σ_i w_i × p_i(y)
Σ_i w_i = 1
```

Pooling đều cho mỗi câu trọng số giống nhau. Pooling có trọng số dùng điểm ranker để câu được ưu tiên nhận trọng số lớn hơn (softmax scores). Sau khi gộp, lớp có xác suất cao nhất là verdict.

### Top-k có nghĩa gì?

`top 20` nghĩa là giữ tối đa 20 ứng viên tốt nhất theo bước đang nói tới; `top 5` là tối đa năm câu chuyển sang verifier. Nếu bài viết có ít câu thì số câu thực tế nhỏ hơn. Ranker chỉ sắp xếp những gì BM25 đã tìm thấy: nó không thể chọn câu gold mà BM25 chưa đưa vào danh sách top 20.

---

## 5. Năm cấu hình: M0, M1, A1, M2, A2

Đọc các cấu hình như một chuỗi so sánh có kiểm soát. “Model tốt nhất” trả lời mô hình nào có điểm cao nhất; ablation giúp hỏi thành phần nào có thể giải thích chênh lệch.

| Mã | Truy hồi / chọn câu | Gộp xác suất | Cách train ranker | Vai trò so sánh |
|---|---|---|---|---|
| **M0** | BM25 top 5 trực tiếp | Đều | Không có reranker học được | Baseline pipeline đơn giản. |
| **M1** | BM25 top 20 → reranker top 5 | Theo trọng số ranker | Reranker được pretrain rồi đóng băng trong verifier training | Hệ thống rerank + weighted pooling. |
| **A1** | Cùng selector/reranker đóng băng như M1 | Đều | Đóng băng | Cô lập tác dụng của weighted pooling khi so với M1. |
| **M2** | BM25 top 20 → reranker top 5 | Theo trọng số ranker | Joint training: verifier feedback có thể truyền tới ranker, cùng evidence supervision | Đề xuất kết hợp đầy đủ trong nghiên cứu. |
| **A2** | Như M2 | Theo trọng số ranker nhưng chặn đường feedback verdict→ranker | Evidence loss vẫn cập nhật ranker | Ablation kiểm tra đóng góp của verification feedback. |

### M0 — mốc nền

M0 lấy tối đa năm câu BM25 và gộp dự đoán đồng đều. Đây là baseline nội bộ để so sánh thay đổi trong cùng implementation. Không nên gọi nó là toàn bộ các baseline đã công bố trong paper gốc nếu report không chạy trực tiếp những cấu hình đó.

### M1 — thêm reranker và pooling có trọng số

M1 lấy tới 20 câu từ BM25, rerank để chọn tối đa năm, sau đó dùng score/weight ranker khi tổng hợp xác suất. Reranker được pretrain và đóng băng trong giai đoạn verifier train. So sánh M1 với M0 thay đổi cả selection pipeline lẫn pooling, nên riêng cặp này không tách được đóng góp từng yếu tố.

### A1 — cùng câu được chọn, gộp đều

A1 dùng cùng kiểu selector frozen như M1 nhưng gộp đều thay vì dùng trọng số ranker. Vì vậy **M1 − A1** là so sánh trực tiếp hơn cho tác dụng của weighted pooling trong điều kiện selection gần như giữ nguyên.

### M2 — joint training

M2 train verifier và ranker cùng nhau. Tín hiệu verdict có thể tác động ngược đến điểm ranker qua trọng số gộp; evidence supervision cũng dạy ranker câu nào là gold evidence. Top-k chọn câu là thao tác rời rạc nên gradient không đi xuyên qua chính quyết định chọn top-k; đường feedback khả vi ở đây đi qua điểm/trọng số được dùng để gộp.

### A2 — chặn verifier feedback

A2 giữ thiết lập gần M2 nhưng detach verification weights trước khi dùng để gộp. Vì vậy verdict loss không truyền gradient ngược vào ranker qua weights. **Evidence loss vẫn có thể cập nhật ranker** trên ví dụ eligible. So sánh M2−A2 ước lượng đóng góp của nhánh feedback verdict trong protocol này; nó không phải so sánh “có train” với “không train” toàn bộ ranker.

### Bốn phép trừ nên nhớ

| So sánh | Câu hỏi gần đúng |
|---|---|
| `A1 − M0` | Thay pipeline chọn câu bằng selector/reranker M1, trong khi pooling đều, có thay đổi kết quả không? Lưu ý selector và số ứng viên cũng đổi. |
| `M1 − A1` | Weighted pooling có ích hơn pooling đều khi selector được giữ tương tự không? |
| `M2 − M1` | Joint training/evidence supervision theo cấu hình M2 có thêm lợi ích so với M1 frozen không? Đây là nhiều khác biệt liên quan, cần xem implementation để diễn giải chính xác. |
| `M2 − A2` | Cho verdict feedback về ranker có thêm lợi ích so với chặn đường gradient đó, khi evidence supervision vẫn còn không? |

Mọi hiệu ứng này là hiệu ứng trong **protocol và dataset hiện tại**; không tự động khái quát thành định luật cho mọi fact-checking model.

---

## 6. Loss: mô hình bị phạt như thế nào khi sai?

Loss là một con số mà quá trình tối ưu cố gắng làm nhỏ. Nó khác metric báo cáo: training cập nhật theo loss; checkpoint/report có thể chọn theo Macro-F1 trên dev.

### 6.1 Verdict loss

Với nhãn gold `y`, mô hình bị phạt khi xác suất gán cho lớp đúng nhỏ. Dạng cross-entropy cho một ví dụ:

```text
L_verdict = −log P(y_gold | claim, context)
```

Nếu đúng lớp nhận xác suất cao, loss thấp; nếu nhận xác suất rất thấp, loss cao. Loss được tính sau khi xác suất từ các câu được gộp thành xác suất của claim.

### 6.2 Evidence loss

Ranker cần học phân biệt câu gold evidence với câu không gold. Trong code, evidence loss là binary cross-entropy trên scores, cân bằng trung bình giữa nhóm positive và negative để một nhóm đông hơn không tự lấn át nhóm còn lại.

Evidence loss chỉ áp dụng khi claim `SUPPORTED`/`REFUTED` có alignment hợp lệ và candidate set có cả gold positive lẫn negative. **NEI không có evidence loss** trong cấu hình này; nó vẫn nhận verdict loss. Claim không align được cũng không nhận evidence loss nhưng vẫn có thể tham gia huấn luyện verdict.

### 6.3 Tổng loss và lambda

Với M2, mục tiêu có dạng:

```text
L_total = L_verdict + λ × L_evidence
```

`λ` điều chỉnh độ mạnh của evidence supervision tương đối với verdict loss. V2 tuning trên seed 42/dev chọn `λ = 0.1`; thử các giá trị `0.1, 0.5, 1.0`. Đây là weight của loss, không phải xác suất, không phải learning rate và không phải tỷ lệ evidence đúng.

Ở các cấu hình không dùng evidence loss, phần này không đóng góp vào objective. A2 dùng evidence loss nhưng chặn nhánh gradient từ verdict loss tới ranker. Không nên hiểu “loss đã hội tụ” chỉ từ Macro-F1 cuối: để kết luận hội tụ theo nghĩa tối ưu cần xem đường loss/train-dev qua epoch, gradient và trạng thái huấn luyện. Report này trình bày checkpoint selection và diagnostics; điểm cuối tốt tự nó không chứng minh tối ưu toàn cục.

---

## 7. Training v2 và ý nghĩa best-of-five

Protocol v2 sử dụng train/dev chính thức, tune trên seed 42/dev rồi khóa tham số cho các seed còn lại. Report ghi nhận LR `1e-5`, evidence lambda `0.1`. Test dùng cả 1,447 claim cho từng model/seed sau khi hoàn tất việc training; test không dùng để chọn hyperparameter hay checkpoint trong protocol v2.

Quy trình verifier:

1. Reranker được pretrain 3 epoch và chọn theo **dev MRR**.
2. Train 3 epoch warmup cho weighted verifier chung; lưu `warmup_best.pt` và trạng thái cuối epoch 3 `warmup_end.pt`.
3. M1/M2/A2 bắt đầu continuation từ đúng trạng thái cuối warmup, gồm cả optimizer/scaler/RNG, rồi chạy global epoch 4–5.
4. M0/A1 train verifier 5 epoch từ common initial verifier weights.
5. Chọn checkpoint dự đoán theo **dev Macro-F1 tốt nhất trong cả năm epoch**; nếu hòa, chọn epoch sớm hơn.

Do đó “best-of-five” nghĩa là kiểm tra năm checkpoint epoch bằng dev Macro-F1 rồi chọn checkpoint tốt nhất. Nó **không** có nghĩa lấy trung bình năm model hay ensemble năm lần chạy. Với M2/A2, checkpoint warmup có thể thắng continuation; điều này không mâu thuẫn với việc terminal state đã được dùng để kiểm tra gradient/encoder update. Checkpoint được chọn để dự đoán và trạng thái terminal để resume/training là hai mục đích khác nhau.

### Seed là gì trong thí nghiệm này?

Seed là giá trị điều khiển các nguồn ngẫu nhiên như khởi tạo trọng số, thứ tự mẫu và dropout. Chạy cùng cấu hình với seed khác tạo ba lần huấn luyện độc lập hơn về randomness, còn split train/dev/test vẫn giống nhau. Nếu kết quả gần nhau, đó là dấu hiệu phương pháp ít nhạy với các lần khởi tạo đã thử; 3 seeds vẫn là mẫu nhỏ, không chứng minh ổn định trên mọi lần chạy hay dataset.

Các seeds là **42, 13, 2026**. Chúng không phải ba phần dữ liệu, không phải ba nhãn, và không phải ba fold cross-validation. Bảng mean ± sample SD là tóm tắt điểm của ba model riêng; inference report không phải ensemble lấy phiếu từ ba seed.

---

## 8. Metrics: mỗi con số trả lời câu hỏi nào?

### 8.1 Metrics phân loại verdict

#### Accuracy

Tỷ lệ tổng số claim mà verdict dự đoán khớp nhãn gold:

```text
accuracy = số dự đoán đúng / tổng số claim
```

Ưu điểm: dễ hiểu, trả lời “trên toàn bộ mẫu có bao nhiêu phần đúng?”. Hạn chế: có thể che việc một lớp bị làm kém hơn lớp khác. Luôn đọc cùng per-label metrics và Macro-F1.

#### Precision theo lớp

Trong tất cả các claim mô hình **dự đoán là lớp X**, tỷ lệ thật sự là X. Ví dụ REFUTED precision thấp nghĩa là nhiều claim bị gọi là REFUTED nhưng gold label thuộc lớp khác.

#### Recall theo lớp

Trong tất cả claim gold thuộc lớp X, tỷ lệ mô hình tìm/dự đoán đúng là X. REFUTED recall thấp nghĩa là nhiều claim thực sự bị bác bỏ nhưng mô hình bỏ sót, thường nhầm sang SUPPORTED hoặc NEI.

#### F1 theo lớp

F1 là trung bình điều hòa precision và recall:

```text
F1 = 2 × precision × recall / (precision + recall)
```

F1 chỉ cao khi cả precision lẫn recall tương đối tốt. Nó làm rõ từng verdict, không gộp mất khó khăn riêng của một lớp.

#### Macro-F1

Lấy F1 riêng của SUPPORTED, REFUTED, NEI rồi tính trung bình đều. Mỗi lớp có trọng số ngang nhau trong phép trung bình, bất kể support. Đây là metric chính để chọn checkpoint trong v2. Vì nó chú ý cả lớp yếu, Macro-F1 có thể thấp hơn accuracy.

#### Support

Số ví dụ gold thuộc mỗi lớp. Ví dụ test support của SUPPORTED là 508 trong confusion matrix seed 42. Support cho biết mẫu số và giúp nhận ra metric nào được tính trên ít ví dụ hơn.

#### Confusion matrix

Ma trận đếm lớp gold theo hàng và lớp mô hình dự đoán theo cột. Đường chéo là dự đoán đúng; ngoài đường chéo là kiểu nhầm. Trong report, thứ tự hàng/cột là SUPPORTED, REFUTED, NEI. Nếu REFUTED→SUPPORTED lớn, mô hình đang nhầm một phát biểu bị bác bỏ thành được hỗ trợ; nếu REFUTED→NEI lớn, mô hình thường không đưa ra kết luận đủ mạnh.

### 8.2 Metrics truy hồi evidence

Chỉ dùng 539 claim eligible như đã giải thích ở phần Dataset. `k` là số câu đầu trong ranking đang đánh giá.

#### Hit@k

Tỷ lệ claim mà **ít nhất một** gold evidence sentence xuất hiện trong top-k. Hit@5 cao cho biết thường có một câu bằng chứng trong năm câu đầu; nó không đảm bảo đã lấy đủ toàn bộ bằng chứng.

#### Sentence Recall@k

Với mỗi claim, tính số gold evidence sentences được tìm thấy trong top-k chia cho tổng số gold evidence sentences, rồi lấy trung bình qua claim. Nếu claim cần 2 câu và top-k chỉ lấy 1, recall cho claim đó là 1/2.

#### Complete gold@k

Tỷ lệ claim mà top-k chứa **tất cả** gold evidence sentences đã align. Đây là tiêu chuẩn nghiêm hơn Hit@k. Nếu bài toán có nhiều evidence sentence cho một claim, có thể Hit cao nhưng complete-gold thấp.

#### MRR (Mean Reciprocal Rank)

Với mỗi claim, lấy thứ hạng của gold sentence đầu tiên: đứng đầu đóng góp 1; thứ hai đóng góp 1/2; thứ ba đóng góp 1/3; không thấy gold đóng góp 0. MRR cao nghĩa là bằng chứng đầu tiên thường được xếp gần đầu danh sách. MRR không đo xem các gold còn lại đã đủ hết chưa.

#### Alignment coverage / eligible count

Coverage báo tỷ lệ record có alignment hợp lệ theo report; eligible count là mẫu số thực tế dùng tính retrieval. Đây là thông tin về độ bao phủ nhãn evidence và mẫu đánh giá, không phải chất lượng ranker tự thân.

### 8.3 Độ dao động qua seeds và khoảng tin cậy

**Mean ± sample SD** tóm tắt điểm ba lần train. SD nhỏ nghĩa là ba điểm đã quan sát gần nhau; với ba seeds không nên tuyên bố độ ổn định tuyệt đối. Chênh lệch giữa hai cấu hình được tính theo từng seed rồi mới tổng hợp.

Report còn có paired article-cluster bootstrap 95% CI cho chênh lệch trên test của từng seed. Bootstrap lấy cụm theo bài viết để claim từ cùng article không bị xem như hoàn toàn độc lập; paired nghĩa là so hai mô hình trên cùng mẫu/cụm. CI này **có điều kiện trên các checkpoint đã fit**: nó không thay thế độ biến thiên do training seed. Vì các seed đánh giá cùng test claim, không coi 15 kết quả model×seed là 15 test set độc lập.

CI chứa 0 nghĩa là bootstrap interval được báo cáo chưa tách rõ chênh lệch dương với 0 ở mức khoảng đó; không tự động chứng minh hai model bằng nhau. CI không chứa 0 cho thấy chênh lệch nhất quán theo phép bootstrap có điều kiện đã dùng; không giải quyết mọi hạn chế về split, dataset hay thiết kế.

### 8.4 Diagnostics và chi phí

| Chỉ số | Nó nói gì? | Nó không tự nói gì? |
|---|---|---|
| Error stages | Claim sai gắn với alignment unavailable, BM25 thiếu gold, selector thiếu gold, verifier sai dù có đủ gold, hoặc lỗi NEI. | Không phải tỷ lệ nguyên nhân nhân quả hoàn hảo; phân loại phụ thuộc alignment và định nghĩa stage trong code. |
| Truncation counts | Bao nhiêu claim/câu bị cắt do giới hạn token. | Không đảm bảo phần bị cắt gây ra lỗi. |
| Seconds per claim | Thời gian inference trung bình theo cách đo/report. | Không so được máy khác nếu phần cứng/điều kiện khác. |
| Peak CUDA allocated/reserved | Bộ nhớ GPU lớn nhất mô hình dùng/giữ trong evaluation. | Không phải yêu cầu VRAM tối thiểu chính xác cho mọi môi trường. |
| Training wall-clock | Thời gian thực tế của study, gồm gián đoạn và phần việc lặp lại như report ghi. | Không phải dự báo chắc chắn cho lần chạy mới hoặc laptop khác. |

---

## 9. Kết quả Listwise v2 hiện tại

Các số dưới là mean ± sample SD qua seeds 42, 13, 2026; không phải ensemble. Mỗi model được đánh giá trên cùng test split.

| Model | Test Macro-F1 | Test Accuracy |
|---|---:|---:|
| M0 | 0.5871 ± 0.0158 | 0.5911 ± 0.0105 |
| M1 | 0.6733 ± 0.0140 | 0.6750 ± 0.0128 |
| A1 | 0.5919 ± 0.0038 | 0.5941 ± 0.0054 |
| **M2** | **0.7066 ± 0.0057** | **0.7081 ± 0.0055** |
| A2 | 0.6799 ± 0.0047 | 0.6809 ± 0.0049 |

### Diễn giải từng chênh lệch

- `A1 − M0 = +0.0048` Macro-F1 trung bình: đổi sang selector/reranker nhưng pooling đều chỉ tăng rất ít trong trung bình ba seed. Chênh lệch theo seed khác dấu/độ lớn; không nên quảng bá đây là cải thiện chắc chắn lớn.
- `M1 − A1 = +0.0814`: weighted pooling có lợi rõ trong cặp này, với selection gần như giống nhau. Đây là đối chiếu rõ nhất cho pooling có trọng số.
- `M2 − M1 = +0.0332`: cấu hình joint M2 cao hơn M1 về Macro-F1 trung bình. Kết quả phù hợp với lợi ích từ thiết kế M2, nhưng so sánh bao gồm các thành phần/lịch huấn luyện liên quan; cần mô tả đúng ablation thay vì gán toàn bộ chênh lệch cho một chi tiết duy nhất.
- `M2 − A2 = +0.0266`: kết quả ủng hộ việc cho phép verifier feedback truyền tới ranker trong cấu hình đã thử, trong khi A2 vẫn giữ evidence loss. CI theo từng seed và giới hạn ba seeds vẫn cần được lưu ý.

M2 là model có điểm tốt nhất trong năm cấu hình được báo cáo, và Macro-F1 của nó khá gần nhau ở ba seed (0.7016, 0.7052, 0.7128; SD 0.0057). Đây là bằng chứng tích cực **trên split và protocol này**, không phải chứng minh mô hình luôn thắng mọi baseline/paper, cũng không phải đảm bảo hiệu quả trên dữ liệu ngoài miền.

### Một điều dễ đọc nhầm ở retrieval

M2 có verdict Macro-F1 cao nhất nhưng **không có Hit@5 tốt hơn M1 một cách nhất quán**. Các giá trị Hit@5 ở ba seed dao động và M1/A1 cũng rất cao. Vì vậy kết luận an toàn là M2 cải thiện phân loại verdict trong thí nghiệm; không nên nói “M2 thắng vì tìm được nhiều gold evidence hơn” nếu retrieval table không ủng hộ cách giải thích đó. Có thể pooling, tương tác giữa evidence/claim và cách dùng các câu được chọn góp phần; muốn xác định nguyên nhân cần phân tích sâu hơn.

### Câu nói ngắn dùng khi present

> “Trên ViFactCheck test split, cấu hình M2 đạt Macro-F1 trung bình 0.7066 qua ba seed, cao hơn M1 0.0332 và A2 0.0266. Ablation M1–A1 cho thấy weighted pooling đóng góp đáng kể trong thiết kế này; M2–A2 phù hợp với giả thuyết rằng verdict feedback hữu ích cho ranker. Retrieval Hit@5 không tăng tương ứng một cách nhất quán, nên chúng tôi không quy toàn bộ cải thiện verdict cho việc truy hồi được nhiều gold evidence hơn.”

---

## 10. Những giới hạn cần trình bày trung thực

1. **Mẫu retrieval nhỏ hơn classification.** Chỉ 539 claim eligible dùng cho retrieval; alignment coverage không bao phủ mọi test item. Cần nêu mẫu số khi báo retrieval metrics.
2. **Ba seeds là số lượng còn nhỏ.** Mean/SD là bằng chứng ban đầu về độ ổn định, không phải bảo đảm tổng quát.
3. **Test split không phải benchmark hoàn toàn mới.** Đây là official test trong protocol v2, nhưng test đã xuất hiện trong các nghiên cứu/phiên bản trước của dự án. Khi viết paper, cần ghi rõ lịch sử sử dụng để tránh diễn đạt như một blind test chưa từng xem.
4. **RQ về kiểu mâu thuẫn chưa được trả lời trọn vẹn.** Cần taxonomy/annotation và phân tích lỗi tương ứng; số tổng hợp hiện tại không cho biết M2 tốt hơn riêng với phủ định, số liệu, thời gian, thực thể hay nguyên nhân.
5. **M0 là baseline nội bộ của pipeline.** Đừng đồng nhất M0 với baseline PhoBERT trong paper gốc nếu chưa chạy đúng cấu hình, dữ liệu và protocol của baseline đó.
6. **Liên hệ với paper tham khảo là nguồn cảm hứng/thiết kế, không nhất thiết tái lập nguyên bản.** List-aware attention là implementation riêng; report ghi rõ không phải reproduction đầy đủ của M-ReRank hoặc RAV.
7. **Điểm classification không tự chứng minh lý do mô hình tốt hơn.** Cần ghép confusion matrix, retrieval, stage errors và ablations; tương quan giữa metric không tự tạo thành giải thích nhân quả.
8. **Checkpoint được chọn theo dev Macro-F1.** Đó là quy tắc selection đã định trong protocol; không suy từ checkpoint cuối cùng rằng loss train đã hội tụ tuyệt đối.

---

## 11. Cách đọc report, mở file nào trước

1. Mở [README](../../README.md) để nắm trạng thái study và các lệnh/nhánh nghiên cứu.
2. Mở [protocol Listwise v2](../protocol/v2.md) để hiểu cách train, tune, resume và chọn checkpoint.
3. Đọc phần đầu [REPORT.md](../../reports/experiments/listwise-best5-v2-three-seeds-20261112/REPORT.md): protocol và bảng aggregate scores.
4. Xem **Per-seed scores** để chắc rằng mean không che một seed thất bại.
5. Xem **Per-label F1** và **Confusion matrices** để biết verdict nào còn khó.
6. Xem **Retrieval and alignment** cùng eligible count để hiểu evidence metrics áp dụng cho subset nào.
7. Xem **Paired differences** và CI để đánh giá độ bất định của các đối chiếu.
8. Xem **Error stages**, **Truncation**, **runtime/memory** để tìm lỗi và chi phí vận hành.
9. Khi cần chi tiết implementation, đọc `metrics.py`, `model.py`, `training.py`/`full.py`; dùng Proposal để đối chiếu câu hỏi nghiên cứu và thiết kế dự kiến.

Report hiện lưu confusion matrices và diagnostics chi tiết theo seed/model; nên bắt đầu với seed 42 để học cách đọc một bảng, sau đó kiểm tra hai seed kia để xem mẫu hình có lặp lại không.

---

## 12. Những câu hỏi tự kiểm tra trước buổi trình bày

1. Verdict khác evidence ở điểm nào?
2. NEI có nghĩa claim sai không? Vì sao không?
3. Vì sao retrieval denominator là 539 nhưng classification count là 1,447?
4. Hit@5 cao có đồng nghĩa mọi gold evidence đã được tìm thấy không? Metric nào kiểm tra điều đó?
5. M1 và A1 khác nhau ở đâu; vì sao cặp này giúp kiểm tra weighted pooling?
6. M2 và A2 khác nhau ở đường gradient nào; A2 có còn evidence supervision không?
7. `λ=0.1` điều chỉnh điều gì? Nó có phải confidence không?
8. Macro-F1 khác accuracy ra sao, và vì sao v2 dùng Macro-F1 để chọn checkpoint?
9. Mean ± SD qua ba seeds chứng minh được điều gì, và điều gì còn chưa chứng minh?
10. Nếu M2 có Macro-F1 cao hơn nhưng Hit@5 không hơn M1, bạn sẽ nói kết luận nào an toàn?
11. Vì sao không gọi ba seeds là ba fold hoặc ensemble?
12. Tại sao `best-of-five` không có nghĩa là lấy trung bình năm checkpoint?

Nếu bạn trả lời được 1–8 bằng lời của mình và giải thích giới hạn ở 9–12, bạn đã nắm được xương sống của dự án để đọc paper và trình bày kết quả mà không chỉ đọc lại bảng số.

---

## Bảng tra nhanh

| Muốn hiểu… | Đọc… |
|---|---|
| Verdict / nhãn | Phần 1 và 3 |
| Toàn bộ pipeline | Phần 4 |
| Năm ablation | Phần 5 |
| Loss, lambda, gradient | Phần 6 |
| Epoch, checkpoint, seed | Phần 7 |
| Cách hiểu từng metric | Phần 8 |
| Kết quả mới nhất và cách diễn giải | Phần 9 |
| Điểm yếu và câu hỏi phản biện | Phần 10–12 |
