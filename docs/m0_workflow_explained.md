# M0 hoạt động như thế nào? — Luồng dữ liệu và vai trò từng file

Tài liệu giải thích code M0 hiện tại theo từng bước, dành cho việc đọc và học code. Các điểm số minh họa bên dưới là giả định, không phải kết quả thực nghiệm. Xem [hướng dẫn chạy M0](m0.md) để biết các lệnh và trạng thái huấn luyện.

## 1. M0 cần giải quyết việc gì?

Đầu vào là một **claim** (phát biểu cần kiểm chứng) và **context** (văn bản nguồn đi kèm). M0 phải dự đoán một trong ba nhãn:

| Nhãn | Ý nghĩa |
|---|---|
| `SUPPORTED` | Bằng chứng hỗ trợ claim |
| `REFUTED` | Bằng chứng bác bỏ claim |
| `NOT_ENOUGH_INFO` | Chưa đủ thông tin để kết luận |

```text
Claim + context
    → chọn những câu liên quan trong context bằng BM25
    → PhoBERT kiểm chứng claim với từng câu được chọn
    → tổng hợp dự đoán của các cặp thành dự đoán cho claim
```

BM25 chỉ chọn câu theo điểm đối chiếu token. PhoBERT mới là thành phần học dự đoán ba nhãn. M0 không có reranker và không sinh câu trả lời tự do bằng LLM.

## 2. Sơ đồ tổng thể

```text
Dữ liệu ViFactCheck nguồn
    → vifactcheck.py: download → normalize → validate → publish
    → record chuẩn: id, claim, context, evidence, label, split, ...
                       │
                       ├── claim + context
                       │       → sentences.py: tách context thành câu
                       │       → tokenizer.py: tạo token BM25 cho câu và claim
                       │       → context_bm25.py: xếp hạng trong context, lưu top-20
                       │       → prepare_m0.py: ghi file inputs không chứa gold/label
                       │
                       └── evidence + label
                               → evidence_alignment.py: ánh xạ gold vào câu khi phù hợp
                               → prepare_m0.py: ghi targets riêng

Inputs
    → M0Dataset: lấy tối đa 5 câu đầu theo thứ hạng BM25
    → M0Collator: ghép claim với từng câu
    → VnCoreNLP: tách từ tiếng Việt
    → tokenizer PhoBERT: chuyển thành token IDs
    → M0Verifier: dự đoán từng cặp
    → uniform_log_pool: trung bình xác suất theo claim
                       │
                       ├── khi train: ghép nhãn claim → loss → cập nhật verifier
                       └── khi predict: argmax → nhãn dự đoán

Dự đoán + targets
    → evaluate_m0.py: đánh giá nhãn và chất lượng chọn bằng chứng
```

Gold evidence không được đưa vào thuật toán chọn câu. Khi chuẩn bị dữ liệu có hai nhánh riêng: một nhánh tạo input, một nhánh tạo thông tin giám sát/đánh giá.

## 3. Bước 1 — Chuẩn hóa ViFactCheck

**File:** `src/factcheck/data/schemas.py` và `src/factcheck/data/vifactcheck.py`.

```text
Dataset nguồn
    → download_dataset(): tải các split nguồn
    → normalize_dataset(): chuyển từng dòng về DatasetRecord
    → validate_dataset(): kiểm tra cấu trúc, ID và split
    → publish_splits(): ghi dữ liệu processed và manifest
```

Ví dụ các trường chính của một record đã chuẩn hóa:

```python
{
    "id": "example:train:1",
    "claim": "Hà Nội là thủ đô Việt Nam.",
    "context": "Hà Nội là thủ đô Việt Nam. Thành phố có nhiều hồ. Du lịch phát triển.",
    "evidence": [
        {"evidence_id": "example:train:1:e1", "text": "Hà Nội là thủ đô Việt Nam."}
    ],
    "label": "SUPPORTED",
    "dataset": "vifactcheck",
    "split": "train"
}
```

`context` là văn bản để hệ thống tìm câu; `evidence` là chú thích gold dùng để đối chiếu. Hai trường này có vai trò khác nhau.

Tên file nguồn `dev.jsonl` được xuất thành `validation.jsonl`. Hàm publish không chia ngẫu nhiên lại dữ liệu.

## 4. Bước 2 — Tách context thành câu

**File:** `src/factcheck/data/sentences.py`.

```text
Context nguyên văn
    → context_id(): tạo ID từ SHA-256 của nội dung
    → split_sentences(): tìm ranh giới theo dấu câu, dòng trống và quy tắc ngoại lệ
    → tạo danh sách Sentence
        → sentence_id: ID câu
        → context_id: câu thuộc context nào
        → index: thứ tự trong context
        → text: nội dung nguyên văn
        → start/end: vị trí ký tự trong context
```

Ví dụ:

```text
"Hà Nội là thủ đô Việt Nam. Thành phố có nhiều hồ. Du lịch phát triển."
    → câu 0: "Hà Nội là thủ đô Việt Nam."
    → câu 1: "Thành phố có nhiều hồ."
    → câu 2: "Du lịch phát triển."
```

Luôn có quan hệ `context[sentence.start:sentence.end] == sentence.text`. Vị trí `end` không nằm trong đoạn được lấy.

Bộ tách câu hiện dùng quy tắc, hạn chế tách nhầm tại `TS.`, `3.14`, `01.10.2026`, dấu ba chấm và dấu ngoặc đóng. Nó chưa bảo đảm mọi ranh giới ngôn ngữ đều đúng; pilot cần được rà soát.

## 5. Bước 3 — Tạo token cho BM25 và xếp hạng

**File:** `src/factcheck/retrieval/tokenizer.py` và `src/factcheck/retrieval/context_bm25.py`.

```text
Context
    → sentences.py tách thành câu
    → tokenize_vietnamese() chuyển mỗi câu thành token
    → ContextBM25Retriever.build() xây index riêng cho context

Claim
    → tokenize_vietnamese() chuyển thành token
    → ContextBM25Retriever.retrieve() tính điểm với các câu trong index
    → sắp xếp điểm giảm dần; hòa điểm theo thứ tự câu
    → lưu tối đa 20 câu điểm cao nhất
```

Ví dụ token BM25:

```text
"Hà Nội là thủ đô Việt Nam!"
    → chuẩn hóa Unicode NFKC
    → chuyển chữ thường
    → lấy các cụm chữ/số, bỏ dấu câu và dấu gạch dưới
    → ["hà", "nội", "là", "thủ", "đô", "việt", "nam"]
```

Cả claim và câu dùng cùng tokenizer. BM25 dựa trên sự xuất hiện của token, độ phổ biến token trong các câu và độ dài câu để tính điểm. Thống kê này được tính trong **context của claim**, không trên một corpus gộp mọi bài báo.

```text
Claim A → index của context A → các câu của context A
Claim B → index của context B → các câu của context B
```

Các claim có context giống hệt có thể dùng chung index trong lần chuẩn bị. Điểm BM25 không phải xác suất claim đúng. Điểm bằng 0 vẫn được xếp hạng; khi hòa điểm, câu đứng trước trong context được ưu tiên.

BM25 tokenizer có thể tách `3.14` thành `['3', '14']`. Chỉ biểu diễn dùng cho retrieval bị biến đổi; raw text lưu trong ứng viên vẫn giữ `3.14` để verifier đọc.

## 6. Bước 4 — Căn chỉnh gold ở nhánh riêng

**File:** `src/factcheck/data/evidence_alignment.py`.

```text
Gold evidence + context + danh sách câu
    → chuẩn hóa nhẹ khoảng trắng, có ánh xạ về offset gốc
    → tìm vị trí xuất hiện của evidence trong context
    → tìm các câu chồng lên đoạn evidence
    → gold_sentence_ids + alignment_status
```

Ví dụ evidence là câu đầu trong context thì gold sentence ID tương ứng câu 0. Nếu evidence kéo dài qua hai câu, gold có thể gồm hai sentence IDs.

Các trạng thái chính:

| Trạng thái | Ý nghĩa |
|---|---|
| `aligned` | Mọi snippet đều khớp duy nhất |
| `unmatched` | Không tìm được vị trí khớp |
| `ambiguous` | Có nhiều vị trí khớp |
| `partial` | Các snippet có trạng thái khớp khác nhau |
| `no_gold` | Không có evidence để căn chỉnh |
| `not_applicable` | Nhánh chuẩn bị đánh dấu cho NEI |

M0 sử dụng kết quả này cho đánh giá retrieval, không dùng gold sentence IDs để chọn câu hoặc tính loss verifier. Loss verifier sử dụng nhãn claim.

Hàm `normalize_for_evidence_match()` trong `retrieval/tokenizer.py` thuộc phần đánh giá chunk trước đây, không phải hàm alignment của M0 hiện tại.

## 7. Bước 5 — Ghi dữ liệu chuẩn bị thành hai phần

**File:** `src/factcheck/data/prepare_m0.py`.

```text
prepare_dataset()
    → kiểm tra manifest/checksum và các record nguồn
    → tách câu, tạo index và lấy BM25 top-20
    → tính alignment ở nhánh gold riêng
    → ghi inputs
    → ghi targets
    → ghi audit, pilot và manifest đầu ra
```

```text
{split}.inputs.jsonl
    → id
    → claim
    → context_id
    → sentence_count
    → candidates: sentence_id, index, text, start, end, score

{split}.targets.jsonl
    → id
    → label
    → gold_sentence_ids
    → alignment_status
    → chi tiết alignment
```

Inputs có văn bản các câu ứng viên, không chứa full context, nhãn hoặc gold evidence. Hai file được ghép bằng `id` khi cần; không dựa vào việc chúng tình cờ cùng thứ tự dòng.

Train/validation được chuẩn bị mặc định. Test cần được yêu cầu riêng theo giao thức đánh giá cuối.

## 8. Bước 6 — Dataset chọn top-5 cho verifier

**File:** `src/factcheck/training/m0_dataset.py`, lớp `M0Dataset`.

```text
File inputs chứa tối đa 20 ứng viên mỗi claim
    → đọc và kiểm tra ID, schema, score, thứ tự BM25
    → lấy candidates[:5]
    → nếu context có ít hơn 5 câu: lấy toàn bộ câu thực có
    → khi train: ghép label từ targets bằng claim ID
    → trả một mẫu cho DataLoader
```

Phân biệt hai con số:

```text
Top-20 được lưu → phục vụ retrieval metrics và ứng viên cho các bước phát triển sau
Top-5 được dùng → đầu vào verifier của M0
```

Ví dụ một context có 3 câu thì M0 dùng cả 3. Khi tổng hợp sẽ chia cho 3, không chia cho 5. Với ví dụ context 3 câu ở trên, BM25 chỉ thay đổi thứ tự; nếu muốn thấy ảnh hưởng của việc loại câu, cần context có hơn 5 câu.

## 9. Bước 7 — Tách từ, tạo token IDs và batch

**File:** `src/factcheck/data/word_segmentation.py` và lớp `M0Collator` trong `src/factcheck/training/m0_dataset.py`.

**Phân biệt Dataset, DataLoader và Collator**

Cả ba cùng tham gia chuẩn bị dữ liệu, nhưng có trách nhiệm khác nhau:

| Thành phần | Đại diện/xử lý gì? | Đầu ra |
|---|---|---|
| `M0Dataset` | Toàn bộ tập mẫu; mỗi mẫu là một claim và tối đa 5 câu BM25 chọn | `dataset[i]` trả mẫu thứ `i`, kèm nhãn claim nếu có targets |
| `DataLoader` | Lấy các mẫu từ Dataset, có thể xáo trộn thứ tự khi train, rồi gom theo `batch_size` | Nhóm mẫu được chuyển cho Collator; DataLoader trả batch đã được Collator xử lý |
| `M0Collator` | Một nhóm claim trong một batch tại thời điểm được gọi | Các cặp claim–câu đã tokenize, padding, chuyển thành tensor và metadata |

Dataset không chỉ chứa một claim; **mỗi phần tử của Dataset** mới là một claim. Collator không xử lý toàn bộ dataset một lần; nó nhận những mẫu của batch hiện tại. Các câu BM25 chọn là bằng chứng ứng viên, chưa chắc trùng gold evidence.

```text
Dataset: chứa 5.062 claim train
    ↓ DataLoader lấy batch_size=2

Nhóm mẫu của batch hiện tại:
    Claim A + 5 câu
    Claim B + 3 câu (context chỉ có 3 câu)
    ↓ Collator

8 cặp claim–câu:
    A1, A2, A3, A4, A5, B1, B2, B3

bag_index:
    [0, 0, 0, 0, 0, 1, 1, 1]
    ↓ tokenize, padding và tạo tensor

Batch đầu vào mô hình
    → PhoBERT dự đoán 8 cặp
    → pooling gom về 2 dự đoán claim
```

Ví dụ trên minh họa một batch, không mô tả hai record cụ thể trong dữ liệu. Luồng trách nhiệm là: **Dataset cung cấp từng mẫu → DataLoader gom mẫu → Collator biến nhóm mẫu thành batch đầu vào mô hình.** Sau đó, DataLoader tiếp tục lấy nhóm mẫu tiếp theo.

```text
Một claim và n câu được chọn
    → tạo n cặp:
        (claim, câu 1)
        (claim, câu 2)
        ...
        (claim, câu n)

Văn bản claim / văn bản từng câu
    → VnCoreNLPSegmenter: tách từ tiếng Việt
    → tokenizer PhoBERT: tạo subword token IDs
    → giới hạn claim tối đa 96 token
    → cấp ngân sách token còn lại cho từng câu
    → thêm special tokens cho cặp
    → padding các cặp trong batch
    → tensor đầu vào mô hình
```

VnCoreNLP được gọi qua `VnCoreNLPBridge.java`. Ví dụ biểu diễn đã tách từ có thể có dạng:

```text
"Hà Nội là thủ đô Việt Nam."
    → "Hà_Nội là thủ_đô Việt_Nam ."
    → tokenizer PhoBERT chuyển thành các ID số
```

Không đưa danh sách token BM25 vào PhoBERT. Verifier dùng raw text của câu được chọn rồi tách từ/tokenize theo quy trình riêng.

| Thành phần | Đầu ra | Dùng cho |
|---|---|---|
| `sentences.py` | Các câu có offset | Đơn vị truy xuất |
| `retrieval/tokenizer.py` | List token chữ/số | BM25 |
| VnCoreNLP | Văn bản đã tách từ | Tiền xử lý PhoBERT |
| Tokenizer PhoBERT | Các token ID | Đầu vào mô hình |

Mỗi cặp có tối đa 256 subword token, tính cả special tokens. Claim cap 96 không có nghĩa mọi claim đều dùng 96 token; claim ngắn để lại nhiều ngân sách hơn cho câu.

```text
Ví dụ một batch có 2 claim:
    Claim A có 2 câu → cặp A1, A2
    Claim B có 3 câu → cặp B1, B2, B3

Sau khi flatten:
    input_ids      → 5 hàng token IDs
    attention_mask → 5 hàng, phân biệt token thật và token padding
    bag_index      → [0, 0, 1, 1, 1]
    num_claims     → 2
    labels         → 2 nhãn claim, chỉ có khi train
```

`bag_index` cho biết cặp nào thuộc claim nào để gom lại sau khi dự đoán. Không có câu giả được thêm để đủ 5; padding ở đây là padding độ dài token của các cặp thật.

### 9.1. Special tokens — token đặc biệt

Special tokens đánh dấu cấu trúc đầu vào hoặc phục vụ thao tác như padding. Chúng không phải các từ trong văn bản gốc, nhưng cũng có ID số trong bộ từ vựng của tokenizer.

Với cặp claim–câu, tokenizer PhoBERT tạo cấu trúc:

```text
<s> token_của_claim </s> </s> token_của_câu </s>
```

| Token | Vai trò trong đầu vào M0 |
|---|---|
| `<s>` | Đánh dấu bắt đầu chuỗi |
| `</s>` | Đánh dấu kết thúc/phân cách các phần của cặp |
| `<pad>` | Bù độ dài cho cặp ngắn hơn khi tạo batch |

Trong `M0Collator`, `build_inputs_with_special_tokens()` thêm các dấu mốc cho cặp. Code hỏi tokenizer số special tokens cần dùng để trừ khỏi ngân sách 256; không tự nối những chuỗi ký hiệu này vào văn bản rồi đoán ID.

Ví dụ claim dùng 30 subword token, cấu trúc cặp dùng 4 special tokens thì câu còn tối đa `256 - 30 - 4 = 222` token. Độ dài thực tế vẫn có thể ngắn hơn 256.

### 9.2. Tensor — cấu trúc dữ liệu số của mô hình

Tensor có thể hình dung là một mảng số với một hoặc nhiều chiều. PyTorch dùng tensor để lưu đầu vào, tham số mô hình và kết quả tính toán.

```python
import torch

x = torch.tensor([
    [10, 20, 30],
    [40, 50, 60],
])

print(x.shape)
# torch.Size([2, 3]): 2 hàng, mỗi hàng 3 phần tử
```

Các số trên chỉ minh họa một tensor, không phải token IDs thật của một câu PhoBERT.

Tensor có `shape` (kích thước các chiều), `dtype` (kiểu dữ liệu) và `device` (CPU/GPU...). Trong M0, `input_ids` chứa số nguyên; logits/xác suất chứa số thực. Tensor ID đầu vào không phải thứ được optimizer học: gradient dùng để cập nhật các tham số của verifier.

### 9.3. Padding — bù độ dài token

Hai cặp có thể có số token khác nhau. Để tạo một tensor hai chiều có các hàng bằng nhau, tokenizer thêm `<pad>` vào cặp ngắn hơn.

```text
Trước padding:
    Cặp A1: [t1, t2, t3, t4, t5, t6]
    Cặp A2: [u1, u2, u3, u4]

Sau padding:
    Cặp A1: [t1, t2, t3, t4, t5, t6]
    Cặp A2: [u1, u2, u3, u4, PAD, PAD]
```

`t1`, `u1`, `PAD` ở đây là ký hiệu minh họa; dữ liệu đưa vào model là ID số. Padding không thêm bằng chứng mới và khác với truncation: padding bù cặp ngắn, truncation cắt phần vượt ngân sách của cặp dài.

Code gọi `tokenizer.pad(..., padding=True, return_tensors="pt")`: bù đến độ dài cặp dài nhất trong batch và trả tensor PyTorch. Không phải batch nào cũng dài 256. M0 không thêm câu giả để mỗi claim đủ 5 câu.

### 9.4. attention_mask — phân biệt token thật và padding

`attention_mask` có cùng kích thước với `input_ids`. Theo cách tạo batch hiện tại, `1` chỉ vị trí token thật và `0` chỉ padding.

```text
input_ids:
    A1: [t1, t2, t3, t4, t5, t6]
    A2: [u1, u2, u3, u4, PAD, PAD]

attention_mask:
    A1: [ 1,  1,  1,  1,  1,  1]
    A2: [ 1,  1,  1,  1,   0,   0]
```

Mask giúp mô hình không attention đến các vị trí padding như nội dung đầu vào thật. Các special tokens đánh dấu cấu trúc như `<s>` và `</s>` vẫn có mask bằng 1; `<pad>` có mask bằng 0.

Mask không phải điểm quan trọng của token/câu, không đánh dấu đúng–sai và không phải trọng số pooling của M0.

### 9.5. Batch và batch tensor

Batch là một nhóm mẫu được gom để xử lý trong một lượt. Trong M0, DataLoader gom theo **claim**, sau đó collator tạo nhiều cặp claim–câu từ nhóm claim đó.

```text
Một batch gồm 2 claim:
    Claim A có 2 câu → A1, A2
    Claim B có 3 câu → B1, B2, B3

PhoBERT nhận 5 cặp, dù DataLoader chỉ gom 2 claim.
```

Giả sử cặp dài nhất trong nhóm có 40 token sau khi thêm special tokens:

```text
input_ids.shape      = [5, 40]
attention_mask.shape = [5, 40]
                       │   │
                       │   └── chiều dài sau padding
                       └────── số cặp claim–câu
```

“Batch tensor” là tensor chứa dữ liệu của cả nhóm. Trong code, toàn bộ batch là một dictionary chứa nhiều tensor cùng metadata (`ids`, `sentence_ids`, thống kê cắt token...), không phải mọi thành phần đều nằm trong một tensor duy nhất.

`batch_size` cấu hình là số claim mỗi microbatch. `gradient_accumulation_steps` cho phép tích lũy gradient qua nhiều microbatch trước một lần cập nhật tham số; đó là khái niệm riêng với padding và số cặp.

### 9.6. bag_index — mỗi cặp thuộc claim nào?

Sau khi flatten các cặp, cần giữ thông tin nhóm để tổng hợp dự đoán về mức claim.

```text
Thứ tự hàng:  A1  A2  B1  B2  B3
bag_index:    0   0   1   1   1
```

`0` là claim đầu tiên trong batch, `1` là claim thứ hai. Đây không phải nhãn, thứ hạng BM25 hoặc ID cố định trong dataset; chỉ số nhóm được tạo lại cho từng batch.

```text
PhoBERT trả dự đoán của 5 cặp
    → bag_index == 0: gom A1, A2 → trung bình xác suất → claim A
    → bag_index == 1: gom B1, B2, B3 → trung bình xác suất → claim B
```

`bag_index` được phần pooling trong `M0Verifier` sử dụng; nó không được truyền vào encoder PhoBERT như một token. Mỗi claim chia cho số câu thật của chính nó: A chia 2, B chia 3.

### 9.7. Đối chiếu kích thước trong một ví dụ chung

Với 2 claim, tổng 5 cặp và chiều dài sau padding là 40:

| Thành phần | Shape hoặc giá trị | Ý nghĩa |
|---|---|---|
| `input_ids` | Shape `[5, 40]` | ID token của 5 cặp |
| `attention_mask` | Shape `[5, 40]` | Token thật/padding của từng cặp |
| `bag_index` | Giá trị `[0, 0, 1, 1, 1]`, shape `[5]` | Nhóm claim của mỗi cặp |
| `num_claims` | Số nguyên `2` | Số claim trong batch |
| `labels` nếu có targets | Shape `[2]`, ví dụ giá trị `[0, 1]` | Nhãn S/R/NEI cho từng claim |
| `pair_logits` | Shape `[5, 3]` | Ba điểm nhãn của mỗi cặp |
| Xác suất sau pooling | Shape `[2, 3]` | Ba xác suất cho mỗi claim |

Đọc `shape [2]` là “tensor có 2 phần tử”, không có nghĩa giá trị nhãn là 2. Nhãn thật chỉ được dùng tính loss theo claim; `attention_mask` và `bag_index` đều không chứa đáp án.

## 10. Bước 8 — PhoBERT dự đoán từng cặp

**File:** `src/factcheck/models/verifier.py`, lớp `M0Verifier`.

```text
input_ids + attention_mask (+ token_type_ids nếu tokenizer cung cấp)
    → pair_model: PhoBERT + head phân loại
    → mỗi cặp sinh 3 logits
    → log_softmax theo chiều nhãn
    → log xác suất của từng cặp, dùng để pooling ổn định
```

Thứ tự nhãn được cố định:

```text
0 → SUPPORTED
1 → REFUTED
2 → NOT_ENOUGH_INFO
```

Logits là điểm thô, chưa phải xác suất. Sau softmax, ba xác suất của mỗi cặp cộng lại bằng 1, trong giới hạn sai số số học.

### 10.1. Logits là gì?

Logits là các điểm số thô mà mô hình tạo ra cho từng nhãn, trước khi chuyển thành xác suất. Trong M0, mỗi cặp claim–câu có ba logits theo thứ tự SUPPORTED, REFUTED, NOT_ENOUGH_INFO.

```text
Nhãn:     SUPPORTED   REFUTED   NOT_ENOUGH_INFO
Logits:       2          1            0
```

Ví dụ này cho biết mô hình ưu tiên SUPPORTED hơn hai nhãn còn lại. Giá trị `2` không có nghĩa là xác suất 200%.

- Logits có thể âm hoặc dương, không bị giới hạn trong khoảng 0–1.
- Tổng logits không cần bằng 1.
- Logit lớn hơn được softmax chuyển thành xác suất lớn hơn.
- Khoảng cách tương đối giữa logits quyết định mức chênh lệch xác suất. Cộng cùng một số vào mọi logit không làm thay đổi softmax: `[2, 1, 0]` và `[12, 11, 10]` cho cùng phân phối xác suất.

Trong `M0Verifier.forward()`:

```python
pair_output = self.pair_model(**pair_inputs)
pair_logits = pair_output.logits
```

`pair_logits` có shape `[số_cặp, 3]`: mỗi hàng chứa ba điểm thô cho một cặp claim–câu, chưa phải kết quả đã tổng hợp cho claim.

### 10.2. Softmax là gì?

Softmax chuyển một danh sách logits thành một phân phối xác suất: mỗi giá trị nằm trong khoảng 0–1 và tổng bằng 1, trong giới hạn sai số số học.

Công thức cho nhãn thứ i:

```text
p_i = exp(z_i) / Σ_j exp(z_j)
```

`z_i` là logit của nhãn i; `exp(z_i)` nghĩa là e mũ z_i, với e xấp xỉ 2,71828. Tổng ở mẫu số chạy qua cả ba nhãn của cùng một cặp, không chạy qua các câu khác nhau.

Ví dụ với logits `[2, 1, 0]`:

```text
Bước 1 — Lấy hàm mũ:
    exp(2) ≈ 7,389
    exp(1) ≈ 2,718
    exp(0) = 1

Bước 2 — Tính tổng:
    7,389 + 2,718 + 1 ≈ 11,107

Bước 3 — Chia từng giá trị cho tổng:
    SUPPORTED:       7,389 / 11,107 ≈ 0,665
    REFUTED:         2,718 / 11,107 ≈ 0,245
    NOT_ENOUGH_INFO:  1    / 11,107 ≈ 0,090
```

```text
Logits:   [2,     1,     0]
                 ↓ softmax
Xác suất: [0,665; 0,245; 0,090]
```

Nếu ba logits bằng nhau, mỗi nhãn nhận xác suất 1/3. Softmax tạo ra xác suất mô hình gán cho nhãn; giá trị 0,665 không tự bảo đảm mô hình đúng 66,5% trong thực tế.

### 10.3. Vì sao code dùng log_softmax?

Trong `uniform_log_pool()`:

```python
pair_log_probs = F.log_softmax(logits, dim=-1)
```

`log_softmax` tính log tự nhiên của xác suất softmax bằng cách ổn định về số học. Ý nghĩa toán học là `log(softmax(logits))`, nhưng nên dùng hàm chuyên dụng thay vì tự tính softmax rồi lấy log, đặc biệt khi xác suất rất nhỏ.

`dim=-1` chỉ chiều cuối của tensor. Với shape `[số_cặp, 3]`, đó là chiều ba nhãn; mỗi cặp được chuẩn hóa riêng.

```text
Xác suất softmax: [0,665; 0,245; 0,090]
                         ↓ log tự nhiên
Log xác suất:    [-0,408; -1,408; -2,408] (xấp xỉ)
```

Log xác suất không cộng lại bằng 1 và có thể âm; đó không phải lỗi. Lấy `exp()` sẽ chuyển log xác suất trở lại xác suất.

Luồng thực tế trong M0:

```text
Logits của từng cặp
    → log_softmax: log xác suất từng nhãn của từng cặp
    → bag_index: xác định những cặp thuộc cùng claim
    → logsumexp và trừ log(số câu): log của trung bình xác suất
    → exp(): xác suất cuối cho claim
```

Ý nghĩa vẫn là **softmax riêng từng cặp rồi trung bình xác suất theo claim**. Code không lấy trung bình logits trước softmax, cũng không lấy trung bình trực tiếp các log xác suất.

## 11. Bước 9 — Tổng hợp dự đoán theo claim

**File:** `src/factcheck/models/verifier.py`, hàm `uniform_log_pool()`.

```text
Xác suất từng cặp + bag_index
    → gom các cặp thuộc cùng claim
    → cộng xác suất tương ứng của từng nhãn
    → chia cho số câu thực có của claim
    → lấy nhãn có xác suất lớn nhất
```

Ví dụ minh họa một claim có 3 câu:

| Cặp | SUPPORTED | REFUTED | NEI |
|---|---:|---:|---:|
| Claim–câu 1 | 0,80 | 0,10 | 0,10 |
| Claim–câu 2 | 0,50 | 0,20 | 0,30 |
| Claim–câu 3 | 0,20 | 0,10 | 0,70 |
| Trung bình | **0,50** | **0,1333** | **0,3667** |

Nhãn cuối là `SUPPORTED`. Đây không phải bỏ phiếu theo nhãn từng câu, cũng không phải lấy trung bình logits trước softmax.

Code dùng `log_softmax` và `logsumexp` để tính log của trung bình xác suất một cách ổn định. Công thức có cùng ý nghĩa:

```text
P_claim(y) = (P_pair1(y) + ... + P_pair_n(y)) / n
```

Điểm BM25 chỉ dùng chọn câu, không dùng làm trọng số pooling. Trong M0, mỗi câu thực có trọng số `1/n`.

## 12. Bước 10 — Khi huấn luyện: tính loss và cập nhật verifier

**File:** `src/factcheck/training/train_m0.py`; hàm hỗ trợ trong `src/factcheck/training/m0_runtime.py`.

```text
train_m0.main()
    → đọc m0.yaml qua M0Config
    → đặt seed, chọn device, kiểm tra dữ liệu chuẩn bị
    → nạp PhoBERT/tokenizer và VnCoreNLP
    → tạo Dataset, Collator và DataLoader
    → tạo AdamW optimizer và learning-rate scheduler

Mỗi batch train
    → forward các cặp
    → pooling thành xác suất cho từng claim
    → lấy label thật của từng claim
    → NLL loss ở mức claim
    → backward: tính gradient
    → tích lũy gradient theo số claim thật
    → đủ nhóm accumulation hoặc hết epoch:
        chuẩn hóa gradient → clip → optimizer.step() → scheduler.step()
```

Với ví dụ phía trên, nếu nhãn thật là SUPPORTED:

```text
Loss của claim = -ln(0,50) ≈ 0,6931
```

Loss batch lấy trung bình theo claim, không theo tổng số câu. Không mặc nhiên gán nhãn SUPPORTED cho cả ba câu rồi tính ba loss độc lập.

```text
Cuối mỗi epoch
    → model.eval(), dự đoán validation không cập nhật tham số
    → tính Macro-F1 validation
    → nếu tốt hơn mốc trước: lưu model/tokenizer tốt nhất
    → nếu bằng điểm: giữ mốc sớm hơn
    → ghi loss, metric, thời gian và thống kê cắt token
```

Cấu hình đầy đủ hiện là 3+2 epoch với optimizer/scheduler liên tục, không reset ở ranh giới epoch 3. Chế độ `--smoke` chỉ kiểm tra kỹ thuật bằng một tập nhỏ.

BM25 không được cập nhật bằng gradient; chỉ verifier được huấn luyện. NEI và các claim không có gold alignment vẫn tham gia học verdict.

## 13. Bước 11 — Khi dự đoán: nạp checkpoint và xuất kết quả

**File:** `src/factcheck/inference/predict_m0.py` và `m0_runtime.predict_batches()`.

```text
Checkpoint đã train + file inputs đã chuẩn bị
    → đọc config/seed của run
    → nạp model và tokenizer đã lưu
    → M0Dataset chọn top-5
    → Collator tạo các cặp và tensor
    → model.eval() + inference_mode()
    → dự đoán từng cặp và pooling
    → argmax thành verdict
    → ghi id, sentence_ids, probabilities, label, seed/model
    → ghi metadata/checksum và thời gian
```

**CLI predict hiện nhận inputs đã chuẩn bị**, không trực tiếp nhận context thô. Bước BM25 đã chạy trước trong prepare. Predict không mở file targets và không cần nhãn thật để đưa ra dự đoán.

Các tác vụ hỗ trợ trong `m0_runtime.py` gồm chọn CPU/CUDA/MPS, nạp model, chuyển tensor sang device, cố định seed, ghi JSON và kiểm tra checksum.

## 14. Bước 12 — Đánh giá verdict và retrieval riêng

**File:** `src/factcheck/evaluation/evaluate_m0.py`.

```text
Dự đoán + nhãn thật, ghép theo claim ID
    → verdict_metrics()
    → accuracy, Macro-F1, precision/recall/F1 từng lớp, confusion matrix
    → tính trên mọi claim hợp lệ

Danh sách BM25 đã xếp hạng + gold sentence IDs
    → retrieval_metrics()
    → chỉ lấy SUPPORTED/REFUTED có alignment hợp lệ
    → Hit@5/20, macro sentence Recall@5/20, MRR@5/20, complete-gold
    → báo số mẫu và alignment coverage
```

- **Hit@k:** trong top-k có ít nhất một câu gold hay không?
- **Recall@k:** lấy được bao nhiêu phần trong tập câu gold? Tính cho từng claim rồi trung bình.
- **MRR@k:** gold đầu tiên xuất hiện ở hạng nào? Hạng càng cao thì điểm càng lớn; không có trong top-k thì 0.
- **Complete-gold:** top-k có chứa toàn bộ tập câu gold không?
- **Coverage:** bao nhiêu claim S/R có thể đánh giá bằng gold alignment hợp lệ?

NEI không được gán một recall evidence giả định. Claim chưa khớp gold vẫn có mặt trong verdict metrics. Chọn được câu gold chưa bảo đảm verifier dự đoán đúng; ngược lại, dự đoán đúng cũng chưa chứng minh câu được chọn trùng gold.

## 15. Lệnh terminal đi vào code như thế nào?

```text
bash scripts/setup_m0.sh
    → run_m0.sh setup
    → data/m0_assets.py
    → tải tài nguyên PhoBERT và VnCoreNLP đã pin phiên bản

bash scripts/run_m0.sh prepare
    → data/prepare_m0.py
    → tạo dữ liệu inputs/targets, audit và manifest

bash scripts/run_m0.sh train --seed 42
    → training/train_m0.py
    → huấn luyện và chọn checkpoint theo validation

bash scripts/run_m0.sh predict [các tham số]
    → inference/predict_m0.py
    → nạp checkpoint, dự đoán trên inputs đã chuẩn bị

bash scripts/run_m0.sh evaluate [các tham số]
    → evaluation/evaluate_m0.py
    → đối chiếu dự đoán và gold để tính metric
```

Cấu hình nằm trong `configs/experiments/m0.yaml`, được đọc và kiểm tra bởi `src/factcheck/m0_config.py`. Hướng dẫn lệnh đầy đủ nằm trong [m0.md](m0.md).

## 16. Những điểm cần nhớ khi đọc code

1. Đơn vị dữ liệu cuối cùng là một **claim**, dù một claim tạo nhiều cặp model.
2. Đơn vị retrieval là **câu**, với phạm vi tìm kiếm là context của claim.
3. Tokenizer BM25 và tokenizer PhoBERT phục vụ hai giai đoạn khác nhau.
4. Gold evidence phục vụ đánh giá retrieval M0; nhãn claim phục vụ loss verifier.
5. M0 dùng **trung bình xác suất với trọng số đều**, không có reranking hoặc trọng số theo BM25.
6. Dữ liệu top-20 có sẵn không có nghĩa verifier đọc 20 câu; nó chỉ đọc tối đa 5.
7. Chạy pipeline thành công hoặc chạy smoke không đồng nghĩa mô hình đã được huấn luyện đủ để báo cáo kết quả nghiên cứu.

Ở thời điểm viết, M0 đã có code và kiểm tra smoke/model reload; chưa có kết quả huấn luyện đầy đủ 5 epoch/3 seed. Learning rate còn tạm thời chờ lựa chọn chung bằng M1; pilot tách câu và giới hạn alignment vẫn cần được ghi nhận trước thực nghiệm chính thức.
