# Audit logic nguồn ViFactCheck v2

Ngày kiểm tra: 11/11/2026. Phạm vi: source tại `release/v2`, Proposal v2, dữ liệu raw trong repo và kết quả listwise seed 42 đã lưu ở commit `baa0bdc`. Chưa kiểm tra checkpoint/log của run ba seed v2 đang chạy.

## Chuẩn đối chiếu

- [ViFactCheck](https://arxiv.org/abs/2412.15308) cung cấp dataset và bài toán; [PhoBERT](https://aclanthology.org/2020.findings-emnlp.92/) cung cấp backbone. M0 không phải baseline PhoBERT full-context hay gold-evidence của paper ViFactCheck: M0 dùng BM25 top 5 và trung bình xác suất theo câu.
- [M-ReRank](https://aclanthology.org/2024.findings-emnlp.428.pdf) là nguồn ý tưởng cho pipeline nhiều giai đoạn và list-aware reranking. Paper dùng thêm dense retrieval, HybRank/HLATR, dữ liệu câu và bảng, cùng objective contrastive/margin ranking. Source hiện tại dùng BM25 trong từng bài, PhoBERT với một candidate-attention layer và balanced BCE. Đây là biến thể của nhóm, không phải bản tái hiện M-ReRank.
- [RAV](https://aclanthology.org/2024.findings-acl.551.pdf) là nguồn gần nhất cho huấn luyện chung và gradient verdict quay về ranker. Ablation gốc của RAV bỏ retriever hoặc ranker; A1/A2 không lấy nguyên từ bảng ablation của paper.
- [FER](https://aclanthology.org/2023.findings-emnlp.422.pdf) dùng feedback dựa trên chênh lệch utility giữa evidence truy xuất và gold. FER còn nêu rằng retriever và verifier của họ được tối ưu độc lập. Source của nhóm không tính utility divergence; chỉ lấy cảm hứng từ câu hỏi evidence có hữu ích cho verdict hay không.
- `reports/Proposal_v2.docx` là đặc tả trực tiếp của M0/M1/M2/A1/A2, bốn phép đối chứng, D0 và các giới hạn về top-k. Paper [BERT-based Vietnamese Fact Verification](https://arxiv.org/abs/2503.00356) và [SemViQA](https://aclanthology.org/2026.acl-industry.94/) là bối cảnh liên quan, không định nghĩa ablation của nhóm.

## Kết luận đã kiểm tra

**Khớp Proposal ở lõi năm cấu hình.** `data.py:14-20` tạo BM25 top 20 trong `Context`; `model.py:152-178` cho M0 lấy top 5 BM25, A1 gộp đều, M1 gộp có trọng số, M2 giữ gradient verdict qua trọng số, A2 chặn riêng đường đó bằng `detach()`. `model.py:95-104` dùng BCE cân bằng dương/âm và bỏ evidence loss khi NEI, alignment không hợp lệ hoặc không có cả dương lẫn âm. `training.py:271-304` huấn luyện reranker trên mẫu có supervision, sau đó dùng verdict loss và evidence loss ở continuation khi đủ điều kiện. `training.py:506-588` giữ khởi tạo chung và kiểm tra reranker đóng băng của M1/A1. Không thấy dấu hiệu gold evidence bị chèn vào top 20 hoặc top 5 khi train/predict.

**Đường gradient hoạt động như đặc tả.** Kiểm tra tensor nhỏ với PyTorch cho gradient verdict M2 trên ba score là `[-0.6116, +0.6116, 0]`: ứng viên không được chọn có gradient verdict bằng 0. A2 cho gradient verdict tới score là `None`, nhưng gradient evidence trên cả ba score khác 0. Candidate-attention ở chế độ eval giữ tính hoán vị tương đương, sai số lớn nhất `1.2e-7`. Đây là kiểm tra cấu phần, chưa thay cho chạy full model.

**Protocol v2 trên source phù hợp với tài liệu:** reranker chọn bằng dev MRR, M1/M2/A2 cùng nhánh từ warmup cuối epoch 3, checkpoint prediction chọn dev Macro-F1 tốt nhất trong năm epoch, hòa điểm ưu tiên epoch trước. `full.py:349-455` chọn LR và lambda bằng dev seed 42, hoàn tất train trước khi đọc test. Các test trong `tests/test_protocol_v2.py` bao phủ chọn epoch, resume và warmup winner; ở máy hiện tại không chạy được toàn bộ suite do thiếu `rank_bm25`.

## Phát hiện và ảnh hưởng

### F1 — Cao: retrieval metric chỉ đo trên phần gold căn chỉnh được

`text.py:128-160` chỉ chấp nhận `Evidence` khớp duy nhất như một đoạn liên tục sau chuẩn hóa. `data.py:91-97` loại mẫu không căn chỉnh được khỏi evidence supervision; `metrics.py:15-32` loại chúng khỏi retrieval metric. Trên raw test có 976 claim SUPPORTED/REFUTED: 539 evidence khớp duy nhất, 431 không xuất hiện nguyên đoạn, 6 xuất hiện nhiều lần. `reports/experiments/listwise-seed42/results.csv` xác nhận mẫu số retrieval là 539; Hit@5 của M1 là `0.9889` **trên 539 mẫu này**. Macro-F1 `0.7046` của M2 vẫn tính đủ 1.447 claim, không bị thay đổi bởi cách tính retrieval metric.

Ví dụ raw test `index=840` có evidence ghép ý từ hai câu trong context và lược dấu phẩy/cụm từ; `index=628` ghép hai đoạn ở các vị trí khác nhau. Vì vậy không thể mặc định 431 trường hợp là lỗi tách câu, cũng không thể gán nhãn bằng substring/fuzzy match mà chưa rà soát. Ở train, chỉ 1.873/3.409 claim SUPPORTED/REFUTED có evidence khớp duy nhất trước khi xét BM25 top 20. Điều này giới hạn tập dùng để học và đánh giá reranker; checkpoint reranker cũng được chọn bằng dev MRR trên phần căn chỉnh được. Nếu thay chính sách căn chỉnh thì phải tạo dữ liệu mới và train lại, giữ kết quả cũ là lịch sử.

**Hướng xử lý:** báo cáo đồng thời mẫu số S/R, số căn chỉnh hợp lệ và tỷ lệ coverage có điều kiện; lấy mẫu các ca không khớp để phân loại evidence ghép đoạn, diễn đạt lại, lỗi dữ liệu hoặc lỗi chuẩn hóa. Chỉ mở rộng supervision khi có quy tắc đối chiếu được kiểm chứng thủ công.

### F2 — Cao đối với mục tiêu nghiên cứu: D0 và phân tích mâu thuẫn chưa có

Proposal mô tả D0 dùng **toàn bộ câu gold**, kể cả khi quá 5, không train lại verifier; đồng thời H3 cần phân tích thủ công dạng mâu thuẫn. `full.py:246` và `full_reporting.py:176-180` ghi rõ D0 và manual contradiction annotation chưa chạy; không có nhánh D0 trong `model.py`, `training.py` hay CLI. Năm cấu hình chính vẫn có thể báo điểm, nhưng nghiên cứu chưa hoàn tất phần chẩn đoán và H3 theo Proposal.

**Hướng xử lý:** thêm runner D0 chỉ dùng claim S/R có alignment hợp lệ và checkpoint verifier đã chọn, gộp đều mọi câu gold, lưu số câu thực dùng; không dùng D0 để chọn checkpoint. Chuẩn bị hướng dẫn gán nhãn mâu thuẫn và mẫu kiểm tra thủ công trước khi phát biểu về H3.

### F3 — Trung bình: `alignment_coverage` dễ bị hiểu sai mẫu số

`metrics.py:15-17` cho `count` là số claim S/R có gold hợp lệ, nhưng `alignment_coverage` lấy số claim alignment `valid` chia **tất cả** claim, gồm NEI. Bản test seed 42 ghi `count=539` và `alignment_coverage=857/1447=59.23%`; coverage đúng cho tập S/R của retrieval là `539/976=55.23%`. Cả hai số đều tính đúng theo công thức riêng, nhưng đặt cạnh nhau mà không ghi mẫu số khiến người đọc tưởng cùng quần thể.

**Hướng xử lý:** xuất riêng `overall_alignment_coverage` và `sr_alignment_coverage`, ghi tử/mẫu số trong báo cáo. Giữ tên cũ như alias chỉ khi cần đọc artifact lịch sử; không âm thầm đổi nghĩa của số cũ.

### F4 — Trung bình: checkpoint M2/A2 có thể chưa chứa joint update

`training.py:244-259` và `training.py:324-345` cho phép warmup epoch 1–3 thắng dev Macro-F1 của M2/A2. `full.py:146-155` kiểm tra gradient và thay đổi encoder trên **terminal continuation state**, không nhất thiết trên checkpoint được chọn để dự đoán. Đây là chính sách v2 đã ghi trong `docs/protocol/v2.md`, không phải lỗi save/load. Tuy nhiên nếu warmup thắng, kết quả dự đoán được gắn nhãn M2/A2 không chứng minh tác dụng của update joint ở checkpoint đó.

**Hướng xử lý:** với từng seed, báo `selected_stage`, `selected_global_epoch` và cờ `selected_checkpoint_has_joint_updates`. Khi warmup thắng, diễn giải tương phản M2–M1/M2–A2 theo checkpoint thực tế; không dựa vào gradient ở terminal state để khẳng định checkpoint prediction đã học joint.

### F5 — Thấp cho run hiện tại, lỗi mã xác nhận được: metric âm thầm cắt dữ liệu

`metrics.py:16` dùng `zip(rows, rankings)` mà không kiểm tra độ dài. Tái hiện với hai claim và một ranking: hàm trả `count=1`, Hit@5 `=1.0` nhưng `alignment_status_counts` báo hai claim. Pipeline `evaluate()` hiện tạo một ranking cho mỗi claim, nên chưa có bằng chứng điểm đã công bố bị ảnh hưởng; hàm metric vẫn có thể tạo báo cáo sai mà không báo lỗi nếu caller khác đưa dữ liệu thiếu.

**Hướng xử lý:** kiểm tra `len(rows) == len(rankings)` trước khi tính, ném `ValueError` nếu khác; thêm test cho thiếu và thừa ranking.

### F6 — Giới hạn đã dự kiến: top 5 không thể chứa toàn bộ gold của mọi claim

Paper ViFactCheck có mẫu cần nhiều evidence. Trong prediction archive seed 42, 2/539 claim S/R được căn chỉnh có 6 hoặc 8 gold sentence; `complete_gold@5` luôn bằng 0 cho chúng dù thứ hạng tốt. Đây là giới hạn top 5 của Proposal, không phải bug. D0 cần giữ toàn bộ gold như đặc tả để khảo sát các trường hợp này.

### F7 — Thấp: tài liệu nguồn tham khảo chưa đầy đủ

`docs/research/README.md` không liệt kê FER dù Proposal viện dẫn trực tiếp; `docs/research/paper_04_multistage_reranking.md:6` có URL ACL sai (`aclantholbâyogy.org`). Sửa link và ghi rõ mức độ kế thừa phương pháp để tránh gọi source hiện tại là bản tái hiện nguyên paper.

## Thứ tự sửa và điều kiện nghiệm thu

1. Sửa F5 và cách đặt tên/mẫu số ở F3, kèm test metric. Chỉ sửa báo cáo mới; báo cáo seed 42 cũ giữ nguyên và thêm chú thích giải nghĩa nếu trích dẫn.
2. Rà soát F1 trên mẫu evidence không khớp, lưu nhãn nguyên nhân và tỷ lệ theo split/label. Nếu thay cách căn chỉnh hoặc evidence supervision, tăng phiên bản protocol/dữ liệu, tạo run mới cho cả năm cấu hình và không so sánh điểm mới như cùng điều kiện với điểm cũ.
3. Bổ sung trạng thái checkpoint được chọn theo F4 vào báo cáo, rồi thực hiện D0 và phân tích H3 theo F2 khi có checkpoint phù hợp. D0 không tham gia tuning.
4. Chỉnh mô tả nguồn paper và link theo F7. Báo cáo phải phân biệt “lấy cảm hứng” với “tái hiện” và phân biệt điểm ViFactCheck full-context/gold-evidence với M0.

Kiểm tra đã chạy: `.venv/bin/python -m pytest -q tests/test_assets.py tests/test_metrics_model.py` → **6 passed**. Lần chạy test rộng hơn bị chặn khi import `rank_bm25` trong `.venv`; Python hệ thống còn thiếu `torch`. Không suy ra lỗi source từ thiếu dependency này. Chưa kiểm tra checkpoint, loss curve hoặc kết quả ba seed v2.
