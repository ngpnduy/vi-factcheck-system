# Phân tích Khoảng trống Nghiên cứu (Research Gap Analysis) — ViFactCheck

> Cập nhật: 2025-09-19 · Dựa trên 24 bài báo đã khảo sát

---

## 1. Tổng hợp các bài báo đã đọc (24 bài)

### Nhóm A — Vietnamese Fact-Checking (Baseline & Dataset)
| # | File | Bài báo | Venue | Ý chính |
|---|---|---|---|---|
| 1 | `paper_01` | ViFactCheck | AAAI 2025 | Benchmark đa lĩnh vực, PLM + LLM baseline, hybrid retrieval |
| 2 | `paper_02` | ViWikiFC | arXiv 2024 | 20K claims từ Wikipedia tiếng Việt, quy mô lớn |
| 3 | `paper_03` | ViNumFCR | INLG 2025 | Numerical reasoning fact-checking tiếng Việt |
| 12 | `paper_12` | SemViQA | arXiv 2025 | QA-based 2 bước, state-of-the-art trên ISE-DSC01/ViWikiFC |

### Nhóm B — Evidence Retrieval
| # | File | Bài báo | Venue | Ý chính |
|---|---|---|---|---|
| 4 | `paper_04` | Multi-stage Reranking | EMNLP 2024 | Cross-Encoder reranker cải thiện recall mạnh |
| 6 | `paper_06` | RAV | ACL Findings 2024 | "Retrieval is almost all you need" — end-to-end training |

### Nhóm C — LLM for Fact-Checking
| # | File | Bài báo | Venue | Ý chính |
|---|---|---|---|---|
| 5 | `paper_05` | LLM Verification Survey | arXiv 2024 | Khảo sát toàn cảnh LLM + RAG trong fact-checking |
| 7a | `paper_07_openfactcheck` | OpenFactCheck | 2024 | Framework đánh giá factuality của LLM |
| 7b | `paper_07_token_uncertainty` | Token Uncertainty | 2024 | Phát hiện hallucination qua logprobs |
| 7c | `paper_07_reflex` | REFLEX | 2024 | Explanation-guided steering, faithful reasoning |
| 17 | `paper_17` | MiniCheck | arXiv 2024 | SLM 770M đạt ngang GPT-4, chi phí giảm 400x |
| 18 | `paper_18` | FIRE | arXiv 2024 | Iterative retrieval + verification, giảm 7.6x chi phí |

### Nhóm D — Low-resource / Cross-lingual
| # | File | Bài báo | Venue | Ý chính |
|---|---|---|---|---|
| 10 | `paper_10` | EX-Claim | arXiv 2025 | Entity-aware cross-lingual claim detection (27 ngôn ngữ) |
| 11 | `paper_11` | SemEval-2025 Task 7 | SemEval 2025 | Multilingual fact-checked claim retrieval |
| 16 | `paper_16` | Vistral | 2024 | LLM tiếng Việt 7B, fine-tuned cho chat |

### Nhóm E — Explainable Fact-Checking
| # | File | Bài báo | Venue | Ý chính |
|---|---|---|---|---|
| 13 | `paper_13` | CorXFact | COLING 2025 | Claim-evidence correlation → sinh giải thích |
| 14 | `paper_14` | ClaimVer | 2024 | Human-centric, Knowledge Graph attribution |
| 15 | `paper_15` | DEFAME | 2025 | Multimodal 6-stage zero-shot pipeline |

### Nhóm F — RAG for Fact-Checking
| # | File | Bài báo | Venue | Ý chính |
|---|---|---|---|---|
| 19a | `paper_19_safe` | SAFE | 2024 | Multi-agent RAG cho COVID fact-checking |
| 19b | `paper_19_face_the_facts` | Face the Facts | INLG 2025 | Benchmark RAG theo chuẩn fact-checker chuyên nghiệp |
| 19c | `paper_19_raguard` | RAGuard | 2024 | RAG + misleading evidence → tệ hơn zero-shot |

### Nhóm G — Multimodal / Social Media / Numerical
| # | File | Bài báo | Venue | Ý chính |
|---|---|---|---|---|
| 24a | `paper_24_defame` | DEFAME (multimodal) | ICML 2024 | Dynamic multi-modal verification |
| 24b | `paper_24_communitynotes` | CommunityNotes | ACL 2024 | Giải thích hữu ích cho mạng xã hội |
| 24c | `paper_24_quantemp` | QuanTemp | SIGIR 2024 | Open-domain numerical + temporal reasoning |

---

## 2. Phân tích 7 Khoảng Trống Nghiên cứu

### Gap 1: Cross-Encoder Reranking cho Retrieval tiếng Việt
- **Hiện trạng:** ViFactCheck gốc dùng Hybrid thô sơ (nhân BM25 × cosine SBERT). SemViQA dùng token matching nhưng chưa fine-tune reranker.
- **Gap:** Chưa ai áp dụng **Cross-Encoder Reranking** (paper 04, 06) chuyên cho tiếng Việt trên ViFactCheck dataset.
- **Khả thi:** ⭐⭐⭐⭐⭐ — Có sẵn codebase retrieval, chỉ cần fine-tune PhoBERT/mMiniLM làm reranker. Chạy được trên 1 GPU.
- **Novelty:** Trung bình — Kỹ thuật đã biết, nhưng chưa áp dụng cho tiếng Việt.

### Gap 2: Explainable Fact-Checking cho tiếng Việt (Sinh lời giải thích)
- **Hiện trạng:** Tất cả hệ thống FC tiếng Việt (ViFactCheck, SemViQA, ViWikiFC) chỉ xuất nhãn {Supported, Refuted, NEI}. Không hệ thống nào sinh **lời giải thích** (explanation/justification).
- **Gap:** Chưa có nghiên cứu nào về **Explainable Vietnamese Fact-Checking**. Papers 13 (CorXFact), 7c (REFLEX) chỉ làm trên tiếng Anh.
- **Khả thi:** ⭐⭐⭐⭐ — Dùng LoRA fine-tune Gemma/Vistral trên ~1-2K mẫu instruction. Cần GPT-4 API tạo dữ liệu (~$10-20).
- **Novelty:** ⭐⭐⭐⭐⭐ — **Rất cao** cho tiếng Việt. Chưa ai làm.

### Gap 3: Numerical Reasoning trong Fact-Checking tiếng Việt
- **Hiện trạng:** ViNumFCR (paper 03) đặt vấn đề nhưng chưa đề xuất giải pháp chuyên biệt. PLM truyền thống yếu với các claim chứa con số.
- **Gap:** Thiếu **module suy luận số liệu** (so sánh, tính toán) tích hợp vào pipeline FC tiếng Việt.
- **Khả thi:** ⭐⭐⭐⭐ — Rule-based extraction + LLM nhẹ. Dữ liệu có sẵn (ViNumFCR).
- **Novelty:** Cao — Ngách hẹp, ít cạnh tranh.

### Gap 4: Iterative/Agentic Retrieval cho Fact-Checking tiếng Việt
- **Hiện trạng:** Tất cả hệ thống FC tiếng Việt dùng single-turn retrieval. FIRE (paper 18) chứng minh iterative tốt hơn + rẻ hơn.
- **Gap:** Chưa ai áp dụng **agentic iterative retrieval** cho FC tiếng Việt.
- **Khả thi:** ⭐⭐⭐ — Cần LLM follow instructions tốt bằng tiếng Việt, latency cao, khó reproduce.
- **Novelty:** Cao.

### Gap 5: Entity-aware Retrieval cho tiếng Việt
- **Hiện trạng:** ViFactCheck không dùng NER. EX-Claim (paper 10) chứng minh entity-aware tăng hiệu quả cross-lingual.
- **Gap:** Tích hợp **PhoNER** vào retrieval pipeline để ưu tiên bằng chứng chứa cùng thực thể.
- **Khả thi:** ⭐⭐⭐⭐⭐ — Rất dễ implement. PhoNER có sẵn.
- **Novelty:** ⭐⭐ — Đóng góp mỏng, khó đứng một mình thành bài.

### Gap 6: RAG Pipeline End-to-End cho tiếng Việt
- **Hiện trạng:** Nhiều paper RAG cho FC (papers 19a-c) nhưng toàn tiếng Anh. RAGuard cảnh báo misleading evidence rất nguy hiểm.
- **Gap:** Chưa có RAG-based fact-checking pipeline cho tiếng Việt với đánh giá faithfulness.
- **Khả thi:** ⭐⭐⭐ — Scope lớn, cần nhiều engineering. Có thể quá sức cho 3 tháng.
- **Novelty:** Cao.

### Gap 7: Distillation từ LLM lớn → SLM chuyên biệt cho FC tiếng Việt
- **Hiện trạng:** MiniCheck (paper 17) chứng minh SLM 770M ≈ GPT-4 cho FC tiếng Anh. Chưa ai làm cho tiếng Việt.
- **Gap:** Distill GPT-4/Claude → PhoBERT-base/Qwen2.5-1.5B chuyên cho FC tiếng Việt.
- **Khả thi:** ⭐⭐⭐ — Cần budget API (~$30-50), kết quả khó đảm bảo trong 3 tháng.
- **Novelty:** Cao.

---

## 3. 🏆 Top 3 Đề Xuất cho Team 3 Người × 3 Tháng

> **Mục tiêu:** Workshop paper hoặc Short paper tại PACLIC, RIVF, SOICT, KSE, hoặc EMNLP/ACL Workshop.

---

### 🥇 Đề xuất 1: Explainable Vietnamese Fact-Checking với LLM nhẹ

> **Tên bài tiềm năng:** *"Towards Explainable Vietnamese Fact-Checking: Evidence-Grounded Justification Generation with Lightweight LLMs"*

**Tại sao chọn:**
- **Novelty cực cao** — Chưa có bất kỳ nghiên cứu nào sinh lời giải thích cho FC tiếng Việt.
- Bắt kịp xu hướng nóng nhất (CorXFact COLING 2025, REFLEX 2024, CommunityNotes ACL 2024).
- Xây dựng trên codebase có sẵn (Phase 5 LLM đã scaffolded).

**Kế hoạch 3 tháng:**

| Tháng | Người 1 (Data) | Người 2 (Model) | Người 3 (Eval & Writing) |
|---|---|---|---|
| **T1** | Tạo 1.5K mẫu instruction (claim + evidence → verdict + giải thích) bằng GPT-4 API từ ViFactCheck test set | Hoàn thiện inference pipeline cho LoRA (Phase 5 codebase) | Xây dựng evaluation framework: LLM-as-a-judge + human eval rubric |
| **T2** | Quality check & augment data, test với Vistral/Qwen2.5-7B prompting | LoRA fine-tune Gemma-2-2B hoặc Qwen2.5-7B (4-bit) trên data instruction | Chạy baseline: zero-shot prompting (không fine-tune) để so sánh |
| **T3** | Phân tích lỗi (error analysis) trên các loại claim khó | Thử nghiệm CorXFact-style correlation weighting | Viết bài, tổng hợp kết quả, submit |

**Tài nguyên cần:**
- GPU: Google Colab Pro (~$10/tháng) hoặc 1× RTX 3090
- API: GPT-4 mini/Claude Haiku tạo data (~$15-20)
- Dataset: ViFactCheck (đã có)

---

### 🥈 Đề xuất 2: Multi-stage Reranking + Entity-aware Retrieval cho ViFactCheck

> **Tên bài tiềm năng:** *"Improving Evidence Retrieval for Vietnamese Fact-Checking with Cross-Encoder Reranking and Named Entity Matching"*

**Tại sao chọn:**
- Kết hợp **Gap 1 + Gap 5** → đủ contribution cho 1 short paper.
- Cực kỳ khả thi: codebase retrieval đã chạy, chỉ cần thêm 2 module.
- Dễ đo lường: ACC@K, MRR, NDCG có sẵn.

**Kế hoạch 3 tháng:**

| Tháng | Người 1 (Reranker) | Người 2 (NER) | Người 3 (Eval) |
|---|---|---|---|
| **T1** | Tích hợp mMiniLM / PhoBERT làm Cross-Encoder reranker sau BM25 | Tích hợp PhoNER, xây entity-boosting score | Xây evaluation pipeline end-to-end (retrieval → classification) |
| **T2** | Fine-tune reranker trên ViFactCheck với Hard Negatives | Thử nghiệm entity weighting strategies | Chạy ablation study: BM25 → +SBERT → +Reranker → +Entity |
| **T3** | Tối ưu hyperparams | Error analysis trên entity types | Viết bài, so sánh với ViFactCheck gốc & SemViQA |

**Tài nguyên cần:**
- GPU: Colab free đủ (PhoBERT-base nhẹ)
- Dataset: ViFactCheck (đã có)
- Libraries: `sentence-transformers`, `underthesea` (PhoNER)

---

### 🥉 Đề xuất 3: Numerical Claim Verification Plugin cho ViFactCheck

> **Tên bài tiềm năng:** *"NumViFC: A Numerical Reasoning Module for Vietnamese Fact-Checking Systems"*

**Tại sao chọn:**
- Ngách hẹp, ít cạnh tranh (chỉ có ViNumFCR + QuanTemp).
- Đóng góp rõ ràng: module plug-in, dễ reproduce.
- Phù hợp sinh viên thích systems work hơn NLP thuần.

**Kế hoạch 3 tháng:**

| Tháng | Người 1 (Extraction) | Người 2 (Reasoning) | Người 3 (Integration) |
|---|---|---|---|
| **T1** | Regex + NER trích xuất số từ claim & evidence | Xây rule-based comparator (>, <, =, ≈, %) | Lọc subset numerical claims từ ViFactCheck |
| **T2** | Xử lý edge cases (đơn vị, format VN: "1.000" vs "1,000") | Thử LLM nhẹ (Qwen2.5-1.5B) cho multi-step numerical reasoning | Tích hợp dual-stream: text pipeline ∥ numerical pipeline |
| **T3** | Test trên ViNumFCR cross-evaluation | Ablation: rule-based vs LLM vs hybrid | Viết bài |

---

## 4. Ma trận So sánh Nhanh

| Tiêu chí | 🥇 Explainable FC | 🥈 Reranking+NER | 🥉 Numerical |
|---|---|---|---|
| **Novelty** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Khả thi (3 tháng)** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **GPU cần** | Medium (Colab Pro) | Low (Colab Free) | Low-Medium |
| **Rủi ro** | Data quality từ GPT-4 | Novelty mỏng | Edge cases tiếng Việt |
| **Venue phù hợp** | EMNLP Workshop, PACLIC | RIVF, SOICT, KSE | INLG Workshop, PACLIC |
| **Mở rộng được** | → Full paper | → Kết hợp với Đề xuất 1 | → Kết hợp với Đề xuất 2 |
