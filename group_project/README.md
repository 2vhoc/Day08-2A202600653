# Bài Tập Nhóm — RAG Chatbot + Evaluation Pipeline

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm hoàn thiện cả **RAG Chatbot demo** và
**RAG Evaluation Pipeline** để đánh giá end-to-end pipeline hỏi đáp về pháp luật
ma túy và tin tức liên quan.

Trạng thái hiện tại: **hoàn thành**.

- Golden dataset: 15 Q&A pairs
- Evaluation script: ưu tiên RAGAS thật, có offline fallback khi thiếu LLM/API
- A/B comparison: hybrid + rerank/fallback vs dense-only
- Báo cáo kết quả: `group_project/evaluation/results.md`
- Web chatbot: `app.py` ở thư mục gốc project

---

## Yêu cầu 1: Sản phẩm nhóm RAG Chatbot — Hoàn thành

Xây dựng chatbot trả lời câu hỏi về pháp luật ma tuý và tin tức liên quan.

**Yêu cầu đã đáp ứng:**
- Giao diện chat web chạy local bằng Python standard library
- Trả lời có citation dựa trên `src/task10_generation.py`
- Hỗ trợ follow-up questions bằng conversation memory phía client/server
- Hiển thị source documents đã dùng ở panel bên phải

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

Chạy app:

```bash
source ~/global/bin/activate
python app.py --host 127.0.0.1 --port 8501
```

Mở trình duyệt tại `http://127.0.0.1:8501`.

---

## Yêu cầu 2: RAG Evaluation Pipeline — Đã chọn

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

Nhóm chọn **RAGAS**. File `eval_pipeline.py` đã implement nhánh gọi
`ragas.evaluate` với `datasets.Dataset`, `ChatOpenAI` và `OpenAIEmbeddings`.
Nếu `OPENAI_API_KEY` chưa hợp lệ, script tự fallback sang evaluator offline để
vẫn demo được trên máy local.

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

### Code mẫu — DeepEval

```python
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

# Tạo test cases từ golden dataset
test_cases = []
for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    test_case = LLMTestCase(
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["expected_answer"],
        retrieval_context=[c["content"] for c in result["sources"]],
    )
    test_cases.append(test_case)

# Chạy evaluation
metrics = [
    FaithfulnessMetric(threshold=0.7),
    AnswerRelevancyMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
]

results = evaluate(test_cases, metrics)
```

### Code mẫu — RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

# Chuẩn bị data
eval_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [],
}

for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    eval_data["question"].append(item["question"])
    eval_data["answer"].append(result["answer"])
    eval_data["contexts"].append([c["content"] for c in result["sources"]])
    eval_data["ground_truth"].append(item["expected_answer"])

dataset = Dataset.from_dict(eval_data)

# Chạy evaluation
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)
print(result.to_pandas())
```

### Code mẫu — TruLens

```python
from trulens.apps.custom import TruCustomApp, instrument
from trulens.core import Feedback
from trulens.providers.openai import OpenAI as TruOpenAI

provider = TruOpenAI()

# Define feedback functions
f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
f_relevance = Feedback(provider.relevance).on_input_output()
f_context_relevance = Feedback(provider.context_relevance).on_input()

# Wrap RAG pipeline
tru_rag = TruCustomApp(
    rag_pipeline,
    app_name="DrugLaw_RAG",
    feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
)

# Run evaluation
with tru_rag as recording:
    for item in golden_dataset:
        rag_pipeline.generate_with_citation(item["question"])

# View dashboard
from trulens.dashboard import run_dashboard
run_dashboard()
```

### Deliverable Evaluation

- [x] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [x] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [x] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [x] So sánh A/B ít nhất 2 configs

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân: Task 4-10 được nối thành RAG end-to-end.
2. **Demo hoạt động được**: chạy local bằng `python app.py`.
3. **Evaluation pipeline** chạy được và có báo cáo kết quả.
4. **Code push lên repository** chung của nhóm.
5. **README** mô tả kiến trúc, lệnh chạy và phân công.

---

## Kiến Trúc Hệ Thống

```
data/landing/
  ├── legal PDFs
  └── news JSON
        │
        ▼
Task 3: Markdown Standardization
data/standardized/
        │
        ▼
Task 4: Chunking + Local Vector Index
data/index/vector_index.json
        │
        ├───────────────┐
        ▼               ▼
Task 5 Semantic     Task 6 BM25 Lexical
        │               │
        └──────┬────────┘
               ▼
Task 9 Hybrid Retrieval
  - RRF fusion
  - rerank/fallback logic
  - PageIndex-style local fallback
               │
               ▼
Task 10 Generation with Citation
               │
               ▼
Group Evaluation Pipeline
  - Golden dataset 15 Q&A
  - Faithfulness
  - Answer Relevance
  - Context Recall
  - Context Precision
  - A/B comparison
```

Web demo:

```
Browser UI
  ├── POST /api/chat
  │     └── Task 10 generate_with_citation
  │           └── Task 9 retrieve
  ├── GET /api/evaluation
  │     └── group_project/evaluation/results.md
  └── GET /api/health
```

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| Vũ Văn Học | N/A | Data collection, crawling, markdown conversion | Hoàn thành |
| Vũ Văn Học | N/A | Chunking, indexing, semantic/BM25 retrieval | Hoàn thành |
| Vũ Văn Học | N/A | Reranking, fallback retrieval, cited generation | Hoàn thành |
| Vũ Văn Học | N/A | Web chatbot, golden dataset, evaluation script, results report | Hoàn thành |

---

## Hướng Dẫn Chạy

```bash
# Từ thư mục gốc project
cd /home/vuvanhoc/Study/AI_ThucChien/Applied_AI_Talent/8_Day/Day08_RAG_pipeline_cohort2

# Kích hoạt môi trường ảo global
source ~/global/bin/activate

# Chạy toàn bộ test cá nhân
python -m unittest tests.test_individual -v

# Chạy web chatbot
python app.py --host 127.0.0.1 --port 8501

# Chạy evaluation nhóm và sinh lại results.md
python group_project/evaluation/eval_pipeline.py
```

## Deploy Railway

Các file deploy đã chuẩn bị ở thư mục gốc:

```text
Dockerfile
.dockerignore
railway.json
requirements-railway.txt
app.py
```

Railway sẽ tự inject biến `PORT`; `app.py` đã tự đọc `PORT` và bind
`0.0.0.0` khi chạy trên Railway. Healthcheck dùng endpoint `/api/health`.

Biến môi trường cần set trên Railway nếu muốn gọi API thật:

```env
OPENAI_API_KEY=...
OPENAI_USE_API=1
PAGEINDEX_API_KEY=...
PAGEINDEX_USE_API=1
JINA_API_KEY=...
```

Nếu không set các biến trên, app vẫn chạy bằng retrieval/generation fallback
local từ `data/index` và `data/standardized`.

Để chạy RAGAS thật thay vì fallback, cần có OpenAI key hợp lệ trong `.env`:

```env
OPENAI_API_KEY=...
RAGAS_LLM_MODEL=gpt-4o-mini
RAGAS_EMBEDDING_MODEL=text-embedding-3-small
```

Kết quả kỳ vọng:

```text
Loaded 15 test cases
Hybrid average: 94.2%
Dense-only average: 91.8%
Saved report: .../group_project/evaluation/results.md
```

---

## Kết Quả Evaluation

File báo cáo: `group_project/evaluation/results.md`

| Metric | Config A (hybrid + rerank/fallback) | Config B (dense-only) | Δ |
|--------|-------------------------------------|------------------------|---|
| Faithfulness | 98.9% | 99.0% | -0.0 pts |
| Answer Relevance | 92.9% | 89.4% | +3.5 pts |
| Context Recall | 88.2% | 83.5% | +4.7 pts |
| Context Precision | 96.7% | 95.5% | +1.2 pts |
| **Average** | **94.2%** | **91.8%** | **+2.4 pts** |

Kết luận: config hybrid tốt hơn dense-only trên bộ golden dataset vì BM25
bắt keyword điều luật/tên người tốt hơn, còn semantic search giúp bổ sung các
truy vấn diễn đạt tự nhiên.

---

## Cấu Trúc Deliverables

```text
app.py                      # web chatbot demo ở thư mục gốc
group_project/
├── README.md
└── evaluation/
    ├── golden_dataset.json   # 15 câu hỏi + expected answer/context
    ├── eval_pipeline.py      # RAGAS-first evaluator + offline fallback + A/B comparison
    └── results.md            # báo cáo kết quả đã sinh
```

---

## Lưu ý: Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
