# Bài Tập Nhóm — RAG Chatbot + Evaluation Pipeline

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm hoàn thiện cả **RAG Chatbot demo** và
**RAG Evaluation Pipeline** để đánh giá end-to-end pipeline hỏi đáp về pháp luật
ma túy và tin tức liên quan.

---

## Yêu cầu 1: Sản phẩm nhóm RAG Chatbot — Hoàn thành

Xây dựng chatbot trả lời câu hỏi về pháp luật ma tuý và tin tức liên quan.

**Yêu cầu đã đáp ứng:**
- Giao diện chat web chạy local bằng Python standard library
- Trả lời có citation dựa trên `src/task10_generation.py`
- Hỗ trợ follow-up questions bằng conversation memory phía client/server
- Hiển thị source documents đã dùng ở panel bên phải


---

## Yêu cầu 2: RAG Evaluation Pipeline — Đã chọn

### Framework lựa chọn

Nhóm chọn **RAGAS**. File `eval_pipeline.py` đã implement nhánh gọi
`ragas.evaluate` với `datasets.Dataset`, `ChatOpenAI` và `OpenAIEmbeddings`
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
| Vũ Văn Học | 2A202600653 | Code | Hoàn thành |
| Trần Tiến Đạt | 2A202600978 | Deploy | Hoàn thành |
| Hồ Trọng Nhật Minh | 2A202600768 | Q&A + kết quả test | Hoàn thành |
| Nguyễn Đức Thành | 2A202600955 | giao diện  | Hoàn thành |

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
