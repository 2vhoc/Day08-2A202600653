# RAG Evaluation Results

## Framework sử dụng

Framework chính: **RAGAS unavailable - offline fallback**.

RAGAS đã được tích hợp trong `eval_pipeline.py`, nhưng lần chạy này dùng offline fallback vì API/LLM chưa sẵn sàng. Offline fallback vẫn tính 4 metric tương ứng bằng lexical/evidence overlap để demo chạy local.

RAGAS status: `OPENAI_API_KEY is missing or too short for RAGAS LLM/embedding evaluation.`

## Overall Scores

| Metric | Config A (hybrid + rerank/fallback) | Config B (dense-only) | Δ |
|--------|-------------------------------------|------------------------|---|
| Faithfulness | 98.9% | 99.0% | -0.0 pts |
| Answer Relevance | 92.9% | 89.4% | +3.5 pts |
| Context Recall | 88.2% | 83.5% | +4.7 pts |
| Context Precision | 96.7% | 95.5% | +1.2 pts |
| **Average** | 94.2% | 91.8% | +2.4 pts |

## A/B Comparison Analysis

**Config A:** hybrid retrieval kết hợp semantic search, BM25 lexical search, RRF fusion, reranking/fallback logic và cited generation.

**Config B:** dense-only retrieval chỉ dùng semantic index local, không dùng BM25, RRF hoặc PageIndex fallback.

**Kết luận:** Config A có average cao hơn trong bộ golden dataset. Hybrid thường tăng context recall nhờ BM25 bắt đúng keyword pháp luật, trong khi dense-only có thể bỏ sót điều khoản hoặc bài báo có tên riêng.

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Điều 251 Bộ luật Hình sự xử lý hành vi mua bán trái phép chất ma túy như thế nào? | 100.0% | 85.7% | 60.7% | retrieval_recall | Retriever chưa lấy đủ evidence kỳ vọng. |
| 2 | Bài Dân Trí về Châu Việt Cường nêu chất ma túy nào được sử dụng trong vụ án? | 100.0% | 87.8% | 66.3% | retrieval_recall | Retriever chưa lấy đủ evidence kỳ vọng. |
| 3 | Nguồn tài chính cho phòng, chống ma túy gồm những nguồn nào? | 92.3% | 76.4% | 92.9% | answering | Câu trả lời chưa bám sát câu hỏi hoặc expected answer. |

## Recommendations

### Cải tiến 1
**Action:** OCR hoặc thay nguồn text-based cho `nghi-dinh-105.pdf` và các tài liệu scan.

**Expected impact:** Tăng context recall cho câu hỏi cần nghị định/danh mục chất ma túy.

### Cải tiến 2
**Action:** Dùng embedding multilingual thật như `BAAI/bge-m3` khi có GPU/CPU đủ mạnh.

**Expected impact:** Tăng semantic recall cho câu hỏi diễn đạt khác keyword trong văn bản.

### Cải tiến 3
**Action:** Chuẩn hóa metadata theo điều luật, ngày báo, cơ quan ban hành và source URL.

**Expected impact:** Citation chính xác hơn và giảm lỗi khi generation chọn nhầm nguồn.

## Per-case Details

| # | Retrieval | Avg | Question |
|---|-----------|-----|----------|
| 1 | hybrid | 92.7% | Điều 249 Bộ luật Hình sự quy định gì về tội tàng trữ trái phép chất ma túy? |
| 2 | hybrid | 90.8% | Điều 250 Bộ luật Hình sự nói về hành vi nào liên quan đến ma túy? |
| 3 | hybrid | 83.8% | Điều 251 Bộ luật Hình sự xử lý hành vi mua bán trái phép chất ma túy như thế nào? |
| 4 | hybrid | 92.5% | Luật Phòng, chống ma túy 2021 điều chỉnh những nội dung nào? |
| 5 | hybrid | 94.4% | Các biện pháp cai nghiện ma túy theo Luật Phòng, chống ma túy 2021 gồm những gì? |
| 6 | hybrid | 97.3% | Quy trình cai nghiện ma túy gồm những giai đoạn nào? |
| 7 | hybrid | 97.8% | Thời hạn cai nghiện ma túy tự nguyện tại gia đình, cộng đồng là bao lâu? |
| 8 | hybrid | 100.0% | Người cai nghiện ma túy bắt buộc từ đủ 12 tuổi đến dưới 18 tuổi có thời hạn cai nghiện ... |
| 9 | hybrid | 90.4% | Nguồn tài chính cho phòng, chống ma túy gồm những nguồn nào? |
| 10 | hybrid | 95.4% | Gia đình người sử dụng trái phép chất ma túy có trách nhiệm gì? |
| 11 | hybrid | 100.0% | Diễn viên Hữu Tín bị truy tố về tội gì trong bài báo Tuổi Trẻ? |
| 12 | hybrid | 96.4% | Trong vụ Hữu Tín, cơ quan chức năng phát hiện những ai dương tính với chất ma túy? |
| 13 | hybrid | 97.6% | Ca sĩ Chu Bin được báo VietNamNet mô tả liên quan đến vụ việc gì? |
| 14 | hybrid | 98.5% | Châu Việt Cường bị khởi tố theo tội danh nào trong bài VnExpress? |
| 15 | hybrid | 85.1% | Bài Dân Trí về Châu Việt Cường nêu chất ma túy nào được sử dụng trong vụ án? |
