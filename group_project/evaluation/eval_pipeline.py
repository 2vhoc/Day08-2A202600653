"""
Offline RAG Evaluation Pipeline.

The course README suggests DeepEval/RAGAS/TruLens. Those packages are optional
and often require LLM/API access, so this script implements the same four
evaluation dimensions with deterministic lexical/evidence heuristics:

- Faithfulness: answer terms should be grounded in retrieved context.
- Answer relevance: answer should overlap both the question and expected answer.
- Context recall: retrieved context should cover expected answer/context terms.
- Context precision: retrieved chunks should be useful for the question.

It also runs an A/B comparison:
- Config A: Task 9 hybrid retrieval + Task 10 local cited generation.
- Config B: dense-only retrieval + the same local cited generation style.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import types
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task4_chunking_indexing import tokenize
from src.task5_semantic_search import semantic_search
from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import local_generate_answer, reorder_for_llm

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def load_env_file():
    """Load .env without requiring python-dotenv."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_env_file()


@dataclass
class EvalCaseResult:
    question: str
    answer: str
    expected_answer: str
    expected_context: str
    faithfulness: float
    answer_relevance: float
    context_recall: float
    context_precision: float
    retrieval_source: str
    failure_stage: str
    root_cause: str

    @property
    def average(self) -> float:
        return statistics.mean(
            [
                self.faithfulness,
                self.answer_relevance,
                self.context_recall,
                self.context_precision,
            ]
        )


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    return json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))


def token_set(text: str) -> set[str]:
    """Tokenize and remove ultra-common Vietnamese glue words."""
    stopwords = {
        "la",
        "ve",
        "va",
        "cua",
        "cho",
        "theo",
        "trong",
        "nhung",
        "gi",
        "nao",
        "duoc",
        "cac",
        "mot",
        "co",
        "bi",
        "voi",
        "tu",
        "den",
        "tai",
        "doi",
        "nguoi",
    }
    return {token for token in tokenize(text) if len(token) >= 3 and token not in stopwords}


def coverage(needle: str, haystack: str) -> float:
    """How much of needle's information appears in haystack."""
    needle_tokens = token_set(needle)
    if not needle_tokens:
        return 0.0

    haystack_tokens = token_set(haystack)
    return len(needle_tokens & haystack_tokens) / len(needle_tokens)


def context_text(sources: list[dict]) -> str:
    """Concatenate retrieval context and metadata for scoring."""
    parts = []
    for source in sources:
        metadata = source.get("metadata", {})
        parts.append(source.get("content", ""))
        parts.append(" ".join(str(value) for value in metadata.values()))
    return "\n".join(parts)


def context_precision_score(question: str, expected_answer: str, sources: list[dict]) -> float:
    """Average usefulness of retrieved chunks for the question/expected answer."""
    if not sources:
        return 0.0

    target = f"{question} {expected_answer}"
    chunk_scores = [coverage(target, source.get("content", "")) for source in sources]
    useful_chunks = [score for score in chunk_scores if score >= 0.08]
    usefulness_ratio = len(useful_chunks) / len(sources)
    mean_overlap = statistics.mean(chunk_scores) if chunk_scores else 0.0
    return min(1.0, 0.65 * usefulness_ratio + 0.35 * mean_overlap * 2)


def infer_failure_stage(result: EvalCaseResult) -> tuple[str, str]:
    """Classify the weakest part of a failed example."""
    scores = {
        "generation": result.faithfulness,
        "answering": result.answer_relevance,
        "retrieval_recall": result.context_recall,
        "retrieval_precision": result.context_precision,
    }
    stage = min(scores, key=scores.get)
    causes = {
        "generation": "Câu trả lời chứa nhiều từ/cụm không được context hỗ trợ.",
        "answering": "Câu trả lời chưa bám sát câu hỏi hoặc expected answer.",
        "retrieval_recall": "Retriever chưa lấy đủ evidence kỳ vọng.",
        "retrieval_precision": "Một phần context truy xuất chưa hữu ích cho câu hỏi.",
    }
    return stage, causes[stage]


def score_case(item: dict, answer: str, sources: list[dict], retrieval_source: str) -> EvalCaseResult:
    """Compute all four offline metrics for one golden item."""
    joined_context = context_text(sources)
    citation_bonus = 0.08 if "[" in answer and "]" in answer else 0.0

    faithfulness = min(1.0, coverage(answer, joined_context) * 1.15 + citation_bonus)
    answer_relevance = min(
        1.0,
        0.45 * coverage(item["question"], answer)
        + 0.55 * coverage(item["expected_answer"], answer),
    )
    context_recall = min(
        1.0,
        0.75 * coverage(item["expected_answer"], joined_context)
        + 0.25 * coverage(item["expected_context"], joined_context),
    )
    context_precision = context_precision_score(
        item["question"],
        item["expected_answer"],
        sources,
    )

    result = EvalCaseResult(
        question=item["question"],
        answer=answer,
        expected_answer=item["expected_answer"],
        expected_context=item["expected_context"],
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        context_recall=context_recall,
        context_precision=context_precision,
        retrieval_source=retrieval_source,
        failure_stage="",
        root_cause="",
    )
    result.failure_stage, result.root_cause = infer_failure_stage(result)
    return result


def run_hybrid_pipeline(question: str, top_k: int = 5) -> tuple[str, list[dict], str]:
    """Config A: full Task 9/10 pipeline."""
    sources = retrieve(question, top_k=top_k, use_reranking=True)
    reordered = reorder_for_llm(sources)
    answer = local_generate_answer(question, reordered)
    retrieval_source = sources[0].get("source", "none") if sources else "none"
    return answer, sources, retrieval_source


def run_dense_only_pipeline(question: str, top_k: int = 5) -> tuple[str, list[dict], str]:
    """Config B: semantic retrieval only, no lexical/RRF/fallback."""
    sources = semantic_search(question, top_k=top_k)
    for source in sources:
        source["source"] = "dense_only"
    reordered = reorder_for_llm(sources)
    answer = local_generate_answer(question, reordered)
    return answer, sources, "dense_only"


def build_ragas_rows(golden_dataset: list[dict], runner=run_hybrid_pipeline) -> list[dict]:
    """Build rows using RAGAS 0.4.x column names."""
    rows = []
    for item in golden_dataset:
        answer, sources, retrieval_source = runner(item["question"])
        rows.append(
            {
                "user_input": item["question"],
                "response": answer,
                "retrieved_contexts": [source.get("content", "") for source in sources],
                "reference": item["expected_answer"],
                "retrieval_source": retrieval_source,
            }
        )
    return rows


def install_ragas_vertexai_shim():
    """
    RAGAS 0.4.3 imports a legacy LangChain VertexAI path that is absent in
    langchain-community 0.4.x. The class is only needed for type checks unless
    VertexAI is used, so this shim lets OpenAI-based RAGAS evaluation import.
    """
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    module = types.ModuleType(module_name)

    class ChatVertexAI:  # pragma: no cover - compatibility shim only
        pass

    module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = module


def has_usable_openai_key() -> bool:
    """RAGAS needs an LLM/embedding provider; current implementation uses OpenAI."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    return len(api_key.strip()) > 20


def evaluate_config(golden_dataset: list[dict], config_name: str, runner) -> dict:
    """Evaluate one pipeline config on the full golden dataset."""
    case_results = []

    for item in golden_dataset:
        answer, sources, retrieval_source = runner(item["question"])
        case_results.append(score_case(item, answer, sources, retrieval_source))

    return {
        "config_name": config_name,
        "cases": case_results,
        "summary": summarize(case_results),
    }


def summarize(cases: list[EvalCaseResult]) -> dict[str, float]:
    """Average metrics across cases."""
    return {
        "faithfulness": statistics.mean(case.faithfulness for case in cases),
        "answer_relevance": statistics.mean(case.answer_relevance for case in cases),
        "context_recall": statistics.mean(case.context_recall for case in cases),
        "context_precision": statistics.mean(case.context_precision for case in cases),
        "average": statistics.mean(case.average for case in cases),
    }


# =============================================================================
# Framework-compatible wrappers
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """DeepEval-style offline evaluation wrapper."""
    return evaluate_config(golden_dataset, "hybrid_rerank", run_hybrid_pipeline)


def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Run real RAGAS when available and configured.

    RAGAS 0.4.x requires an LLM for faithfulness/context metrics and embeddings
    for answer relevancy. We use OpenAI via langchain-openai when
    OPENAI_API_KEY is usable. If API access is unavailable, return the offline
    heuristic result with a clear status flag.
    """
    offline = evaluate_config(golden_dataset, "hybrid_rerank", run_hybrid_pipeline)

    if not has_usable_openai_key():
        offline["framework"] = "RAGAS unavailable - offline fallback"
        offline["ragas_status"] = "skipped"
        offline["ragas_error"] = "OPENAI_API_KEY is missing or too short for RAGAS LLM/embedding evaluation."
        return offline

    try:
        install_ragas_vertexai_shim()
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        rows = build_ragas_rows(golden_dataset, run_hybrid_pipeline)
        dataset = Dataset.from_list(rows)
        llm = ChatOpenAI(
            model=os.getenv("RAGAS_LLM_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
        embeddings = OpenAIEmbeddings(
            model=os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        ragas_result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            ],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
            show_progress=False,
        )
        ragas_scores = dict(ragas_result)
        summary = {
            "faithfulness": float(ragas_scores.get("faithfulness", 0.0) or 0.0),
            "answer_relevance": float(ragas_scores.get("answer_relevancy", 0.0) or 0.0),
            "context_recall": float(ragas_scores.get("context_recall", 0.0) or 0.0),
            "context_precision": float(ragas_scores.get("context_precision", 0.0) or 0.0),
        }
        summary["average"] = statistics.mean(summary.values())
        offline["summary"] = summary
        offline["framework"] = "RAGAS"
        offline["ragas_status"] = "completed"
        offline["ragas_raw"] = ragas_scores
        return offline
    except Exception as exc:
        offline["framework"] = "RAGAS attempted - offline fallback"
        offline["ragas_status"] = "failed"
        offline["ragas_error"] = f"{type(exc).__name__}: {exc}"
        return offline


def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """TruLens-style offline evaluation wrapper."""
    return evaluate_config(golden_dataset, "hybrid_rerank", run_hybrid_pipeline)


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """Compare full hybrid retrieval against dense-only retrieval."""
    return {
        "hybrid_rerank": evaluate_config(
            golden_dataset,
            "hybrid_rerank",
            run_hybrid_pipeline,
        ),
        "dense_only": evaluate_config(
            golden_dataset,
            "dense_only",
            run_dense_only_pipeline,
        ),
    }


# =============================================================================
# Export Results
# =============================================================================

def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def truncate(text: str, length: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= length else text[: length - 3] + "..."


def export_results(results: dict, comparison: dict):
    """Export evaluation results to results.md"""
    hybrid = comparison["hybrid_rerank"]["summary"]
    dense = comparison["dense_only"]["summary"]
    delta = {metric: hybrid[metric] - dense[metric] for metric in hybrid}
    cases = comparison["hybrid_rerank"]["cases"]
    worst = sorted(cases, key=lambda case: case.average)[:3]

    content = "# RAG Evaluation Results\n\n"
    content += "## Framework sử dụng\n\n"
    framework = results.get("framework", "RAGAS-compatible offline evaluator")
    status = results.get("ragas_status", "offline")
    content += f"Framework chính: **{framework}**.\n\n"
    if status == "completed":
        content += (
            "RAGAS thật đã chạy với `ragas.evaluate`, HuggingFace `Dataset`, "
            "`ChatOpenAI` và `OpenAIEmbeddings` cho 4 metric: faithfulness, "
            "answer_relevancy, context_recall, context_precision.\n\n"
        )
    else:
        content += (
            "RAGAS đã được tích hợp trong `eval_pipeline.py`, nhưng lần chạy này "
            "dùng offline fallback vì API/LLM chưa sẵn sàng. Offline fallback vẫn "
            "tính 4 metric tương ứng bằng lexical/evidence overlap để demo chạy local.\n\n"
        )
        if results.get("ragas_error"):
            content += f"RAGAS status: `{results['ragas_error']}`\n\n"

    content += "## Overall Scores\n\n"
    content += "| Metric | Config A (hybrid + rerank/fallback) | Config B (dense-only) | Δ |\n"
    content += "|--------|-------------------------------------|------------------------|---|\n"
    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevance": "Answer Relevance",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
        "average": "**Average**",
    }
    for metric, label in metric_labels.items():
        content += f"| {label} | {pct(hybrid[metric])} | {pct(dense[metric])} | {delta[metric] * 100:+.1f} pts |\n"

    winner = "Config A" if hybrid["average"] >= dense["average"] else "Config B"
    content += "\n## A/B Comparison Analysis\n\n"
    content += (
        "**Config A:** hybrid retrieval kết hợp semantic search, BM25 lexical search, "
        "RRF fusion, reranking/fallback logic và cited generation.\n\n"
    )
    content += (
        "**Config B:** dense-only retrieval chỉ dùng semantic index local, không dùng "
        "BM25, RRF hoặc PageIndex fallback.\n\n"
    )
    content += (
        f"**Kết luận:** {winner} có average cao hơn trong bộ golden dataset. "
        "Hybrid thường tăng context recall nhờ BM25 bắt đúng keyword pháp luật, "
        "trong khi dense-only có thể bỏ sót điều khoản hoặc bài báo có tên riêng.\n\n"
    )

    content += "## Worst Performers (Bottom 3)\n\n"
    content += "| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |\n"
    content += "|---|----------|-------------|-----------|--------|---------------|------------|\n"
    for index, case in enumerate(worst, 1):
        content += (
            f"| {index} | {truncate(case.question)} | {pct(case.faithfulness)} | "
            f"{pct(case.answer_relevance)} | {pct(case.context_recall)} | "
            f"{case.failure_stage} | {case.root_cause} |\n"
        )

    content += "\n## Recommendations\n\n"
    content += "### Cải tiến 1\n"
    content += "**Action:** OCR hoặc thay nguồn text-based cho `nghi-dinh-105.pdf` và các tài liệu scan.\n\n"
    content += "**Expected impact:** Tăng context recall cho câu hỏi cần nghị định/danh mục chất ma túy.\n\n"
    content += "### Cải tiến 2\n"
    content += "**Action:** Dùng embedding multilingual thật như `BAAI/bge-m3` khi có GPU/CPU đủ mạnh.\n\n"
    content += "**Expected impact:** Tăng semantic recall cho câu hỏi diễn đạt khác keyword trong văn bản.\n\n"
    content += "### Cải tiến 3\n"
    content += "**Action:** Chuẩn hóa metadata theo điều luật, ngày báo, cơ quan ban hành và source URL.\n\n"
    content += "**Expected impact:** Citation chính xác hơn và giảm lỗi khi generation chọn nhầm nguồn.\n\n"
    content += "## Per-case Details\n\n"
    content += "| # | Retrieval | Avg | Question |\n|---|-----------|-----|----------|\n"
    for index, case in enumerate(cases, 1):
        content += f"| {index} | {case.retrieval_source} | {pct(case.average)} | {truncate(case.question)} |\n"

    RESULTS_PATH.write_text(content, encoding="utf-8")


def main():
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    results = evaluate_with_ragas(None, golden_dataset)
    comparison = compare_configs(None, golden_dataset)
    export_results(results, comparison)

    hybrid_average = comparison["hybrid_rerank"]["summary"]["average"]
    dense_average = comparison["dense_only"]["summary"]["average"]
    print(f"Hybrid average: {pct(hybrid_average)}")
    print(f"Dense-only average: {pct(dense_average)}")
    print(f"Evaluation framework: {results.get('framework', 'offline')}")
    print(f"Saved report: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
