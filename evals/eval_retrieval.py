"""
Phase 10 — RAG Pipeline Evaluator (RAGAS)

Evaluates the Qdrant-based memory retrieval pipeline using RAGAS metrics:
- Faithfulness: Does the output match what was retrieved?
- Context precision/recall: Are we pulling the right memories vs. noise?
- Answer relevancy: Does the output address the task?

This script can be run standalone or imported as a module.
"""

import os
import logging
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy RAGAS import — gracefully degrade if the dependency is broken
# ---------------------------------------------------------------------------

try:
    from ragas import evaluate as ragas_evaluate
    from ragas import EvaluationDataset
    from ragas.metrics import (
        faithfulness,
        context_precision,
        context_recall,
        answer_relevancy,
    )
    from ragas.dataset_schema import SingleTurnSample
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning(
        "RAGAS is not available. Install with: pip install ragas langchain-google-vertexai"
    )


def evaluate_retrieval_pipeline(
    questions: list[str],
    retrieved_contexts: list[list[str]],
    ground_truths: list[list[str]],
    answers: list[str],
) -> dict[str, float]:
    """
    Run RAGAS evaluation on a set of Q/A pairs with retrieved contexts.

    Parameters
    ----------
    questions : list[str]
        The user queries.
    retrieved_contexts : list[list[str]]
        The text chunks retrieved from Qdrant for each question.
    ground_truths : list[list[str]]
        The ideal/reference contexts for each question.
    answers : list[str]
        The final answer produced by the aggregator (or any agent).

    Returns
    -------
    dict[str, float]
        A dict mapping metric names to scores (0.0–1.0).
        Returns empty dict if RAGAS is unavailable.
    """
    if not RAGAS_AVAILABLE:
        logger.error("RAGAS is not installed — cannot run evaluation.")
        return {}

    if not (len(questions) == len(retrieved_contexts) == len(ground_truths) == len(answers)):
        raise ValueError("All input lists must have the same length")

    samples = []
    for q, ctxs, gts, ans in zip(questions, retrieved_contexts, ground_truths, answers):
        sample = SingleTurnSample(
            user_input=q,
            retrieved_contexts=ctxs,
            reference_contexts=gts,
            response=ans,
        )
        samples.append(sample)

    metrics = [faithfulness, context_precision, context_recall, answer_relevancy]

    try:
        dataset = EvaluationDataset(samples)
        result = ragas_evaluate(dataset=dataset, metrics=metrics)
        scores = {
            "faithfulness": result[faithfulness.name],
            "context_precision": result[context_precision.name],
            "context_recall": result[context_recall.name],
            "answer_relevancy": result[answer_relevancy.name],
        }
        logger.info("RAGAS evaluation complete: %s", scores)
        return scores
    except Exception as e:
        logger.error("RAGAS evaluation failed: %s", e)
        return {}


def run_demo_evaluation() -> dict[str, float]:
    """
    Run a minimal demo evaluation with synthetic data.

    This demonstrates the evaluation harness without requiring a live
    Qdrant instance or real LLM calls.
    """
    questions = [
        "What is the Fibonacci sequence?",
        "How do I implement a binary search?",
    ]
    retrieved_contexts = [
        [
            "The Fibonacci sequence is a series where each number is the sum of the two preceding ones.",
            "It starts with 0 and 1.",
        ],
        [
            "Binary search is an O(log n) algorithm for finding an element in a sorted array.",
            "It works by repeatedly dividing the search interval in half.",
        ],
    ]
    ground_truths = [
        [
            "Fibonacci: F(0)=0, F(1)=1, F(n)=F(n-1)+F(n-2).",
            "Commonly used in algorithm demonstrations.",
        ],
        [
            "Binary search requires a sorted array.",
            "Time complexity: O(log n), Space: O(1).",
        ],
    ]
    answers = [
        "The Fibonacci sequence is defined by F(0)=0, F(1)=1, and F(n)=F(n-1)+F(n-2) for n>1.",
        "Binary search finds an element in a sorted array in O(log n) time by repeatedly halving the search space.",
    ]

    return evaluate_retrieval_pipeline(questions, retrieved_contexts, ground_truths, answers)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scores = run_demo_evaluation()
    if scores:
        print("RAGAS Evaluation Scores:")
        for metric, score in scores.items():
            print(f"  {metric}: {score:.4f}")
    else:
        print("RAGAS evaluation skipped (not available or failed).")