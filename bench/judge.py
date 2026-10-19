"""Optional judged answers (SPEC.md M7.6): a model answers each question from the
retrieved context only, and a second call judges the answer against the reference.
Both go through OpenMCP's model client (openmcp.llm), so any Open Responses or Chat
Completions endpoint works. Retrieval metrics stay the headline; this shows whether
what was retrieved is enough to answer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bench.datasets import Question
from bench.runner import Retrieved
from openmcp import llm

ANSWER_INSTRUCTIONS = (
    "Answer the question using only the conversation excerpts provided. Be brief. If the excerpts don't "
    "contain the answer, say you don't know."
)
JUDGE_INSTRUCTIONS = (
    "You grade answers. Given a question, a reference answer and a response, reply with exactly one word: "
    "correct if the response gives the same answer as the reference (wording may differ), otherwise incorrect. "
    "If the reference says the question can't be answered, the response is correct only if it says it doesn't know."
)


def answerer(model: llm.Model, k: int) -> Callable[[Question, Retrieved], dict[str, Any]]:
    def answer(question: Question, retrieved: Retrieved) -> dict[str, Any]:
        excerpts = "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(retrieved.texts[:k])) or "(none)"
        date = question.date.date().isoformat() if question.date else "unknown"
        try:
            response = llm.complete(model, ANSWER_INSTRUCTIONS,
                                    f"Today's date: {date}\n\nExcerpts:\n{excerpts}\n\nQuestion: {question.text}")
            verdict = llm.complete(model, JUDGE_INSTRUCTIONS,
                                   f"Question: {question.text}\nReference answer: {question.answer}\nResponse: {response}")
        except llm.ModelError as e:
            return {"judge_error": str(e)[:200]}
        word = verdict.strip().lower().strip(".!\"'")
        return {"correct": 1.0 if word.startswith("correct") else 0.0}
    return answer
