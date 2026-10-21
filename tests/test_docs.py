"""Docs that quote the code must match it (SPEC.md M8.5: kept current by tests)."""

from pathlib import Path

from openmemore import consolidate, llm
from openmemore.models import CONVERSATION_BATCH_CHARS

DOCS = Path(__file__).resolve().parent.parent / "docs"


def test_the_consolidation_endpoint_page_matches_the_code():
    page = (DOCS / "consolidation-endpoint.md").read_text(encoding="utf-8")
    assert consolidate.INSTRUCTIONS in page
    assert f'"max_output_tokens": {llm.MAX_OUTPUT_TOKENS}' in page and f'"max_tokens": {llm.MAX_OUTPUT_TOKENS}' in page
    assert f"{int(llm.TIMEOUT_S)} seconds" in page
    assert f"up to {consolidate.RELATED_MEMORIES} of your memories" in page
    assert f"{CONVERSATION_BATCH_CHARS:,} characters" in page
    assert consolidate.KEY_ENV in page and consolidate.KEY_FILE in page
