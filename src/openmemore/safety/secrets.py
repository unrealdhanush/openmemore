"""Secret / PII filter applied to everything written to the store.

Memories are rejected outright (`find_secret`); captured conversation exchanges
are redacted instead (`redact`), so a pasted key doesn't cost the whole record.
Neither ever returns the matched text, so the secret can't be echoed back to the
model or into logs.
"""

from __future__ import annotations

import re

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        # The whole block, or everything after the header if the END line is missing.
        re.compile(
            r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"
            r"(?:[\s\S]*?-----END (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----|[\s\S]*)"
        ),
        "contains a private key",
    ),
    (re.compile(r"\bsk-(?:proj-|ant-|live-|test-)?[A-Za-z0-9_-]{20,}"), "contains an API key (sk-...)"),
    (re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}"), "contains a Stripe API key"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"), "contains a GitHub token"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}"), "contains a GitHub token"),
    (re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), "contains an AWS access key"),
    (re.compile(r"(?i)\baws_secret_access_key\b\s*[:=]"), "contains an AWS secret key"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "contains a Google API key"),
    (re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"), "contains a Slack token"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), "contains a JWT"),
    (
        re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
        "contains what looks like a US Social Security number",
    ),
    (
        re.compile(r"(?i)\b(?:password|passwd|passphrase|pwd)\b\s*(?:is\s*)?[:=]\s*\S+"),
        "contains what looks like a password",
    ),
    (
        re.compile(
            r"(?i)\b(?:api[_ -]?key|secret[_ -]?key|client[_ -]?secret|access[_ -]?token|auth[_ -]?token|bearer)"
            r"\b\s*[:=]?\s*[A-Za-z0-9_\-./+=]{16,}"
        ),
        "contains what looks like a credential",
    ),
]

# 13-19 digits, optionally grouped by single spaces or dashes.
_CARD_CANDIDATE = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


_CARD_REASON = "contains what looks like a payment card number"

REDACTED = "[REDACTED]"


def _is_card(candidate: str) -> bool:
    digits = re.sub(r"\D", "", candidate)
    return 13 <= len(digits) <= 19 and _luhn_ok(digits)


def find_secret(text: str) -> str | None:
    """The reason `text` must not be stored, or None if it looks safe."""
    for pattern, reason in _PATTERNS:
        if pattern.search(text):
            return reason
    if any(_is_card(m.group()) for m in _CARD_CANDIDATE.finditer(text)):
        return _CARD_REASON
    return None


def redact(text: str) -> tuple[str, list[str]]:
    """`text` with every secret replaced by [REDACTED], and the reasons (deduplicated)."""
    reasons: list[str] = []
    for pattern, reason in _PATTERNS:
        text, n = pattern.subn(REDACTED, text)
        if n and reason not in reasons:
            reasons.append(reason)

    def card(m: re.Match[str]) -> str:
        if not _is_card(m.group()):
            return m.group()
        if _CARD_REASON not in reasons:
            reasons.append(_CARD_REASON)
        return REDACTED

    return _CARD_CANDIDATE.sub(card, text), reasons
