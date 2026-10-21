import pytest

from openmemore.safety.secrets import find_secret, redact

# Synthetic values only: shaped like real secrets, valid for nothing.
SECRETS = {
    "openai": "my key is sk-proj-AbCdEfGhIjKlMnOpQrStUvWx1234",
    "anthropic": "use sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWx",
    "github": "token ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789",
    "github_pat": "github_pat_11ABCDEFG0123456789_abcdefghijklmnop",
    "aws": "AKIAIOSFODNN7EXAMPLE is the access key",
    "aws_secret": "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG",
    "jwt": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    "private_key": "-----BEGIN RSA PRIVATE KEY-----\nMIIEow...",
    "openssh_key": "-----BEGIN OPENSSH PRIVATE KEY-----",
    "card": "card 4111 1111 1111 1111 exp 12/29",
    "card_dashes": "4111-1111-1111-1111",
    "ssn": "SSN 123-45-6789",
    "password": "wifi password: hunter2",
    "password_eq": "PASSWORD=correcthorse",
    "credential": "api_key: 9f8e7d6c5b4a39281706f5e4d3c2b1a0",
    "slack": "xoxb-1234567890-abcdefghij",
}

SAFE = [
    "Prefers Python and terse answers.",
    "Decided to use passwordless login (passkeys) for the app.",
    "Rotates API keys every 90 days and stores them in 1Password.",
    "Phone extension 4111, order number 1234567890123",  # 13 digits, fails Luhn
    "Born 1990-04-12; moved to Berlin in 2019.",
    "Uses the sk-learn library a lot",  # 'sk-' prefix but not key-shaped
    "Bearer of good news: the launch went fine.",
]


@pytest.mark.parametrize("name", SECRETS)
def test_detects_secrets(name):
    assert find_secret(SECRETS[name]) is not None


@pytest.mark.parametrize("text", SAFE)
def test_allows_ordinary_text(text):
    assert find_secret(text) is None


@pytest.mark.parametrize("name", SECRETS)
def test_reason_never_echoes_the_secret(name):
    secret_part = max(SECRETS[name].split(), key=len)
    assert secret_part not in find_secret(SECRETS[name])


def test_redact_replaces_secrets_and_keeps_the_rest():
    text, reasons = redact("Deploy failed with key sk-proj-AbCdEfGhIjKlMnOpQrStUvWx1234, how do I fix it?")
    assert text == "Deploy failed with key [REDACTED], how do I fix it?"
    assert reasons == ["contains an API key (sk-...)"]


def test_redact_removes_whole_private_key_block():
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\nabc123\n-----END RSA PRIVATE KEY-----"
    text, _ = redact(f"here:\n{pem}\nwhy won't ssh accept it?")
    assert "MIIE" not in text and "abc123" not in text
    assert text == "here:\n[REDACTED]\nwhy won't ssh accept it?"


def test_redact_private_key_without_end_line_drops_the_remainder():
    text, _ = redact("key: -----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXk")
    assert text == "key: [REDACTED]"


def test_redact_cards_only_when_luhn_valid():
    text, reasons = redact("card 4111 1111 1111 1111, order 1234567890123")
    assert text == "card [REDACTED], order 1234567890123"
    assert reasons == ["contains what looks like a payment card number"]


@pytest.mark.parametrize("text", SAFE)
def test_redact_leaves_ordinary_text_alone(text):
    assert redact(text) == (text, [])


@pytest.mark.parametrize("name", SECRETS)
def test_redact_covers_everything_find_secret_catches(name):
    text, reasons = redact(SECRETS[name])
    assert reasons and find_secret(text) is None
