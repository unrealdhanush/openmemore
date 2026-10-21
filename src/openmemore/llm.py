"""A language model OpenMemore calls itself, through the open standard for LLM
APIs rather than any provider's SDK:

- Open Responses (openresponses.org, spec 2026-04-24): POST {endpoint}/responses with
  `model`, `instructions` and `input`; the answer is in `output` items of type
  "message", as "output_text" parts. Implemented by OpenAI, Ollama, LM Studio, vLLM,
  OpenRouter, Hugging Face, and others.
- OpenAI-style Chat Completions, POST {endpoint}/chat/completions, for servers that
  don't speak Open Responses yet (some providers' compatibility endpoints).

The endpoint is a base URL such as http://127.0.0.1:11434/v1. A local one (loopback)
is called with proxy settings ignored and without following redirects, so nothing
leaves the machine; anything else needs the user's permission first (openmemore.cli).
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

import httpx

Api = Literal["responses", "chat-completions"]
TIMEOUT_S = 300.0
MAX_OUTPUT_TOKENS = 4096


class ModelError(Exception):
    """The model couldn't be reached or didn't answer; the message is for the user."""


def is_local(endpoint: str) -> bool:
    """Whether the endpoint is on this machine: localhost or a loopback address. Names
    aren't resolved, so a name that happens to point here still counts as remote."""
    host = (urlparse(endpoint).hostname or "").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def host(endpoint: str) -> str:
    return urlparse(endpoint).hostname or endpoint


@dataclass(frozen=True)
class Model:
    endpoint: str
    model: str
    api: Api = "responses"
    api_key: str | None = None
    timeout: float = TIMEOUT_S

    def client(self, transport: httpx.BaseTransport | None = None) -> httpx.Client:
        local = is_local(self.endpoint)
        return httpx.Client(
            timeout=self.timeout,
            follow_redirects=False,  # a redirect could lead off the machine
            trust_env=not local,  # HTTP(S)_PROXY would route even a local call elsewhere
            transport=transport,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
        )


def _error(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:300] or response.reason_phrase
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    return str(error or body)[:300]


def _responses_text(body: Any) -> str:
    if not isinstance(body, dict):
        raise ModelError("the endpoint didn't return a response object")
    if body.get("status") == "failed" or body.get("error"):
        error = body.get("error") or {}
        raise ModelError(f"the model failed: {error.get('message') if isinstance(error, dict) else error}")
    parts = [
        part["text"]
        for item in body.get("output") or []
        if isinstance(item, dict) and item.get("type") == "message"
        for part in item.get("content") or []
        if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str)
    ]
    return "".join(parts)


def _chat_text(body: Any) -> str:
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ModelError("the endpoint didn't return a chat completion") from None
    if isinstance(content, list):  # some servers return content parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content if isinstance(content, str) else ""


def complete(model: Model, instructions: str, prompt: str, *, transport: httpx.BaseTransport | None = None) -> str:
    """The model's text answer to `prompt`, following `instructions`."""
    base = model.endpoint.rstrip("/")
    if model.api == "responses":
        url, payload = f"{base}/responses", {
            "model": model.model, "instructions": instructions, "input": prompt,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        }
    else:
        url, payload = f"{base}/chat/completions", {
            "model": model.model, "max_tokens": MAX_OUTPUT_TOKENS,
            "messages": [{"role": "system", "content": instructions}, {"role": "user", "content": prompt}],
        }
    try:
        with model.client(transport) as client:
            response = client.post(url, json=payload)
    except httpx.HTTPError as e:
        raise ModelError(f"couldn't reach {host(model.endpoint)}: {type(e).__name__}: {e}") from None
    if response.is_redirect:
        raise ModelError(f"{host(model.endpoint)} redirected elsewhere; set the endpoint to where it points")
    if response.status_code >= 400:
        hint = " (if this server doesn't support Open Responses, set api = \"chat-completions\")" \
            if model.api == "responses" and response.status_code in (404, 405) else ""
        raise ModelError(f"{host(model.endpoint)} answered {response.status_code}: {_error(response)}{hint}")
    try:
        body = response.json()
    except ValueError:
        raise ModelError(f"{host(model.endpoint)} didn't return JSON") from None
    return _responses_text(body) if model.api == "responses" else _chat_text(body)
