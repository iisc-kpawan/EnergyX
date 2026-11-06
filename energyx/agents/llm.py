"""Provider-agnostic LLM adapter.

One interface, several backends, plus a null backend so the whole pipeline runs
with no model configured. The deterministic planner rides on ``NullLLM`` and
doubles as the floor baseline in the results table -- the number that says what
the tools achieve without any agentic reasoning at all.

Adding a provider means adding a subclass, not touching agents.
"""
from __future__ import annotations

import json
import socket
import os
import re
import time
from dataclasses import dataclass
from typing import Any


#: A transient 500 from a model server must not read as a refusal to plan.
#: Six attempts, widening pauses, then the error is real.
#: A reasoning model's scratchpad. Not part of the response.
THINK = re.compile(r"<think>.*?</think>", re.S)

RETRIES = 6
RETRY_BACKOFF_S = 2.0
#: Ollama answers a concurrent request on a loaded model with HTTP 500
#: rather than queueing it. The failure is transient but can outlast a
#: short linear backoff, so the wait doubles: 2+4+8+16+32 = 62s in all.
RETRY_BACKOFF_CAP_S = 32.0


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_s: float = 0.0
    model: str = ""
    provider: str = ""

    def json(self) -> Any:
        """Parse a JSON object from the response, tolerating prose or fences.

        A reasoning model thinks out loud first. deepseek-r1 wraps that in
        <think>...</think>, and the thinking quotes the very JSON it is about
        to emit -- so the first "{" in the reply belongs to a sentence about
        the answer rather than to the answer, and the brace-span fallback
        swallowed the prose between them. Dropping the block is not a
        tolerance: what is inside it is explicitly not the response.
        """
        text = THINK.sub("", self.text or "").strip()
        # An unclosed block means the reply was cut off mid-thought; whatever
        # follows the last close is the only part that was ever an answer.
        if "<think>" in text:
            text = text.rsplit("</think>", 1)[-1].strip()
            text = text.split("<think>", 1)[0].strip() if "<think>" in text else text
        fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
        if fence:
            text = fence.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if 0 <= start < end:
                return json.loads(text[start:end + 1])
            raise


class LLM:
    """Base adapter."""

    provider = "base"

    def __init__(self, model: str = "", temperature: float = 0.0,
                 timeout: int = 120, **options):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.options = options

    @property
    def available(self) -> bool:
        return False

    @property
    def unavailable_reason(self) -> str:
        return "" if self.available else f"{self.provider}: not configured"

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"


class NullLLM(LLM):
    """No model. Forces the deterministic path and keeps runs reproducible."""

    provider = "null"

    @property
    def available(self) -> bool:
        return True

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        return LLMResponse(text="", provider=self.provider, model="none")


class OllamaLLM(LLM):
    """Local models through an Ollama server."""

    provider = "ollama"

    def __init__(self, model: str = "qwen2.5:14b", host: str | None = None,
                 num_ctx: int = 32768, think: bool | None = False,
                 # A planning response is a JSON plan: across 226 calls from
                 # three local models the largest was 2,410 tokens and the
                 # median 606. A small model that loses the thread does not
                 # produce a long plan, it produces an unbounded one -- on one
                 # question a 9B held the server for over ten minutes, and the
                 # client's timeout then retried the request behind the one it
                 # had abandoned. This ceiling is 3.4x the largest response
                 # ever observed, so it cannot truncate a real plan; what it
                 # truncates is a loop, and it turns a stalled run into one
                 # failed question.
                 num_predict: int = 8192, **kw):
        super().__init__(model=model, **kw)
        # OLLAMA_HOST is shared with the ollama CLI, which writes it without a
        # scheme ("127.0.0.1:11435"). Used verbatim that string builds a URL
        # urllib cannot open, and the failure surfaces as "unknown url type"
        # far from the setting that caused it.
        raw = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.host = raw if "://" in raw else f"http://{raw}"
        # A tool catalogue plus a plan does not fit an 8K window; a truncated
        # response is unparseable JSON, which reads as a model failure when it
        # is a context failure.
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        # Reasoning models emit a long preamble before the answer. Where the
        # server supports it, turn that off: it costs completion budget the
        # plan itself needs.
        self.think = think

    def installed_models(self) -> list[str]:
        import urllib.error
        import urllib.request
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as r:
                return [m.get("name", "") for m in json.loads(r.read()).get("models", [])]
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            return []

    @property
    def available(self) -> bool:
        """The endpoint answering is not enough -- the model must be pulled.

        A running Ollama with no models would otherwise report as available and
        then fail every request, which reads as a model answering badly rather than
        as a model that was never there.
        """
        models = self.installed_models()
        if not models:
            return False
        base = self.model.split(":")[0]
        return any(m == self.model or m.split(":")[0] == base for m in models)

    @property
    def unavailable_reason(self) -> str:
        if self.available:
            return ""
        models = self.installed_models()
        if not models:
            return (f"ollama at {self.host} has no models pulled "
                    f"(try: ollama pull {self.model})")
        return (f"model {self.model!r} not pulled; available: "
                f"{', '.join(models[:5])}")

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        import urllib.error
        import urllib.request
        payload = {"model": self.model, "prompt": prompt, "stream": False,
                   "options": {"temperature": self.temperature,
                               "num_ctx": self.num_ctx,
                               "num_predict": self.num_predict}}
        if self.think is not None:
            payload["think"] = self.think
        if system:
            payload["system"] = system
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        started = time.time()
        # Ollama returns a 500 intermittently -- most often while a model is
        # loading or being swapped. The planner treats any exception as "the
        # model produced no usable plan" and quietly falls back to the
        # rule-based planner, so a transient server error reads downstream as
        # a model that chose not to answer: zero tokens, a rule-based route,
        # and a table column that is not measuring the model at all, which
        # shows up as scores moving between runs for no visible reason.
        # Retried rather than surfaced,
        # because a retry is what a human would do and the alternative is a
        # silent downgrade.
        last: Exception | None = None
        for attempt in range(RETRIES):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read())
                break
            except urllib.error.HTTPError as exc:
                # "HTTP Error 500" on its own names nothing. Ollama puts the
                # reason in the body -- a context overflow, a model that failed
                # to load -- and without it a persistent failure is
                # indistinguishable from a transient one.
                try:
                    detail = exc.read().decode("utf-8", "replace")[:300]
                except Exception:                           # noqa: BLE001
                    detail = ""
                last = RuntimeError(
                    f"HTTP {exc.code} from {self.provider}"
                    + (f": {detail}" if detail else ""))
                if attempt == RETRIES - 1:
                    raise last from exc
                time.sleep(min(RETRY_BACKOFF_S * 2 ** attempt,
                                RETRY_BACKOFF_CAP_S))
                continue
            except Exception as exc:                        # noqa: BLE001
                last = exc
                if attempt == RETRIES - 1:
                    raise
                time.sleep(min(RETRY_BACKOFF_S * 2 ** attempt,
                                RETRY_BACKOFF_CAP_S))
        return LLMResponse(
            text=body.get("response", ""),
            prompt_tokens=body.get("prompt_eval_count", 0),
            completion_tokens=body.get("eval_count", 0),
            duration_s=time.time() - started,
            model=self.model, provider=self.provider)


class OpenAICompatibleLLM(LLM):
    """Any OpenAI-compatible /chat/completions endpoint."""

    provider = "openai_compatible"

    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", **kw):
        super().__init__(model=model, **kw)
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.api_key_env = api_key_env

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    @property
    def unavailable_reason(self) -> str:
        return "" if self.available else f"${self.api_key_env} is not set"

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        import urllib.request
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps({"model": self.model, "messages": messages,
                             "temperature": self.temperature}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        started = time.time()
        # A hosted endpoint rations requests, and a sweep of sixty questions
        # is exactly the shape of traffic that trips the ration: on a free
        # tier the answer to "too many requests" is to wait, not to fail the
        # question. The server usually says how long to wait, and its number
        # beats a guess. Anything that is not a rate limit or an outage is
        # raised straight away -- a bad key will not fix itself in 32 seconds.
        body = None
        for attempt in range(RETRIES):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read())
                break
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", "replace")[:300]
                except Exception:                           # noqa: BLE001
                    detail = ""
                failure = RuntimeError(
                    f"HTTP {exc.code} from {self.provider}"
                    + (f": {detail}" if detail else ""))
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    raise failure from exc
                if attempt == RETRIES - 1:
                    raise failure from exc
                try:
                    wait = float(exc.headers.get("Retry-After") or 0)
                except (TypeError, ValueError):
                    wait = 0.0
                time.sleep(max(wait, min(RETRY_BACKOFF_S * 2 ** attempt,
                                         RETRY_BACKOFF_CAP_S)))
            except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
                # A read that times out is the same transient as a 503: the
                # request reached a busy host and the answer did not come back
                # in time. Only the HTTP errors were being retried, so three
                # questions were lost outright to "TimeoutError: The read
                # operation timed out" while the quota they needed was still
                # unspent.
                if attempt == RETRIES - 1:
                    raise RuntimeError(
                        f"{type(exc).__name__} from {self.provider} after "
                        f"{RETRIES} attempts: {exc}") from exc
                time.sleep(min(RETRY_BACKOFF_S * 2 ** attempt,
                               RETRY_BACKOFF_CAP_S))
        usage = body.get("usage", {})
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        # A reasoning model bills tokens it does not return. Gemini reports
        # prompt 8, completion 1 and total 110: the 101 it thought with are
        # charged, count against the context, and are the reason a "cheap"
        # model can cost more per plan than a local one. Counting only the two
        # named fields understated a run tenfold, so the remainder is carried
        # where it was generated rather than dropped.
        total = usage.get("total_tokens", 0)
        if total > prompt + completion:
            completion = total - prompt
        return LLMResponse(
            text=body["choices"][0]["message"]["content"],
            prompt_tokens=prompt,
            completion_tokens=completion,
            duration_s=time.time() - started,
            model=self.model, provider=self.provider)


class AnthropicLLM(LLM):
    """Anthropic Messages API."""

    provider = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5", api_key_env: str = "ANTHROPIC_API_KEY",
                 max_tokens: int = 4096, **kw):
        super().__init__(model=model, **kw)
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    @property
    def unavailable_reason(self) -> str:
        return "" if self.available else f"${self.api_key_env} is not set"

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        import urllib.request
        payload: dict = {"model": self.model, "max_tokens": self.max_tokens,
                         "temperature": self.temperature,
                         "messages": [{"role": "user", "content": prompt}]}
        if system:
            payload["system"] = system
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "x-api-key": self.api_key or "",
                     "anthropic-version": "2023-06-01"})
        started = time.time()
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read())
        usage = body.get("usage", {})
        return LLMResponse(
            text="".join(b.get("text", "") for b in body.get("content", [])),
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            duration_s=time.time() - started,
            model=self.model, provider=self.provider)


class GeminiLLM(OpenAICompatibleLLM):
    """Gemini through its OpenAI-compatible endpoint.

    Google serves a /chat/completions surface that speaks the same dialect, so
    the only differences from any other OpenAI-compatible host are the base
    URL, the environment variable holding the key, and the fact that the
    response bills reasoning tokens the base class already accounts for.
    """

    provider = "gemini"

    def __init__(self, model: str = "gemini-3.8-flash",
                 base_url: str | None = None,
                 api_key_env: str = "GEMINI_API_KEY", **kw):
        super().__init__(
            model=model, api_key_env=api_key_env,
            base_url=(base_url or os.environ.get("GEMINI_BASE_URL")
                      or "https://generativelanguage.googleapis.com/v1beta/openai"),
            **kw)


PROVIDERS = {"null": NullLLM, "ollama": OllamaLLM,
             "openai": OpenAICompatibleLLM, "anthropic": AnthropicLLM,
             "gemini": GeminiLLM}


def make_llm(provider: str = "null", **kwargs) -> LLM:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; valid: {sorted(PROVIDERS)}")
    return PROVIDERS[provider](**kwargs)
