"""Swappable LLM backends. Standard library only (no SDKs to install).

Pick one with --backend / PROMO_BACKEND:
  gemini   Google AI Studio (free key). Env: GEMINI_API_KEY, model via --model (default gemini-3.8-flash)
  openai   Any OpenAI-compatible endpoint: OpenAI, Groq, OpenRouter, Together, a local Ollama...
           Env: OPENAI_API_KEY, OPENAI_BASE_URL (default https://api.openai.com/v1)
  offline  No model. The pipeline outputs the rule-based picks only.

Each backend takes a prompt and returns a parsed JSON dict. Adding a new
provider = one class with a `complete_json(prompt) -> dict` method.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    pass


def _post(url: str, body: dict, headers: dict, timeout=120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def extract_json(text: str) -> dict:
    """Models sometimes wrap JSON in ```json fences or add chatter. Pull out the object."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise LLMError(f"No JSON object in model reply: {text[:200]}")
    return json.loads(text[start:end + 1])


class Backend:
    name = "base"
    model = ""

    def __init__(self, model: str | None = None, temperature: float = 0.3, max_retries: int = 3):
        self.model = model or self.model
        self.temperature = temperature
        self.max_retries = max_retries

    def _call(self, prompt: str) -> str:
        raise NotImplementedError

    def complete_json(self, prompt: str) -> dict:
        delay = 5.0
        last = None
        for attempt in range(self.max_retries):
            try:
                return extract_json(self._call(prompt))
            except urllib.error.HTTPError as e:
                last = e
                body = e.read().decode(errors="replace")[:300]
                if e.code == 429 and ("limit: 0" in body or "free_tier" in body.lower() and "quota" in body.lower() and attempt >= 1):
                    raise LLMError(f"HTTP 429 no free quota for {self.model}: {body[:200]}") from e
                if e.code in (429, 500, 502, 503, 504):  # rate limit / overload: back off and retry
                    print(f"  [{self.name}] {self.model}: HTTP {e.code}, retrying in {delay:.0f}s ... {body[:120]!r}")
                    time.sleep(delay)
                    delay = min(delay * 2, 90)
                    continue
                raise LLMError(f"HTTP {e.code}: {body}") from e
            except (json.JSONDecodeError, LLMError) as e:
                last = e
                print(f"  [{self.name}] bad JSON ({e}); retrying ...")
                time.sleep(2)
            except urllib.error.URLError as e:
                last = e
                time.sleep(delay)
        raise LLMError(f"{self.name} failed after {self.max_retries} tries: {last}")


class GeminiBackend(Backend):
    name = "gemini"
    model = "gemini-3.8-flash"
    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.key:
            raise LLMError("Set GEMINI_API_KEY (free key from https://aistudio.google.com/apikey)")
        extra = os.environ.get("PROMO_FALLBACK_MODELS", "")
        self.fallbacks = [m.strip() for m in extra.split(",") if m.strip()]
        self._discovered = False

    def _discover_fallbacks(self):
        """Ask the API which Flash models this key can use (only when the main model keeps failing)."""
        self._discovered = True
        try:
            req = urllib.request.Request(f"{self.BASE}/models?pageSize=200", headers={"x-goog-api-key": self.key})
            with urllib.request.urlopen(req, timeout=30) as r:
                models = json.loads(r.read().decode()).get("models", [])
        except Exception as e:  # discovery is best-effort
            print(f"  [gemini] could not list models: {e}")
            return
        names = [m["name"].split("/", 1)[-1] for m in models
                 if "generateContent" in m.get("supportedGenerationMethods", []) and "flash" in m["name"]
                 and not any(x in m["name"] for x in ("image", "tts", "audio", "live", "embed"))]
        def rank(n):  # stable 'latest'/'lite' aliases first, previews and experiments last
            risky = any(x in n for x in ("preview", "exp", "omni", "thinking"))
            return (risky, "latest" not in n, "lite" not in n, [-ord(c) for c in n])
        names = sorted(set(names) - {self.model} - set(self.fallbacks), key=rank)
        self.fallbacks += names[:6]
        print(f"  [gemini] fallback models available: {', '.join(self.fallbacks) or 'none'}")

    def complete_json(self, prompt: str) -> dict:
        """Try the main model; if it stays overloaded, switch to a fallback and keep using it."""
        tried = []
        while True:
            try:
                return super().complete_json(prompt)
            except LLMError as e:
                if not any(c in str(e) for c in ("503", "500", "502", "504", "429", "403", "404", "overloaded", "UNAVAILABLE")):
                    raise
                tried.append(self.model)
                if not self._discovered:
                    self._discover_fallbacks()
                nxt = next((m for m in self.fallbacks if m not in tried), None)
                if not nxt:
                    raise
                print(f"  [gemini] {self.model} is overloaded, switching to {nxt}")
                self.model = nxt

    def _call(self, prompt: str) -> str:
        url = f"{self.BASE}/models/{self.model}:generateContent"
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature, "responseMimeType": "application/json"},
        }
        data = _post(url, body, {"x-goog-api-key": self.key})
        try:
            return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError):
            raise LLMError(f"Unexpected Gemini reply: {json.dumps(data)[:300]}")


class OpenAICompatBackend(Backend):
    name = "openai"
    model = "gpt-4o-mini"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.key = os.environ.get("OPENAI_API_KEY", "")
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    def _call(self, prompt: str) -> str:
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        data = _post(f"{self.base}/chat/completions", body, headers)
        return data["choices"][0]["message"]["content"]


BACKENDS = {"gemini": GeminiBackend, "openai": OpenAICompatBackend}


def get_backend(name: str, model: str | None = None) -> Backend | None:
    if name in ("offline", "none", ""):
        return None
    if name not in BACKENDS:
        raise LLMError(f"Unknown backend '{name}'. Choose from: {', '.join(BACKENDS)} or offline")
    return BACKENDS[name](model=model)
