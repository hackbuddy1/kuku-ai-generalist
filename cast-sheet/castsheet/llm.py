"""Swappable LLM backends (standard library only).

  gemini  Google AI Studio free key. Env GEMINI_API_KEY; model via CAST_MODEL (default gemini-3.8-flash).
          If a model is overloaded or has no free quota, it asks the API for other Flash models and switches.
  openai  Any OpenAI-compatible endpoint (OpenAI, Groq, OpenRouter, Ollama). Env OPENAI_API_KEY, OPENAI_BASE_URL.

Every backend exposes: generate(prompt, json_mode=True, audio=None) -> str
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    pass


def _post(url, body, headers, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b == -1:
        raise LLMError(f"No JSON in reply: {text[:200]}")
    return json.loads(text[a:b + 1])


class Backend:
    name, model = "base", ""

    def __init__(self, model=None, temperature=0.3, max_retries=3):
        self.model = model or self.model
        self.temperature = temperature
        self.max_retries = max_retries

    def _call(self, prompt, json_mode, audio):
        raise NotImplementedError

    def generate(self, prompt, json_mode=True, audio=None) -> str:
        delay, last = 4.0, None
        for attempt in range(self.max_retries):
            try:
                return self._call(prompt, json_mode, audio)
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:400]
                last = f"HTTP {e.code}: {body}"
                if e.code == 429 and "limit: 0" in body:
                    raise LLMError(f"HTTP 429 no free quota for {self.model}") from e
                if e.code in (429, 500, 502, 503, 504):
                    print(f"  [{self.name}] {self.model}: HTTP {e.code}, retry in {delay:.0f}s")
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise LLMError(last) from e
            except urllib.error.URLError as e:
                last = str(e)
                time.sleep(delay)
        raise LLMError(f"{self.name}:{self.model} failed after {self.max_retries} tries: {last}")

    def generate_json(self, prompt, audio=None) -> dict:
        for _ in range(2):
            try:
                return extract_json(self.generate(prompt, True, audio))
            except (json.JSONDecodeError, LLMError) as e:
                if "HTTP" in str(e):
                    raise
                last = e
        raise LLMError(f"Model did not return valid JSON: {last}")


class GeminiBackend(Backend):
    name, model = "gemini", "gemini-3.8-flash"
    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.key:
            raise LLMError("Set GEMINI_API_KEY (free key from https://aistudio.google.com/apikey)")
        self.fallbacks, self._discovered = [], False

    def _discover(self):
        self._discovered = True
        try:
            req = urllib.request.Request(f"{self.BASE}/models?pageSize=200", headers={"x-goog-api-key": self.key})
            with urllib.request.urlopen(req, timeout=30) as r:
                models = json.loads(r.read().decode()).get("models", [])
        except Exception as e:
            print(f"  [gemini] could not list models: {e}")
            return
        names = {m["name"].split("/", 1)[-1] for m in models
                 if "generateContent" in m.get("supportedGenerationMethods", []) and "flash" in m["name"]
                 and not any(x in m["name"] for x in ("image", "tts", "live", "embed"))}
        rank = lambda n: (any(x in n for x in ("preview", "exp", "omni")), "latest" not in n, "lite" not in n, n)
        self.fallbacks = sorted(names - {self.model}, key=rank)[:6]

    def generate(self, prompt, json_mode=True, audio=None):
        tried = []
        while True:
            try:
                return super().generate(prompt, json_mode, audio)
            except LLMError as e:
                if not any(c in str(e) for c in ("429", "500", "502", "503", "504", "404")):
                    raise
                tried.append(self.model)
                if not self._discovered:
                    self._discover()
                nxt = next((m for m in self.fallbacks if m not in tried), None)
                if not nxt:
                    raise
                print(f"  [gemini] {self.model} unavailable, switching to {nxt}")
                self.model = nxt

    def _call(self, prompt, json_mode, audio):
        parts = [{"text": prompt}]
        if audio:
            data, mime = audio
            parts.append({"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}})
        cfg = {"temperature": self.temperature}
        if json_mode:
            cfg["responseMimeType"] = "application/json"
        data = _post(f"{self.BASE}/models/{self.model}:generateContent",
                     {"contents": [{"role": "user", "parts": parts}], "generationConfig": cfg},
                     {"x-goog-api-key": self.key})
        try:
            return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError):
            raise LLMError(f"Unexpected Gemini reply: {json.dumps(data)[:300]}")


class OpenAICompatBackend(Backend):
    name, model = "openai", "gpt-4o-mini"

    def _call(self, prompt, json_mode, audio):
        if audio:
            raise LLMError("Voice input needs the gemini backend")
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        key = os.environ.get("OPENAI_API_KEY", "")
        body = {"model": self.model, "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        data = _post(f"{base}/chat/completions", body, {"Authorization": f"Bearer {key}"} if key else {})
        return data["choices"][0]["message"]["content"]


BACKENDS = {"gemini": GeminiBackend, "openai": OpenAICompatBackend}


def get_backend(name=None, model=None) -> Backend:
    name = name or os.environ.get("CAST_BACKEND", "gemini")
    model = model or os.environ.get("CAST_MODEL")
    if name not in BACKENDS:
        raise LLMError(f"Unknown backend {name}")
    return BACKENDS[name](model=model)
