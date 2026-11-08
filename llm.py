"""Model-agnostic chat client. Zero dependencies (stdlib only) so it runs anywhere, incl. Colab.

Swap the model with two settings:  provider  and  model   (CLI flags or env PROMO_PROVIDER / PROMO_MODEL).
  gemini     -> Google Gemini API            key: GEMINI_API_KEY        (free tier in AI Studio)
  anthropic  -> Claude API                    key: ANTHROPIC_API_KEY
  openai     -> any OpenAI-compatible API     key: OPENAI_API_KEY, optional OPENAI_BASE_URL (Groq, OpenRouter, ...)
"""
import json, os, re, time, urllib.request, urllib.error

DEFAULT_MODELS = {"gemini": "gemini-2.5-flash", "anthropic": "claude-sonnet-4-5", "openai": "gpt-4o-mini"}


class LLM:
    def __init__(self, provider=None, model=None, min_interval=None):
        self.provider = (provider or os.getenv("PROMO_PROVIDER") or "gemini").lower()
        self.model = model or os.getenv("PROMO_MODEL") or DEFAULT_MODELS[self.provider]
        # free tiers are rate limited (Gemini flash ~10 req/min); space calls out instead of failing
        self.min_interval = float(os.getenv("PROMO_MIN_INTERVAL", min_interval if min_interval is not None else
                                            (7 if self.provider == "gemini" else 0.5)))
        self._last = 0.0
        self.calls = 0

    def _key(self):
        name = {"gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}[self.provider]
        k = os.getenv(name) or (os.getenv("GOOGLE_API_KEY") if self.provider == "gemini" else None)
        if not k:
            raise SystemExit(f"Missing API key: set the {name} environment variable (or use --mode heuristic).")
        return k

    def _post(self, url, headers, body):
        data = json.dumps(body).encode()
        for attempt in range(6):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0: time.sleep(wait)
            self._last = time.time()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    self.calls += 1
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:300]
                if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                    time.sleep(min(60, 8 * (attempt + 1))); continue
                raise RuntimeError(f"{self.provider} HTTP {e.code}: {msg}")
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < 5: time.sleep(5 * (attempt + 1)); continue
                raise RuntimeError(f"network error: {e}")

    def complete(self, system: str, user: str, temperature=0.2) -> str:
        key = self._key()
        if self.provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            out = self._post(url, {"x-goog-api-key": key}, {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"}})
            return "".join(p.get("text", "") for p in out["candidates"][0]["content"]["parts"])
        if self.provider == "anthropic":
            out = self._post("https://api.anthropic.com/v1/messages",
                             {"x-api-key": key, "anthropic-version": "2023-06-01"},
                             {"model": self.model, "max_tokens": 8000, "temperature": temperature, "system": system,
                              "messages": [{"role": "user", "content": user}]})
            return "".join(b.get("text", "") for b in out["content"])
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        out = self._post(f"{base}/chat/completions", {"Authorization": f"Bearer {key}"},
                         {"model": self.model, "temperature": temperature,
                          "response_format": {"type": "json_object"},
                          "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        return out["choices"][0]["message"]["content"]

    def complete_json(self, system, user, tries=3):
        last = ""
        for _ in range(tries):
            last = self.complete(system, user)
            txt = re.sub(r"^```(?:json)?|```$", "", last.strip(), flags=re.M).strip()
            try:
                return json.loads(txt), last
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", txt, re.S)
                if m:
                    try: return json.loads(m.group(0)), last
                    except json.JSONDecodeError: pass
                user += "\n\nYour last reply was not valid JSON. Reply with ONLY the JSON object."
        raise RuntimeError("model did not return valid JSON: " + last[:200])
