"""
Smart Resume — LLM Client  v3.0
================================
Magic tricks used:
  1. llama-3.1-8b-instant for generation (30k tok/min vs 6k for 70b = 5x headroom)
  2. llama-3.3-70b-versatile for evaluation/doubt questions (needs reasoning)
  3. Exponential backoff: on 429, wait and retry same key before rotating
  4. In-memory prompt cache: identical prompt → skip API call entirely
  5. Key rotation: up to 6 Groq keys, Together AI as final fallback
  6. Retry-after header: respect server's own wait time on 429

Add keys to .env:
  GROQ_API_KEY=...            (primary — required)
  GROQ_API_KEY_2=...          (optional extra Groq key)
  GROQ_API_KEY_3=...          (optional)
  GROQ_API_KEY_4=...          (optional)
  TOGETHER_API_KEY=...        (free at api.together.xyz — fallback)
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import re
import time
from typing import Optional

import httpx

logger = logging.getLogger("smart_resume.llm_client")

# ── Models ───────────────────────────────────────────────────────────────────
GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"

# 8b: 30,000 tok/min — used for ALL generation tasks (fast + high limit)
GROQ_FAST      = "llama-3.1-8b-instant"
# 70b: 6,000 tok/min — used ONLY for doubt questions (needs reasoning)
GROQ_SMART     = "llama-3.3-70b-versatile"

# ── In-memory cache ───────────────────────────────────────────────────────────
# Key: md5(model + prompt) → response string
# Saves tokens when same prompt is retried (e.g. page refresh)
_CACHE: dict[str, str] = {}

def _cache_key(model: str, prompt: str, system_msg: str) -> str:
    return hashlib.md5(f"{model}||{system_msg}||{prompt}".encode()).hexdigest()


def _all_groq_keys(primary: str) -> list[str]:
    keys = [primary] + [os.getenv(f"GROQ_API_KEY_{i}", "") for i in range(2, 7)]
    return [k.strip() for k in keys if k and k.strip()]


def _groq_call(
    key: str,
    model: str,
    messages: list[dict],
    body_base: dict,
    timeout: int = 60,
) -> tuple[Optional[str], str]:
    """
    Single Groq call with exponential backoff on 429.
    Returns (content, error_msg).
    """
    MAX_RETRIES = 2   # only 2 retries per key — then rotate fast
    wait = 3          # 3s first wait, 6s second → max 9s per key before rotating

    for attempt in range(MAX_RETRIES):
        try:
            resp = httpx.post(
                GROQ_URL,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"},
                json={**body_base, "model": model, "messages": messages},
                timeout=timeout,
            )

            if resp.status_code == 429:
                # Respect Retry-After header. Groq often asks for 20-30s. 
                # Capping at 8s was causing "thundering herd" re-triggering.
                retry_after = resp.headers.get("retry-after", "")
                actual_wait = min(int(retry_after), 45) if retry_after.isdigit() else wait
                
                # Add 1-2s jitter to avoid synchronized retries
                import random
                actual_wait += random.uniform(0.5, 2.0)

                logger.warning(
                    "429 Rate Limit (model %s, attempt %d/%d) — waiting %.1fs",
                    model, attempt + 1, MAX_RETRIES, actual_wait
                )
                time.sleep(actual_wait)
                wait = min(wait * 2.5, 45) # Faster growth for wait time
                continue

            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            logger.info("Groq %s OK (%d chars)", model, len(content))
            return content, ""

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                time.sleep(wait)
                wait = min(wait * 2, 30)
                continue
            elif e.response.status_code in (413, 500, 503):
                logger.warning("Groq call failed with status %d: %s", e.response.status_code, e.response.text)
                return None, f"HTTP {e.response.status_code}: {e.response.text}"
            return None, f"HTTP {e.response.status_code}: {e.response.text}"
        except Exception as e:
            logger.warning("Groq call error (attempt %d): %s", attempt + 1, e)
            if attempt == MAX_RETRIES - 1:
                return None, str(e)

    return None, "Rate-limited after retries"


def call_llm(
    prompt: str,
    api_key: str,
    *,
    json_mode: bool = True,
    max_tokens: int = 500,
    smart: bool = False,      # False → 8b (30k tok/min). True → 70b (reserved, unused)
    system_msg: str | None = None,
    use_cache: bool = True,
    force_gemini: bool = False,
) -> str:
    """
    Main LLM call with full reliability stack:
      force_gemini → cache → Groq 8b/70b with backoff → key rotation → Gemini → Ollama → error
    """
    model = "gemini-2.5-flash" if force_gemini else (GROQ_SMART if smart else GROQ_FAST)

    if system_msg is None:
        system_msg = (
            "You are Smart Resume, a structured resume evaluation engine. "
            "Always respond with a single valid JSON object. "
            "Never add explanations, markdown, or text outside the JSON."
            if json_mode else
            "You are a professional resume writer. "
            "Output ONLY what is requested. No markdown. No explanation. No preamble."
        )

    # ── Cache check ───────────────────────────────────────────────────────────
    ck = _cache_key(model, prompt, system_msg)
    if use_cache and ck in _CACHE:
        logger.info("Cache HIT — skipping LLM call")
        return _CACHE[ck]

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": prompt},
    ]
    body_base: dict = {
        "max_tokens":  max_tokens,
        "temperature": 0.15,
    }
    if json_mode:
        body_base["response_format"] = {"type": "json_object"}

    errors: list[str] = []

    # ── 0. Force Gemini ──────────────────────────────────────────────────────
    if force_gemini:
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not gemini_key:
            raise RuntimeError("GEMINI_API_KEY requested but not found in environment")
        # Proceed to Gemini block below by skipping Groq loop
    else:
        # ── 1. Groq — try all keys with backoff ──────────────────────────────
        for idx, key in enumerate(_all_groq_keys(api_key)):
            content, error_msg = _groq_call(key, model, messages, body_base)
            if content is not None:
                if use_cache:
                    _CACHE[ck] = content
                return content
            errors.append(f"Groq key {idx+1}: {error_msg}")

    # ── 2. Gemini Fallback ───────────────────────────────────────────────────
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        try:
            logger.info("Groq exhausted, falling back to Gemini API")
            
            gemini_body = {
                "systemInstruction": {"parts": [{"text": system_msg}]},
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": 8192,
                    "temperature": 0.15,
                }
            }
            if json_mode:
                gemini_body["generationConfig"]["responseMimeType"] = "application/json"
                
            resp = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}",
                headers={"Content-Type": "application/json"},
                json=gemini_body, timeout=120,
            )
            
            resp.raise_for_status()
            
            # Gemini response parsing
            res_json = resp.json()
            if "candidates" in res_json and len(res_json["candidates"]) > 0:
                content = res_json["candidates"][0]["content"]["parts"][0]["text"]
                logger.info("Gemini OK (%d chars). Prefix: %s", len(content), content[:200])
                if res_json["candidates"][0].get("finishReason") != "STOP":
                    logger.warning("Gemini finished with reason: %s. Content: %s", res_json["candidates"][0].get("finishReason"), content)
                if use_cache:
                    _CACHE[ck] = content
                return content
            else:
                logger.warning("Gemini empty or blocked. Raw JSON: %s", res_json)
                errors.append("Gemini API: Empty response or blocked by safety settings")
        except Exception as e:
            errors.append(f"Gemini API: {e}")
            logger.warning("Gemini failed: %s", e)

    # ── 3. Ollama Fallback ───────────────────────────────────────────────────
    ollama_enabled = os.getenv("OLLAMA_ENABLED", "false").lower() == "true"
    if ollama_enabled:
        from core.ollama_pipeline import call_ollama
        content = call_ollama(prompt, system_msg=system_msg, max_tokens=max_tokens)
        if content:
            logger.info("Ollama OK (%d chars)", len(content))
            return content
        errors.append("Ollama: Failed directly")

    # ── 4. All failed ─────────────────────────────────────────────────────────
    raise RuntimeError(
        "All LLM providers reached their rate limits or failed.\n"
        f"Details: {'; '.join(errors)}\n"
        "Fix: Add GEMINI_API_KEY (free 15 requests/min) or GROQ_API_KEY_4 to your .env file.\n"
        "Get free Gemini key at: aistudio.google.com/app/apikey"
    )


def call_llm_json(
    prompt: str,
    api_key: str,
    max_tokens: int = 500,
    system_msg: str | None = None,
    smart: bool = True,
) -> dict:
    """Call LLM and parse JSON. Uses 8b by default (all tasks via smart=False)."""
    raw = call_llm(
        prompt, api_key,
        json_mode=True, max_tokens=max_tokens,
        smart=smart, system_msg=system_msg,
    )
    raw = re.sub(r"```json\s*|```\s*", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("JSONDecodeError Details: %s", e)
        logger.error("Raw LLM response causing crash:\n%s\n---END RAW---", raw)
        with open("failed_llm_response.txt", "w", encoding="utf-8") as f:
            f.write(raw)
        raise


def call_llm_text(
    prompt: str,
    api_key: str,
    max_tokens: int = 400,
    system_msg: str | None = None,
    force_gemini: bool = False,
) -> str:
    """Call LLM for plain text (generation). Uses 8b fast model unless forced to Gemini."""
    return call_llm(
        prompt, api_key,
        json_mode=False, max_tokens=max_tokens,
        smart=False, system_msg=system_msg,
        force_gemini=force_gemini,
    )



def call_llm_json_gemini(
    prompt: str,
    api_key: str,
    max_tokens: int = 2000,
    system_msg: str | None = None,
) -> dict:
    """Optimized helper to force Gemini for complex JSON generation."""
    raw = call_llm(
        prompt, api_key,
        json_mode=True, max_tokens=max_tokens,
        force_gemini=True, system_msg=system_msg
    )
    raw = re.sub(r"```json\s*|```\s*", "", raw).strip()
    return json.loads(raw)