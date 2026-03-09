"""
Smart Resume — Shared LLM Client
==================================
Single place for all LLM calls. Every module imports from here.
Provider order: Groq key 1 → Groq key 2/3/4/5 → Together AI → raise

Add keys to .env:
  GROQ_API_KEY=...           (primary)
  GROQ_API_KEY_2=...         (optional extra free Groq key)
  GROQ_API_KEY_3=...         (optional)
  TOGETHER_API_KEY=...       (optional — free at api.together.xyz)
"""
from __future__ import annotations
import json
import logging
import os
import re

import httpx

logger = logging.getLogger("smart_resume.llm_client")

# ── Endpoints & models ────────────────────────────────────────────────────────
GROQ_URL      = "https://api.groq.com/openai/v1/chat/completions"
TOGETHER_URL  = "https://api.together.xyz/v1/chat/completions"

GROQ_MODEL    = "llama-3.3-70b-versatile"   # evaluation & doubt questions
GROQ_FAST     = "llama-3.1-8b-instant"      # resume generation (higher limit)
TOGETHER_MODEL= "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"


def _all_groq_keys(primary: str) -> list[str]:
    keys = [primary] + [os.getenv(f"GROQ_API_KEY_{i}", "") for i in range(2, 7)]
    return [k.strip() for k in keys if k and k.strip()]


def call_llm(
    prompt: str,
    api_key: str,
    *,
    json_mode: bool = True,
    max_tokens: int = 2000,
    fast: bool = False,       # True → use lighter model with higher rate limits
    system_msg: str | None = None,
) -> str:
    """
    Call LLM with automatic key rotation and provider fallback.

    Raises RuntimeError only if ALL providers fail — never returns empty string
    or silently falls back to rule-based logic.
    """
    if system_msg is None:
        system_msg = (
            "You are Smart Resume, a structured resume evaluation engine. "
            "Always respond with a single valid JSON object. "
            "Never add explanations, markdown, or text outside the JSON."
            if json_mode else
            "You are a professional resume writer. "
            "Write clean plain text resumes only. No JSON. No markdown."
        )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": prompt},
    ]
    body_base: dict = {
        "max_tokens":  max_tokens,
        "temperature": 0.1,
        "messages":    messages,
    }
    if json_mode:
        body_base["response_format"] = {"type": "json_object"}

    errors: list[str] = []

    # ── 1. Try all Groq keys ─────────────────────────────────────────────────
    model = GROQ_FAST if fast else GROQ_MODEL
    for idx, key in enumerate(_all_groq_keys(api_key)):
        try:
            body = {**body_base, "model": model}
            resp = httpx.post(
                GROQ_URL,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"},
                json=body, timeout=60,
            )
            if resp.status_code == 429:
                logger.warning("Groq key %d rate limited — trying next", idx + 1)
                errors.append(f"Groq key {idx+1}: 429")
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            logger.info("Groq key %d OK (%d chars)", idx + 1, len(content))
            return content
        except httpx.HTTPStatusError:
            raise   # non-429 HTTP errors propagate immediately
        except Exception as e:
            logger.warning("Groq key %d error: %s", idx + 1, e)
            errors.append(f"Groq key {idx+1}: {e}")

    # ── 2. Try Together AI ───────────────────────────────────────────────────
    together_key = os.getenv("TOGETHER_API_KEY", "").strip()
    if together_key:
        try:
            body = {**body_base, "model": TOGETHER_MODEL}
            resp = httpx.post(
                TOGETHER_URL,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {together_key}"},
                json=body, timeout=90,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            logger.info("Together AI OK (%d chars)", len(content))
            return content
        except Exception as e:
            logger.warning("Together AI failed: %s", e)
            errors.append(f"Together AI: {e}")
    else:
        logger.debug("No TOGETHER_API_KEY set")

    # ── 3. All failed — raise with clear instructions ────────────────────────
    raise RuntimeError(
        "All LLM providers are rate-limited or unavailable.\n"
        f"Errors: {'; '.join(errors)}\n"
        "Solutions:\n"
        "  1. Wait 1 minute and try again\n"
        "  2. Add GROQ_API_KEY_2=... to .env (get free at console.groq.com)\n"
        "  3. Add TOGETHER_API_KEY=... to .env (get free at api.together.xyz)"
    )


def call_llm_json(prompt: str, api_key: str, max_tokens: int = 2000,
                  system_msg: str | None = None) -> dict:
    """Convenience: call LLM and parse JSON response."""
    raw = call_llm(prompt, api_key, json_mode=True, max_tokens=max_tokens,
                   system_msg=system_msg)
    raw = re.sub(r"```json\s*|```\s*", "", raw).strip()
    return json.loads(raw)


def call_llm_text(prompt: str, api_key: str, max_tokens: int = 1500,
                  system_msg: str | None = None) -> str:
    """Convenience: call LLM for plain text (resume generation)."""
    return call_llm(prompt, api_key, json_mode=False, max_tokens=max_tokens,
                    fast=True, system_msg=system_msg)