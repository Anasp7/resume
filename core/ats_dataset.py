"""
ATS Dataset Collector
=====================
Silently collects (input, output) resume pairs during normal system usage.
- SHA-256 deduplication: identical resumes are never stored twice
- Zero impact on pipeline: all writes fire-and-forget in background thread
- Manual activation: does NOT affect evaluator output until user enables MODEL_ACTIVE

Usage:
    from core.ats_dataset import collect_sample
    collect_sample(input_text, output_text, role, evaluator_response)
"""

from __future__ import annotations
import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("smart_resume.ats_dataset")

# ── Storage paths ──────────────────────────────────────────────────────────────
_DATA_DIR         = Path(__file__).parent.parent / "data"
_DATASET_PATH     = _DATA_DIR / "resume_dataset.jsonl"
_PARSER_PATH      = _DATA_DIR / "parser_dataset.jsonl"
_HASHES_PATH      = _DATA_DIR / "seen_hashes.txt"
_WRITE_LOCK       = threading.Lock()

# ── Deduplication ──────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Normalize text before hashing to catch near-duplicates."""
    t = text.lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]", "", t)
    return t.strip()


def _hash(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


def _load_seen_hashes() -> set:
    if not _HASHES_PATH.exists():
        return set()
    try:
        return set(_HASHES_PATH.read_text(encoding="utf-8").splitlines())
    except Exception:
        return set()


def _save_hash(h: str) -> None:
    with _HASHES_PATH.open("a", encoding="utf-8") as f:
        f.write(h + "\n")


def is_duplicate(resume_text: str) -> bool:
    """Returns True if this resume has already been collected."""
    h = _hash(resume_text)
    return h in _load_seen_hashes()


# ── Label derivation ───────────────────────────────────────────────────────────

def _derive_labels(
    input_text: str,
    output_text: str,
    role: str,
    response: dict,
) -> dict:
    """
    Automatically derive multi-task labels from evaluator response.
    No manual annotation needed.
    """
    from core.evaluator import ats_score

    in_ats  = ats_score(input_text)
    out_ats = ats_score(output_text) if output_text else in_ats

    input_score  = response.get("job_match_analysis", {}).get("weighted_score", 0) or 0
    output_score = out_ats["score"] / 100.0

    matched   = response.get("job_match_analysis", {}).get("matched_skills", []) or []
    required  = response.get("job_match_analysis", {}).get("required_skills", []) or (
        response.get("job_match_analysis", {}).get("missing_skills", []) or []
    )
    total_req = len(matched) + len(required)
    skill_match = round(len(matched) / total_req, 3) if total_req else 0.0

    verdict = str(response.get("job_match_analysis", {}).get("verdict", "")).lower()
    role_fit_map = {"strong": "Strong", "moderate": "Moderate", "weak": "Weak"}
    role_fit = role_fit_map.get(verdict, "Moderate")

    # hallucination_risk: 1 if factual evaluation flags issues
    fe = response.get("factual_evaluation", {})
    hallu_risk = 1 if "low" in str(fe.get("confidence", "")).lower() else 0

    improvement_delta = round(output_score - (input_score / 100.0 if input_score > 1 else input_score), 3)

    return {
        "ats_parseable":        1 if in_ats["score"] >= 60 else 0,
        "content_quality":      round(output_score, 3),
        "skill_match_potential": skill_match,
        "hallucination_risk":   hallu_risk,
        "improvement_delta":    improvement_delta,
        "role_fit":             role_fit,
    }


def _derive_meta(input_text: str, output_text: str, response: dict) -> dict:
    from core.evaluator import run_ats_checks
    flags = [i["issue"][:40] for i in run_ats_checks(input_text)]
    return {
        "ats_flags":      flags,
        "weighted_score": response.get("job_match_analysis", {}).get("weighted_score", 0),
        "llm_verdict":    response.get("job_match_analysis", {}).get("verdict", ""),
        "word_count_in":  len(input_text.split()),
        "word_count_out": len(output_text.split()) if output_text else 0,
    }


# ── Core collection function ───────────────────────────────────────────────────

def _write_sample(record: dict, path: Path) -> None:
    """Append one JSONL record, thread-safe."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_sample(
    input_text:  str,
    output_text: str,
    role:        str,
    response:    Optional[dict] = None,
) -> None:
    """
    Silently collect one training sample in the background.
    Call this from the evaluator pipeline — it never blocks or raises.
    """
    def _run():
        try:
            if not input_text or len(input_text.strip()) < 100:
                return

            h = _hash(input_text)
            seen = _load_seen_hashes()
            if h in seen:
                logger.debug("ats_dataset: duplicate skipped (hash=%s...)", h[:12])
                return

            resp = response or {}
            labels = _derive_labels(input_text, output_text or "", role, resp)
            meta   = _derive_meta(input_text, output_text or "", resp)

            record = {
                "id":             f"sha256:{h}",
                "input_resume":   input_text,
                "output_resume":  output_text or "",
                "target_role":    role or "",
                "labels":         labels,
                "meta":           meta,
            }

            _write_sample(record, _DATASET_PATH)
            _save_hash(h)

            count = sum(1 for _ in _DATASET_PATH.open(encoding="utf-8"))
            logger.info("ats_dataset: sample saved (total=%d)", count)

        except Exception as e:
            logger.debug("ats_dataset: silent error — %s", e)

    # Fire-and-forget: never blocks the pipeline
    threading.Thread(target=_run, daemon=True).start()


# ── Parser dataset collection ──────────────────────────────────────────────────

def collect_parse_sample(raw_text: str, parsed: dict) -> None:
    """
    Collect (raw_text → parsed_fields) pairs for parser self-training.
    Uses the same dedup index as the scoring dataset.
    """
    def _run():
        try:
            if not raw_text or len(raw_text.strip()) < 80:
                return

            h = _hash(raw_text)
            seen = _load_seen_hashes()
            if h in seen:
                return

            record = {
                "id":         f"sha256:{h}",
                "raw_text":   raw_text,
                "parsed":     parsed,
            }
            _write_sample(record, _PARSER_PATH)
            _save_hash(h)

        except Exception as e:
            logger.debug("ats_dataset(parser): silent error — %s", e)

    threading.Thread(target=_run, daemon=True).start()


# ── Dataset stats (for CLI / admin) ───────────────────────────────────────────

def dataset_stats() -> dict:
    """Return current dataset size. Used by the /api/score-resume endpoint."""
    try:
        n_score  = sum(1 for _ in _DATASET_PATH.open(encoding="utf-8")) if _DATASET_PATH.exists() else 0
        n_parser = sum(1 for _ in _PARSER_PATH.open(encoding="utf-8")) if _PARSER_PATH.exists() else 0
        n_hashes = len(_load_seen_hashes())
        return {
            "scoring_samples": n_score,
            "parser_samples":  n_parser,
            "unique_resumes":  n_hashes,
            "ready_to_train":  n_score >= 50,
        }
    except Exception:
        return {"scoring_samples": 0, "parser_samples": 0, "unique_resumes": 0, "ready_to_train": False}
