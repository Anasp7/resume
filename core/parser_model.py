"""
Parser Model — Learned Fallback for Unusual Resume Formats
===========================================================
The rule-based parser (core/parser.py, core/smart_parser.py) is always PRIMARY.
This model activates ONLY when the rule parser fails:
  - name is None on a non-empty resume
  - experience is [] on a resume with 200+ words

Activation: MANUAL — only loads if models/parser_model/ exists.
If model directory is missing, this module is a no-op.

Training data: core/ats_dataset.py collect_parse_sample() accumulates
(raw_text → parsed_fields) pairs from every parser run into data/parser_dataset.jsonl.
Run train_parser_model.py when ready.
"""

from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("smart_resume.parser_model")

MODEL_DIR = Path(__file__).parent.parent / "models" / "parser_model"

# ── Field patterns (used both for feature extraction and as CRF fallback) ──────

_FIELD_PATTERNS = {
    "email":   r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    "phone":   r"(\+?\d[\d\s\-\(\)]{8,}\d|\b\d{10}\b)",
    "github":  r"github\.com/(\S+)",
    "linkedin":r"linkedin\.com/in/(\S+)",
}

_SECTION_HEADERS = [
    "experience", "work history", "employment",
    "education", "projects", "skills", "certifications",
    "summary", "objective", "profile",
]


def _rule_extract_basic(text: str) -> dict:
    """Fast regex extraction — same logic as rule parser, used for feature alignment."""
    out = {}
    for field, pat in _FIELD_PATTERNS.items():
        m = re.search(pat, text, re.I)
        out[field] = m.group(0) if m else None

    # Name heuristic: first non-empty line that looks like a name
    for line in text.splitlines():
        line = line.strip()
        if line and re.match(r"^[A-Z][a-z]+\s+[A-Z][a-z]+", line):
            out["name"] = line
            break
    else:
        out["name"] = None

    return out


def rule_parser_failed(parsed: dict, raw_text: str) -> bool:
    """
    Returns True if the rule-based parser produced an empty/invalid result
    on a resume that clearly has content.
    """
    if len(raw_text.strip().split()) < 100:
        return False  # too short to be a real resume — don't try ML
    name_missing = not parsed.get("name")
    exp_empty    = not parsed.get("experience") and not parsed.get("projects")
    return name_missing or exp_empty


class ParserModel:
    """
    Wraps the trained spaCy NER model.
    All methods return None / empty dict if model is not loaded.
    """

    _instance: Optional["ParserModel"] = None

    def __init__(self):
        self._nlp   = None
        self._ready = False

    @classmethod
    def get(cls) -> "ParserModel":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance

    def _load(self):
        if not MODEL_DIR.exists():
            logger.debug("ParserModel: model dir not found — running in collection-only mode")
            return
        try:
            import spacy
            self._nlp   = spacy.load(str(MODEL_DIR))
            self._ready = True
            logger.info("ParserModel: loaded from %s", MODEL_DIR)
        except Exception as e:
            logger.warning("ParserModel: load failed — %s", e)

    def is_ready(self) -> bool:
        return self._ready

    def parse(self, raw_text: str) -> Optional[dict]:
        """
        Run the learned NER parser on raw resume text.
        Returns None if model is not trained/loaded.
        """
        if not self._ready or not self._nlp:
            return None

        try:
            doc = self._nlp(raw_text[:5000])  # cap input length

            result: dict = {
                "name":       None,
                "email":      None,
                "phone":      None,
                "github":     None,
                "linkedin":   None,
                "skills":     [],
                "experience": [],
                "projects":   [],
                "education":  [],
            }

            for ent in doc.ents:
                label = ent.label_.lower()
                val   = ent.text.strip()
                if label == "person" and not result["name"]:
                    result["name"] = val
                elif label == "skill":
                    result["skills"].append(val)
                elif label == "org":
                    result.setdefault("_orgs", []).append(val)

            # Merge regex basics for contact fields
            basics = _rule_extract_basic(raw_text)
            for field in ("email", "phone", "github", "linkedin"):
                if not result.get(field) and basics.get(field):
                    result[field] = basics[field]

            return result

        except Exception as e:
            logger.debug("ParserModel.parse error — %s", e)
            return None


# ── Convenience: try ML parse only if rule parse failed ───────────────────────

def try_ml_parse_if_needed(rule_result: dict, raw_text: str) -> dict:
    """
    Called after the rule parser. If rule parser clearly failed,
    try the ML model as fallback. If ML model also not ready, return rule_result as-is.
    """
    if not rule_parser_failed(rule_result, raw_text):
        return rule_result

    pm = ParserModel.get()
    if not pm.is_ready():
        return rule_result

    ml_result = pm.parse(raw_text)
    if not ml_result:
        return rule_result

    # Merge: ML fills in missing fields from rule result
    merged = dict(rule_result)
    for key, val in ml_result.items():
        if not merged.get(key) and val:
            merged[key] = val
            logger.debug("ParserModel filled missing field: %s", key)

    return merged
