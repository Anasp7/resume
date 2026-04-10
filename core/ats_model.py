"""
ATS Scorer — Multi-Task Resume Intelligence Model
==================================================
6-task XGBoost model that scores resumes across quality dimensions.

Activation: MANUAL — only loads if models/ats_scorer.pkl exists.
If model file is missing, all predict() calls return None (zero system impact).

Tasks:
  - ats_parseable        (binary)
  - content_quality      (regression 0-1)
  - skill_match potential (regression 0-1)
  - hallucination_risk   (binary)
  - improvement_delta    (regression -1 to 1)
  - role_fit             (3-class: Weak/Moderate/Strong)
"""

from __future__ import annotations
import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("smart_resume.ats_model")

MODEL_PATH = Path(__file__).parent.parent / "models" / "ats_scorer.pkl"

# ── Feature extraction ─────────────────────────────────────────────────────────

_SECTION_PATTERNS = {
    "has_experience": r"\b(EXPERIENCE|WORK HISTORY|EMPLOYMENT)\b",
    "has_education":  r"\b(EDUCATION|DEGREE|UNIVERSITY)\b",
    "has_projects":   r"\b(PROJECTS?|PORTFOLIO)\b",
    "has_skills":     r"\b(SKILLS?|TECHNICAL|TECHNOLOGIES)\b",
    "has_summary":    r"\b(SUMMARY|OBJECTIVE|PROFILE)\b",
    "has_certs":      r"\b(CERTIFICATIONS?|CERTIFICATES?|AWARDS?)\b",
}

_QUALITY_PATTERNS = {
    "has_email":       r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}",
    "has_phone":       r"(\+?\d[\d\s\-\(\)]{8,}\d|\b\d{10}\b)",
    "has_github":      r"github\.com/\S+",
    "has_linkedin":    r"linkedin\.com/in/\S+",
    "has_metrics":     r"\b\d+\s*(%|percent|users?|clients?|ms|seconds?)\b",
    "has_action_verbs": r"\b(developed|engineered|designed|built|implemented|optimized|led|architected)\b",
}


def extract_features(input_text: str, output_text: str = "", role: str = "") -> list:
    """
    Extract a fixed-length feature vector from resume texts.
    Returns a list of floats suitable for XGBoost.
    """
    t_in  = input_text or ""
    t_out = output_text or t_in

    feats = []

    # Structural features (on input)
    for key, pat in _SECTION_PATTERNS.items():
        feats.append(1.0 if re.search(pat, t_in, re.I) else 0.0)

    # Quality signals (on output — optimized version)
    for key, pat in _QUALITY_PATTERNS.items():
        feats.append(1.0 if re.search(pat, t_out, re.I) else 0.0)

    # Word count features
    wc_in  = len(t_in.split())
    wc_out = len(t_out.split())
    feats.extend([
        min(wc_in / 500.0, 2.0),     # normalized, capped
        min(wc_out / 500.0, 2.0),
        min((wc_out - wc_in) / 200.0, 2.0),  # improvement delta in words
    ])

    # ATS flag count (on input)
    try:
        from core.evaluator import run_ats_checks
        flags = run_ats_checks(t_in)
        high   = sum(1 for f in flags if f["severity"] == "high")
        medium = sum(1 for f in flags if f["severity"] == "medium")
        feats.extend([len(flags) / 10.0, high / 5.0, medium / 5.0])
    except Exception:
        feats.extend([0.0, 0.0, 0.0])

    # Role keyword overlap
    if role:
        role_words = set(re.findall(r"\w+", role.lower()))
        out_words  = set(re.findall(r"\w+", t_out.lower()))
        overlap = len(role_words & out_words) / max(len(role_words), 1)
        feats.append(min(overlap, 1.0))
    else:
        feats.append(0.0)

    # Bullet point density (proxy for content richness)
    bullets = len(re.findall(r"^\s*[-•*]\s+\S", t_out, re.MULTILINE))
    feats.append(min(bullets / 20.0, 1.0))

    return feats


FEATURE_COUNT = (
    len(_SECTION_PATTERNS)    # 6
    + len(_QUALITY_PATTERNS)  # 6
    + 3                       # word counts
    + 3                       # ats flag counts
    + 1                       # role overlap
    + 1                       # bullet density
)  # = 20


# ── Model wrapper ──────────────────────────────────────────────────────────────

class ATSScorer:
    """
    Wraps 6 XGBoost sub-models.
    Loads lazily only if models/ats_scorer.pkl exists.
    All public methods are safe to call when untrained (return None).
    """

    _instance: Optional["ATSScorer"] = None

    def __init__(self):
        self._models: Optional[dict] = None
        self._tfidf = None
        self._ready = False

    @classmethod
    def get(cls) -> "ATSScorer":
        """Singleton loader — zero cost if model file is missing."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance

    def _load(self):
        if not MODEL_PATH.exists():
            logger.debug("ATSScorer: model file not found — running in collection-only mode")
            return
        try:
            import pickle
            with MODEL_PATH.open("rb") as f:
                bundle = pickle.load(f)
            self._models = bundle["models"]
            self._tfidf  = bundle["tfidf"]
            self._ready  = True
            logger.info("ATSScorer: loaded — %d tasks", len(self._models))
        except Exception as e:
            logger.warning("ATSScorer: load failed — %s", e)

    def is_ready(self) -> bool:
        return self._ready

    def predict(
        self,
        input_text:  str,
        output_text: str = "",
        role:        str = "",
    ) -> Optional[dict]:
        """
        Returns None if model is not yet trained/activated.
        Otherwise returns multi-label score dict.
        """
        if not self._ready:
            return None

        try:
            import numpy as np
            struct = extract_features(input_text, output_text, role)
            tfidf  = self._tfidf.transform([input_text]).toarray()[0]
            X      = np.array(struct + list(tfidf)).reshape(1, -1)

            result = {}
            for task, model in self._models.items():
                pred = model.predict(X)[0]
                if task in ("ats_parseable", "hallucination_risk"):
                    result[task] = int(round(pred))
                elif task == "role_fit":
                    classes = model.classes_
                    result[task] = str(classes[int(pred)])
                else:
                    result[task] = round(float(pred), 3)

            # Human-readable improvement tag
            delta = result.get("improvement_delta", 0)
            result["improvement_tag"] = (
                f"+{int(delta*100)}% improvement" if delta > 0.05
                else "minimal change" if abs(delta) <= 0.05
                else f"{int(delta*100)}% regression"
            )
            result["model_version"] = "v1"
            return result

        except Exception as e:
            logger.debug("ATSScorer.predict error — %s", e)
            return None


# ── Module-level singleton ────────────────────────────────────────────────────
# Loaded once at import; no-op if model file missing
_scorer = ATSScorer.get()


def predict(input_text: str, output_text: str = "", role: str = "") -> Optional[dict]:
    """Convenience function — returns None if model not activated."""
    return _scorer.predict(input_text, output_text, role)


def is_model_ready() -> bool:
    return _scorer.is_ready()
