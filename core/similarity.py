"""
Smart Resume — Similarity Engine & Backend Decision Logic (Phase 1)
=================================================================
Responsibilities:
  - Compute semantic similarity score (0–100) between resume and JD
  - Detect job domain from JD keywords
  - Select template based on candidate profile
  - Decide needs_optimization flag
  - Parse JD into structured form

The AI prompt NEVER sees raw math — it only receives the decisions made here.
"""

from __future__ import annotations

import re
import logging
from typing import Optional

from core.schemas import (
    Domain,
    ParsedJobDescription,
    ParsedResume,
    TemplateType,
)

logger = logging.getLogger("smart_resume.similarity")

# ─── Optional heavy import ───────────────────────────────────────────────────

# sentence-transformers: auto-enabled when package is installed.
# Set SMART_RESUME_USE_ST=0 to explicitly force Jaccard fallback.
# Model: all-MiniLM-L6-v2 (~90MB, downloads once from HuggingFace on first use)
import os as _os
_HAS_ST = False
_FORCE_DISABLED = _os.getenv("SMART_RESUME_USE_ST") == "0"

if not _FORCE_DISABLED:
    try:
        from sentence_transformers import SentenceTransformer, util as st_util
        import torch
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _HAS_ST = True
        logger.info("SentenceTransformer loaded: all-MiniLM-L6-v2 (semantic similarity active)")
    except ImportError:
        logger.info("sentence-transformers not installed — using Jaccard fallback (pip install sentence-transformers to enable)")
    except Exception as e:
        logger.warning("sentence-transformers failed to load (%s) — using Jaccard fallback", e)
else:
    logger.info("Similarity: Jaccard mode forced (SMART_RESUME_USE_ST=0)")


def get_similarity_mode() -> str:
    """Returns current similarity engine mode for health/status reporting."""
    return "sentence-transformers (all-MiniLM-L6-v2)" if _HAS_ST else "Jaccard skill-overlap"


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_similarity(resume_preprocessed: str, jd_preprocessed: str) -> float:
    """
    Returns a float in [0, 100].
    Primary: sentence-transformers cosine similarity.
    Fallback: Jaccard overlap on token sets.
    """
    if _HAS_ST:
        return _st_similarity(resume_preprocessed, jd_preprocessed)
    return _jaccard_similarity(resume_preprocessed, jd_preprocessed)


def _st_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity via sentence-transformers, scaled to 0–100."""
    emb_a = _model.encode(text_a, convert_to_tensor=True)
    emb_b = _model.encode(text_b, convert_to_tensor=True)
    cosine = float(st_util.pytorch_cos_sim(emb_a, emb_b)[0][0])
    # cosine in [-1, 1] → map to [0, 100]
    score = (cosine + 1) / 2 * 100
    logger.info("ST cosine similarity: %.2f → score: %.2f", cosine, score)
    return round(score, 2)


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Phase 3 upgrade: Skill-overlap similarity.
    Compares extracted tech terms from resume vs JD — not raw word bags.
    This gives meaningful scores even for short resumes.

    Components:
      40% — skill term overlap (most important)
      35% — weighted recall (resume coverage of JD terms)
      25% — token jaccard (general vocabulary)
    """
    from core.parser import KNOWN_TECH_TERMS

    def tokenize(text: str) -> set:
        return set(
            t.lower() for t in re.findall(r"[a-zA-Z0-9#+.]+", text)
            if len(t) > 2 or t.lower() in {"ml", "ai", "c#", "c++", "go", "r", "c"}
        )

    def extract_skills(text: str) -> set:
        tl = text.lower()
        return {
            term for term in KNOWN_TECH_TERMS
            if re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", tl)
        }

    # Component 1: skill term overlap
    skills_a = extract_skills(text_a)
    skills_b = extract_skills(text_b)

    if skills_a and skills_b:
        skill_intersection = skills_a & skills_b
        # Recall-oriented: how many JD skills does resume cover?
        skill_recall  = len(skill_intersection) / len(skills_b) * 100 if skills_b else 0
        skill_jaccard = len(skill_intersection) / len(skills_a | skills_b) * 100 if (skills_a | skills_b) else 0
        skill_score   = 0.65 * skill_recall + 0.35 * skill_jaccard
    else:
        skill_score = 0.0

    # Component 2: token recall (resume covers JD vocabulary)
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)
    STOP = {"the","and","for","with","you","are","our","have","will","this",
            "that","from","your","they","must","also","role","work","team","able",
            "good","strong","plus","years","year","experience","knowledge","skills"}
    tokens_a -= STOP
    tokens_b -= STOP

    if tokens_a and tokens_b:
        token_intersection = tokens_a & tokens_b
        token_recall  = len(token_intersection) / len(tokens_b) * 100 if tokens_b else 0
        token_jaccard = len(token_intersection) / len(tokens_a | tokens_b) * 100 if (tokens_a | tokens_b) else 0
        token_score   = 0.7 * token_recall + 0.3 * token_jaccard
    else:
        token_score = 0.0

    # Blend components
    if skills_a and skills_b:
        score = 0.55 * skill_score + 0.45 * token_score
    else:
        # No tech terms extracted — fall back to token only
        score = token_score

    score = min(score, 99.0)
    logger.info(
        "Similarity: skill_score=%.1f token_score=%.1f final=%.1f "
        "(resume_skills=%d jd_skills=%d overlap=%d)",
        skill_score, token_score, score,
        len(skills_a), len(skills_b),
        len(skills_a & skills_b) if skills_a and skills_b else 0
    )
    return round(score, 2)


# ─────────────────────────────────────────────────────────────────────────────
# JOB DESCRIPTION PARSER
# ─────────────────────────────────────────────────────────────────────────────

_DOMAIN_KEYWORDS: dict[Domain, list[str]] = {
    Domain.BACKEND:  [
        "backend", "api", "rest", "graphql", "microservices", "server", "database",
        "sql", "nosql", "fastapi", "django", "flask", "spring", "node",
    ],
    Domain.ML: [
        "machine learning", "deep learning", "nlp", "computer vision", "llm",
        "pytorch", "tensorflow", "model training", "inference", "mlops",
        "feature engineering", "data scientist",
    ],
    Domain.DATA: [
        "data engineer", "data pipeline", "etl", "spark", "hadoop", "airflow",
        "data warehouse", "bigquery", "redshift", "dbt", "analytics",
    ],
    Domain.FRONTEND: [
        "frontend", "react", "vue", "angular", "typescript", "html", "css",
        "ui/ux", "responsive", "next.js", "web developer",
    ],
    Domain.DEVOPS: [
        "devops", "ci/cd", "kubernetes", "docker", "terraform", "aws", "gcp",
        "azure", "infrastructure", "sre", "site reliability", "helm",
    ],
}

_SKILL_EXTRACTION_PATTERN = re.compile(
    r"(?:required|preferred|must have|nice to have|experience with|proficiency in)"
    r"\s*:?\s*([^\n.]+)",
    re.IGNORECASE,
)

_EXPERIENCE_PATTERN = re.compile(
    r"(\d+)\+?\s*(?:to\s*\d+\s*)?years?\s*(?:of\s*)?(?:professional\s*)?experience",
    re.IGNORECASE,
)


def parse_job_description(raw_jd: str, target_role: str) -> ParsedJobDescription:
    """
    Extract structured info from raw JD text.
    Returns ParsedJobDescription.
    """
    domain           = _detect_domain(raw_jd)
    required_skills  = _extract_skills(raw_jd, required=True)
    preferred_skills = _extract_skills(raw_jd, required=False)
    responsibilities = _extract_responsibilities(raw_jd)
    req_years        = _extract_required_years(raw_jd)

    return ParsedJobDescription(
        raw_text=raw_jd,
        target_role=target_role,
        detected_domain=domain,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        required_experience_years=req_years,
        key_responsibilities=responsibilities,
    )


def _detect_domain(text: str) -> Domain:
    text_lower = text.lower()
    scores: dict[Domain, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else Domain.UNKNOWN


def _extract_skills(text: str, required: bool) -> list[str]:
    """
    Extract skills from JD.
    required=True  → looks for 'required / must have' sections.
    required=False → looks for 'preferred / nice to have'.
    Uses known tech term scanning as primary method.
    """
    from core.parser import KNOWN_TECH_TERMS

    scope_keywords = (
        ["required", "must have", "minimum qualifications"]
        if required else
        ["preferred", "nice to have", "bonus", "plus"]
    )

    text_lower = text.lower()
    # Check if relevant scope section exists
    scope_found = any(kw in text_lower for kw in scope_keywords)

    if scope_found:
        # Try to narrow scope
        relevant_chunk = _extract_scope_chunk(text_lower, scope_keywords)
    else:
        relevant_chunk = text_lower

    found = [
        term for term in KNOWN_TECH_TERMS
        if re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", relevant_chunk)
    ]
    return sorted(set(found))


def _extract_scope_chunk(text: str, keywords: list[str]) -> str:
    """Return text starting from the first keyword match up to ~800 chars."""
    for kw in keywords:
        idx = text.find(kw)
        if idx != -1:
            return text[idx: idx + 800]
    return text


def _extract_responsibilities(text: str) -> list[str]:
    """Extract bullet-point style responsibilities from JD."""
    lines = text.splitlines()
    responsibilities = []
    in_resp_section = False

    for line in lines:
        stripped = line.strip()
        if re.search(r"\b(responsibilit|duties|you will|what you.ll do|role)\b",
                     stripped, re.IGNORECASE):
            in_resp_section = True
            continue
        if in_resp_section:
            if re.search(r"\b(requirement|qualification|about us|benefit)\b",
                         stripped, re.IGNORECASE):
                break  # end of responsibilities section
            if stripped.startswith(("-", "•", "*", "·")) or (
                stripped and stripped[0].isdigit() and stripped[1:3] in (". ", ") ")
            ):
                responsibilities.append(stripped.lstrip("-•*·0123456789.) ").strip())

    return responsibilities[:15]  # cap at 15


def _extract_required_years(text: str) -> Optional[float]:
    match = _EXPERIENCE_PATTERN.search(text)
    if match:
        return float(match.group(1))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND DECISION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

OPTIMIZATION_THRESHOLD = 72.0   # similarity score below this → needs optimization


def decide_template(parsed_resume: ParsedResume) -> TemplateType:
    """
    Rule-based template selection (Section 9 logic).
    Pure math — no AI reasoning here.
    """
    yoe  = parsed_resume.years_of_experience
    proj = parsed_resume.project_count
    exp  = parsed_resume.experience_count
    skills_count = len(parsed_resume.skills)

    if yoe < 1:
        return TemplateType.FRESHER_ACADEMIC

    if proj > exp:
        return TemplateType.PROJECT_FOCUSED

    if exp >= proj and exp > 0:
        return TemplateType.EXPERIENCE_FOCUSED

    if skills_count > 10 and proj == 0 and exp == 0:
        return TemplateType.SKILL_FOCUSED

    # Default
    return TemplateType.PROJECT_FOCUSED


def decide_needs_optimization(similarity_score: float) -> bool:
    """Score < OPTIMIZATION_THRESHOLD → needs_optimization = True."""
    return similarity_score < OPTIMIZATION_THRESHOLD


def compute_weighted_score(
    similarity_score: float,
    matched_skills: list,
    required_skills: list,
    years_of_experience: float,
    required_experience_years: Optional[float],
) -> dict:
    """
    Phase 3 — Deterministic weighted scoring formula.
    Replaces LLM-opinion verdict with consistent math.

    Weights:
      40% — skill overlap (matched / required)
      35% — semantic similarity score
      25% — experience match

    Returns: { score: 0-100, verdict: Strong/Moderate/Weak, breakdown: {...} }
    """
    # Component 1: skill overlap
    if required_skills:
        req_lower     = {s.lower() for s in required_skills}
        matched_lower = {s.lower() for s in matched_skills}
        skill_overlap = len(req_lower & matched_lower) / len(req_lower) * 100
    else:
        skill_overlap = similarity_score  # fall back to sim if no JD skills

    # Component 2: similarity (already 0-100)
    sim_component = similarity_score

    # Component 3: experience match
    if required_experience_years and required_experience_years > 0:
        exp_ratio     = min(years_of_experience / required_experience_years, 1.0)
        exp_component = exp_ratio * 100
    else:
        exp_component = 75.0  # no experience requirement → neutral

    # Weighted blend
    final_score = (
        0.40 * skill_overlap +
        0.35 * sim_component +
        0.25 * exp_component
    )
    final_score = round(min(final_score, 99.0), 1)

    # Verdict thresholds
    if final_score >= 68:
        verdict = "Strong"
    elif final_score >= 38:
        verdict = "Moderate"
    else:
        verdict = "Weak"

    logger.info(
        "Weighted score: skill=%.1f sim=%.1f exp=%.1f → final=%.1f (%s)",
        skill_overlap, sim_component, exp_component, final_score, verdict
    )

    return {
        "score":   final_score,
        "verdict": verdict,
        "breakdown": {
            "skill_overlap":    round(skill_overlap, 1),
            "similarity":       round(sim_component, 1),
            "experience_match": round(exp_component, 1),
        }
    }


def compute_proficiency_evidence_scores(
    user_proficiencies: list,   # list[SkillProficiency]
    parsed_resume: ParsedResume,
) -> dict[str, dict]:
    """
    For each declared skill+proficiency, count evidence level (0–3)
    across projects and experience.

    Returns: { skill_name: { "declared": level, "evidence": 0-3, "gap": int } }
    The AI uses this dict for reasoning — never re-computes it.
    """
    from core.schemas import ProficiencyLevel

    _LEVEL_EXPECTED: dict[ProficiencyLevel, int] = {
        ProficiencyLevel.BEGINNER:     0,  # beginner needs no project evidence
        ProficiencyLevel.INTERMEDIATE: 2,
        ProficiencyLevel.ADVANCED:     2,
        ProficiencyLevel.EXPERT:       3,
    }

    all_text_chunks: list[str] = []
    for proj in parsed_resume.projects:
        all_text_chunks.append(proj.description)
        all_text_chunks.extend(proj.technologies)
    for exp in parsed_resume.experience:
        all_text_chunks.extend(exp.responsibilities)
        all_text_chunks.extend(exp.technologies)

    combined = " ".join(all_text_chunks).lower()

    results = {}
    for sp in user_proficiencies:
        skill = sp.skill_name.lower()
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        occurrences = len(re.findall(pattern, combined))

        # Map occurrences → evidence level
        if occurrences == 0:
            evidence = 0
        elif occurrences == 1:
            evidence = 1
        elif occurrences <= 4:
            evidence = 2
        else:
            evidence = 3

        expected = _LEVEL_EXPECTED[sp.level]
        gap = max(0, expected - evidence)  # 0 = aligned, >0 = under-evidenced

        results[skill] = {
            "declared":  sp.level.value,
            "evidence":  evidence,
            "expected":  expected,
            "gap":       gap,
            "aligned":   gap == 0,
        }

    return results