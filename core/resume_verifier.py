"""
Smart Resume — Resume Verifier v2
===================================
Post-generation pass that checks generated resume against original content.

Two layers of protection:
  1. Skill fabrication check — any tech term in the generated resume that does
     NOT appear in the candidate's original content gets flagged and removed.
  2. Hardcoded bad-phrase removal — generic filler, soft-skill preamble, etc.
  3. Format fixes — section headings, "Not Specified", blank lines.
"""

from __future__ import annotations
import re
import logging
from typing import Optional

logger = logging.getLogger("smart_resume.verifier")

# ── Tech terms that the verifier watches for fabrication ──────────────────────
# Subset of KNOWN_TECH_TERMS — only specific/unambiguous names worth checking
WATCHLIST_TERMS = {
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "xgboost",
    "langchain", "llamaindex", "openai", "huggingface", "transformers",
    "opencv", "spacy", "nltk", "gensim",
    "fastapi", "flask", "django", "spring", "express", "nestjs", "laravel",
    "react", "vue", "angular", "nextjs", "svelte",
    "docker", "kubernetes", "terraform", "ansible",
    "aws", "gcp", "azure", "heroku", "vercel",
    "postgresql", "mongodb", "redis", "elasticsearch", "cassandra", "dynamodb",
    "spark", "hadoop", "airflow", "dbt", "bigquery", "redshift", "snowflake",
    "ros", "ros2", "gazebo", "moveit", "arduino", "raspberry pi",
    "kotlin", "swift", "flutter", "react native",
    "rust", "scala", "elixir", "haskell", "solidity",
    "graphql", "grpc", "kafka", "rabbitmq",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "jenkins", "github actions", "gitlab ci", "argocd",
    "prometheus", "grafana", "datadog", "elk",
    "pytest", "selenium", "playwright", "cypress", "jest",
    "streamlit", "gradio", "pydantic", "sqlalchemy",
}

FABRICATION_PHRASES = [
    r"\brelational database\b",
    r"\bcontrol algorithm\b",
    r"\bteam player\b",
    r"\bfast learner\b",
    r"\bself[- ]motivated\b",
    r"\bstrong communication\b",
    r"\bpassionate about\b",
    r"\bseeking a challenging\b",
    r"\bproblem[- ]solving skills\b",
    r"\beager to learn\b",
    r"\bquick learner\b",
    r"\bexcellent communication\b",
    r"\bresults[- ]driven\b",
    r"\bdetail[- ]oriented\b",
]


def verify_resume(
    generated_text: str,
    original_resume: dict,
    clarification_answers: list,
) -> tuple[str, list[str]]:
    """
    Verify generated resume against original content.
    Returns (cleaned_text, list_of_issues_fixed).
    """
    if not generated_text or not generated_text.strip():
        return generated_text, []

    issues = []
    text   = generated_text

    # ── 1. Build allowed vocabulary from original ─────────────────────────────
    allowed_skills = set()
    for skill in original_resume.get("skills", []):
        allowed_skills.add(skill.lower().strip())

    # Also extract from project descriptions + experience bullets
    all_original_text_parts = [
        original_resume.get("summary", "") or "",
        " ".join(
            " ".join(e.get("responsibilities", [])) + " " + e.get("role", "")
            if isinstance(e, dict) else
            " ".join(e.responsibilities) + " " + e.role
            for e in original_resume.get("experience", [])
        ),
        " ".join(
            (p.get("description", "") if isinstance(p, dict) else p.description)
            + " " +
            " ".join(p.get("technologies", []) if isinstance(p, dict) else p.technologies)
            for p in original_resume.get("projects", [])
        ),
    ]

    # Add clarification answers — these are verified facts
    for ans in clarification_answers:
        if hasattr(ans, "answer"):
            all_original_text_parts.append(ans.answer)
        elif isinstance(ans, str):
            all_original_text_parts.append(ans)

    all_original_text = " ".join(all_original_text_parts).lower()

    # ── 1b. Extract denied techs from answers ────────────────────────────────
    # If candidate said "c is not used", "nil", "didn't use X" → remove from resume
    NIL_WORDS = {"nil", "none", "no", "nothing", "not used", "didn't use", "did not use",
                 "not applicable", "n/a", "na", "not mentioned", "i didn't", "i did not",
                 "not related", "not relevant", "never used"}
    denied_techs = set()
    for ca in clarification_answers:
        ans = ca.answer.strip().lower() if hasattr(ca, 'answer') else str(ca).lower()
        if ans in NIL_WORDS or len(ans) <= 3:
            continue
        # Extract explicit denials
        import re as _re2
        for pat in [r"(\w[\w+#]*)\s+is\s+not\s+used",
                    r"(\w[\w+#]*)\s+not\s+used",
                    r"didn.t\s+use\s+(\w[\w+#]*)",
                    r"did\s+not\s+use\s+(\w[\w+#]*)",
                    r"not\s+using\s+(\w[\w+#]*)",
                    r"(\w[\w+#]*)\s+not\s+(?:in|for|used\s+in)"]:
            for m in _re2.finditer(pat, ans):
                denied_techs.add(m.group(1).strip())

    # Strip denied techs from generated text (remove entire line)
    if denied_techs:
        for term in denied_techs:
            text = re.sub(
                r"^[^\n]*(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])[^\n]*$",
                "",
                text,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            issues.append(f"Removed denied tech from output: {term}")
        text = re.sub(r"\n{3,}", "\n\n", text)

        # ── 2. Skill fabrication check — catch any WATCHLIST term with NO evidence ─
    # Checks EVERY tech term in generated resume against original + clarification answers
    fabricated_skills = []
    text_lower = text.lower()

    # Expand WATCHLIST to also include any tech-like words in the generated resume
    # that aren't in the watchlist (catches long-tail terms like "gazebo", "motion planning")
    # We check every word/phrase in generated bullets against allowed content
    for term in WATCHLIST_TERMS:
        in_generated = bool(re.search(
            r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text_lower
        ))
        in_original  = bool(re.search(
            r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", all_original_text
        )) or term in allowed_skills

        if in_generated and not in_original:
            fabricated_skills.append(term)

    # ── 2b. Extended check — catch fabricated terms NOT in WATCHLIST ──────────
    # Extract all multi-word tech-like phrases from generated bullets
    EXTRA_TECH_PATTERNS = [
        r"\bmotion planning\b",
        r"\bsensor fusion\b",
        r"\bpath planning\b",
        r"\bobject detection\b",
        r"\bneural network\b",
        r"\bmachine learning\b",
        r"\bdeep learning\b",
        r"\bcomputer vision\b",
        r"\bnatural language\b",
        r"\boci\b", r"\baws\b", r"\bgcp\b", r"\bazure\b",
        r"\bflask\b", r"\bfastapi\b", r"\bdjango\b", r"\bspring\b",
        r"\bkubernetes\b", r"\bdocker\b", r"\bterraform\b",
        r"\bangular\b", r"\breact\b", r"\bvue\b",
        r"\bpytorch\b", r"\btensorflow\b", r"\bkeras\b",
        r"\bopencv\b", r"\bspacy\b", r"\blangchain\b",
    ]
    for pat in EXTRA_TECH_PATTERNS:
        m = re.search(pat, text_lower, re.IGNORECASE)
        if m:
            found_term = m.group().strip()
            in_original = bool(re.search(
                r"(?<![a-z0-9])" + re.escape(found_term) + r"(?![a-z0-9])", all_original_text
            )) or found_term in allowed_skills
            if not in_original and found_term not in fabricated_skills:
                fabricated_skills.append(found_term)

    if fabricated_skills:
        logger.warning("Fabricated terms detected: %s", fabricated_skills)
        for term in fabricated_skills:
            # Remove entire line — use [^\r\n]* to handle Windows CRLF endings
            text = re.sub(
                r"^[^\r\n]*(?<![a-zA-Z0-9])" + re.escape(term) + r"(?![a-zA-Z0-9])[^\r\n]*\r?$",
                "",
                text,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            # Also try phrase-level removal within a line (in case whole line has valid content too)
            text = re.sub(
                r",?\s*(?:utilizing|using|via|with|for|in)\s+[^,\n]*(?<![a-zA-Z0-9])"
                + re.escape(term) + r"(?![a-zA-Z0-9])[^,\n]*",
                "",
                text,
                flags=re.IGNORECASE,
            )
            issues.append(f"Removed fabricated term: {term}")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\r\n", "\n", text)  # normalise CRLF → LF

    # ── 3. Hardcoded bad-phrase removal ───────────────────────────────────────
    for phrase in FABRICATION_PHRASES:
        if re.search(phrase, text, re.IGNORECASE):
            text = re.sub(
                r"[^.!?\n]*" + phrase + r"[^.!?\n]*[.!?\n]?",
                "",
                text,
                flags=re.IGNORECASE,
            )
            issues.append(f"Removed generic phrase matching: {phrase}")

    # ── 4. Format fixes ───────────────────────────────────────────────────────
    text = re.sub(r"^(SUMMARY|PROFILE|PROFESSIONAL SUMMARY)\s*$",
                  "OBJECTIVE", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"\n+NOTE[:\s].*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\n+REMARK[:\s].*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("Not Specified", "").replace("not specified", "")

    # ── 5. Clean up ───────────────────────────────────────────────────────────
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if issues:
        logger.info("Verifier fixed %d issues: %s", len(issues), issues)

    return text, issues