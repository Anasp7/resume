"""
Smart Resume — Resume Verifier v3
===================================
Post-generation pass that checks generated resume against original content.

Changes from v2:
- No longer removes entire lines for fabricated terms — only removes the term inline
- Role-description terms (machine learning, deep learning, etc.) are NEVER removed
  even if not in original resume — they describe the target role, not invented experience
- Flask/FastAPI etc. checked against full project description text not just skills list
- Soft-skill phrase removal kept but gentler (removes phrase, not whole sentence)
"""

from __future__ import annotations
import re
import logging

logger = logging.getLogger("smart_resume.verifier")

# ── Terms that are NEVER fabrication — they describe the role being applied for ──
# These appear naturally in objectives/summaries even if not in original resume.
ROLE_DESCRIPTION_TERMS = {
    "machine learning", "deep learning", "artificial intelligence", "ai",
    "data science", "data analysis", "natural language processing", "nlp",
    "computer vision", "robotics", "automation", "software engineering",
    "backend development", "frontend development", "full stack",
    "embedded systems", "control systems", "autonomous systems",
    "software developer", "software engineer", "data scientist",
    "ml engineer", "robotics engineer", "backend developer",
}

# ── Tech terms worth checking for fabrication ────────────────────────────────
# Only specific unambiguous framework/library names — NOT general CS terms
WATCHLIST_TERMS = {
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "xgboost",
    "langchain", "llamaindex", "openai", "huggingface", "transformers",
    "spacy", "nltk", "gensim",
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
    "prometheus", "grafana", "datadog",
    "pytest", "selenium", "playwright", "cypress", "jest",
    "streamlit", "gradio", "pydantic", "sqlalchemy",
    "opencv",
}

FABRICATION_PHRASES = [
    r"\bteam player\b",
    r"\bfast learner\b",
    r"\bpassionate about\b",
    r"\bseeking a challenging\b",
    r"\beager to learn\b",
    r"\bquick learner\b",
    r"\bexcellent communication skills\b",
    r"\bresults[- ]driven professional\b",
]


def verify_resume(
    generated_text: str,
    original_resume: dict,
    clarification_answers: list,
) -> tuple[str, list[str]]:
    """
    Verify generated resume against original content.
    Returns (cleaned_text, list_of_issues_fixed).

    Philosophy: Protect against invented tech stacks (e.g. adding PyTorch when
    candidate never used it). Never strip role-description language or terms that
    appear in project descriptions.
    """
    if not generated_text or not generated_text.strip():
        return generated_text, []

    issues = []
    text   = generated_text

    # ── 1. Build allowed vocabulary from original ─────────────────────────────
    allowed_terms: set[str] = set()

    for skill in original_resume.get("skills", []):
        allowed_terms.add(skill.lower().strip())

    # Collect all original text — skills, experience, projects, summary
    original_parts = [original_resume.get("summary", "") or ""]

    for e in original_resume.get("experience", []):
        if isinstance(e, dict):
            original_parts.append(e.get("role", ""))
            original_parts.append(e.get("company", ""))
            original_parts.extend(e.get("responsibilities", []))
            original_parts.extend(e.get("technologies", []))
        else:
            original_parts.append(getattr(e, "role", "") or "")
            original_parts.extend(getattr(e, "responsibilities", []) or [])
            original_parts.extend(getattr(e, "technologies", []) or [])

    for p in original_resume.get("projects", []):
        if isinstance(p, dict):
            original_parts.append(p.get("title", ""))
            original_parts.append(p.get("description", ""))
            original_parts.extend(p.get("technologies", []))
        else:
            original_parts.append(getattr(p, "title", "") or "")
            original_parts.append(getattr(p, "description", "") or "")
            original_parts.extend(getattr(p, "technologies", []) or [])

    # Add clarification answers — these are verified facts
    for ans in clarification_answers:
        if hasattr(ans, "answer"):
            original_parts.append(ans.answer or "")
        elif isinstance(ans, str):
            original_parts.append(ans)

    all_original_text = " ".join(str(p) for p in original_parts).lower()

    # Pre-populate allowed_terms from all original text tokens
    for word in re.findall(r"[a-z][a-z0-9+#._-]{1,20}", all_original_text):
        allowed_terms.add(word)

    # ── 2. Build denied techs from explicit denial answers ────────────────────
    NIL_WORDS = {"nil","none","no","nothing","not used","didn't use","did not use",
                 "not applicable","n/a","na","not mentioned","i didn't","i did not",
                 "not related","not relevant","never used"}
    denied_techs: set[str] = set()
    for ca in clarification_answers:
        ans = (ca.answer.strip().lower() if hasattr(ca, "answer") else str(ca).lower())
        if ans in NIL_WORDS or len(ans) <= 3:
            continue
        for pat in [
            r"(\w[\w+#]*)\s+is\s+not\s+used",
            r"(\w[\w+#]*)\s+not\s+used",
            r"didn.t\s+use\s+(\w[\w+#]*)",
            r"did\s+not\s+use\s+(\w[\w+#]*)",
            r"not\s+using\s+(\w[\w+#]*)",
        ]:
            for m in re.finditer(pat, ans):
                denied_techs.add(m.group(1).strip().lower())

    # Remove denied techs inline (not whole line)
    for term in denied_techs:
        count_before = len(re.findall(re.escape(term), text, re.IGNORECASE))
        if count_before:
            text = re.sub(
                r",?\s*(?:using|via|with|in|and)?\s*(?<![a-zA-Z0-9])"
                + re.escape(term) + r"(?![a-zA-Z0-9])",
                "",
                text,
                flags=re.IGNORECASE,
            )
            issues.append(f"Removed denied tech: {term}")

    # ── 3. Fabrication check — WATCHLIST terms only ───────────────────────────
    # Only flag if: term is in generated text AND not in any original content
    # AND not a role-description term
    text_lower = text.lower()
    fabricated: list[str] = []

    for term in WATCHLIST_TERMS:
        # Skip if it's a role-description term
        if term in ROLE_DESCRIPTION_TERMS:
            continue
        in_generated = bool(re.search(
            r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text_lower
        ))
        if not in_generated:
            continue
        in_original = (
            term in allowed_terms or
            bool(re.search(
                r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])",
                all_original_text,
            ))
        )
        if not in_original:
            fabricated.append(term)

    if fabricated:
        logger.warning("Fabricated terms detected: %s", fabricated)
        for term in fabricated:
            # Remove inline — just the term and surrounding connector words
            # Do NOT remove entire lines — other content on the line is valid
            text = re.sub(
                r",?\s*(?:utilizing|using|leveraging|via|with|and|in|for)?\s*"
                r"(?<![a-zA-Z0-9])" + re.escape(term) + r"(?![a-zA-Z0-9])",
                "",
                text,
                flags=re.IGNORECASE,
            )
            issues.append(f"Removed fabricated term: {term}")
        # Clean up any double spaces or orphaned punctuation
        text = re.sub(r"  +", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

    # ── 4. Soft-skill phrase removal — phrase only, not whole line ────────────
    for phrase in FABRICATION_PHRASES:
        if re.search(phrase, text, re.IGNORECASE):
            text = re.sub(phrase, "", text, flags=re.IGNORECASE)
            text = re.sub(r"  +", " ", text)
            issues.append(f"Removed generic phrase: {phrase}")

    # ── 5. Format fixes ───────────────────────────────────────────────────────
    text = re.sub(
        r"^(SUMMARY|PROFILE|PROFESSIONAL SUMMARY)\s*$",
        "OBJECTIVE", text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    text = re.sub(r"\n+NOTE[:\s].*$",   "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\n+REMARK[:\s].*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("Not Specified", "").replace("not specified", "")

    # ── Remove ALL placeholder/empty lines ───────────────────────────────────
    # "- " alone, "- -", "-  " etc
    text = re.sub(r"^\s*-\s*-?\s*$", "", text, flags=re.MULTILINE)
    # "- -" inline within a line
    text = re.sub(r"^\s*-\s+-\s*$", "", text, flags=re.MULTILINE)
    # Any line that is just dashes and spaces
    text = re.sub(r"^\s*[-–]+\s*$", "", text, flags=re.MULTILINE)

    # ── Remove Class X / Class XII placeholder lines ──────────────────────────
    # "Class X: -"  "Class X:  "  "Class X: -  "
    text = re.sub(r"^.*?Class\s*X(?:II)?[:\s]*[-–]?\s*$",
                  "", text, flags=re.MULTILINE | re.IGNORECASE)
    # "Class 10: -"  "Class 12: -"
    text = re.sub(r"^.*?Class\s*(?:10|12)[:\s]*[-–]?\s*$",
                  "", text, flags=re.MULTILINE | re.IGNORECASE)
    # Class X/XII with just a number (hallucinated like "Class X: 10")
    text = re.sub(r"^.*?Class\s*X(?:II)?[:\s]+\d{1,3}\s*$",
                  "", text, flags=re.MULTILINE | re.IGNORECASE)

    # ── Remove entirely empty sections ────────────────────────────────────────
    # Section header followed only by blank lines / dash lines before next header
    section_names = r"(?:OBJECTIVE|SUMMARY|SKILLS|TECHNICAL SKILLS|EXPERIENCE|WORK EXPERIENCE|PROJECTS|EDUCATION|CERTIFICATIONS)"
    text = re.sub(
        r"^(" + section_names + r")\s*\n(?:\s*\n|\s*-+\s*\n)*(?=" + section_names + r")",
        "", text, flags=re.MULTILINE | re.IGNORECASE
    )

    # ── Fix truncated objective sentences ─────────────────────────────────────
    # "Motivated B.Tech Computer Science student role as an ml engineer" — missing verb
    text = re.sub(
        r"(Motivated B\.Tech[^\n]+?)\s+role\s+as\s+an?\s+([^\n]+)",
        r"\1 seeking a \2 position",
        text, flags=re.IGNORECASE
    )

    # ── 6. Clean up ───────────────────────────────────────────────────────────
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\r\n", "\n", text)
    text = text.strip()

    if issues:
        logger.info("Verifier fixed %d issues: %s", len(issues), issues)

    return text, issues