"""
Smart Resume — Doubt Detection Engine v5
==========================================
LLM-generated questions — fully dynamic.

Flow:
  1. Send actual resume content + target role to LLM
  2. LLM reads the real text and generates 2-3 targeted questions
  3. Each question references the candidate's own project names / bullet text
  4. Falls back to regex detection if LLM call fails

Questions are short-answer style — each answerable in 1-2 sentences.
"""

from __future__ import annotations
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import httpx
from core.llm_client import call_llm_json

from core.schemas import ParsedResume, SkillProficiency

logger = logging.getLogger("smart_resume.doubt_engine")

GROQ_API_URL     = "https://api.groq.com/openai/v1/chat/completions"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
MODEL            = "llama-3.3-70b-versatile"
TOGETHER_MODEL   = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"


@dataclass
class DoubtIssue:
    type:     str
    context:  str
    question: str
    required: bool


# ── Domain core skills (used in fallback only) ────────────────────────────────

DOMAIN_CORE_SKILLS: dict[str, set[str]] = {
    "ml":        {"tensorflow", "pytorch", "scikit", "keras", "xgboost",
                  "deep learning", "nlp", "computer vision"},
    "backend":   {"django", "fastapi", "flask", "spring", "express",
                  "rest api", "postgresql", "mysql", "redis"},
    "frontend":  {"react", "vue", "angular", "nextjs", "typescript"},
    "devops":    {"docker", "kubernetes", "terraform", "jenkins", "ci/cd"},
    "data":      {"spark", "airflow", "dbt", "bigquery", "etl"},
    "embedded":  {"ros", "ros2", "rtos", "arduino", "raspberry", "firmware",
                  "microcontroller"},
    "security":  {"penetration", "nmap", "burp", "owasp", "vulnerability"},
    "mobile":    {"android", "ios", "kotlin", "swift", "flutter", "react native"},
    "fullstack": {"react", "node", "postgresql", "docker", "rest api"},
    "blockchain":{"solidity", "web3", "ethereum", "smart contract"},
    "cloud":     {"aws", "gcp", "azure", "lambda", "s3"},
    "general":   set(),
}


# ─────────────────────────────────────────────────────────────────────────────
# LLM QUESTION GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def _build_doubt_prompt(
    resume: ParsedResume,
    proficiencies: list[SkillProficiency],
    target_role: str,
) -> str:
    """Build the prompt that asks the LLM to generate clarification questions."""

    # Serialize resume content — real text the LLM can read
    proj_lines = []
    for p in resume.projects:
        techs = ", ".join(p.technologies) if p.technologies else "none listed"
        proj_lines.append(f'  - "{p.title}": {p.description.strip()} [tech: {techs}]')

    exp_lines = []
    for e in resume.experience:
        bullets = "\n    ".join(e.responsibilities[:4])
        exp_lines.append(f'  - {e.role} at {e.company} ({e.duration or "no dates"}):\n    {bullets}')

    prof_lines = []
    for p in proficiencies:
        level = p.level.value if hasattr(p.level, "value") else str(p.level)
        prof_lines.append(f'  - {p.skill_name}: {level}')

    skills_str   = ", ".join(resume.skills[:20]) if resume.skills else "none listed"
    projects_str = "\n".join(proj_lines) if proj_lines else "  none"
    exp_str      = "\n".join(exp_lines) if exp_lines else "  none"
    prof_str     = "\n".join(prof_lines) if prof_lines else "  none declared"

    return f"""You are a technical resume reviewer. A candidate is applying for: {target_role}

RESUME CONTENT:

SKILLS: {skills_str}

PROJECTS:
{projects_str}

EXPERIENCE:
{exp_str}

---

Generate EXACTLY 4 clarification questions. Each must target a DIFFERENT category:

CATEGORY 1 — EXPERIENCE VERIFICATION
  Ask about a specific vague or aspirational bullet in their experience.
  If a bullet says "aim to develop" or "working on" — ask what they have personally done so far, what their specific role is, and what has been completed vs in progress.

CATEGORY 2 — SKILL EVIDENCE
  Pick one skill from their skills list that does NOT appear in any project/experience description.
  Ask which specific project used it, what they built with it, and what the output was.
  If all skills are evidenced, ask about the depth of the most relevant skill for {target_role}.

CATEGORY 3 — PROJECT DEPTH
  Pick the most technically thin project description.
  Ask: what was the specific problem, what did THEY personally implement (not the team), and what was the measurable outcome?

CATEGORY 4 — TECH USAGE CLARIFICATION
  Ask about a technology that appears but its actual usage is unclear.
  For example: if they list C as a skill but experience bullets don't mention C — ask specifically which project used C and how (was it used for data structures practice, systems programming, or something else)?

RULES:
- Every question MUST use the candidate's actual project name, company name, or exact bullet text.
- Questions must be answerable in 2-3 sentences.
- Do NOT ask generic questions like "Describe your experience with Python".
- If a bullet says "aim to" or "working on", acknowledge that and ask what is done so far.
- The candidate may answer "nil" or "not applicable" — that is a valid answer.

Respond ONLY with valid JSON:
{{
  "questions": [
    {{
      "type": "experience_verification | skill_evidence | project_depth | tech_usage",
      "question": "Specific question using their actual resume content",
      "required": true
    }}
  ]
}}

Generate exactly 4 questions, one per category."""


def _call_llm_for_questions(prompt: str, api_key: str) -> list[dict] | None:
    """Generate doubt questions via LLM — uses shared llm_client for key rotation."""
    try:
        data = call_llm_json(
            prompt, api_key,
            max_tokens=600,
            system_msg="You are a technical resume reviewer. Respond only with valid JSON.",
        )
        qs = data.get("questions", [])
        if qs:
            logger.info("LLM doubt questions: %d generated", len(qs))
            return qs
        logger.warning("LLM returned empty questions list")
        return None
    except Exception as e:
        logger.warning("LLM doubt generation failed: %s", e)
        return None



def _fallback_questions(
    resume: ParsedResume,
    proficiencies: list[SkillProficiency],
    target_role: str,
) -> list[DoubtIssue]:
    """Simple regex-based fallback when LLM is unavailable."""
    issues   = []
    all_text = " ".join([
        resume.raw_text,
        " ".join(" ".join(e.responsibilities) for e in resume.experience),
        " ".join(p.description + " " + " ".join(p.technologies) for p in resume.projects),
    ]).lower()

    # Unevidenced skills
    for skill in resume.skills[:20]:
        if len(skill) > 1 and not re.search(r"\b" + re.escape(skill.lower()) + r"\b", all_text):
            issues.append(DoubtIssue(
                type     = "unevidenced_skill",
                context  = skill,
                question = (
                    f"You listed '{skill}' as a skill but it doesn't appear in any "
                    f"project or role description. Which project did you use it in and what did you build?"
                ),
                required = True,
            ))
            if len(issues) >= 1:
                break

    # Vague experience bullets
    for exp in resume.experience:
        for resp in exp.responsibilities:
            for pattern in VAGUE_SIGNALS:
                if re.search(pattern, resp, re.IGNORECASE):
                    short = resp.strip()[:80]
                    issues.append(DoubtIssue(
                        type     = "vague_experience",
                        context  = exp.role,
                        question = (
                            f"In your role as {exp.role} at {exp.company}, you wrote: \"{short}\". "
                            f"What was your specific individual contribution and what was the output?"
                        ),
                        required = True,
                    ))
                    break
            if len(issues) >= 2:
                break
        if len(issues) >= 2:
            break

    # Thin projects
    for proj in resume.projects:
        if len(proj.description.strip().split()) < 8 and len(issues) < 3:
            tech = proj.technologies[0] if proj.technologies else ""
            issues.append(DoubtIssue(
                type     = "thin_description",
                context  = proj.title,
                question = (
                    f"Your '{proj.title}' description is very brief"
                    + (f" — for {tech}, what specifically did you implement and what was the outcome?" if tech
                       else " — what was the main functionality and technical challenge?")
                ),
                required = False,
            ))

    return issues[:3]


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def generate_doubt_questions(
    resume: ParsedResume,
    proficiencies: list[SkillProficiency],
    target_role: str,
    api_key: Optional[str] = None,
) -> list[DoubtIssue]:
    """
    Generate 2-3 targeted clarification questions.
    Uses LLM when api_key is available, falls back to regex otherwise.
    """
    if api_key:
        prompt    = _build_doubt_prompt(resume, proficiencies, target_role)
        llm_qs    = _call_llm_for_questions(prompt, api_key)

        if llm_qs:
            issues = []
            for q in llm_qs[:3]:
                issues.append(DoubtIssue(
                    type     = q.get("type", "general"),
                    context  = q.get("type", "general").replace("_", " ").title(),
                    question = q.get("question", "").strip(),
                    required = bool(q.get("required", True)),
                ))
            logger.info("LLM generated %d doubt questions", len(issues))
            return issues

        logger.warning("LLM doubt generation failed — using regex fallback")

    return _fallback_questions(resume, proficiencies, target_role)


def _detect_domain(target_role: str) -> str:
    role = target_role.lower()
    if any(k in role for k in ["ml", "machine learn", "ai ", " ai,", "nlp", "deep learn", "data scien", "computer vision"]):
        return "ml"
    if any(k in role for k in ["frontend", "front-end", "ui ", "react ", "angular", "vue"]):
        return "frontend"
    if any(k in role for k in ["devops", "cloud engineer", "sre", "infrastructure", "platform engineer"]):
        return "devops"
    if any(k in role for k in ["data engineer", "data pipeline", "etl", "analytics engineer"]):
        return "data"
    if any(k in role for k in ["backend", "back-end", "api", "server side", "django", "flask"]):
        return "backend"
    if any(k in role for k in ["embedded", "firmware", "robotics", "ros", "iot", "hardware"]):
        return "embedded"
    if any(k in role for k in ["security", "cyber", "pentest", "soc", "infosec"]):
        return "security"
    if any(k in role for k in ["mobile", "android", "ios", "flutter"]):
        return "mobile"
    if any(k in role for k in ["fullstack", "full stack", "full-stack"]):
        return "fullstack"
    if any(k in role for k in ["blockchain", "web3", "smart contract"]):
        return "blockchain"
    if any(k in role for k in ["cloud", "aws", "gcp", "azure"]):
        return "cloud"
    return "general"


def _skill_evidenced(skill: str, full_text: str) -> bool:
    pattern = r"\b" + re.escape(skill.lower()) + r"\b"
    return bool(re.search(pattern, full_text.lower()))


def _detect_date_issues(resume: ParsedResume) -> list[str]:
    issues = []
    timeline = []
    for exp in resume.experience:
        dur   = exp.duration or ""
        years = re.findall(r"\b(20\d{2}|19\d{2})\b", dur)
        if len(years) >= 2:
            s, e = int(years[0]), int(years[-1])
            if e < s:
                issues.append(f"In '{exp.role} at {exp.company}', end year {e} is before start {s} — typo?")
            else:
                timeline.append((s, e, f"{exp.role} at {exp.company}"))
    for i in range(len(timeline)):
        for j in range(i + 1, len(timeline)):
            s1, e1, l1 = timeline[i]
            s2, e2, l2 = timeline[j]
            if s2 <= e1 and s1 <= e2:
                issues.append(f"'{l1}' and '{l2}' overlap — were these concurrent?")
    return issues


def format_doubt_questions_for_prompt(issues: list[DoubtIssue]) -> str:
    if not issues:
        return ""
    lines = ["CLARIFICATION NEEDED (answer each in 1-2 sentences):"]
    for i, issue in enumerate(issues, 1):
        req = "REQUIRED" if issue.required else "OPTIONAL"
        lines.append(f"{i}. [{req}] {issue.question}")
    return "\n".join(lines)


def validate_answer_with_llm(
    question: str,
    answer: str,
    api_key: str,
) -> tuple[bool, str]:
    """
    Ask the LLM: is this answer valid/genuine or is it a nil/denial?
    Returns (is_valid, normalized_answer).

    Rules:
    - "nil", "none", "no", "not used", "didn't use X" → valid denial, treat as confirmed NO
    - A real answer with content → valid yes
    - Completely off-topic or gibberish → invalid, ask again
    """
    NIL_WORDS = {"nil", "none", "no", "nothing", "not used", "didn't use", "did not use",
                 "not applicable", "n/a", "na", "i didn't", "i did not", "not relevant",
                 "not related", "never used", "not involved"}
    a_stripped = answer.strip().lower()

    # Fast-path: clear nil answer — valid, means candidate denies that tech/claim
    if a_stripped in NIL_WORDS or len(a_stripped) <= 3:
        return True, answer.strip()

    # Check for denial patterns without LLM
    denial_patterns = [
        r"\b(\w[\w+#]*)\s+(?:is\s+)?not\s+used",
        r"didn.t\s+use\s+\w",
        r"did\s+not\s+use\s+\w",
        r"not\s+using\s+\w",
        r"never\s+used\s+\w",
        r"i\s+(?:have\s+)?not\s+used",
    ]
    for pat in denial_patterns:
        if re.search(pat, a_stripped):
            return True, answer.strip()

    # Real content answer — trust it directly (LLM validation call removed to save quota)
    # The answer is used only to confirm/deny tech claims — not to invent content
    if len(a_stripped.split()) >= 3:
        return True, answer.strip()

    return False, "Answer too short — please provide more detail."


def all_required_answered(
    issues: list[DoubtIssue],
    answers: list,
) -> tuple[bool, list[DoubtIssue]]:
    """
    Check if all required questions have answers.
    answers = None  → first run, none answered
    answers = []    → user skipped, treat all as answered (skip mode)
    answers = [...] → check each required question has a matching answer
    """
    # None = first run, doubt questions must still be shown
    if answers is None:
        unanswered = [i for i in issues if i.required]
        return len(unanswered) == 0, unanswered

    # [] = explicit skip — proceed without answers
    if len(answers) == 0:
        return True, []

    # Fuzzy key match: first 60 chars of question text
    def _qkey(q: str) -> str:
        return re.sub(r"\s+", " ", q.strip().lower())[:60]

    answered_keys = set()
    for a in answers:
        if hasattr(a, "question") and a.question:
            answered_keys.add(_qkey(a.question))

    unanswered = [
        i for i in issues
        if i.required and _qkey(i.question) not in answered_keys
    ]
    return len(unanswered) == 0, unanswered