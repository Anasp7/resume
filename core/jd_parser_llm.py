"""
Smart Resume — LLM-Powered JD Parser + Virtual JD Generator
=============================================================
Two modes:
  1. JD provided  → parse actual JD to extract skills/domain/responsibilities
  2. No JD        → generate "virtual JD" from LLM knowledge of target role
                    so similarity, skill gap, and matching work exactly the same
"""

from __future__ import annotations
import json
import logging
import re

import httpx

logger = logging.getLogger("smart_resume.jd_parser")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL        = "llama-3.1-8b-instant"  # routed via llm_client

# ── Prompt: parse a real JD ───────────────────────────────────────────────────

JD_PARSE_PROMPT = """Parse this job description and return ONLY valid JSON.

Extract:
- domain: one of ML, Backend, Frontend, DevOps, Data, Embedded, Security, Mobile, Unknown
- required_skills: list of technical skills explicitly required
- preferred_skills: list of skills mentioned as preferred/nice-to-have
- required_experience_years: number (null if not specified)
- key_responsibilities: list of 3-5 main technical responsibilities
- seniority: one of Intern, Junior, Mid, Senior, Lead, Unknown
- virtual_jd_text: reconstruct a clean plain-text summary of the role requirements
  (used for similarity computation — 3-5 sentences covering role, skills, responsibilities)

Rules:
- Skills = specific technical terms only (Python, TensorFlow, Docker etc)
- Never include soft skills or generic phrases
- If skill in both required and preferred, put in required only

Return exactly:
{
  "domain": "ML",
  "required_skills": ["python", "tensorflow", "sql"],
  "preferred_skills": ["pytorch", "aws"],
  "required_experience_years": 2,
  "key_responsibilities": ["Train ML models", "Deploy to production"],
  "seniority": "Mid",
  "virtual_jd_text": "We are looking for an ML Engineer with Python, TensorFlow and SQL skills..."
}

Job Description:
"""

# ── Prompt: generate virtual JD from role name ───────────────────────────────

VIRTUAL_JD_PROMPT = """You are a technical recruiter. Generate a realistic job requirements profile for the role below.

Return ONLY valid JSON — no markdown, no explanation.

Role: {role}

Based on your knowledge of what this role typically requires in 2025, generate:
- domain: one of ML, Backend, Frontend, DevOps, Data, Embedded, Security, Mobile, Unknown
- required_skills: 6-10 technical skills that are ALWAYS required for this role
- preferred_skills: 4-6 skills that are commonly preferred/bonus for this role
- required_experience_years: typical years of experience required (use 0-1 for intern/fresher roles)
- key_responsibilities: 4-6 main technical responsibilities for this role
- seniority: infer from role name — Intern, Junior, Mid, Senior, Lead, Unknown
- virtual_jd_text: 4-5 sentence plain text description of the role requirements
  (written as if it were a real JD — used for similarity scoring against resumes)

Important:
- Be specific and realistic — not generic
- Skills must be actual tool/language names, not phrases
- virtual_jd_text must mention the key skills naturally so similarity scoring works

Return exactly:
{{
  "domain": "",
  "required_skills": [],
  "preferred_skills": [],
  "required_experience_years": null,
  "key_responsibilities": [],
  "seniority": "",
  "virtual_jd_text": ""
}}
"""


def parse_jd_with_llm(raw_jd: str, api_key: str) -> dict:
    """Parse a real job description — uses shared llm_client."""
    try:
        return call_llm_json(
            JD_PARSE_PROMPT + "\n\nJOB DESCRIPTION:\n" + raw_jd,
            api_key, max_tokens=600, smart=False,
            system_msg="You are a JD parser. Return only valid JSON.",
        )
    except Exception as e:
        logger.warning("JD parse LLM failed: %s", e)
        return {}


# ── Role knowledge base — deterministic fallback, zero LLM tokens ────────────
# Maps normalised role keywords → typical required skills + domain
# LLM extends this when available; this always guarantees a non-empty result.

_ROLE_KB: dict[str, dict] = {
    # Software / Backend
    "software engineer":    {"domain":"Backend",  "req":["python","git","linux","sql","rest api","data structures","algorithms"], "pref":["docker","aws","system design"]},
    "backend developer":    {"domain":"Backend",  "req":["python","sql","rest api","git","linux","postgresql"], "pref":["docker","redis","fastapi","django"]},
    "backend engineer":     {"domain":"Backend",  "req":["python","sql","rest api","git","linux","postgresql"], "pref":["docker","redis","fastapi","django"]},
    "full stack":           {"domain":"Frontend", "req":["javascript","react","node","sql","git","html","css"], "pref":["typescript","docker","aws","mongodb"]},
    "frontend developer":   {"domain":"Frontend", "req":["javascript","react","html","css","git","typescript"], "pref":["vue","tailwind","nextjs","figma"]},
    "web developer":        {"domain":"Frontend", "req":["javascript","html","css","git","react"], "pref":["typescript","nodejs","sql","docker"]},
    # ML / AI
    "ml engineer":          {"domain":"ML",       "req":["python","pytorch","tensorflow","scikit","sql","git","pandas","numpy"], "pref":["mlflow","docker","aws","spark"]},
    "machine learning":     {"domain":"ML",       "req":["python","pytorch","tensorflow","scikit","sql","git","pandas","numpy"], "pref":["mlflow","docker","aws","spark"]},
    "data scientist":       {"domain":"Data",     "req":["python","pandas","numpy","sql","matplotlib","scikit","statistics"], "pref":["pytorch","spark","tableau","r"]},
    "data engineer":        {"domain":"Data",     "req":["python","sql","spark","kafka","airflow","aws","etl","postgresql"], "pref":["scala","docker","dbt","redshift"]},
    "data analyst":         {"domain":"Data",     "req":["python","sql","excel","pandas","tableau","statistics"], "pref":["powerbi","r","matplotlib","aws"]},
    "ai engineer":          {"domain":"ML",       "req":["python","pytorch","langchain","openai","fastapi","git"], "pref":["docker","aws","huggingface","transformers"]},
    "nlp engineer":         {"domain":"ML",       "req":["python","pytorch","transformers","nltk","spacy","sql"], "pref":["huggingface","langchain","docker","aws"]},
    # DevOps / Cloud
    "devops engineer":      {"domain":"DevOps",   "req":["linux","docker","kubernetes","git","jenkins","aws","terraform"], "pref":["ansible","grafana","prometheus","helm"]},
    "cloud engineer":       {"domain":"DevOps",   "req":["aws","linux","docker","kubernetes","terraform","python"], "pref":["gcp","azure","ansible","jenkins"]},
    "sre":                  {"domain":"DevOps",   "req":["linux","docker","kubernetes","python","monitoring","aws"], "pref":["prometheus","grafana","terraform","go"]},
    "site reliability":     {"domain":"DevOps",   "req":["linux","docker","kubernetes","python","monitoring","aws"], "pref":["prometheus","grafana","terraform","go"]},
    # Embedded / Robotics
    "embedded systems":     {"domain":"Embedded", "req":["c","c++","rtos","microcontrollers","gpio","uart","spi","i2c","cmake"], "pref":["ros2","python","stm32","esp32","keil"]},
    "embedded engineer":    {"domain":"Embedded", "req":["c","c++","rtos","microcontrollers","gpio","uart","spi","cmake"], "pref":["ros2","python","stm32","esp32"]},
    "robotics engineer":    {"domain":"Embedded", "req":["ros2","python","c++","gazebo","linux","cmake","opencv"], "pref":["slam","moveit","pytorch","matlab"]},
    "robotics":             {"domain":"Embedded", "req":["ros2","python","c++","gazebo","linux","cmake"], "pref":["slam","moveit","opencv","pytorch"]},
    "firmware engineer":    {"domain":"Embedded", "req":["c","c++","rtos","arm","uart","spi","i2c","cmake","git"], "pref":["python","stm32","esp32","openocd"]},
    "iot engineer":         {"domain":"Embedded", "req":["c","python","mqtt","linux","gpio","i2c","spi","rtos"], "pref":["aws","docker","esp32","arduino"]},
    # Mobile
    "android developer":    {"domain":"Mobile",   "req":["kotlin","java","android","git","rest api","sqlite"], "pref":["jetpack","firebase","rxjava","mvvm"]},
    "ios developer":        {"domain":"Mobile",   "req":["swift","xcode","ios","git","rest api","objc"], "pref":["swiftui","combine","firebase","mvvm"]},
    "mobile developer":     {"domain":"Mobile",   "req":["flutter","dart","git","rest api","firebase"], "pref":["kotlin","swift","redux","sqlite"]},
    # Security
    "security engineer":    {"domain":"Security", "req":["python","linux","networking","wireshark","nmap","bash"], "pref":["metasploit","burp","docker","aws"]},
    "cybersecurity":        {"domain":"Security", "req":["python","linux","networking","wireshark","bash","cryptography"], "pref":["docker","metasploit","burp","aws"]},
    # General / Fallback
    "intern":               {"domain":"Backend",  "req":["python","git","linux","sql","data structures"], "pref":["javascript","docker","rest api"]},
    "fresher":              {"domain":"Backend",  "req":["python","git","linux","sql","data structures"], "pref":["javascript","docker","rest api"]},
    "trainee":              {"domain":"Backend",  "req":["python","git","linux","sql","data structures"], "pref":["javascript","docker","rest api"]},
}

def _match_role_kb(role: str) -> dict:
    """Find best matching entry in _ROLE_KB for the given role string."""
    role_lower = role.lower().strip()
    # Exact match first
    if role_lower in _ROLE_KB:
        return _ROLE_KB[role_lower]
    # Partial match — find first key that appears in role string
    for key, val in _ROLE_KB.items():
        if key in role_lower:
            return val
    # No match — return generic software engineer
    return _ROLE_KB["software engineer"]


def generate_virtual_jd(target_role: str, api_key: str) -> dict:
    """
    Generate a virtual JD when none is provided.
    Strategy: deterministic role knowledge base first (zero tokens),
    then optionally enhanced by LLM if available.
    Guarantees a non-empty result even if LLM is rate-limited.
    """
    # ── Step 1: deterministic base from role knowledge ────────────────────
    kb = _match_role_kb(target_role)
    domain       = kb["domain"]
    req_skills   = kb["req"].copy()
    pref_skills  = kb["pref"].copy()
    responsibilities = _default_responsibilities(target_role, domain)
    virtual_text = _build_virtual_text(target_role, req_skills, pref_skills, responsibilities)

    base_result = {
        "domain":                   domain,
        "required_skills":          req_skills,
        "preferred_skills":         pref_skills,
        "required_experience_years": 0,
        "key_responsibilities":     responsibilities,
        "seniority":                "Junior" if any(w in target_role.lower() for w in ("intern","fresher","junior","trainee")) else "Mid",
        "virtual_jd_text":          virtual_text,
        "source":                   "deterministic",
    }
    logger.info("Virtual JD (deterministic) for %r: domain=%s, %d required skills",
                target_role, domain, len(req_skills))

    # ── Step 2: try LLM enhancement — adds role-specific nuance ──────────
    # Ollama Phi-3 first (free, local) → Groq fallback
    if api_key:
        compact_prompt = (
            "For the role " + repr(target_role) + ", list the 8 most important technical skills "
            "(required) and 4 preferred skills in 2025. "
            "Also write 3 key responsibilities (1 line each). "
            "Return ONLY JSON: {required_skills:[...],preferred_skills:[...],key_responsibilities:[...]}"
        )
        try:
            from core.hybrid_llm import call_llm_json_with_fallback
            llm_result = call_llm_json_with_fallback(
                compact_prompt, api_key,
                max_tokens=350,
                system_msg="Return only valid JSON. No explanation.",
                prefer_ollama=True,   # ← Ollama Phi-3 first
            )
            # Merge: LLM skills take priority, deterministic as fallback
            llm_req   = [s.lower() for s in llm_result.get("required_skills", []) if s]
            llm_pref  = [s.lower() for s in llm_result.get("preferred_skills", []) if s]
            llm_resp  = llm_result.get("key_responsibilities", [])

            if llm_req:
                # Combine: LLM skills + any deterministic skills not already included
                merged_req  = llm_req[:8]
                for s in req_skills:
                    if s not in merged_req: merged_req.append(s)
                merged_pref = llm_pref[:4] or pref_skills
                merged_resp = llm_resp[:5] or responsibilities
                virtual_text = _build_virtual_text(target_role, merged_req, merged_pref, merged_resp)
                base_result.update({
                    "required_skills":      merged_req,
                    "preferred_skills":     merged_pref,
                    "key_responsibilities": merged_resp,
                    "virtual_jd_text":      virtual_text,
                    "source":               "llm_enhanced",
                })
                logger.info("Virtual JD enhanced (Ollama-first): %d req skills", len(merged_req))
        except Exception as e:
            logger.warning("Virtual JD LLM enhancement skipped (%s) — using deterministic", e)

    return base_result



def _default_responsibilities(role: str, domain: str) -> list[str]:
    """Generate default responsibilities from domain."""
    templates = {
        "Backend":  [f"Design and implement RESTful APIs for {role} features",
                     "Write and optimise SQL queries and database schemas",
                     "Build scalable server-side logic and integrate third-party services"],
        "Frontend": [f"Build responsive UI components for {role} applications",
                     "Integrate frontend with backend APIs and manage application state",
                     "Optimise web performance and cross-browser compatibility"],
        "ML":       [f"Design, train and evaluate machine learning models for {role} tasks",
                     "Prepare and clean datasets for model training and validation",
                     "Deploy models to production and monitor performance metrics"],
        "Data":     [f"Build and maintain data pipelines for {role} workflows",
                     "Analyse large datasets and produce actionable insights",
                     "Create dashboards and reports for stakeholder communication"],
        "DevOps":   ["Build and maintain CI/CD pipelines for automated deployment",
                     "Manage cloud infrastructure using IaC tools",
                     "Monitor system reliability, performance and incident response"],
        "Embedded": [f"Develop firmware and embedded software for {role} applications",
                     "Interface with hardware peripherals using SPI, I2C, UART protocols",
                     "Debug and validate hardware-software integration"],
        "Mobile":   [f"Develop and maintain mobile applications for {role}",
                     "Implement UI/UX designs and integrate REST APIs",
                     "Publish and maintain apps on app stores"],
        "Security": [f"Conduct security assessments and penetration testing for {role}",
                     "Identify and remediate vulnerabilities in systems and code",
                     "Implement security controls and monitor for threats"],
    }
    return templates.get(domain, [f"Contribute to {role} development and implementation",
                                   "Collaborate with team members on technical solutions",
                                   "Write clean, maintainable code following best practices"])


def _build_virtual_text(role: str, req: list, pref: list, resp: list) -> str:
    """Build a readable virtual JD text for similarity scoring."""
    req_str  = ", ".join(req[:6])
    pref_str = ", ".join(pref[:3])
    resp_str = " ".join(resp[:2])
    return (
        f"We are looking for a {role} with strong skills in {req_str}. "
        f"Knowledge of {pref_str} is preferred. "
        f"{resp_str} "
        f"The ideal candidate has hands-on experience with {req_str} "
        f"and can contribute effectively from day one."
    )