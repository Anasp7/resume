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
MODEL        = "llama-3.3-70b-versatile"

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
    """Parse a real job description using Groq LLM."""
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model":           MODEL,
        "max_tokens":      1000,
        "temperature":     0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a JD parser. Return only valid JSON."},
            {"role": "user",   "content": JD_PARSE_PROMPT + raw_jd[:2000]},
        ],
    }
    try:
        resp = httpx.post(GROQ_API_URL, headers=headers, json=body, timeout=20)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        content = re.sub(r"```json\s*|```\s*", "", content).strip()
        parsed  = json.loads(content)
        parsed["jd_provided"] = True
        logger.info("JD parsed: domain=%s required=%d preferred=%d",
                    parsed.get("domain"), len(parsed.get("required_skills", [])),
                    len(parsed.get("preferred_skills", [])))
        return parsed
    except Exception as e:
        logger.warning("LLM JD parse failed: %s", e)
        return {}


def generate_virtual_jd(target_role: str, api_key: str) -> dict:
    """
    When no JD is provided, generate a virtual JD from LLM knowledge.
    Returns same structure as parse_jd_with_llm so downstream code is identical.
    """
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    prompt = VIRTUAL_JD_PROMPT.format(role=target_role)
    body = {
        "model":           MODEL,
        "max_tokens":      1000,
        "temperature":     0.2,  # slight creativity to get realistic variations
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a technical recruiter. Return only valid JSON."},
            {"role": "user",   "content": prompt},
        ],
    }
    try:
        resp = httpx.post(GROQ_API_URL, headers=headers, json=body, timeout=20)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        content = re.sub(r"```json\s*|```\s*", "", content).strip()
        parsed  = json.loads(content)
        parsed["jd_provided"] = False
        logger.info("Virtual JD generated for '%s': domain=%s required=%d",
                    target_role, parsed.get("domain"), len(parsed.get("required_skills", [])))
        return parsed
    except Exception as e:
        logger.warning("Virtual JD generation failed: %s", e)
        return {}