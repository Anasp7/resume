"""
Smart Resume — LLM-Powered Resume Parser v2
=============================================
Phase 3 fix: Better project detection with fuzzy section headers
"""

from __future__ import annotations
import json
import logging
import re
from typing import Optional

import httpx
from core.llm_client import call_llm_json
from core.schemas import ParsedResume, ProjectEntry, ExperienceEntry, EducationEntry, CertificationEntry

logger = logging.getLogger("smart_resume.llm_parser")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL        = "llama-3.3-70b-versatile"

EXTRACTION_PROMPT = """You are an expert resume parser. Extract ALL structured data precisely.

Return ONLY valid JSON — no explanation, no markdown, no code fences.

JSON structure:
{
  "name": "full name — first non-header line of resume",
  "email": "string or null",
  "phone": "string or null",
  "location": "city, state or district — string or null",
  "linkedin": "linkedin URL or handle or null",
  "github": "github URL or username or null",
  "summary": "full text of summary/objective section or null",
  "skills": ["technical skills only"],
  "experience": [
    {
      "role": "exact job title",
      "company": "exact company or org name",
      "duration": "date range string or null",
      "responsibilities": ["each bullet as separate string — preserve full sentence"]
    }
  ],
  "projects": [
    {
      "title": "project name",
      "description": "complete description — combine all lines under this project",
      "technologies": ["tech stack used"],
      "metrics": ["any measurable outcomes — empty list if none"]
    }
  ],
  "education": [
    {
      "institution": "school/college/university name",
      "degree": "full degree name",
      "graduation_year": number or null,
      "gpa": "GPA value as string or null"
    }
  ],
  "certifications": [
    {
      "name": "certification or course name",
      "issuer": "issuing organization or null",
      "year": number or null
    }
  ],
  "years_of_experience": number
}

CRITICAL EXTRACTION RULES:

SKILLS:
- Extract technical skills ONLY: programming languages, frameworks, libraries, tools, platforms, protocols
- Include: Python, C, SQL, ROS2, Docker, Git, FastAPI, TensorFlow, etc.
- NEVER include: English, Malayalam, Hindi, communication, teamwork, leadership, time management
- Extract skills from ALL sections — skills section, project descriptions, experience bullets

TECHNOLOGIES PER PROJECT — CRITICAL:
- technologies[] for each project must list ONLY tools actually used IN THAT project
- Do NOT copy global skills into every project's technologies[]
- Alumni Connect (DBMS/backend project): technologies should include SQL and any backend language mentioned
- Autonomous Vehicle / BAJA: technologies should include ROS2 and any simulation tool mentioned
- If a skill like "C" appears globally but is NOT mentioned in a project description, do NOT add it to that project's technologies[]

EXPERIENCE vs PROJECTS — strict separation:
- EXPERIENCE: candidate had a ROLE/POSITION at an ORGANIZATION
  → Put in experience[]: internships, jobs, club/team membership, competition team roles
  → Examples: "Automation Team Member at BAJA", "Backend Developer at Web Creation"
  → A competition team like BAJA SAE is an ORGANIZATION — the candidate's role there is EXPERIENCE
- PROJECTS: candidate BUILT/CREATED something (app, system, model, tool, simulation)
  → Put in projects[]: the actual thing they built, regardless of where they built it
  → Examples: "Autonomous Vehicle (built for BAJA)", "Alumni Connect Platform (built for Web Creation)"
  → The same work appears in BOTH: as a responsibility bullet in experience AND as a standalone project
- Rule: if "Student" or "Team Member" or "Developer" → EXPERIENCE. If "Platform", "System", "Model", "Simulation" → PROJECT.

TECHNOLOGIES in projects[]:
- ONLY list technologies that are EXPLICITLY mentioned in THAT project's description or bullets
- Do NOT copy the global skills list into every project
- Do NOT add C, Python, SQL to a project just because they are in the skills section
- Example: Alumni Connect Platform → technologies: ["sql"] (only SQL is in the description)
- Example: Autonomous Vehicle → technologies: ["python", "ros2"] (only if mentioned in that project's text)

RESPONSIBILITIES:
- Each responsibility bullet = separate string in array
- Preserve complete sentences — do not truncate
- If responsibilities are missing, use the role description

EDUCATION:
- Extract GPA/CGPA as string (e.g. "7.79", "8.5/10")
- graduation_year: the year of graduation or expected graduation

CONTACT:
- Extract LinkedIn URL or just the username after linkedin.com/in/
- Extract GitHub URL or username after github.com/

years_of_experience:
- Calculate from date ranges in experience section
- 0 for students/freshers with no work history
- Do not count projects as work experience

NEVER invent data. Use null or [] if information is missing.

Resume text:
"""



def parse_resume_with_llm(raw_text: str, api_key: str) -> dict:
    """Parse resume with multi-provider LLM — uses shared llm_client."""
    try:
        parsed = call_llm_json(
            EXTRACTION_PROMPT + raw_text, api_key,
            max_tokens=2000,
            system_msg="You are a resume parser. Return only valid JSON. No markdown, no explanation.",
        )
        logger.info("LLM extraction: %d skills, %d exp, %d projects",
                    len(parsed.get("skills", [])),
                    len(parsed.get("experience", [])),
                    len(parsed.get("projects", [])))
        return parsed
    except Exception as e:
        logger.warning("LLM resume parse failed: %s", e)
        return {}


def llm_output_to_parsed_resume(
    llm_data: dict,
    raw_text: str,
) -> Optional[ParsedResume]:
    """Convert LLM JSON output to ParsedResume schema."""
    if not llm_data:
        return None

    # ── Skills ────────────────────────────────────────────────────────────────
    ALWAYS_KEEP = {"c", "r", "go"}
    raw_skills  = llm_data.get("skills", [])
    skills = [
        s.lower().strip() for s in raw_skills
        if isinstance(s, str) and (len(s.strip()) > 1 or s.strip().lower() in ALWAYS_KEEP)
    ]

    # ── Experience ────────────────────────────────────────────────────────────
    experience = []
    for e in llm_data.get("experience", []):
        if not isinstance(e, dict):
            continue
        role    = (e.get("role") or "").strip()
        company = (e.get("company") or "").strip()
        if not role and not company:
            continue
        resp = [r for r in e.get("responsibilities", []) if isinstance(r, str) and r.strip()]
        experience.append(ExperienceEntry(
            role             = role,
            company          = company,
            duration         = e.get("duration") or "",
            responsibilities = resp,
            technologies     = [],
        ))

    # ── Projects ─────────────────────────────────────────────────────────────
    projects = []
    for p in llm_data.get("projects", []):
        if not isinstance(p, dict):
            continue
        title = (p.get("title") or "").strip()
        desc  = (p.get("description") or "").strip()
        if not title:
            continue
        # If description is empty but title has content — use title as desc
        if not desc:
            desc = title
        techs   = [t.lower() for t in p.get("technologies", []) if isinstance(t, str)]
        metrics = p.get("metrics", [])
        projects.append(ProjectEntry(
            title        = title,
            description  = desc,
            technologies = techs,
            metrics      = metrics if isinstance(metrics, list) else [],
        ))

    # ── Education ────────────────────────────────────────────────────────────
    education = []
    for e in llm_data.get("education", []):
        if not isinstance(e, dict):
            continue
        inst = (e.get("institution") or "").strip()
        deg  = (e.get("degree") or "").strip()
        if not inst and not deg:
            continue
        gpa_raw = e.get("gpa")
        try:
            gpa = float(str(gpa_raw).replace("/10", "").replace("/4", "")) if gpa_raw else None
        except Exception:
            gpa = None
        education.append(EducationEntry(
            institution     = inst,
            degree          = deg,
            graduation_year = e.get("graduation_year"),
            gpa             = gpa,
        ))

    # ── Certifications ────────────────────────────────────────────────────────
    certifications = []
    for c in llm_data.get("certifications", []):
        if not isinstance(c, dict):
            continue
        name = (c.get("name") or "").strip()
        if name:
            certifications.append(CertificationEntry(
                name   = name,
                year   = c.get("year"),
            ))

    # ── YoE ──────────────────────────────────────────────────────────────────
    yoe = float(llm_data.get("years_of_experience", 0) or 0)

    # Build contact line with LinkedIn/GitHub if present
    linkedin  = llm_data.get("linkedin") or ""
    github    = llm_data.get("github") or ""
    location  = llm_data.get("location") or ""

    return ParsedResume(
        name                 = llm_data.get("name") or "",
        email                = llm_data.get("email") or "",
        phone                = llm_data.get("phone") or "",
        location             = " | ".join(filter(None, [location, linkedin, github])) if (linkedin or github) else location,
        summary              = llm_data.get("summary") or "",
        raw_text             = raw_text,
        skills               = skills,
        projects             = projects,
        experience           = experience,
        education            = education,
        certifications       = certifications,
        claims_with_metrics  = [],
        years_of_experience  = yoe,
        project_count        = len(projects),
        experience_count     = len(experience),
    )