"""
Smart Resume — FastAPI App v3
==============================
Lives at project ROOT. Run with: python main.py
"""

from __future__ import annotations
import json, logging, re, uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_env_path)
    print(f"[startup] Loaded .env from {_env_path}")
except ImportError:
    pass

from core.parser import (
    parse_document, preprocess_for_embedding, SOFT_SKILLS,
    JUNK_TITLE_PATTERNS, is_junk_line, is_real_project_title,
    compute_ats_score as _compute_ats,
)
_is_junk_line          = is_junk_line
_is_real_project_title = is_real_project_title

from core.schemas import (
    BackendPayload, ClarificationAnswer, CertificationEntry,
    ParsedResume, ProjectEntry, ExperienceEntry, EducationEntry, SkillProficiency,
)
from core.similarity import (
    compute_similarity,
    decide_template, parse_job_description,
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if GROQ_API_KEY:
    print(f"[startup] GROQ_API_KEY loaded ({len(GROQ_API_KEY)} chars)")
else:
    print("[startup] WARNING: GROQ_API_KEY not set — AI features disabled")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("smart_resume.app")

app = FastAPI(title="Smart Resume", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_MAX_SESSIONS = 200
_sessions: dict[str, BackendPayload] = {}

def _store_session(key: str, value) -> None:
    if len(_sessions) >= _MAX_SESSIONS:
        oldest = next(iter(_sessions))
        del _sessions[oldest]
        logger.info("Session cap reached — evicted %s", oldest)
    _sessions[key] = value


# ─────────────────────────────────────────────────────────────────────────────
# ATS CHECKS (defined once)
# ─────────────────────────────────────────────────────────────────────────────

ATS_CHECKS = [
    (lambda t: bool(re.search(r"\|{3,}|\+[-+]{5,}", t)),
     "Possible table detected — ATS parsers like Taleo strip table content", "high"),
    (lambda t: bool(re.search(r"\b(references available|references on request)\b", t, re.I)),
     "Remove 'References available' — wastes space and modern ATS ignores it", "medium"),
    (lambda t: len(re.findall(r"[^\x00-\x7F]", t)) > 5,
     "Non-ASCII characters detected — may cause ATS garbling", "high"),
    (lambda t: bool(re.search(r"(CURRICULUM VITAE|\bCV\b)", t, re.I)),
     "Use 'Resume' not 'CV' for US/Canada ATS systems", "medium"),
    (lambda t: not bool(re.search(r"(EXPERIENCE|WORK|EMPLOYMENT)", t, re.I)),
     "No EXPERIENCE section detected — ATS requires it", "high"),
    (lambda t: not bool(re.search(r"(EDUCATION|DEGREE|UNIVERSITY|COLLEGE)", t, re.I)),
     "No EDUCATION section detected — ATS requires it", "high"),
    (lambda t: not bool(re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", t)),
     "No email address detected — ATS cannot identify candidate without it", "high"),
    (lambda t: bool(re.search(r"(?m)^.{121,}", t)),
     "Lines over 120 chars — some ATS parsers truncate long lines", "low"),
]

def run_ats_checks(resume_text: str) -> list[dict]:
    results = []
    if not resume_text:
        return results
    for check_fn, message, severity in ATS_CHECKS:
        try:
            if check_fn(resume_text):
                results.append({"issue": message, "severity": severity})
        except Exception:
            pass
    return results


# ─────────────────────────────────────────────────────────────────────────────
# RESUME BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _clean_summary(text: str) -> str:
    if not text: return ""
    import re
    text = re.sub(r"(?i)\d+\.?\d*/10\.?0?\s*(?:CGPA|GPA)?", "", text)
    text = re.sub(r"(?i)\d+\.?\d*%\s*(?:marks|percentage|aggregate)?", "", text)
    text = re.sub(r"(?i)\s+(?:at|from)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _build_parsed_resume_from_smart(smart_data: dict, parser_output: dict) -> ParsedResume:
    from core.schemas import ParsedResume, ExperienceEntry, ProjectEntry, EducationEntry
    from core.smart_parser import parse_resume as smart_parse, _extract_skills, _split_sections

    raw_text = parser_output.get("raw_text", "")

    _from_ollama  = smart_data.get("_from_ollama", False)
    _ollama_exp   = smart_data.get("experience", []) if _from_ollama else []
    _ollama_proj  = smart_data.get("projects", [])   if _from_ollama else []
    _ollama_skills = smart_data.get("skills", [])    if _from_ollama else []

    _ollama_has_exp = _from_ollama

    import hashlib
    from core.parser import _COL_CACHE
    col_info = None
    for h, info in _COL_CACHE.items():
        if info.get("is_two_col"):
            col_info = info
            break

    if col_info:
        left_text  = col_info["left"]
        right_text = col_info["right"]
        logger.info("Two-column parse: left=%d chars, right=%d chars",
                    len(left_text), len(right_text))

        left_parsed  = smart_parse(left_text)
        right_parsed = smart_parse(right_text)

        left_sections = _split_sections(left_text)
        skill_text = (left_sections.get("skills", "") or
                      left_sections.get("technical skills", "") or
                      left_text)
        mini_sec = {"technical skills": skill_text,
                    "experience": right_parsed.get("experience_raw", "")}
        skills = _extract_skills(mini_sec, left_text + "\n" + right_text)

        parser_contact = parser_output.get("contact_info", {})
        phone    = parser_contact.get("phone", "") or left_parsed.get("phone", "")
        email    = left_parsed.get("email", "") or parser_contact.get("email", "")

        loc_m = re.search(
            r"(?:Pattambi|Kerala|Palakkad|[A-Z][a-z]+)[.,\s]+(?:[A-Z][a-z]+)",
            left_text
        )
        location = loc_m.group().strip("., ") if loc_m else ""
        if not location:
            loc_m2 = re.search(r"[A-Z][a-z]{2,}[.,][A-Z][a-z]{2,}", left_text)
            location = loc_m2.group().replace(".", ", ") if loc_m2 else ""

        name = ""
        for line in right_text.splitlines():
            l = line.strip()
            if l and len(l.split()) >= 2 and not re.search(r"[@\d]", l):
                if l.upper() == l or l.istitle() or l[0].isupper():
                    name = l
                    break

        right_sections = _split_sections(right_text)
        exp_text_r  = right_sections.get("experience", "") or right_sections.get("work experience", "")
        proj_text_r = right_sections.get("projects", "")
        summary     = (right_sections.get("profile", "") or
                       right_sections.get("summary", "") or
                       right_sections.get("objective", ""))

        _DURATION_RE2 = re.compile(
            r"(\b20\d\d|\b19\d\d)\s*[-–—·•]\s*(\b20\d\d|\b19\d\d|present|current|now|)",
            re.I
        )
        _YEAR_RE2 = re.compile(r"\b(20\d\d|19\d\d)\b")

        experience = []
        _cur_block2: list[str] = []
        exp_raw_blocks2: list[list[str]] = []
        for _line in exp_text_r.splitlines():
            _stripped = _line.strip()
            _clean2   = _stripped.lstrip("•-* ")
            _has_year2 = bool(_YEAR_RE2.search(_stripped))
            _is_caps2  = (
                _clean2.upper() == _clean2 and 2 < len(_clean2) < 50 and
                not re.match(r"(Aim|Worked|Built|Developed|Implemented)", _clean2, re.I)
            )
            if (_has_year2 or _is_caps2) and _cur_block2:
                exp_raw_blocks2.append(_cur_block2)
                _cur_block2 = [_stripped]
            elif not _stripped and _cur_block2:
                exp_raw_blocks2.append(_cur_block2)
                _cur_block2 = []
            elif _stripped:
                _cur_block2.append(_stripped)
        if _cur_block2:
            exp_raw_blocks2.append(_cur_block2)

        for block_lines in exp_raw_blocks2:
            company2, role2, duration2 = "", "", ""
            resp2: list[str] = []
            for _l in block_lines:
                _c = _l.lstrip("•-* ").strip()
                if not _c:
                    continue
                if "|" in _c:
                    _parts = [p.strip() for p in _c.split("|")]
                    role2 = _parts[0]
                    company2 = _parts[1] if len(_parts) > 1 else company2
                    if len(_parts) > 2: duration2 = _parts[2]
                    continue
                _dm2 = _DURATION_RE2.search(_c)
                _ym2 = _YEAR_RE2.search(_c)
                if _dm2:
                    if not duration2: duration2 = _dm2.group()
                    _before2 = _c[:_dm2.start()].strip().lstrip("•-* ")
                    if _before2 and not company2: company2 = _before2
                    continue
                if _ym2 and not duration2:
                    _before2 = _c[:_ym2.start()].strip().lstrip("•-* ")
                    if _before2 and not company2:
                        company2 = _before2
                        duration2 = _ym2.group()
                    elif not company2:
                        duration2 = _c
                    continue
                if company2 and not role2:
                    role2 = _c
                    continue
                if _c and len(_c) > 5:
                    resp2.append(_c)
            if company2 or role2:
                experience.append(ExperienceEntry(
                    role=role2 or company2,
                    company=company2 or "Unknown",
                    duration=duration2,
                    responsibilities=resp2,
                    technologies=[s for s in skills if re.search(r'(?<![a-zA-Z0-9])' + re.escape(s) + r'(?![a-zA-Z0-9])', " ".join(resp2).lower())],
                ))

        from core.smart_parser import _reclassify_experience_entries
        experience_objs = [
            {
                "role":             e.role,
                "company":          e.company,
                "duration":         e.duration,
                "responsibilities": e.responsibilities,
                "technologies":     e.technologies,
            }
            for e in experience
        ]
        reclassified_exp_dicts, reclassified_proj_dicts = _reclassify_experience_entries(
            experience_objs, []
        )
        experience = [
            ExperienceEntry(
                role=e["role"], company=e["company"],
                duration=e["duration"], responsibilities=e["responsibilities"],
                technologies=e["technologies"],
            )
            for e in reclassified_exp_dicts
        ]

        left_sections2 = _split_sections(left_text)
        edu_text_l = left_sections2.get("education", "")
        _DEGREE_RE2 = re.compile(
            r"bachelor|b\.?\s*tech|b\.?\s*e\.?\b|master|phd|diploma|"
            r"plus two|10\+2|hsc|sslc|12th|10th|secondary", re.I
        )
        _GPA_RE2 = re.compile(r"(?:cgpa|gpa|grade)\s*[:/]?\s*([\d.]+)", re.I)

        educations = []
        edu_lines = [l.strip() for l in edu_text_l.splitlines() if l.strip()]
        ei = 0
        while ei < len(edu_lines):
            eline = edu_lines[ei].lstrip("•-* ").strip()
            only_yr = bool(re.match(r"^\d{4}\s*[-–·]?\s*\d{0,4}\s*$", eline))
            if only_yr:
                years = _YEAR_RE2.findall(eline)
                start_yr = int(years[0]) if years else None
                end_yr   = int(years[1]) if len(years) > 1 else None
                grad_yr  = end_yr or start_yr
                inst2, degree2, gpa2 = "", "", None
                for ej in range(ei+1, min(ei+6, len(edu_lines))):
                    nxt = edu_lines[ej].lstrip("•-* ").strip()
                    if _GPA_RE2.search(nxt):
                        gpa2 = _GPA_RE2.search(nxt).group(1)
                        continue
                    if _DEGREE_RE2.search(nxt):
                        degree2 = nxt
                        continue
                    if _YEAR_RE2.search(nxt):
                        break
                    if not inst2 and nxt:
                        inst2 = nxt
                try:
                    gpa2f = float(gpa2) if gpa2 else None
                except Exception:
                    gpa2f = None
                if inst2 or degree2:
                    educations.append(EducationEntry(
                        institution=inst2, degree=degree2 or inst2,
                        graduation_year=grad_yr, gpa=gpa2f,
                    ))
            ei += 1

        if not educations:
            educations = [EducationEntry(
                institution=edu_lines[0] if edu_lines else "",
                degree=edu_lines[1] if len(edu_lines) > 1 else "Degree",
                graduation_year=None, gpa=None,
            )]

        projects = []
        if proj_text_r.strip():
            from core.smart_parser import _extract_projects
            right_sec_for_proj = {"projects": proj_text_r}
            for p in _extract_projects(right_sec_for_proj):
                projects.append(ProjectEntry(
                    title=p["title"], description=p["description"],
                    technologies=p["technologies"],
                ))

        for p in reclassified_proj_dicts:
            p_title = p.get("title", "").lower()
            is_dupe = any(p_title in (e.company.lower() if e.company else "") for e in experience)
            if not is_dupe:
                projects.append(ProjectEntry(
                    title=p.get("title", "Project"),
                    description=p.get("description", ""),
                    technologies=p.get("technologies", []),
                ))

        if _from_ollama:
            logger.info("Using Ollama skills (%d) — replacing regex skills", len(_ollama_skills))
            skills = _ollama_skills
            logger.info("Using Ollama experience (%d) — replacing regex experience", len(_ollama_exp))
            experience = [
                ExperienceEntry(
                    role            = e.get("role") or "",
                    company         = e.get("company") or "",
                    duration        = e.get("duration") or "",
                    duration_certain= bool(e.get("duration")),
                    responsibilities= e.get("responsibilities") or [],
                    technologies    = e.get("technologies") or [],
                    is_project_mislabeled=False,
                )
                for e in _ollama_exp
            ]
            logger.info("Using Ollama projects (%d) — replacing regex projects", len(_ollama_proj))
            projects = []
            for _p in _ollama_proj:
                projects.append(ProjectEntry(
                    title=_p.get("title") or "Project",
                    description=_p.get("description") or "",
                    technologies=_p.get("technologies") or [],
                ))

        return ParsedResume(
            raw_text  = raw_text,
            name      = name,
            phone     = phone,
            email     = email,
            location  = location,
            linkedin  = left_parsed.get("linkedin", ""),
            github    = left_parsed.get("github", ""),
            skills    = skills,
            experience= experience,
            projects  = projects,
            education = [EducationEntry(**e) for e in smart_data["education"]] if smart_data.get("education") else educations,
            summary   = _clean_summary(summary.strip()),
        )

    else:
        pr = _build_parsed_resume(parser_output)
        from core.smart_parser import _extract_skills
        sections_raw = parser_output.get("sections", {})
        skill_section_text = (sections_raw.get("skills", "") or
                              sections_raw.get("technical skills", ""))
        mini_sections = {
            "technical skills": skill_section_text,
            "experience":       sections_raw.get("experience", ""),
            "projects":         sections_raw.get("projects", ""),
        }
        smart_skills = _extract_skills(mini_sections, raw_text)
        if smart_skills:
            pr.skills = smart_skills

        if "education" in smart_data and smart_data["education"]:
            pr.education = [EducationEntry(**e) for e in smart_data["education"]]

        if _from_ollama:
            logger.info("[single-col] Using Ollama skills (%d)", len(_ollama_skills))
            pr.skills = _ollama_skills
            logger.info("[single-col] Using Ollama experience (%d)", len(_ollama_exp))
            pr.experience = [
                ExperienceEntry(
                    role            = e.get("role") or "",
                    company         = e.get("company") or "",
                    duration        = e.get("duration") or "",
                    duration_certain= bool(e.get("duration")),
                    responsibilities= e.get("responsibilities") or [],
                    technologies    = e.get("technologies") or [],
                    is_project_mislabeled=False,
                )
                for e in _ollama_exp
            ]
            logger.info("[single-col] Using Ollama projects (%d)", len(_ollama_proj))
            pr.projects = []
            for _p in _ollama_proj:
                pr.projects.append(ProjectEntry(
                    title=_p.get("title") or "Project",
                    description=_p.get("description") or "",
                    technologies=_p.get("technologies") or [],
                ))

        pr.summary = _clean_summary(pr.summary)
        return pr


def _build_parsed_resume(parser_output: dict) -> ParsedResume:
    sections   = parser_output["sections"]
    raw_text   = parser_output["raw_text"]
    mismatches = parser_output["mismatches"]

    logger.info("Building resume. Sections: %s", list(sections.keys()))

    skill_text  = sections.get("skills", "")
    raw_text_sk = parser_output.get("raw_text", "")

    LANG_TERMS_P = {"python","java","javascript","typescript","c","c++","c#","go","rust",
                    "kotlin","swift","r","scala","ruby","php","bash","dart","matlab"}
    FWK_TERMS_P  = {"ros2","ros","react","vue","angular","nextjs","django","flask","fastapi",
                    "spring","pytorch","tensorflow","keras","express","flutter","langchain"}
    DB_TERMS_P   = {"sql","postgresql","mysql","sqlite","mongodb","redis","firebase",
                    "cassandra","dynamodb","elasticsearch"}
    TOOL_TERMS_P = {"docker","kubernetes","git","github","linux","aws","gcp","azure",
                    "jenkins","terraform","grafana","postman","gazebo","cmake","arduino"}
    ALWAYS_KEEP_SKILLS = {"c","r","go"}
    LABEL_WORDS  = {"languages","language","softwares","software","tools","tool",
                    "frameworks","framework","libraries","library","others","other",
                    "technical","technologies","technology","stack","skills","skill",
                    "databases","database","and","the","a","an","of","for","in","is","to"}

    raw_candidates = []
    for line in skill_text.splitlines():
        val = re.sub(r"^[A-Za-z ]+[:/]\s*", "", line).strip()
        raw_candidates += re.split(r"[,.|/;\s]+", val)
    for line in raw_text_sk.splitlines():
        if re.search(r"tech\s*:", line, re.I):
            tech = re.sub(r".*tech\s*:\s*", "", line, flags=re.I)
            raw_candidates += re.split(r"[,\s]+", tech)
    raw_candidates += parser_output.get("inline_skills", [])

    classified_sk = set()
    langs_p, fwks_p, dbs_p, tools_p = [], [], [], []
    for s in raw_candidates:
        s = s.strip().lower().strip("()")
        if not s or s in LABEL_WORDS or s in SOFT_SKILLS: continue
        if not (s in ALWAYS_KEEP_SKILLS or (1 < len(s) < 30)): continue
        if s.replace(" ","").isdigit(): continue
        if s in classified_sk: continue
        if   s in LANG_TERMS_P:  langs_p.append(s);  classified_sk.add(s)
        elif s in FWK_TERMS_P:   fwks_p.append(s);   classified_sk.add(s)
        elif s in DB_TERMS_P:    dbs_p.append(s);    classified_sk.add(s)
        elif s in TOOL_TERMS_P:  tools_p.append(s);  classified_sk.add(s)

    skills = list(dict.fromkeys(langs_p + fwks_p + dbs_p + tools_p))
    logger.info("Skills (regex robust): %s", skills)

    exp_text   = sections.get("experience", "")
    experience: list[ExperienceEntry] = []

    _DURATION_RE = re.compile(
        r"(\b20\d\d|\b19\d\d)\s*[-–—·•]\s*(\b20\d\d|\b19\d\d|present|current|now|)",
        re.I
    )
    if exp_text.strip():
        from core.smart_parser import _extract_experience
        exp_sec = {"experience": exp_text}
        for e in _extract_experience(exp_sec):
            experience.append(ExperienceEntry(
                company=e.get("company", ""),
                role=e.get("role", ""),
                duration=e.get("duration", ""),
                duration_certain=bool(e.get("duration", "")),
                responsibilities=e.get("responsibilities", []),
                technologies=e.get("technologies", []),
                is_project_mislabeled=False,
            ))

    from core.smart_parser import _reclassify_experience_entries
    exp_dicts = [
        {"role": e.role, "company": e.company, "duration": e.duration,
         "responsibilities": e.responsibilities, "technologies": e.technologies}
        for e in experience
    ]
    reclassified_exp, reclassified_proj = _reclassify_experience_entries(exp_dicts, [])
    experience = [
        ExperienceEntry(
            role=e["role"], company=e["company"], duration=e["duration"],
            responsibilities=e["responsibilities"], technologies=e["technologies"],
        )
        for e in reclassified_exp
    ]

    proj_text = sections.get("projects", "")
    projects: list[ProjectEntry] = []
    if proj_text.strip():
        current_title = None
        desc_lines = []
        action_verb_re = re.compile(r"^(?:built|developed|designed|created|implemented|collaborated|used|led|managed|worked|participated|assisted|integrated|added|optimized|improved|idea|idea:)\b", re.I)

        def save_current_project():
            if current_title:
                desc = " ".join(desc_lines).strip() if desc_lines else current_title
                techs = [s for s in skills if re.search(r'(?<![a-zA-Z0-9])' + re.escape(s) + r'(?![a-zA-Z0-9])', (current_title + " " + desc).lower())]
                metrics = re.findall(r"\d+[%x]|\d+\s*(?:ms|seconds?|users?|requests?)", desc, re.IGNORECASE)
                if len(desc) > 5 and not is_junk_line(current_title):
                    projects.append(ProjectEntry(title=current_title, description=desc, technologies=techs, metrics=metrics))

        for line in proj_text.splitlines():
            stripped = line.strip()
            if not stripped: continue
            is_desc = (
                stripped.startswith(("-", "•", "*", "·", "▸", "◦")) or
                stripped.endswith(".") or
                action_verb_re.match(stripped) or
                len(stripped) > 85
            )
            if not is_desc and len(stripped) > 2:
                save_current_project()
                current_title = stripped
                desc_lines = []
            elif current_title:
                clean_line = stripped.lstrip("•-*·▸◦ ").strip()
                if clean_line and not is_junk_line(clean_line):
                    desc_lines.append(clean_line)

        save_current_project()

    for p in reclassified_proj:
        projects.append(ProjectEntry(
            title=p.get("title","Project"),
            description=p.get("description",""),
            technologies=p.get("technologies",[]),
        ))

    edu_text   = sections.get("education", "")
    educations: list[EducationEntry] = []
    if edu_text:
        _DEGREE_RE = re.compile(
            r"bachelor|b\.?\s*tech|b\.?\s*e\.?\b|master|m\.?\s*tech|phd|diploma|"
            r"plus two|10\+2|hsc|sslc|12th|10th|secondary|higher secondary", re.I
        )
        _GPA_RE   = re.compile(r"(?:cgpa|gpa|grade|percentage)\s*[:/]?\s*([\d.]+)", re.I)
        _YEAR_RE2 = re.compile(r"\b(20\d\d|19\d\d)\b")
        _RANGE_RE = re.compile(r"(\b(?:20|19)\d\d)\s*[-–]\s*(\b(?:20|19)\d\d|\s*$)", re.M)
        edu_lines = [l.strip() for l in edu_text.splitlines() if l.strip()]
        i = 0
        while i < len(edu_lines):
            line = edu_lines[i]
            clean = line.lstrip("•-* ").strip()
            rng_m  = _RANGE_RE.search(clean)
            only_yr = bool(re.match(r"^\d{4}\s*[-–]?\s*\d{0,4}\s*$", clean))
            if only_yr or rng_m:
                years    = _YEAR_RE2.findall(clean)
                start_yr = int(years[0]) if years else None
                end_yr   = int(years[1]) if len(years) > 1 else None
                grad_yr  = end_yr or start_yr
                institution, degree, gpa = "", "", None
                for j in range(i+1, min(i+6, len(edu_lines))):
                    nxt = edu_lines[j].lstrip("•-* ").strip()
                    gm  = _GPA_RE.search(nxt)
                    if gm: gpa = gm.group(1); continue
                    if _DEGREE_RE.search(nxt): degree = nxt; continue
                    if _YEAR_RE2.search(nxt): break
                    if not institution and nxt: institution = nxt
                try:    gpa_f = float(gpa) if gpa else None
                except: gpa_f = None
                if institution or degree:
                    educations.append(EducationEntry(
                        institution=institution, degree=degree or institution,
                        graduation_year=grad_yr, gpa=gpa_f,
                    ))
            i += 1
        if not educations and edu_lines:
            educations.append(EducationEntry(
                institution=edu_lines[0],
                degree=edu_lines[1] if len(edu_lines) > 1 else "Degree",
                graduation_year=None,
            ))

    certifications: list[CertificationEntry] = []
    for line in sections.get("certifications", "").splitlines():
        line = line.strip()
        if len(line) > 5 and not is_junk_line(line):
            ym = re.search(r"(20\d\d)", line)
            certifications.append(CertificationEntry(
                name=line, year=int(ym.group(1)) if ym else None
            ))

    claims    = re.findall(r"[^.]*\d+[%x][^.]*\.", raw_text)
    years_exp = re.findall(r"(20\d\d|19\d\d)", exp_text)
    yoe = 0.0
    if len(years_exp) >= 2:
        yi  = sorted(set(int(y) for y in years_exp))
        yoe = float(min(yi[-1] - yi[0], 30))

    logger.info("Final: %d skills, %d projects, %d exp, %d edu",
                len(skills), len(projects), len(experience), len(educations))

    return ParsedResume(
        raw_text=raw_text, skills=skills, projects=projects,
        experience=experience, education=educations,
        certifications=certifications, claims_with_metrics=claims[:20],
        years_of_experience=yoe, project_count=len(projects),
        experience_count=len(experience),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    import shutil, subprocess
    node_available = False
    try:
        node_cmd = shutil.which("node")
        if not node_cmd:
            for candidate in [
                r"C:\Program Files\nodejs\node.exe",
                r"C:\Program Files (x86)\nodejs\node.exe",
            ]:
                if os.path.isfile(candidate):
                    node_cmd = candidate
                    break
        if node_cmd:
            result = subprocess.run([node_cmd, "--version"], capture_output=True, timeout=5)
            node_available = result.returncode == 0
    except Exception:
        node_available = False

    try:
        from core.similarity import get_similarity_mode
        sim_mode = get_similarity_mode()
    except Exception:
        sim_mode = "unknown"

    return {
        "status":          "ok",
        "groq_configured": bool(GROQ_API_KEY),
        "node_available":  node_available,
        "session_count":   len(_sessions),
        "session_cap":     _MAX_SESSIONS,
        "similarity_mode": sim_mode,
    }


@app.post("/analyze")
async def analyze(
    resume_file:      UploadFile    = File(...),
    job_description:  Optional[str] = Form(default=None),
    target_role:      Optional[str] = Form(default=None),
    proficiency_json: Optional[str] = Form(default=None),
):
    if not target_role or not target_role.strip():
        target_role = "Software Engineer"
    file_bytes = await resume_file.read()
    if not file_bytes:
        raise HTTPException(400, "Uploaded file is empty.")

    try:
        parser_output = parse_document(file_bytes, resume_file.filename or "resume.pdf")
    except Exception as e:
        raise HTTPException(422, f"Could not parse resume: {e}")

    from core.ollama_pipeline import run_ollama_pipeline, parse_and_structure, parse_education_ollama
    from core.smart_parser import parse_resume as smart_parse

    raw_text     = parser_output["raw_text"]
    target_role_ = target_role or "Software Engineer"

    regex_data = smart_parse(raw_text)
    logger.info(
        "Regex parser: skills=%d exp=%d proj=%d edu=%d",
        len(regex_data.get("skills", [])),
        len(regex_data.get("experience", [])),
        len(regex_data.get("projects", [])),
        len(regex_data.get("education", [])),
    )

    logger.info("Attempting Ollama semantic parse (skills/exp/projects)...")
    llm_data = parse_and_structure(raw_text)

    if llm_data and (llm_data.get("experience") or llm_data.get("skills") or llm_data.get("projects")):
        logger.info(
            "Ollama semantic parse: skills=%d exp=%d proj=%d — merging with Regex",
            len(llm_data.get("skills", [])),
            len(llm_data.get("experience", [])),
            len(llm_data.get("projects", [])),
        )
        llm_education = parse_education_ollama(raw_text)
        raw_summary = llm_data.get("summary") or regex_data.get("summary", "")
        clean_sum   = _clean_summary(raw_summary)
        smart_data = {
            "name":       regex_data.get("name", ""),
            "email":      regex_data.get("email", ""),
            "phone":      regex_data.get("phone", ""),
            "location":   regex_data.get("location", ""),
            "linkedin":   regex_data.get("linkedin", ""),
            "github":     regex_data.get("github", ""),
            "education":  llm_education or llm_data.get("education") or regex_data.get("education", []),
            "summary":    clean_sum,
            "skills":     regex_data.get("skills", []) or llm_data.get("skills", []),
            "experience": llm_data.get("experience", []),
            "projects":   llm_data.get("projects", []),
            "_from_ollama": True,
        }
        logger.info("Hybrid merge complete — Ollama education=%d proj=%d exp=%d",
                    len(smart_data["education"]), len(smart_data["projects"]), len(smart_data["experience"]))
    else:
        logger.info("Ollama unavailable or empty — using Regex data only (fallback)")
        smart_data = regex_data

    parsed_resume = _build_parsed_resume_from_smart(smart_data, parser_output)

    ollama_result = run_ollama_pipeline(raw_text, target_role_, parsed_context=smart_data)
    _ollama_doubt_questions = ollama_result.get("doubt_questions", [])
    logger.info("Ollama pre-generated %d doubt questions", len(_ollama_doubt_questions))

    logger.info(
        "Final parsed resume: skills=%d exp=%d proj=%d edu=%d",
        len(parsed_resume.skills or []),
        len(parsed_resume.experience or []),
        len(parsed_resume.projects or []),
        len(parsed_resume.education or []),
    )

    contact = parser_output.get("contact_info", {})
    if contact.get("name")     and not parsed_resume.name:     parsed_resume.name     = contact["name"]
    if contact.get("email")    and not parsed_resume.email:    parsed_resume.email    = contact["email"]
    if contact.get("phone")    and not parsed_resume.phone:    parsed_resume.phone    = contact["phone"]
    if contact.get("location") and not parsed_resume.location: parsed_resume.location = contact["location"]
    if contact.get("linkedin"): parsed_resume.linkedin = contact["linkedin"]
    if contact.get("github"):   parsed_resume.github   = contact["github"]

    jd_text = (job_description or "").strip()
    if GROQ_API_KEY:
        from core.jd_parser_llm import parse_jd_with_llm, generate_virtual_jd
        if jd_text:
            llm_jd = parse_jd_with_llm(jd_text, GROQ_API_KEY)
        else:
            logger.info("No JD provided — generating virtual JD for role: %s", target_role)
            llm_jd = generate_virtual_jd(target_role, GROQ_API_KEY)
        from core.schemas import Domain
        domain_map = {
            "ml": Domain.ML, "machine learning": Domain.ML,
            "backend": Domain.BACKEND, "frontend": Domain.FRONTEND,
            "devops": Domain.DEVOPS, "data": Domain.DATA, "unknown": Domain.UNKNOWN,
            "embedded": Domain.UNKNOWN, "security": Domain.UNKNOWN, "mobile": Domain.UNKNOWN,
        }
        detected_domain = domain_map.get(llm_jd.get("domain","unknown").lower(), Domain.UNKNOWN)
        from core.schemas import ParsedJobDescription
        virtual_text = llm_jd.get("virtual_jd_text", jd_text or target_role)
        parsed_jd = ParsedJobDescription(
            raw_text                  = virtual_text,
            target_role               = target_role,
            detected_domain           = detected_domain,
            required_skills           = [s.lower() for s in llm_jd.get("required_skills", [])],
            preferred_skills          = [s.lower() for s in llm_jd.get("preferred_skills", [])],
            required_experience_years = llm_jd.get("required_experience_years"),
            key_responsibilities      = llm_jd.get("key_responsibilities", []),
            jd_provided               = bool(jd_text),
        )
    else:
        parsed_jd = parse_job_description(jd_text or target_role, target_role)
        parsed_jd.jd_provided = bool(jd_text)

    user_proficiencies: list[SkillProficiency] = []
    if proficiency_json:
        try:
            user_proficiencies = [SkillProficiency(**i) for i in json.loads(proficiency_json)]
        except Exception:
            pass

    jd_preprocessed = preprocess_for_embedding(parsed_jd.raw_text)
    similarity_score = compute_similarity(
        parser_output["preprocessed_text"], jd_preprocessed
    )
    ats_result = _compute_ats(
        raw_text    = parser_output["raw_text"],
        sections    = parser_output["sections"],
        skills      = parsed_resume.skills,
        target_role = target_role,
    )
    selected_template  = decide_template(parsed_resume)
    needs_optimization = True

    session_id = str(uuid.uuid4())
    payload = BackendPayload(
        resume_raw_text          = parser_output["raw_text"],
        job_description_raw_text = jd_text,
        target_role              = target_role,
        parsed_resume            = parsed_resume,
        parsed_jd                = parsed_jd,
        semantic_similarity_score= similarity_score,
        user_proficiencies       = user_proficiencies,
        clarification_answers    = None,
        selected_template        = selected_template,
        needs_optimization       = needs_optimization,
        session_id               = session_id,
        timestamp                = datetime.now(timezone.utc).isoformat(),
    )
    _store_session(session_id, payload)
    if _ollama_doubt_questions:
        try:
            object.__setattr__(payload, "_ollama_doubt_questions", _ollama_doubt_questions)
        except Exception:
            payload.__dict__["_ollama_doubt_questions"] = _ollama_doubt_questions

    logger.info("Session %s | sim=%.1f | template=%s | ats=%d",
                session_id, similarity_score,
                getattr(selected_template,"value",str(selected_template)),
                ats_result["score"])

    payload_dict = payload.model_dump()
    payload_dict["ats_score"]   = ats_result["score"]
    payload_dict["ats_verdict"] = ats_result["verdict"]
    payload_dict["ats_flags"]   = ats_result["flags"]
    payload_dict["ats_details"] = ats_result["details"]
    return payload_dict


@app.post("/clarify/{session_id}")
async def submit_clarification(session_id: str, answers: list[ClarificationAnswer]):
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found.")
    if _sessions[session_id].clarification_answers is None:
        _sessions[session_id].clarification_answers = []
    _sessions[session_id].clarification_answers.extend(answers)
    return _sessions[session_id]


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found.")
    return _sessions[session_id]


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 ROUTES
# ─────────────────────────────────────────────────────────────────────────────

from core.evaluator import evaluate
from core.schemas import SmartResumeResponse


@app.post("/evaluate/{session_id}", response_model=SmartResumeResponse)
async def evaluate_resume(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found. Run /analyze first.")
    if not GROQ_API_KEY:
        raise HTTPException(400, "GROQ_API_KEY not set. Cannot run evaluation.")
    payload = _sessions[session_id]
    try:
        result = evaluate(payload, GROQ_API_KEY)
        if result.final_resume:
            ats_issues = run_ats_checks(result.final_resume)
            if ats_issues:
                rqa        = result.resume_quality_assessment or {}
                high_count = sum(1 for i in ats_issues if i["severity"] == "high")
                rqa["ats_flags"] = rqa.get("ats_flags",[]) + [
                    i["issue"] for i in ats_issues if i["severity"] == "high"
                ]
                if high_count >= 2: rqa["ats_compatibility"] = "Low"
                elif high_count == 1: rqa["ats_compatibility"] = "Medium"
                result.resume_quality_assessment = rqa
        _store_session(f"eval_{session_id}", result)
        return result
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        raise HTTPException(500, f"Evaluation failed: {e}")


@app.get("/evaluate/{session_id}/result", response_model=SmartResumeResponse)
async def get_evaluation_result(session_id: str):
    key = f"eval_{session_id}"
    if key not in _sessions:
        raise HTTPException(404, "No evaluation found.")
    return _sessions[key]


@app.get("/evaluate/{session_id}/download/docx")
async def download_resume_docx(session_id: str):
    from fastapi.responses import Response
    from core.resume_exporter import export_resume
    key = f"eval_{session_id}"
    if key not in _sessions:
        raise HTTPException(404, "No evaluation found.")
    result = _sessions[key]
    if not result.final_resume:
        raise HTTPException(400, "No optimized resume available.")
    try:
        payload       = _sessions.get(session_id)
        parsed_resume = payload.parsed_resume if payload else None
        if not parsed_resume:
            raise HTTPException(400, "Session data missing.")
        docx_bytes = export_resume(result.final_resume, parsed_resume, fmt="docx")
        safe_name  = re.sub(r"[^a-zA-Z0-9_\-]", "_", parsed_resume.name or "Resume")
        return Response(
            content    = docx_bytes,
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers    = {"Content-Disposition": f"attachment; filename={safe_name}_resume.docx"},
        )
    except Exception as e:
        raise HTTPException(500, f"DOCX generation failed: {e}")


@app.get("/evaluate/{session_id}/download/pdf")
async def download_resume_pdf(session_id: str):
    from fastapi.responses import Response
    from core.resume_exporter import export_resume, compile_latex_to_pdf
    key = f"eval_{session_id}"
    if key not in _sessions:
        raise HTTPException(404, "No evaluation found.")
    result        = _sessions[key]
    payload       = _sessions.get(session_id)
    parsed_resume = payload.parsed_resume if payload else None
    if not result.final_resume and not result.latex_template:
        raise HTTPException(400, "No resume content available.")
    try:
        pdf_bytes = None
        if result.latex_template:
            try:
                pdf_bytes = compile_latex_to_pdf(result.latex_template)
            except Exception as e:
                logger.warning("LaTeX failed, using reportlab: %s", e)
        if not pdf_bytes:
            if not result.final_resume or not parsed_resume:
                raise ValueError("No content for PDF generation")
            pdf_bytes = export_resume(result.final_resume, parsed_resume, fmt="pdf")
        safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_",
                           (parsed_resume.name if parsed_resume else None) or "Resume")
        return Response(
            content    = pdf_bytes,
            media_type = "application/pdf",
            headers    = {"Content-Disposition": f"attachment; filename={safe_name}_optimized.pdf"},
        )
    except Exception as e:
        logger.error("PDF failed for %s: %s", session_id, e, exc_info=True)
        raise HTTPException(500, f"PDF generation failed: {e}")


@app.post("/evaluate/{session_id}/clarify", response_model=SmartResumeResponse)
async def re_evaluate_with_clarification(session_id: str, request: Request):
    from core.schemas import ClarificationAnswer
    from core.doubt_engine import verify_and_map_profile_answers
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found.")
    if not GROQ_API_KEY:
        raise HTTPException(400, "GROQ_API_KEY not set.")
    try:
        body = await request.json()
        raw_answers = body if isinstance(body, list) else body.get("answers", [])
    except Exception:
        raw_answers = []

    payload = _sessions[session_id]
    typed = [
        ClarificationAnswer(**a) if isinstance(a, dict) else a for a in raw_answers
    ]
    payload.clarification_answers = typed
    payload.needs_optimization = True

    NIL_WORDS = {"nil", "none", "no", "skip", "n/a", "na", "not applicable"}

    profile_map_answers = []
    for ca in typed:
        q = (getattr(ca, "question", "") or "").lower()
        a = (getattr(ca, "answer",   "") or "").strip()

        if any(k in q for k in {"github"}):
            profile_map_answers.append({"type": "github", "answer": a})
        elif any(k in q for k in {"linkedin"}):
            profile_map_answers.append({"type": "linkedin", "answer": a})
        elif any(k in q for k in {"plus two", "12th", "hsc", "higher secondary", "class 12"}):
            if a.lower() in NIL_WORDS:
                try:    object.__setattr__(payload.parsed_resume, "school_12th", None)
                except: payload.parsed_resume.__dict__["school_12th"] = None
            else:
                profile_map_answers.append({"type": "school_12th_percentage", "answer": a})
        elif any(k in q for k in {"10th", "sslc", "class 10", "class x"}):
            if a.lower() in NIL_WORDS:
                try:    object.__setattr__(payload.parsed_resume, "school_10th", None)
                except: payload.parsed_resume.__dict__["school_10th"] = None
            else:
                profile_map_answers.append({"type": "school_10th_percentage", "answer": a})
        elif any(k in q for k in {"cgpa", "gpa", "academic percentage"}):
            profile_map_answers.append({"type": "cgpa", "answer": a})
        elif any(k in q for k in {"certification", "course", "nptel", "coursera"}):
            profile_map_answers.append({"type": "certifications", "answer": a})
        elif any(k in q for k in {"hackathon", "competition", "contest", "baja", "robocon"}):
            profile_map_answers.append({"type": "hackathons", "answer": a})
        elif any(k in q for k in {"internship", "intern"}):
            if a.lower() not in NIL_WORDS:
                profile_map_answers.append({"type": "internship", "answer": a})
        elif any(k in q for k in {"workshop", "training", "bootcamp", "seminar"}):
            if a.lower() not in NIL_WORDS:
                profile_map_answers.append({"type": "workshops", "answer": a})
        elif any(k in q for k in {"branch", "specialisation", "specialization", "department"}):
            profile_map_answers.append({"type": "degree_branch", "answer": a})
        elif any(k in q for k in {"technology", "tech stack", "tools used", "what did you use"}):
            profile_map_answers.append({"type": "additional_skills", "answer": a})
        elif any(k in q for k in {"additional project", "other project"}):
            profile_map_answers.append({"type": "additional_projects", "answer": a})
        elif any(k in q for k in {"additional skill", "other skill", "technical skill"}):
            profile_map_answers.append({"type": "additional_skills", "answer": a})

    if profile_map_answers:
        try:
            updated_resume, change_log = verify_and_map_profile_answers(
                payload.parsed_resume, profile_map_answers
            )
            payload.parsed_resume = updated_resume
            logger.info("Clarify: mapped %d profile answers — %s",
                        len(change_log), "; ".join(change_log[:3]))
        except Exception as e:
            logger.warning("Clarify profile mapping failed: %s", e)

    try:
        result = evaluate(payload, GROQ_API_KEY)
        _store_session(f"eval_{session_id}", result)
        return result
    except Exception as e:
        logger.error("Re-evaluation error %s: %s", session_id, e, exc_info=True)
        raise HTTPException(500, f"Re-evaluation failed: {e}")


@app.post("/evaluate/{session_id}/profile-answers", response_model=SmartResumeResponse)
async def submit_profile_answers(session_id: str, request: Request):
    from core.doubt_engine import verify_and_map_profile_answers
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found.")
    if not GROQ_API_KEY:
        raise HTTPException(400, "GROQ_API_KEY not set.")
    try:
        body = await request.json()
        raw_profile_answers = body if isinstance(body, list) else body.get("answers", [])
    except Exception:
        raw_profile_answers = []
    if not raw_profile_answers:
        raise HTTPException(400, "No profile answers provided.")
    payload = _sessions[session_id]
    try:
        updated_resume, change_log = verify_and_map_profile_answers(
            payload.parsed_resume, raw_profile_answers
        )
    except Exception as e:
        logger.error("STEP Y mapping failed: %s", e, exc_info=True)
        raise HTTPException(500, f"Profile answer mapping failed: {e}")
    payload.parsed_resume  = updated_resume
    payload.needs_optimization = True
    object.__setattr__(payload, "_profile_change_log", change_log)
    logger.info("STEP Y: %d changes — %s", len(change_log), "; ".join(change_log[:3]))
    try:
        result = evaluate(payload, GROQ_API_KEY)
        if not result.profile_change_log and change_log:
            result.profile_change_log = change_log
        _store_session(f"eval_{session_id}", result)
        return result
    except Exception as e:
        logger.error("Re-eval after profile answers failed %s: %s", session_id, e, exc_info=True)
        raise HTTPException(500, f"Re-evaluation failed: {e}")


# ─── Career Growth Endpoints ──────────────────────────────────────────────────

class CareerDetectRequest(BaseModel):
    resume_text: str
    parsed_resume_dict: dict

@app.post("/career/detect")
async def detect_career(req: CareerDetectRequest):
    from core.growth.recommender import detect_all_suitable_roles
    roles, level, analysis = detect_all_suitable_roles(req.parsed_resume_dict, req.resume_text)
    return {"roles": roles, "level": level, "analysis": analysis}

class CareerPlanRequest(BaseModel):
    resume_text: str
    parsed_resume_dict: dict
    selected_roles: list
    career_level: str

@app.post("/career/plan")
async def generate_career_plan(req: CareerPlanRequest):
    from core.growth.recommender import generate_detailed_growth_plan
    plan = generate_detailed_growth_plan(
        resume_data=req.parsed_resume_dict,
        selected_roles=req.selected_roles,
        raw_resume_text=req.resume_text,
        career_level=req.career_level
    )
    return plan


# ─── Resume Intelligence Engine Endpoints ────────────────────────────────────

@app.get("/api/dataset-stats")
async def dataset_stats():
    try:
        from core.ats_dataset import dataset_stats as _stats
        return _stats()
    except Exception as e:
        return {"error": str(e), "scoring_samples": 0, "ready_to_train": False}


class ScoreResumeRequest(BaseModel):
    resume_text: str
    optimized_resume: str = ""
    target_role:      str = ""

@app.post("/api/score-resume")
async def score_resume(req: ScoreResumeRequest):
    try:
        from core.ats_model import predict, is_model_ready
        from core.ats_dataset import dataset_stats

        if not is_model_ready():
            stats = dataset_stats()
            return {
                "status":          "model_not_ready",
                "message":         f"Collecting training data — {stats['scoring_samples']} samples so far. Run train_ats_model.py when ready.",
                "scoring_samples": stats["scoring_samples"],
                "ready_to_train":  stats["ready_to_train"],
            }

        result = predict(req.resume_text, req.optimized_resume, req.target_role)
        if result is None:
            return {"status": "model_not_ready", "message": "Model returned no prediction."}

        from core.evaluator import ats_score
        rule_score = ats_score(req.resume_text)

        return {
            "status":       "ok",
            "ml_scores":    result,
            "rule_scores":  rule_score,
        }

    except Exception as e:
        logger.warning("score_resume endpoint error: %s", e)
        return {"status": "error", "message": str(e)}


# ─── Career Recommender Endpoints ─────────────────────────────────────────────

import httpx as _httpx

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"

class CareerRecommendRequest(BaseModel):
    resume_text: str

class CareerDetailRequest(BaseModel):
    resume_text: str
    job_title: str

def _call_groq(prompt: str, groq_key: str, max_tokens: int = 2000) -> str:
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": GROQ_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    resp = _httpx.post(GROQ_API_URL, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def _parse_json_from_text(text: str) -> dict:
    text = re.sub(r"```json|```", "", text).strip()
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON found in response")
    return json.loads(text[start:end+1])

@app.post("/career/recommend")
async def career_recommend(req: CareerRecommendRequest):
    if not GROQ_API_KEY:
        raise HTTPException(400, "GROQ_API_KEY not configured.")
    prompt = f"""Analyse this resume and return ONLY a raw JSON object. No markdown, no backticks.

RESUME:
{req.resume_text}

Extract every skill explicitly written. Compute match % strictly.
Pick best 5 roles from: Full Stack Developer, AI/ML Engineer, Frontend Developer, Backend Developer, Software Engineer.

Return:
{{
  "name": "candidate name",
  "level": "Fresher or Junior or Mid or Senior",
  "readiness_score": 70,
  "readiness_reason": "one sentence",
  "extracted_skills": ["skill1"],
  "top_jobs": [
    {{
      "title": "role",
      "match_percent": 80,
      "skills_you_have": ["skill1"],
      "skills_missing": ["skill1"]
    }}
  ]
}}"""
    try:
        raw  = _call_groq(prompt, GROQ_API_KEY)
        return _parse_json_from_text(raw)
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")

@app.post("/career/recommend/detail")
async def career_recommend_detail(req: CareerDetailRequest):
    if not GROQ_API_KEY:
        raise HTTPException(400, "GROQ_API_KEY not configured.")
    prompt = f"""Given this resume and target job "{req.job_title}", return ONLY a raw JSON object. No markdown, no backticks, no explanation.

RESUME:
{req.resume_text}

CRITICAL RULES:
- skills_have must only include skills EXPLICITLY present in the resume
- You MUST include exactly one course for EVERY skill in skills_missing
- courses must be real courses from Coursera, Udemy, freeCodeCamp, or YouTube
- roadmap must have exactly 6 steps
- timeline must cover full period month by month
- daily_plan must be specific to this exact role
- scores must be calculated from the actual resume
- roadmap steps must NOT teach skills the candidate already has

Return this exact JSON:
{{
  "role": "{req.job_title}",
  "scores": {{"overall": 72, "skills": 75, "projects": 70, "job_readiness": 65}},
  "skills_have": ["skill1", "skill2"],
  "skills_missing": ["skill1", "skill2"],
  "courses": [
    {{"skill": "skill name", "course": "Full course title", "platform": "Coursera", "url": "https://coursera.org/learn/example"}}
  ],
  "roadmap": [
    {{"step": 1, "title": "Step title", "description": "Two sentence description.", "duration": "2 weeks"}}
  ],
  "timeline": [
    {{"period": "Month 1", "goal": "Specific achievable goal"}}
  ],
  "daily_plan": {{
    "morning": "Specific 1-hour learning task",
    "afternoon": "Specific 30-min practice task",
    "evening": "Specific 45-min project task"
  }}
}}"""
    try:
        raw  = _call_groq(prompt, GROQ_API_KEY, max_tokens=2500)
        return _parse_json_from_text(raw)
    except Exception as e:
        raise HTTPException(500, f"Detail failed: {e}")