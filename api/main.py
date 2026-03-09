"""
Smart Resume — FastAPI Backend (Phase 1)
======================================
Endpoints:
  POST /analyze          → Full pipeline: parse → score → build BackendPayload
  POST /clarify          → Submit clarification answers, get updated payload
  GET  /health           → Health check

Run with:
  uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from core.parser import parse_document
from core.schemas import (
    BackendPayload,
    ClarificationAnswer,
    ParsedResume,
    SkillProficiency,
    TemplateType,
)
from core.similarity import (
    compute_proficiency_evidence_scores,
    compute_similarity,
    decide_needs_optimization,
    decide_template,
    parse_job_description,
)

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("smart_resume.api")

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Smart Resume — Resume Optimization Engine",
    description="Dynamic Resume Optimization and Factual Evaluation Engine for CS graduates.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory session store (replace with Redis in production) ───────────────

_sessions: dict[str, BackendPayload] = {}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — Build ParsedResume from parser output
# ─────────────────────────────────────────────────────────────────────────────

def _build_parsed_resume(parser_output: dict) -> ParsedResume:
    """
    Construct ParsedResume from the raw parser output dict.
    This is a best-effort extraction — the AI prompt does deeper analysis.
    """
    import re
    from core.schemas import ProjectEntry, ExperienceEntry, EducationEntry

    sections = parser_output["sections"]
    raw_text = parser_output["raw_text"]
    mismatches = parser_output["mismatches"]

    # ── Skills ───────────────────────────────────────────────────────────────
    skill_text = sections.get("skills", "")
    # Split on common delimiters
    raw_skills = re.split(r"[,|•\n\t/]", skill_text)
    skills = [s.strip().lower() for s in raw_skills if 2 < len(s.strip()) < 40]
    # Add inline-detected skills not already in list
    for term in parser_output.get("inline_skills", []):
        if term not in skills:
            skills.append(term)
    skills = list(dict.fromkeys(skills))  # dedupe preserving order

    # ── Projects (basic extraction) ───────────────────────────────────────────
    proj_text = sections.get("projects", "")
    projects: list[ProjectEntry] = []
    # Split by double newline or numbered markers
    proj_chunks = re.split(r"\n{2,}|(?=\n\d+[\.\)])", proj_text)
    for chunk in proj_chunks:
        chunk = chunk.strip()
        if not chunk or len(chunk) < 20:
            continue
        lines = chunk.splitlines()
        title = lines[0].strip() if lines else "Unnamed Project"
        desc  = " ".join(lines[1:]).strip() if len(lines) > 1 else chunk
        techs = [s for s in skills if s.lower() in desc.lower()]
        metrics = re.findall(r"\d+[%x]|\d+\s*(?:ms|seconds?|users?|requests?)", desc, re.IGNORECASE)
        projects.append(ProjectEntry(
            title=title,
            description=desc,
            technologies=techs,
            metrics=metrics,
        ))

    # ── Experience ────────────────────────────────────────────────────────────
    exp_text = sections.get("experience", "")
    experience: list[ExperienceEntry] = []
    exp_chunks = re.split(r"\n{2,}", exp_text)
    for chunk in exp_chunks:
        chunk = chunk.strip()
        if not chunk or len(chunk) < 20:
            continue
        lines = chunk.splitlines()
        role_line = lines[0].strip() if lines else ""
        # Try to detect company/role pattern "Role @ Company" or "Role, Company"
        role = role_line
        company = "Unknown"
        for sep in [" @ ", " at ", ", ", " | "]:
            if sep in role_line:
                parts = role_line.split(sep, 1)
                role    = parts[0].strip()
                company = parts[1].strip()
                break

        # Duration: look for date patterns
        duration = None
        duration_certain = True
        for line in lines:
            date_match = re.search(
                r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february"
                r"|march|april|june|july|august|september|october|november|december"
                r"|20\d\d|19\d\d|present|current)",
                line, re.IGNORECASE
            )
            if date_match:
                duration = line.strip()
                break
        if not duration:
            duration_certain = False

        responsibilities = [
            ln.strip().lstrip("-•*·").strip()
            for ln in lines[1:]
            if ln.strip() and len(ln.strip()) > 10
        ]
        techs = [s for s in skills if s.lower() in chunk.lower()]

        # Mismatch flag — was this flagged as project-like?
        is_project = any(
            chunk.strip().startswith(m) or m in chunk
            for m in mismatches.get("project_in_experience", [])
        )

        experience.append(ExperienceEntry(
            company=company,
            role=role,
            duration=duration,
            duration_certain=duration_certain,
            responsibilities=responsibilities,
            technologies=techs,
            is_project_mislabeled=is_project,
        ))

    # ── Education ─────────────────────────────────────────────────────────────
    edu_text = sections.get("education", "")
    educations: list[EducationEntry] = []
    graduation_year = None
    year_match = re.search(r"(20\d\d|19\d\d)", edu_text)
    if year_match:
        graduation_year = int(year_match.group(1))
    if edu_text:
        lines = edu_text.splitlines()
        institution = lines[0].strip() if lines else "Unknown"
        degree_line = lines[1].strip() if len(lines) > 1 else ""
        educations.append(EducationEntry(
            institution=institution,
            degree=degree_line or "Degree",
            graduation_year=graduation_year,
        ))

    # ── Certifications ────────────────────────────────────────────────────────
    from core.schemas import CertificationEntry
    cert_text = sections.get("certifications", "")
    certifications: list[CertificationEntry] = []
    for line in cert_text.splitlines():
        line = line.strip()
        if line and len(line) > 5:
            year_m = re.search(r"(20\d\d)", line)
            certifications.append(CertificationEntry(
                name=line,
                year=int(year_m.group(1)) if year_m else None,
            ))

    # ── Metrics ───────────────────────────────────────────────────────────────
    claims_with_metrics = re.findall(
        r"[^.]*\d+[%x][^.]*\.",
        raw_text,
    )

    # ── Years of Experience ───────────────────────────────────────────────────
    yoe = _estimate_years_of_experience(exp_text, edu_text)

    return ParsedResume(
        raw_text=raw_text,
        skills=skills,
        projects=projects,
        experience=experience,
        education=educations,
        certifications=certifications,
        claims_with_metrics=claims_with_metrics[:20],
        years_of_experience=yoe,
        project_count=len(projects),
        experience_count=len(experience),
    )


def _estimate_years_of_experience(exp_text: str, edu_text: str) -> float:
    """
    Heuristic: count year references in experience section.
    Returns float years.
    """
    import re
    years = re.findall(r"(20\d\d|19\d\d)", exp_text)
    if len(years) >= 2:
        years_int = sorted(set(int(y) for y in years))
        span = years_int[-1] - years_int[0]
        return float(min(span, 30))  # sanity cap
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "Smart Resume Phase 1", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/analyze", response_model=BackendPayload)
async def analyze(
    resume_file: UploadFile                 = File(..., description="Resume PDF, DOCX, or TXT"),
    job_description: str                    = Form(..., description="Full job description text"),
    target_role: str                        = Form(..., description="Role being applied to"),
    proficiency_json: Optional[str]         = Form(None, description='JSON: [{"skill_name":"python","level":"Advanced"}]'),
):
    """
    Main pipeline endpoint.

    1. Parse resume file → text + sections
    2. Parse job description → structured JD
    3. Compute semantic similarity
    4. Decide template + optimization flag
    5. Compute proficiency-evidence scores
    6. Assemble and return BackendPayload
    """
    import json

    # ── Parse resume ──────────────────────────────────────────────────────────
    file_bytes = await resume_file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded resume file is empty.")

    try:
        parser_output = parse_document(file_bytes, resume_file.filename or "resume.pdf")
    except Exception as e:
        logger.error("Document parse error: %s", e)
        raise HTTPException(status_code=422, detail=f"Could not parse resume: {e}")

    parsed_resume = _build_parsed_resume(parser_output)

    # ── Parse JD ─────────────────────────────────────────────────────────────
    parsed_jd = parse_job_description(job_description, target_role)

    # ── Proficiencies ─────────────────────────────────────────────────────────
    user_proficiencies: list[SkillProficiency] = []
    if proficiency_json:
        try:
            raw_list = json.loads(proficiency_json)
            user_proficiencies = [SkillProficiency(**item) for item in raw_list]
        except Exception as e:
            logger.warning("Could not parse proficiency JSON: %s", e)

    # ── Similarity ────────────────────────────────────────────────────────────
    similarity_score = compute_similarity(
        parser_output["preprocessed_text"],
        parsed_jd.raw_text,
    )

    # ── Backend decisions ─────────────────────────────────────────────────────
    selected_template  = decide_template(parsed_resume)
    needs_optimization = decide_needs_optimization(similarity_score)

    # Proficiency-evidence numeric scores
    _ = compute_proficiency_evidence_scores(user_proficiencies, parsed_resume)
    # (stored in payload implicitly via user_proficiencies; full scores sent to AI in prompt)

    # ── Assemble payload ──────────────────────────────────────────────────────
    session_id = str(uuid.uuid4())
    payload = BackendPayload(
        resume_raw_text=parser_output["raw_text"],
        job_description_raw_text=job_description,
        target_role=target_role,
        parsed_resume=parsed_resume,
        parsed_jd=parsed_jd,
        semantic_similarity_score=similarity_score,
        user_proficiencies=user_proficiencies,
        clarification_answers=[],
        selected_template=selected_template,
        needs_optimization=needs_optimization,
        session_id=session_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    _sessions[session_id] = payload
    logger.info(
        "Session %s | similarity=%.1f | template=%s | optimize=%s",
        session_id, similarity_score, selected_template, needs_optimization,
    )
    return payload


@app.post("/clarify/{session_id}", response_model=BackendPayload)
async def submit_clarification(
    session_id: str,
    answers: list[ClarificationAnswer],
):
    """
    Round-trip endpoint: user submits clarification answers.
    Updates the session payload and returns refreshed BackendPayload.
    Similarity score is NOT recomputed (answers don't change embedding).
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found. Run /analyze first.")

    payload = _sessions[session_id]
    # Append new answers (don't overwrite — accumulate across rounds)
    payload.clarification_answers.extend(answers)

    logger.info("Session %s | clarification answers added: %d", session_id, len(answers))
    return payload


@app.get("/session/{session_id}", response_model=BackendPayload)
async def get_session(session_id: str):
    """Retrieve a previously computed BackendPayload by session ID."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return _sessions[session_id]