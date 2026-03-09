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

import os
# Load .env BEFORE reading any env vars
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_env_path)
    print(f"[startup] Loaded .env from {_env_path}")
except ImportError:
    pass  # python-dotenv not installed — rely on system env

from core.parser import (
    parse_document, preprocess_for_embedding, SOFT_SKILLS,
    JUNK_TITLE_PATTERNS, is_junk_line, is_real_project_title,
    compute_ats_score as _compute_ats,
)
# backward-compat aliases (some call sites use underscore prefix)
_is_junk_line = is_junk_line
_is_real_project_title = is_real_project_title
from core.llm_parser import parse_resume_with_llm, llm_output_to_parsed_resume
from core.schemas import (
    BackendPayload, ClarificationAnswer, CertificationEntry,
    ParsedResume, ProjectEntry, ExperienceEntry, EducationEntry, SkillProficiency,
)
from core.similarity import (
    compute_similarity, 
    decide_template, parse_job_description,
)

# Read AFTER load_dotenv so .env values are available
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if GROQ_API_KEY:
    print(f"[startup] GROQ_API_KEY loaded ({len(GROQ_API_KEY)} chars)")
else:
    print("[startup] WARNING: GROQ_API_KEY not set — AI features disabled")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("smart_resume.app")

app = FastAPI(title="Smart Resume", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
# Session store with LRU cap — prevents unbounded memory growth on long-running servers
_MAX_SESSIONS = 200
_sessions: dict[str, BackendPayload] = {}

def _store_session(key: str, value) -> None:
    """Store a session, evicting the oldest entry when cap is exceeded."""
    if len(_sessions) >= _MAX_SESSIONS:
        oldest = next(iter(_sessions))
        del _sessions[oldest]
        logger.info("Session cap reached — evicted oldest session %s", oldest)
    _sessions[key] = value




# ── Resume builder ────────────────────────────────────────────────────────────

def _build_parsed_resume(parser_output: dict) -> ParsedResume:
    sections   = parser_output["sections"]
    raw_text   = parser_output["raw_text"]
    mismatches = parser_output["mismatches"]

    logger.info("Building resume. Sections: %s", list(sections.keys()))

    # ── Skills ────────────────────────────────────────────────────────────────
    # ── Robust skill extraction — handles "Languages: SQL.C.Python" style ──────
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

    # Step 1: scan every skills-section line, strip the label, split on all separators
    raw_candidates = []
    for line in skill_text.splitlines():
        # Remove label prefix: "Languages: " "Softwares :" etc.
        val = re.sub(r"^[A-Za-z ]+[:/]\s*", "", line).strip()
        # Split on comma, dot, space, pipe, slash, semicolon
        raw_candidates += re.split(r"[,.|/;\s]+", val)

    # Step 2: scan project tech lines in raw_text
    for line in raw_text_sk.splitlines():
        if re.search(r"tech\s*:", line, re.I):
            tech = re.sub(r".*tech\s*:\s*", "", line, flags=re.I)
            raw_candidates += re.split(r"[,\s]+", tech)

    # Step 3: add inline skills from parser
    raw_candidates += parser_output.get("inline_skills", [])

    # Step 4: classify each candidate
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

    skills = langs_p + fwks_p + dbs_p + tools_p
    skills = list(dict.fromkeys(skills))
    logger.info("Skills (regex robust): %s", skills)

    # ── Experience ────────────────────────────────────────────────────────────
    exp_text   = sections.get("experience", "")
    experience: list[ExperienceEntry] = []

    # Split experience blocks on date patterns or double newlines
    exp_blocks = re.split(r"\n{2,}", exp_text) if exp_text else []
    if not exp_blocks and exp_text:
        exp_blocks = [exp_text]

    for block in exp_blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip() and not is_junk_line(l)]
        if not lines:
            continue

        # Find role line — first non-date, non-bullet line
        role = "Unknown Role"
        company = "Unknown"
        duration = None
        resp_lines = []

        for i, line in enumerate(lines):
            # Date line
            if re.search(r"(20\d\d|19\d\d|present|current)", line, re.IGNORECASE):
                if not duration:
                    duration = line
                continue
            # Role/company line (first meaningful line)
            if role == "Unknown Role":
                for sep in [" @ ", " at ", " | ", ", "]:
                    if sep in line:
                        parts = line.split(sep, 1)
                        role, company = parts[0].strip(), parts[1].strip()
                        break
                else:
                    role = line
                continue
            # Everything else = responsibility
            clean = line.lstrip("-").strip()
            if clean and len(clean) > 5:
                resp_lines.append(clean)

        techs    = [s for s in skills if s in block.lower()]
        is_proj  = any(m in block for m in mismatches.get("project_in_experience", []))

        experience.append(ExperienceEntry(
            company=company, role=role, duration=duration,
            duration_certain=bool(duration),
            responsibilities=resp_lines,
            technologies=techs,
            is_project_mislabeled=is_proj,
        ))

    # ── Projects ──────────────────────────────────────────────────────────────
    proj_text = sections.get("projects", "")
    projects: list[ProjectEntry] = []

    proj_blocks = re.split(r"\n{2,}", proj_text) if proj_text else []
    if not proj_blocks and proj_text:
        proj_blocks = [proj_text]

    for block in proj_blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue

        # Find first line that looks like a real project title
        title_idx = None
        for i, line in enumerate(lines):
            if _is_real_project_title(line):
                title_idx = i
                break

        if title_idx is None:
            continue  # skip blocks with no valid title

        title = lines[title_idx]
        desc_lines = [
            l.lstrip("-").strip() for l in lines[title_idx+1:]
            if l.strip() and not is_junk_line(l)
        ]
        desc  = " ".join(desc_lines) if desc_lines else title
        techs = [s for s in skills if s in block.lower()]
        metrics = re.findall(r"\d+[%x]|\d+\s*(?:ms|seconds?|users?|requests?)", block, re.IGNORECASE)

        if len(desc) > 5:
            projects.append(ProjectEntry(
                title=title, description=desc,
                technologies=techs, metrics=metrics,
            ))

    # ── Education ─────────────────────────────────────────────────────────────
    edu_text   = sections.get("education", "")
    educations: list[EducationEntry] = []

    if edu_text:
        lines = [l.strip() for l in edu_text.splitlines() if l.strip()]
        inst, degree, year = None, None, None

        for line in lines:
            ym = re.search(r"(20\d\d|19\d\d)", line)
            if ym:
                year = int(ym.group(1))

            if re.search(r"\b(college|university|school|institute|hss|higher)\b", line, re.IGNORECASE):
                inst = line
            elif re.search(r"\b(b\.?tech|bachelor|master|m\.?tech|diploma|plus two|12th|10th|sslc|hse)\b", line, re.IGNORECASE):
                degree = line
                if inst:
                    educations.append(EducationEntry(
                        institution=inst, degree=degree, graduation_year=year,
                    ))
                    inst = None

        if not educations and lines:
            educations.append(EducationEntry(
                institution=lines[0],
                degree=lines[1] if len(lines) > 1 else "Degree",
                graduation_year=year,
            ))

    # ── Certifications ────────────────────────────────────────────────────────
    certifications: list[CertificationEntry] = []
    for line in sections.get("certifications", "").splitlines():
        line = line.strip()
        if len(line) > 5 and not is_junk_line(line):
            ym = re.search(r"(20\d\d)", line)
            certifications.append(CertificationEntry(
                name=line, year=int(ym.group(1)) if ym else None
            ))

    # ── YoE & metrics ─────────────────────────────────────────────────────────
    claims     = re.findall(r"[^.]*\d+[%x][^.]*\.", raw_text)
    years_exp  = re.findall(r"(20\d\d|19\d\d)", exp_text)
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


# ── Routes ────────────────────────────────────────────────────────────────────

# ── ATS Compatibility Checker ─────────────────────────────────────────────────

ATS_CHECKS = [
    # (check_fn, issue_message, severity: high/medium/low)
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
    """
    Run deterministic ATS compatibility checks on generated resume text.
    Returns list of {issue, severity} dicts.
    """
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



@app.get("/health")
async def health():
    return {"status": "ok", "service": "Smart Resume", "timestamp": datetime.now(timezone.utc).isoformat()}


# ── ATS Compatibility Checker ─────────────────────────────────────────────────

ATS_CHECKS = [
    # (check_fn, issue_message, severity: high/medium/low)
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
    """
    Run deterministic ATS compatibility checks on generated resume text.
    Returns list of {issue, severity} dicts.
    """
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



@app.get("/health")
async def health_check():
    """Reports server health and capability flags — used by UI to show/hide features."""
    import shutil, subprocess
    node_available = False
    try:
        # Check common node paths
        node_cmd = shutil.which("node")
        if not node_cmd:
            for candidate in [
                r"C:\Program Files\nodejs\node.exe",
                r"C:\Program Files (x86)\nodejs\node.exe",
            ]:
                import os as _os
                if _os.path.isfile(candidate):
                    node_cmd = candidate
                    break
        if node_cmd:
            result = subprocess.run([node_cmd, "--version"], capture_output=True, timeout=5)
            node_available = result.returncode == 0
    except Exception:
        node_available = False

    # Check similarity mode
    try:
        from core.similarity import get_similarity_mode
        sim_mode = get_similarity_mode()
    except Exception:
        sim_mode = "unknown"

    return {
        "status":           "ok",
        "groq_configured":  bool(GROQ_API_KEY),
        "node_available":   node_available,
        "session_count":    len(_sessions),
        "session_cap":      _MAX_SESSIONS,
        "similarity_mode":  sim_mode,
    }


@app.post("/analyze")
async def analyze(
    resume_file:      UploadFile    = File(...),
    job_description:  Optional[str] = Form(None),
    target_role:      str           = Form(...),
    proficiency_json: Optional[str] = Form(None),
):
    file_bytes = await resume_file.read()
    if not file_bytes:
        raise HTTPException(400, "Uploaded file is empty.")

    try:
        parser_output = parse_document(file_bytes, resume_file.filename or "resume.pdf")
    except Exception as e:
        raise HTTPException(422, f"Could not parse resume: {e}")

    # Use LLM parser if API key available, else fall back to regex
    if GROQ_API_KEY:
        try:
            llm_data      = parse_resume_with_llm(parser_output["raw_text"], GROQ_API_KEY)
            parsed_resume = llm_output_to_parsed_resume(llm_data, parser_output["raw_text"])
            if parsed_resume is None:
                raise ValueError("llm_output_to_parsed_resume returned None")
            # Merge: regex parser may catch skills LLM missed (e.g. "SQL.C.Python")
            regex_resume  = _build_parsed_resume(parser_output)
            llm_skill_set = {s.lower() for s in (parsed_resume.skills or [])}
            for sk in (regex_resume.skills or []):
                if sk.lower() not in llm_skill_set:
                    parsed_resume.skills.append(sk)
                    logger.info("Merged regex skill into LLM result: %s", sk)
            logger.info("Used LLM parser successfully. Final skills: %s", parsed_resume.skills)
        except Exception as e:
            logger.warning("LLM parser failed (%s) — falling back to regex parser", e)
            parsed_resume = _build_parsed_resume(parser_output)
    else:
        logger.info("No GROQ_API_KEY set — using regex parser")
        parsed_resume = _build_parsed_resume(parser_output)
    # Enrich parsed_resume with contact fields from parser (github, linkedin, etc.)
    contact = parser_output.get("contact_info", {})
    if contact.get("name")     and not parsed_resume.name:     parsed_resume.name     = contact["name"]
    if contact.get("email")    and not parsed_resume.email:    parsed_resume.email    = contact["email"]
    if contact.get("phone")    and not parsed_resume.phone:    parsed_resume.phone    = contact["phone"]
    if contact.get("location") and not parsed_resume.location: parsed_resume.location = contact["location"]
    if contact.get("linkedin"):  parsed_resume.linkedin = contact["linkedin"]
    if contact.get("github"):    parsed_resume.github   = contact["github"]

    # JD is optional — generate virtual JD from role knowledge when not provided
    jd_text = (job_description or "").strip()
    if GROQ_API_KEY:
        from core.jd_parser_llm import parse_jd_with_llm, generate_virtual_jd
        if jd_text:
            llm_jd = parse_jd_with_llm(jd_text, GROQ_API_KEY)
        else:
            logger.info("No JD provided — generating virtual JD for role: %s", target_role)
            llm_jd = generate_virtual_jd(target_role, GROQ_API_KEY)
        # Build ParsedJobDescription from LLM output
        from core.schemas import Domain
        domain_map = {
            "ml": Domain.ML, "machine learning": Domain.ML,
            "backend": Domain.BACKEND, "frontend": Domain.FRONTEND,
            "devops": Domain.DEVOPS, "data": Domain.DATA, "unknown": Domain.UNKNOWN,
            "embedded": Domain.UNKNOWN, "security": Domain.UNKNOWN, "mobile": Domain.UNKNOWN,
        }
        detected_domain = domain_map.get(
            llm_jd.get("domain", "unknown").lower(), Domain.UNKNOWN
        )
        from core.schemas import ParsedJobDescription
        # Use virtual_jd_text as the raw_text for similarity computation
        virtual_text = llm_jd.get("virtual_jd_text", jd_text or target_role)
        parsed_jd = ParsedJobDescription(
            raw_text                 = virtual_text,
            target_role              = target_role,
            detected_domain          = detected_domain,
            required_skills          = [s.lower() for s in llm_jd.get("required_skills", [])],
            preferred_skills         = [s.lower() for s in llm_jd.get("preferred_skills", [])],
            required_experience_years= llm_jd.get("required_experience_years"),
            key_responsibilities     = llm_jd.get("key_responsibilities", []),
            jd_provided              = bool(jd_text),
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

    # Always compute similarity — raw_text is either real JD or virtual JD
    # Preprocess JD with the same pipeline as resume — prevents asymmetric token sets
    jd_preprocessed = preprocess_for_embedding(parsed_jd.raw_text)
    similarity_score = compute_similarity(
        parser_output["preprocessed_text"], jd_preprocessed
    )

    # ATS simulation — deterministic, no LLM needed
    ats_result = _compute_ats(
        raw_text    = parser_output["raw_text"],
        sections    = parser_output["sections"],
        skills      = parsed_resume.skills,
        target_role = target_role,
    )
    selected_template  = decide_template(parsed_resume)
    needs_optimization = True  # always optimize when submitting

    session_id = str(uuid.uuid4())
    payload = BackendPayload(
        resume_raw_text=parser_output["raw_text"],
        job_description_raw_text=jd_text,
        target_role=target_role,
        parsed_resume=parsed_resume,
        parsed_jd=parsed_jd,
        semantic_similarity_score=similarity_score,
        user_proficiencies=user_proficiencies,
        clarification_answers=None,  # None = first run, triggers doubt question generation
        selected_template=selected_template,
        needs_optimization=needs_optimization,
        session_id=session_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    _store_session(session_id, payload)
    logger.info("Session %s | sim=%.1f | template=%s | optimize=%s | ats=%d",
                session_id, similarity_score, getattr(selected_template,"value",str(selected_template)), needs_optimization, ats_result["score"])
    # Return payload + ATS result as combined dict
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
    """
    Phase 2 — Full evaluation.
    Requires a session from /analyze first.
    Needs GROQ_API_KEY set.
    """
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found. Run /analyze first.")
    if not GROQ_API_KEY:
        raise HTTPException(400, "GROQ_API_KEY not set. Cannot run evaluation.")

    payload = _sessions[session_id]
    try:
        result = evaluate(payload, GROQ_API_KEY)
        # Run ATS compatibility check on generated resume
        if result.final_resume:
            ats_issues = run_ats_checks(result.final_resume)
            if ats_issues:
                rqa = result.resume_quality_assessment or {}
                existing = rqa.get("ats_flags", [])
                rqa["ats_flags"] = existing + [i["issue"] for i in ats_issues if i["severity"] == "high"]
                # Lower ats_compatibility if high-severity issues found
                high_count = sum(1 for i in ats_issues if i["severity"] == "high")
                if high_count >= 2:
                    rqa["ats_compatibility"] = "Low"
                elif high_count == 1:
                    rqa["ats_compatibility"] = "Medium"
                result.resume_quality_assessment = rqa
                logger.info("ATS check: %d issues found (%d high)", len(ats_issues), high_count)
        _store_session(f"eval_{session_id}", result)
        return result
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        raise HTTPException(500, f"Evaluation failed: {e}")


@app.get("/evaluate/{session_id}/result", response_model=SmartResumeResponse)
async def get_evaluation_result(session_id: str):
    """Retrieve a previously computed evaluation result."""
    key = f"eval_{session_id}"
    if key not in _sessions:
        raise HTTPException(404, "No evaluation found. Run POST /evaluate/{session_id} first.")
    return _sessions[key]


@app.get("/evaluate/{session_id}/download/docx")
async def download_resume_docx(session_id: str):
    """Download optimized resume as DOCX."""
    from fastapi.responses import Response
    from core.resume_exporter import export_resume
    key = f"eval_{session_id}"
    if key not in _sessions:
        raise HTTPException(404, "No evaluation found.")
    result = _sessions[key]
    if not result.final_resume:
        raise HTTPException(400, "No optimized resume available. Run evaluation first.")
    try:
        payload    = _sessions.get(session_id)
        parsed_resume = payload.parsed_resume if payload else None
        if not parsed_resume:
            raise HTTPException(400, "Session data missing.")
        docx_bytes = export_resume(result.final_resume, parsed_resume, fmt="docx")
        name       = parsed_resume.name or "Resume"
        safe_name  = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
        return Response(
            content    = docx_bytes,
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers    = {"Content-Disposition": f"attachment; filename={safe_name}_resume.docx"},
        )
    except Exception as e:
        raise HTTPException(500, f"DOCX generation failed: {e}")


@app.get("/evaluate/{session_id}/download/pdf")
async def download_resume_pdf(session_id: str):
    """Download optimized resume as PDF."""
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
        pdf_bytes  = export_resume(result.final_resume, parsed_resume, fmt="pdf")
        name       = parsed_resume.name or "Resume"
        safe_name  = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
        logger.info("PDF download: %d bytes for session %s", len(pdf_bytes), session_id)
        return Response(
            content    = pdf_bytes,
            media_type = "application/pdf",
            headers    = {"Content-Disposition": f"attachment; filename={safe_name}_optimized.pdf"},
        )
    except ImportError as e:
        logger.error("PDF ImportError: %s", e)
        raise HTTPException(500, f"PDF unavailable — run: pip install reportlab  ({e})")
    except Exception as e:
        logger.error("PDF generation failed for session %s: %s", session_id, e, exc_info=True)
        raise HTTPException(500, f"PDF generation failed: {e}")


@app.post("/evaluate/{session_id}/clarify", response_model=SmartResumeResponse)
async def re_evaluate_with_clarification(
    session_id: str,
    request: Request,
):
    """
    Re-run evaluation after user provides clarification answers.
    Accepts a JSON array directly: [{"question": "...", "answer": "..."}]
    """
    from core.schemas import ClarificationAnswer
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found.")
    if not GROQ_API_KEY:
        raise HTTPException(400, "GROQ_API_KEY not set.")

    try:
        body = await request.json()
        # Accept both plain list and {"answers": [...]} format
        if isinstance(body, list):
            raw_answers = body
        elif isinstance(body, dict):
            raw_answers = body.get("answers", [])
        else:
            raw_answers = []
    except Exception:
        raw_answers = []

    payload = _sessions[session_id]
    typed   = [ClarificationAnswer(**a) if isinstance(a, dict) else a for a in raw_answers]
    # [] = skip, [...] = answered. Both are "not None" → bypass doubt gate
    payload.clarification_answers = typed  # replace, not accumulate
    # Ensure resume generation is triggered on re-evaluation
    payload.needs_optimization = True

    try:
        result = evaluate(payload, GROQ_API_KEY)
        _store_session(f"eval_{session_id}", result)
        # Return full result — clarification_required will be False since answers exist
        return result
    except Exception as e:
        logger.error("Re-evaluation error for session %s: %s", session_id, e, exc_info=True)
        raise HTTPException(500, f"Re-evaluation failed: {e}")