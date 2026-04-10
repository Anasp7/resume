"""
core/doubt_engine.py
=====================
Doubt detection engine — STEP X and STEP Y.

STEP X: detect_missing_profile_info(parsed_resume)
  Phase A (conditional) — profile gaps: GitHub, LinkedIn, CGPA, 12th %, 10th %
  Phase B (conditional) — project/experience depth checks
  Phase C (MANDATORY)   — 3 new-entry discovery questions asked for EVERY user

STEP Y: verify_and_map_profile_answers(parsed_resume, raw_answers)
  → Maps answers back into the ParsedResume.

generate_doubt_questions()  — LLM/Ollama factual-integrity questions
all_required_answered()     — Required-answer gating
"""
from __future__ import annotations
import re
import logging
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger("smart_resume.doubt_engine")


# ─────────────────────────────────────────────────────────────────────────────
# STEP X — DETECT MISSING PROFILE INFORMATION
# ─────────────────────────────────────────────────────────────────────────────

def detect_missing_profile_info(parsed_resume, target_role: str = "", raw_resume_text: str = "") -> list:
    """
    STEP X — Detect missing but important resume information.

    Phase A  (conditional — only asked when truly missing)
      1. GitHub URL   — checks parsed JSON field FIRST, then raw text scan
      2. LinkedIn URL — checks parsed JSON field FIRST, then raw text scan
      3. CGPA         — only when a degree entry EXISTS but has no GPA value
      4. 12th marks   — only when a 12th/Plus-Two entry EXISTS but has no marks
      5. 10th marks   — only when a 10th/SSLC entry EXISTS but has no marks

    Phase B  (conditional — project/experience depth)
      6. Project metric   — per project without a quantified result
      7. Thin project     — per project with <= 5-word description
      8. Experience metric— per experience entry with no measurable outcome
      9. Thin skills      — when fewer than 6 skills are present

    Phase C  (MANDATORY — asked for EVERY user, every upload)
      10. Internship / job experience (new entries not in resume)
      11. Workshops, hackathons & competitions (new entries)
      12. Certifications / online courses (new entries)
    """
    issues = []
    pr     = parsed_resume

    # ── Normalise raw text ───────────────────────────────────────────────────
    raw_text = (raw_resume_text or getattr(pr, "raw_text", "") or "").lower()

    def _q(type_, context, question, required=False):
        return SimpleNamespace(
            type=type_,
            context=context,
            question=question,
            required=required,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE A — PROFILE GAPS
    # ═════════════════════════════════════════════════════════════════════════

    # 1. GitHub — check parsed field first (most reliable), then raw text
    github_field = str(getattr(pr, "github", "") or "").strip().lower()
    has_github = (
        "github.com" in github_field           # field set by parser
        or "github.com" in raw_text            # URL in raw text
        or bool(re.search(r"\bgithub\b", raw_text))  # word mention
    )
    if not has_github:
        issues.append(_q(
            "github",
            "Contact information",
            "Do you have a GitHub profile? If yes, share the URL "
            "(e.g. github.com/yourname). Recruiters use this to verify your projects.",
        ))

    # 2. LinkedIn — same dual check
    linkedin_field = str(getattr(pr, "linkedin", "") or "").strip().lower()
    has_linkedin = (
        "linkedin.com" in linkedin_field
        or "linkedin.com" in raw_text
        or bool(re.search(r"\blinkedin\b", raw_text))
    )
    if not has_linkedin:
        issues.append(_q(
            "linkedin",
            "Contact information",
            "Do you have a LinkedIn profile? If yes, please share the URL.",
        ))

    # 3. College CGPA — ONLY ask when a degree entry exists WITHOUT a gpa value
    DEGREE_KWS = {"b.tech", "btech", "bachelor", "b.e", "be", "b.sc", "m.tech", "master", "mba", "phd"}
    has_college_entry = False
    has_cgpa          = False
    for edu in (getattr(pr, "education", []) or []):
        deg_l = (getattr(edu, "degree", "") or "").lower()
        if any(k in deg_l for k in DEGREE_KWS):
            has_college_entry = True
            gpa_val = getattr(edu, "gpa", None)
            if gpa_val and re.search(r"\d", str(gpa_val)):
                has_cgpa = True
            break
    # Also accept if raw text already states CGPA explicitly
    if re.search(r"(cgpa|gpa)\s*[:/]?\s*[\d.]+", raw_text, re.I):
        has_cgpa = True
    if has_college_entry and not has_cgpa:
        issues.append(_q(
            "cgpa",
            "Education — College / University",
            "Your college education entry is missing a CGPA. "
            "What is your current CGPA or academic percentage? "
            "Example: '8.1 / 10'. Say 'skip' if you'd rather not include it.",
        ))

    # 3b. Degree Branch — ONLY ask when degree string has no branch/specialisation
    BRANCH_KWS = {
        "computer science", "cse", "information technology",
        "electronics", "ece", "electrical", "eee", "mechanical", "civil",
        "chemical", "biotechnology", "ai", "artificial intelligence",
        "data science", "robotics", "automobile", "it ",
    }
    has_branch = False
    has_generic_degree = False
    for edu in (getattr(pr, "education", []) or []):
        deg_l = (getattr(edu, "degree", "") or "").lower()
        if any(k in deg_l for k in {"b.tech", "btech", "bachelor", "b.e", "be"}):
            has_generic_degree = True
            # Only check the degree field itself — NOT the raw resume text
            # because branch keywords appear freely in project/experience sections
            if any(b in deg_l for b in BRANCH_KWS):
                has_branch = True
            break
    if has_generic_degree and not has_branch:
        issues.append(_q(
            "degree_branch",
            "Education — Degree Specialisation",
            "Your degree shows 'Bachelor of Technology' but the branch/specialisation is missing. "
            "What is your branch? (e.g. Computer Science, Electronics & Communication, Mechanical, etc.)",
        ))

    # 4. 12th / Plus Two marks — ONLY ask when a 12th entry exists WITHOUT marks
    SCHOOL_12_KWS = {
        "plus two", "12th", "hsc", "xii", "+2", "10+2",
        "higher secondary", "class 12", "intermediate",
    }
    has_12th_entry = False
    has_12th_marks = False
    for edu in (getattr(pr, "education", []) or []):
        combined = (
            (getattr(edu, "degree", "") or "") + " " +
            (getattr(edu, "institution", "") or "")
        ).lower()
        if any(k in combined for k in SCHOOL_12_KWS):
            has_12th_entry = True
            gpa_val = getattr(edu, "gpa", None)
            if gpa_val and re.search(r"\d", str(gpa_val)):
                has_12th_marks = True
            break
    school_12_field = getattr(pr, "school_12th", None) or ""
    if school_12_field and re.search(r"\d", school_12_field):
        has_12th_marks = True
    if re.search(r"(12th|plus two|xii|hsc).*?[\d.]+\s*%", raw_text, re.I):
        has_12th_marks = True
    if has_12th_entry and not has_12th_marks:
        issues.append(_q(
            "school_12th_percentage",
            "Education — 12th / Plus Two",
            "Your Plus Two (12th standard) entry is missing a percentage. "
            "What was your HSC/Plus Two score? Example: '92%'. Say 'skip' to omit.",
        ))

    # 5. 10th / SSLC marks — ONLY ask when a 10th entry exists WITHOUT marks
    SCHOOL_10_KWS = {
        "sslc", "10th", "class 10", "class x", "x std",
        "secondary school", "matric",
    }
    has_10th_entry = False
    has_10th_marks = False
    for edu in (getattr(pr, "education", []) or []):
        combined = (
            (getattr(edu, "degree", "") or "") + " " +
            (getattr(edu, "institution", "") or "")
        ).lower()
        if any(k in combined for k in SCHOOL_10_KWS):
            has_10th_entry = True
            gpa_val = getattr(edu, "gpa", None)
            if gpa_val and re.search(r"\d", str(gpa_val)):
                has_10th_marks = True
            break
    school_10_field = getattr(pr, "school_10th", None) or ""
    if school_10_field and re.search(r"\d", school_10_field):
        has_10th_marks = True
    if re.search(r"(10th|sslc|class x).*?[\d.]+\s*%", raw_text, re.I):
        has_10th_marks = True
    if has_10th_entry and not has_10th_marks:
        issues.append(_q(
            "school_10th_percentage",
            "Education — 10th / SSLC",
            "Your 10th standard (SSLC) entry is missing a percentage. "
            "What was your score? Example: '89%'. Say 'skip' to omit.",
        ))

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE B — (REMOVED)
    # Project/experience depth is now handled strictly and centrally by generate_doubt_questions()
    # ═════════════════════════════════════════════════════════════════════════

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE C — MANDATORY NEW-ENTRY DISCOVERY (ALWAYS ASKED, EVERY USER)
    # These questions actively surface new resume content not in the uploaded doc.
    # ═════════════════════════════════════════════════════════════════════════

    issues.append(_q(
        "internship",
        "Work Experience & Workshops",
        "Have you done any internships, workshops, part-time jobs, or industrial training "
        "that are NOT already listed in your resume? "
        "If yes \u2014 Organization | Role | Duration | What you performed/worked on. "
        "Say 'nil' if none.",
        required=False,
    ))

    issues.append(_q(
        "hackathons",
        "Competitions & Hackathons",
        "Have you attended any hackathons or technical competitions NOT already in your resume? "
        "(e.g. Smart India Hackathon, BAJA, Robocon, coding contests). "
        "Say 'nil' if none.",
        required=False,
    ))

    issues.append(_q(
        "certifications",
        "Certifications & Online Courses",
        "Do you have any certifications or online courses NOT already in your resume? "
        "(e.g. Coursera, NPTEL, AWS, Cisco, Google, etc.) "
        "Please list them. Say 'nil' if none.",
        required=False,
    ))

    logger.info(
        "detect_missing_profile_info: %d questions "
        "(github_missing=%s linkedin_missing=%s cgpa_missing=%s "
        "12th_missing=%s 10th_missing=%s)",
        len(issues),
        not has_github, not has_linkedin,
        has_college_entry and not has_cgpa,
        has_12th_entry and not has_12th_marks,
        has_10th_entry and not has_10th_marks,
    )
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# STEP Y — VERIFY AND MAP PROFILE ANSWERS
# ─────────────────────────────────────────────────────────────────────────────

def verify_and_map_profile_answers(
    parsed_resume,
    raw_answers: list,
) -> tuple:
    """
    STEP Y — Verify and map profile clarification answers into ParsedResume fields.

    raw_answers: list of dicts {\"type\": str, \"answer\": str}

    Returns (updated_parsed_resume, change_log)
    change_log: list of human-readable strings describing what changed.
    """
    from copy import deepcopy

    pr = deepcopy(parsed_resume)
    change_log = []

    NIL_WORDS = {"nil", "none", "no", "nothing", "skip", "n/a", "na", "not applicable"}

    def _is_nil(answer: str) -> bool:
        return answer.strip().lower() in NIL_WORDS or len(answer.strip()) <= 2

    def _setattr_safe(obj, attr, value):
        try:
            object.__setattr__(obj, attr, value)
        except Exception:
            obj.__dict__[attr] = value

    for item in raw_answers:
        if not isinstance(item, dict):
            continue
        answer_type = (item.get("type") or item.get("id") or "").lower().strip()
        answer_text = (item.get("answer") or "").strip()

        if _is_nil(answer_text):
            change_log.append(f"Skipped '{answer_type}' — user said nil/none.")
            continue

        # ── GitHub ────────────────────────────────────────────────────────────
        if answer_type == "github":
            url = answer_text
            if "github.com" not in url.lower():
                url = f"github.com/{answer_text.lstrip('@/')}"
            if not url.startswith("http"):
                url = "https://" + url
            _setattr_safe(pr, "github", url)
            change_log.append(f"Added GitHub: {url}")

        # ── LinkedIn ──────────────────────────────────────────────────────────
        elif answer_type == "linkedin":
            url = answer_text
            if "linkedin.com" not in url.lower():
                url = f"linkedin.com/in/{answer_text.lstrip('@/')}"
            if not url.startswith("http"):
                url = "https://" + url
            _setattr_safe(pr, "linkedin", url)
            change_log.append(f"Added LinkedIn: {url}")

        # ── CGPA ──────────────────────────────────────────────────────────────
        elif answer_type in ("cgpa", "gpa"):
            num_match = re.search(r"[\d.]+", answer_text)
            if num_match:
                gpa_val = float(num_match.group())
                for edu in getattr(pr, "education", []) or []:
                    deg_l = (getattr(edu, "degree", "") or "").lower()
                    is_degree = any(k in deg_l for k in {
                        "b.tech", "btech", "bachelor", "b.e", "be",
                        "b.sc", "master", "mba", "phd",
                    })
                    if is_degree and not getattr(edu, "gpa", None):
                        try:
                            edu.gpa = gpa_val
                        except Exception:
                            edu.__dict__["gpa"] = gpa_val
                        change_log.append(f"Added CGPA {gpa_val} to {getattr(edu, 'institution', '') or getattr(edu, 'degree', '')}")
                        break

        # ── Plus Two / 12th percentage ────────────────────────────────────────
        elif answer_type in ("school_12th_percentage", "school_12th", "plus_two", "12th"):
            pct_match = re.search(r"[\d.]+\s*%?", answer_text)
            pct_val   = pct_match.group().strip() if pct_match else answer_text
            SCHOOL_12_KWS = {"plus two", "12th", "hsc", "xii", "+2", "higher secondary", "class 12"}
            updated = False
            for edu in getattr(pr, "education", []) or []:
                combined = ((getattr(edu, "degree", "") or "") + " " + (getattr(edu, "institution", "") or "")).lower()
                if any(k in combined for k in SCHOOL_12_KWS):
                    if not getattr(edu, "gpa", None):
                        try:
                            edu.gpa = pct_val
                        except Exception:
                            edu.__dict__["gpa"] = pct_val
                        change_log.append(f"Added 12th percentage {pct_val} to {getattr(edu, 'institution', '')}")
                        updated = True
                        break
            if not updated:
                _setattr_safe(pr, "school_12th", pct_val)
                change_log.append(f"Stored 12th percentage: {pct_val}")

        # ── 10th / SSLC percentage ────────────────────────────────────────────
        elif answer_type in ("school_10th_percentage", "school_10th", "sslc", "10th"):
            pct_match = re.search(r"[\d.]+\s*%?", answer_text)
            pct_val   = pct_match.group().strip() if pct_match else answer_text
            SCHOOL_10_KWS = {"sslc", "10th", "class 10", "class x", "x std", "matric"}
            updated = False
            for edu in getattr(pr, "education", []) or []:
                combined = ((getattr(edu, "degree", "") or "") + " " + (getattr(edu, "institution", "") or "")).lower()
                if any(k in combined for k in SCHOOL_10_KWS):
                    if not getattr(edu, "gpa", None):
                        try:
                            edu.gpa = pct_val
                        except Exception:
                            edu.__dict__["gpa"] = pct_val
                        change_log.append(f"Added 10th percentage {pct_val}")
                        updated = True
                        break
            if not updated:
                _setattr_safe(pr, "school_10th", pct_val)
                change_log.append(f"Stored 10th percentage: {pct_val}")

        # ── Degree Branch / Specialisation ───────────────────────────────────
        elif answer_type == "degree_branch":
            branch = answer_text.strip()
            for edu in getattr(pr, "education", []) or []:
                deg_l = (getattr(edu, "degree", "") or "").lower()
                if any(k in deg_l for k in {"b.tech", "btech", "bachelor", "b.e", "be"}):
                    current_deg = getattr(edu, "degree", "") or "Bachelor of Technology"
                    # Avoid double-appending if branch already present
                    if branch.lower() not in current_deg.lower():
                        new_deg = f"{current_deg} in {branch}"
                        try:
                            edu.degree = new_deg
                        except Exception:
                            edu.__dict__["degree"] = new_deg
                        change_log.append(f"Added branch '{branch}' to degree: {new_deg}")
                    break

        # ── Certifications ────────────────────────────────────────────────────
        elif answer_type == "certifications":
            try:
                from core.schemas import CertificationEntry
                _use_schema = True
            except ImportError:
                _use_schema = False
            existing = list(getattr(pr, "certifications", []) or [])
            for line in answer_text.splitlines():
                line = line.strip().lstrip("-•* ")
                if not line: continue
                yr_m = re.search(r"(20\d\d)", line)
                if _use_schema:
                    cert = CertificationEntry(name=line, issuer=None, year=int(yr_m.group(1)) if yr_m else None)
                else:
                    cert = SimpleNamespace(name=line, issuer=None, year=int(yr_m.group(1)) if yr_m else None)
                existing.append(cert)
                change_log.append(f"Added certification: {line}")
            _setattr_safe(pr, "certifications", existing)

        # ── Hackathons / Competitions → PROJECTS ──────────────────
        elif answer_type in ("hackathons", "competitions"):
            try:
                from core.schemas import ProjectEntry
                _use_schema = True
            except ImportError:
                _use_schema = False
            existing = list(getattr(pr, "projects", []) or [])
            for line in answer_text.splitlines():
                line = line.strip().lstrip("-•* ")
                if not line: continue
                if _use_schema:
                    proj = ProjectEntry(title=line, description=line, technologies=[])
                else:
                    proj = SimpleNamespace(title=line, description=line, technologies=[])
                existing.append(proj)
                change_log.append(f"Added {answer_type}: {line} to PROJECTS section")
            _setattr_safe(pr, "projects", existing)

        # ── Additional Skills ──────────────────────────────────────────────────
        elif answer_type == "additional_skills":
            new_skills = [s.strip().lower() for s in re.split(r"[,;|]+", answer_text) if s.strip()]
            existing   = list(getattr(pr, "skills", []) or [])
            added = []
            for s in new_skills:
                if s and s not in existing:
                    existing.append(s)
                    added.append(s)
            _setattr_safe(pr, "skills", existing)
            if added:
                change_log.append(f"Added skills: {', '.join(added)}")

        # ── Internship / Job Experience → EXPERIENCE ───────────────────────────
        elif answer_type == "internship":
            try:
                from core.schemas import ExperienceEntry
                _use_schema = True
            except ImportError:
                _use_schema = False
            existing = list(getattr(pr, "experience", []) or [])
            company, role, duration, responsibilities = "", "", "", []
            _lines = answer_text.splitlines()
            for _l in _lines:
                _l  = _l.strip().lstrip("-•* ")
                _ll = _l.lower()
                if re.match(r"company\s*:", _ll):
                    company = re.sub(r"(?i)company\s*:\s*", "", _l).strip()
                elif re.match(r"role\s*:|position\s*:|title\s*:", _ll):
                    role = re.sub(r"(?i)(role|position|title)\s*:\s*", "", _l).strip()
                elif re.match(r"duration\s*:|period\s*:|from\s*:", _ll):
                    duration = re.sub(r"(?i)(duration|period|from)\s*:\s*", "", _l).strip()
                elif re.match(r"(worked on|responsibilities|i worked|description)\s*:", _ll):
                    resp_text = re.sub(r"(?i)(worked on|responsibilities|i worked|description)\s*:\s*", "", _l).strip()
                    if resp_text: responsibilities.append(resp_text)
                elif _l and len(_l) > 5:
                    responsibilities.append(_l)

            if not company and not role and _lines:
                role = _lines[0].strip().lstrip("-•* ")
                responsibilities = [l.strip().lstrip("-•* ") for l in _lines[1:] if l.strip()]

            if role or company:
                if _use_schema:
                    entry = ExperienceEntry(
                        company=company or "Organization",
                        role=role or "Member",
                        duration=duration or None,
                        responsibilities=responsibilities[:6],
                        technologies=[],
                        experience_type="internship",
                    )
                else:
                    entry = SimpleNamespace(
                        company=company or "Organization",
                        role=role or "Member",
                        duration=duration or None,
                        responsibilities=responsibilities[:6],
                        technologies=[],
                        experience_type="internship",
                    )
                existing.append(entry)
                _setattr_safe(pr, "experience", existing)
                change_log.append(f"Added internship/job: {role} at {company} to EXPERIENCE section")

        else:
            change_log.append(f"Unhandled answer type '{answer_type}': {answer_text[:60]}")

    logger.info("verify_and_map_profile_answers: %d changes", len(change_log))
    return pr, change_log


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _detect_domain(target_role: str) -> str:
    role = (target_role or "").lower()
    if any(k in role for k in {"machine learning", "ml ", "deep learning", "ai ", "artificial intelligence", "data scientist", "nlp", "computer vision"}):
        return "Machine Learning / AI"
    if any(k in role for k in {"robotics", "ros", "autonomous", "embedded", "firmware", "mechatronics", "control systems"}):
        return "Robotics / Embedded Systems"
    if any(k in role for k in {"frontend", "react", "vue", "angular", "ui developer", "web developer"}):
        return "Frontend / Web"
    if any(k in role for k in {"backend", "api developer", "django", "fastapi", "node"}):
        return "Backend / Web"
    if any(k in role for k in {"full stack", "fullstack"}):
        return "Full Stack Web"
    if any(k in role for k in {"devops", "sre", "cloud engineer", "platform engineer"}):
        return "DevOp / Cloud"
    return "Software / Technology"


# ─────────────────────────────────────────────────────────────────────────────
# LLM DOUBT QUESTION GENERATION (called from evaluator.py / ollama pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def generate_doubt_questions(
    parsed_resume,
    user_proficiencies=None,
    target_role: str = "",
    api_key: str = "",
    evaluation_result: dict = None,
    **kwargs,
) -> list[dict]:
    """
    Generate factual-integrity doubt questions for each project and experience
    entry. Questions are SPECIFIC to what the candidate wrote — not generic.

    Returns list of dicts: [{type, context, question, required}]
    """
    issues: list[dict] = []
    pr       = parsed_resume
    projects = getattr(pr, "projects", []) or []
    exp      = getattr(pr, "experience", []) or []

    # De-duplicate projects
    TECH_ONLY_RE = re.compile(
        r"^(postgresql|supabase|mongodb|react|flask|llama|langchain|api|sql|docker"
        r"|git|aws|python|java|javascript|css|html|ros|gazebo|arduino)$", re.I
    )
    seen = set()
    clean_projects = []
    for p in projects:
        t = (getattr(p, "title", "") or "").strip()
        if t and not TECH_ONLY_RE.match(t) and t.lower() not in seen:
            clean_projects.append(p)
            seen.add(t.lower())

    # ── Per-project factual questions ─────────────────────────────────────────
    for proj in clean_projects[:4]:  # cap at 4 projects
        title  = getattr(proj, "title", "Project")
        desc   = (getattr(proj, "description", "") or "").strip()
        techs  = getattr(proj, "technologies", []) or []
        tech_s = ", ".join(techs[:5]) if techs else ""

        # Tech stack missing or very thin
        if not techs or len(techs) <= 1:
            issues.append({
                "type":     "project_tech",
                "context":  f"Project: {title}",
                "question": (
                    f"For your project \u2018{title}\u2019: What did you implement, what tools did you use, and what accuracy rate/metric did you achieve? Say 'nil' if not applicable."
                ),
                "required": True,
            })
        # Description is missing measurable impact (no matter the length)
        elif not re.search(r"\d+%|\d+\s*user|\d+x|\d+\s*ms", desc):
            issues.append({
                "type":     "project_tech",
                "context":  f"Project: {title}",
                "question": (
                    f"For your project \u2018{title}\u2019: What did you implement, what tools did you use, and what accuracy rate/metric did you achieve? Say 'nil' if not applicable."
                ),
                "required": True,
            })

    # ── Per-experience factual questions ─────────────────────────────────────
    for entry in exp[:3]:  # cap at 3 experience entries
        company  = getattr(entry, "company", "") or ""
        role     = getattr(entry, "role",    "") or ""
        resps    = getattr(entry, "responsibilities", []) or []
        exp_type = (getattr(entry, "experience_type", "") or "").lower()

        # Only skip if totally empty
        if not company and not role:
            continue

        resp_text  = " ".join(resps)
        has_metric = bool(re.search(r"\d+%|\d+\s*ms|\d+\s*users|\d+x|\$\d+", resp_text))

        if exp_type == "internship" or "intern" in role.lower() or "workshop" in role.lower():
            label_name = company if company and company.lower() != "unknown" else role
            label = "Workshop" if "workshop" in role.lower() else (f"internship at {label_name} as {role}" if label_name else "internship")
            issues.append({
                "type":     "experience_clarification",
                "context":  label,
                "question": (
                    f"For your {label}: What specific projects did you build or contribute to, what were your core responsibilities, and what technologies did you use?"
                ),
                "required": True,
            })
        elif not has_metric and len(resp_text.split()) > 8:
            role_str = f" as {role}" if role else ""
            issues.append({
                "type":     "experience_clarification",
                "context":  f"Experience at {company}",
                "question": (
                    f"At {company}{role_str}: Can you describe what you built, what problem it solved, and the core technologies used? "
                    f"Quantify your impact if possible (e.g. 'Cut deployment time by 30%')."
                ),
                "required": True,
            })

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# REQUIRED ANSWERS CHECKER
# ─────────────────────────────────────────────────────────────────────────────

def all_required_answered(doubt_issues: list, clarification_answers) -> tuple:
    if not doubt_issues:
        return True, []

    answered_texts: set = set()
    if clarification_answers:
        for ca in clarification_answers:
            q = (getattr(ca, "question", "") or "").strip().lower()
            if q: answered_texts.add(q)

    unanswered = []
    for issue in doubt_issues:
        if hasattr(issue, "required"):
            required = bool(issue.required)
            question = (issue.question or "").strip().lower()
        else:
            required = bool(issue.get("required", False))
            question = (issue.get("question", "") or "").strip().lower()

        if required and question not in answered_texts:
            unanswered.append(issue)

    return len(unanswered) == 0, unanswered