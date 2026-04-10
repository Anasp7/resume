"""
core/ollama_pipeline.py
========================
Ollama phi3 pipeline — three tasks with silent fallback.

  Task 1: parse_and_structure()      — parse raw resume text into structured JSON
  Task 2: classify_and_reclassify()  — experience vs project redistribution
  Task 3: generate_doubt_questions() — detect missing/ambiguous info

On ANY Ollama failure (connection error, timeout, bad JSON):
  Task 1 failure → smart_parser.py handles parsing
  Task 2 failure → _fallback_reclassify() (Python signal words)
  Task 3 failure → _fallback_doubt_questions() (Python rule-based)

Called from app.py /analyze route AFTER document parsing.
"""
from __future__ import annotations
import json
import logging
import re
from typing import Optional

import requests

logger = logging.getLogger("smart_resume.ollama_pipeline")

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "phi3"
OLLAMA_TIMEOUT  = 120  # Give Ollama ample time to parse full resumes



# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL CALLER
# ─────────────────────────────────────────────────────────────────────────────


def _call_ollama(prompt: str, temperature: float = 0.1, max_tokens: int = 3000) -> Optional[str]:
    """
    POST to Ollama /api/generate. Returns raw text response or None on any failure.
    Never raises.
    """
    try:
        payload = {
            "model":       OLLAMA_MODEL,
            "prompt":      prompt,
            "stream":      False,
            "temperature": temperature,
            "options": {
                "num_predict": max_tokens,
                "stop": ["---END---"],
            },
        }
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        if r.status_code == 200:
            text = r.json().get("response", "").strip()
            logger.info("Ollama responded: %d chars", len(text))
            return text
        logger.warning("Ollama HTTP %d: %s", r.status_code, r.text[:200])
        return None
    except requests.exceptions.ConnectionError:
        logger.warning("Ollama not reachable at %s — fallback will be used", OLLAMA_BASE_URL)
        return None
    except requests.exceptions.Timeout:
        logger.warning("Ollama timed out after %ds — fallback will be used", OLLAMA_TIMEOUT)
        return None
    except Exception as e:
        logger.warning("Ollama call failed: %s", e)
        return None


def _extract_json(text: str) -> Optional[dict]:
    """Extract first valid JSON object from Ollama response text."""
    if not text:
        return None
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Remove invalid control characters that break json.loads
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        logger.warning("JSON parse failed: %s", e)
        return None


def ollama_is_available() -> bool:
    """Quick check whether Ollama is reachable."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def call_ollama(prompt: str, system_msg: Optional[str] = None, max_tokens: int = 2500) -> Optional[str]:
    """
    Public generic caller for Ollama.
    If system_msg is provided, prepends it to the prompt.
    """
    full_prompt = prompt
    if system_msg:
        full_prompt = f"System: {system_msg}\n\nUser: {prompt}"
    return _call_ollama(full_prompt, temperature=0.15, max_tokens=max_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# EDUCATION PARSER (separate Ollama call so we can use full prompt focus)
# ─────────────────────────────────────────────────────────────────────────────

_EDU_PARSE_PROMPT = """You are a top-tier, ZERO-HALLUCINATION resume education parser. Extract ALL education entries from the text below.

ABSOLUTE RULES:
1. NEVER invent, assume, or guess an institution name. If no high school name is explicitly provided (e.g., the text just says "Class 10 (CBSE)"), you MUST set institution to "CBSE" or leave it blank "". NEVER invent names like "Government High School".
2. NO EXPLANATIONS: Limit all values to the clean, literal text. NEVER use parentheses to explain what you did.
3. RECONSTRUCT FRAGMENTED NAMES WITH SPACES: If the PDF text randomly broke a university name in half, silently merge them with proper spaces.
4. ISOLATE DATES: NEVER leave date strings inside "institution", "degree", or "field" values. Extract ALL dates into "graduation_year".
5. ACADEMIC ONLY: Extract ONLY legitimate academic degrees, diplomas, and schools.
6. PRESERVE ALL SPACES: Ensure every word is separated by a standard space.
7. YEAR RANGES MUST BE PRESERVED EXACTLY: If the resume shows "2023-2027" or "2023 - 2027", copy it VERBATIM into graduation_year. Do NOT extract only the last year. Do NOT change "2023-2027" to just "2027".
8. NEVER HALLUCINATE A YEAR: If the text does not explicitly state a year, leave graduation_year as "".

Return ONLY this JSON (no markdown, no explanation):
{{
  "education": [
    {{
      "institution": "Full cleaned institution name, or empty string. NO EXPLANATIONS.",
      "degree": "Degree type e.g. Bachelor of Technology or Class 12",
      "field": "Field of study e.g. Computer Science",
      "graduation_year": "Copy verbatim: YYYY, YYYY-YYYY, YYYY - Present, Expected YYYY, or empty string",
      "gpa": "CGPA or percentage value only, e.g. 8.93/10.0 or 94.8%",
      "level": "bachelors|masters|phd|class12|class10"
    }}
  ]
}}

RESUME TEXT:
{resume_text}

Return ONLY valid JSON."""


def parse_education_ollama(raw_text: str) -> Optional[list]:
    """
    Dedicated Ollama education parser.
    Returns list of education dicts on success, None on failure.
    """
    truncated = raw_text[:4000] if len(raw_text) > 4000 else raw_text
    prompt    = _EDU_PARSE_PROMPT.format(resume_text=truncated)

    logger.info("Ollama Education parse: parsing education section...")
    raw_response = _call_ollama(prompt, temperature=0.0, max_tokens=1000)
    if not raw_response:
        return None

    result = _extract_json(raw_response)
    if result and 'education' in result:
        return result['education']
    return None

_PARSE_PROMPT = """You are a top-tier, zero-hallucination resume semantic parser.
Your ONLY job: extract skills, experience, and projects from the resume text below.
Do NOT extract name, email, phone, location, github, linkedin, or education — those are handled by a separate system.

═══════════════════════════════════════════════
DATA INTEGRITY & COMPLETENESS RULES (CRITICAL):
═══════════════════════════════════════════════
1. ZERO-LAZY EXTRACTION: You must extract EVERY SINGLE bullet point/item listed in the resume. 
   - NEVER summarize multiple bullets into one. 
   - NEVER skip bullets to save space. 
   - If an experience entry has 8 bullet points, you MUST return 8 bullet points in the JSON.
2. VERBATIM FACTS, RICHER TONE: 
   - Keep the technical details 100% accurate. 
   - Slightly professionalize the verbs for better impact (e.g., "Worked on" → "Engineered", "Built" → "Architected", "Assisted" → "Collaborated on").
   - DO NOT invent metrics or outcomes. Only improve the vocabulary.

═══════════════════════════════════════════════
ANTI-HALLUCINATION RULES:
═══════════════════════════════════════════════
A. VERBATIM ONLY — copy text exactly as present. Never invent or infer details not present.
B. NEVER use a date or date fragment as a company name.
C. DO NOT REPEAT COMPANY NAMES: If a company name is repeated in different formats (e.g., "Deloitte" and "DeloitteAustralia"), use ONLY the cleanest version with spaces. NEVER return "Company Name CompanyName".
D. NEVER split a multi-word skill into single words.
   ✅ "Machine Learning"   ❌ "Machine", "Learning"
E. If you cannot confidently identify the company name, use empty string "" — NEVER guess.
F. Return ONLY the JSON object. No explanation.

═══════════════════════════════════════════════
HOW TO IDENTIFY AN EXPERIENCE BLOCK:
═══════════════════════════════════════════════
(Same as before... see below for JSON format)

═══════════════════════════════════════════════
RESUME TEXT:
═══════════════════════════════════════════════
{resume_text}

═══════════════════════════════════════════════
OUTPUT — Return this EXACT JSON structure:
═══════════════════════════════════════════════
{{
  "skills": ["exact skill phrase 1", "exact skill phrase 2"],
  "experience": [
    {{
      "role": "professional job title",
      "company": "clean company name with spaces",
      "duration": "date range exactly as written",
      "responsibilities": ["Professionalized bullet 1", "Professionalized bullet 2", "..."],
      "technologies": ["tech explicitly mentioned in bullets"],
      "experience_type": "internship|job|volunteer|workshop|competition|freelance"
    }}
  ],
  "projects": [
    {{
      "title": "exact project name",
      "description": "Professionalized combined description of ALL project bullets. Use spaces to separate ideas. DO NOT TRUNCATE.",
      "technologies": ["tech1", "tech2"],
      "project_type": "academic|personal|competition|open-source|freelance"
    }}
  ]
}}"""


def parse_and_structure(raw_text: str) -> Optional[dict]:
    """
    Task 1: Semantic parse of skills/experience/projects via Ollama phi3.
    Returns structured dict on success, None on failure (triggers smart_parser fallback).
    """
    # Increased to 6000 chars to ensure no data is cut off
    truncated = raw_text[:6000] if len(raw_text) > 6000 else raw_text
    prompt    = _PARSE_PROMPT.format(resume_text=truncated)

    logger.info("Ollama Task 1: semantic parsing (%d chars)...", len(truncated))
    # temperature=0.0 for maximum determinism — no creative guessing
    raw_response = _call_ollama(prompt, temperature=0.0, max_tokens=2500)
    if not raw_response:
        logger.info("Ollama Task 1: no response — smart_parser fallback")
        return None

    result = _extract_json(raw_response)
    if not result:
        logger.warning("Ollama Task 1: non-JSON response — smart_parser fallback")
        return None

    # Validate scoped output — we only expect skills/experience/projects now
    has_content = (
        result.get("skills") or
        result.get("experience") or
        result.get("projects")
    )
    if not has_content:
        logger.warning("Ollama Task 1: empty result — smart_parser fallback")
        return None

    # Strip only name/contact fields — keep education if LLM included it
    for _leaked_key in ("name", "email", "phone", "location", "github", "linkedin"):
        result.pop(_leaked_key, None)

    logger.info(
        "Ollama Task 1 success: skills=%d exp=%d proj=%d",
        len(result.get("skills", [])),
        len(result.get("experience", [])),
        len(result.get("projects", [])),
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — EXPERIENCE vs PROJECT CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

_CLASSIFY_PROMPT = """You are a resume section classifier. Review the experience entries below
and decide whether any should be reclassified as Projects.

CLASSIFICATION RULES:
- EXPERIENCE = paid employment, internship, trainee/apprentice role at a real company.
- PROJECTS = academic builds, DBMS projects, alumni platforms, web apps, hackathons,
  BAJA/SAE competitions, personal tools — any entry describing something the candidate BUILT.
- Competitions like ABAJA, SAE Baja, Robocon → PROJECTS (competition type). Never experience.
- Entries with no real company name (e.g. "WEB CREATION") that describe building something → PROJECTS.
- Internships at named companies → always keep as EXPERIENCE.

EXPERIENCE ENTRIES (0-indexed):
{experience_json}

EXISTING PROJECTS:
{projects_json}

Return ONLY this JSON:
{{
  "keep_in_experience": [0, 2],
  "move_to_projects": [
    {{
      "source_index": 1,
      "title": "project title",
      "description": "what was built",
      "technologies": ["tech1"],
      "project_type": "academic|competition|personal"
    }}
  ]
}}"""


def classify_and_reclassify(
    experience: list[dict],
    projects: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Task 2: Reclassify experience entries as projects where appropriate.
    Falls back to _fallback_reclassify() on Ollama failure.
    Returns (final_experience, final_projects).
    """
    if not experience:
        return experience, projects

    prompt = _CLASSIFY_PROMPT.format(
        experience_json=json.dumps(experience, indent=2),
        projects_json=json.dumps(projects, indent=2),
    )

    logger.info("Ollama Task 2: classifying %d experience entries...", len(experience))
    raw_response = _call_ollama(prompt, temperature=0.05)
    if not raw_response:
        logger.info("Ollama Task 2: no response — Python fallback")
        return _fallback_reclassify(experience, projects)

    result = _extract_json(raw_response)
    if not result:
        logger.warning("Ollama Task 2: bad JSON — Python fallback")
        return _fallback_reclassify(experience, projects)

    keep_indices = set(result.get("keep_in_experience", list(range(len(experience)))))
    move_items   = result.get("move_to_projects", [])

    final_exp  = [exp for i, exp in enumerate(experience) if i in keep_indices]
    extra_proj = []
    for item in move_items:
        src = item.get("source_index")
        if src is not None and 0 <= src < len(experience):
            extra_proj.append({
                "title":        item.get("title") or experience[src].get("role", "Project"),
                "description":  item.get("description") or " ".join(experience[src].get("responsibilities", [])),
                "technologies": item.get("technologies") or experience[src].get("technologies", []),
                "project_type": item.get("project_type", "academic"),
            })

    logger.info(
        "Ollama Task 2: kept %d in experience, moved %d to projects",
        len(final_exp), len(extra_proj),
    )
    return final_exp, extra_proj + projects


def _fallback_reclassify(
    experience: list[dict],
    projects: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Python signal-word fallback for Task 2."""
    COMPETITION = {
        "baja", "abaja", "a baja", "sae baja", "hackathon", "ideathon",
        "datathon", "competition", "contest", "challenge", "olympiad",
        "smart india", "robocon", "robowars",
    }
    PROJECT_SIGS = {
        "dbms", "alumni connect", "alumni", "connect platform", "web creation",
        "web project", "college project", "semester project", "academic project",
        "mini project", "major project", "final year project", "capstone",
        "coursework", "assignment", "platform", "portal", "application",
        "website", "system", "tool", "bot", "dashboard", "module", "pipeline",
        "simulation", "prototype", "demo", "thesis",
    }
    EMPLOYMENT = {
        "intern", "internship", "trainee", "apprentice", "full time",
        "fulltime", "full-time", "part time", "part-time", "employee",
        "staff", "associate", "fellowship",
    }

    final_exp  = []
    extra_proj = []

    for entry in experience:
        text = (
            (entry.get("role", "") or "") + " " +
            (entry.get("company", "") or "") + " " +
            " ".join(entry.get("responsibilities", []))
        ).lower()

        if any(s in text for s in COMPETITION):
            extra_proj.append({
                "title":        entry.get("role") or entry.get("company") or "Competition",
                "description":  " ".join(entry.get("responsibilities", [])),
                "technologies": entry.get("technologies", []),
                "project_type": "competition",
            })
        elif any(s in text for s in EMPLOYMENT):
            final_exp.append(entry)
        elif any(s in text for s in PROJECT_SIGS):
            extra_proj.append({
                "title":        entry.get("role") or entry.get("company") or "Project",
                "description":  " ".join(entry.get("responsibilities", [])),
                "technologies": entry.get("technologies", []),
                "project_type": "academic",
            })
        else:
            final_exp.append(entry)

    return final_exp, extra_proj + projects


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — DOUBT QUESTION GENERATION
# ─────────────────────────────────────────────────────────────────────────────

_DOUBT_PROMPT = """You are an expert technical recruiter reviewing a student resume for a {target_role} role.
Your task is to generate FACTUAL clarification questions based ONLY on what is written below.

CANDIDATE: {name}
TARGET ROLE: {target_role}
SKILLS LISTED: {skills_preview}

PROJECTS IN RESUME:
{projects_summary}

EXPERIENCE IN RESUME:
{exp_summary}

RULES — STRICTLY FOLLOW:
1. Generate EXACTLY ONE question for EVERY INDIVIDUAL project, and EXACTLY ONE question for EVERY INDIVIDUAL experience/internship listed above. Do not skip any!
2. Generate ONLY questions of type "project_tech" or "experience_clarification".
3. Each question MUST reference the EXACT project title or company name from above.
4. DO NOT ask about GitHub, LinkedIn, school marks, or soft skills (these are asked by another system).
5. For projects, you MUST ask EXACTLY: "For your project '[Project Name]': What did you implement, what tools did you use, and what accuracy rate/metric did you achieve? Say 'nil' if not applicable."
6. For internships, workshops, or experience, you MUST ask EXACTLY: "For your experience/role at [Company/Workshop]: What did you practice/perform there? (Focus on what you did, not just skills)."
7. DO NOT make up generic questions. Stick to these exact templates, just filling in the specific Project Name or Company Name.
8. Return ONLY valid JSON — no markdown, no explanation outside JSON.

Return EXACTLY this JSON:
{{
  "doubt_questions": [
    {{
      "type": "project_tech",
      "question": "For your project 'App Name': What did you implement, what tools did you use, and what accuracy rate/metric did you achieve? Say 'nil' if not applicable.",
      "context": "Project - App Name",
      "required": false
    }},
    {{
      "type": "experience_clarification",
      "question": "For your experience/role at Internship Corp: What did you practice/perform there? (Focus on what you did, not just skills).",
      "context": "Experience - Internship Corp",
      "required": false
    }}
  ]
}}"""


def generate_doubt_questions(parsed_data: dict, target_role: str) -> list[dict]:
    """
    Task 3: Generate doubt/clarification questions via Ollama.
    Uses actual project titles, descriptions, and experience bullets as context
    so questions are SPECIFIC and NOT generic.
    Falls back to _fallback_doubt_questions() on failure.
    """
    # Build rich project context — title + type + full description
    projects_summary = "\n".join(
        f"  - \"{p.get('title','?')}\" ({p.get('project_type','project')}): "
        f"{(p.get('description','') or '')[:150]}"
        f"{' [tech: ' + ', '.join(p.get('technologies',[])) + ']' if p.get('technologies') else ''}"
        for p in parsed_data.get("projects", [])
    ) or "  None listed"

    # Build rich experience context — role, company, responsibilities
    exp_summary = "\n".join(
        f"  - {e.get('role','?')} at {e.get('company','?')}"
        f" [{e.get('experience_type','job')}]"
        f": {'; '.join((e.get('responsibilities') or [])[:3])}"
        for e in parsed_data.get("experience", [])
    ) or "  None listed"

    prompt = _DOUBT_PROMPT.format(
        target_role      = target_role,
        name             = parsed_data.get("name", "Unknown"),
        skills_preview   = ", ".join(parsed_data.get("skills", [])[:12]) or "None listed",
        projects_summary = projects_summary,
        exp_summary      = exp_summary,
    )

    logger.info("Ollama Task 3: generating doubt questions for role=%s...", target_role)
    raw_response = _call_ollama(prompt, temperature=0.15, max_tokens=900)
    if not raw_response:
        logger.info("Ollama Task 3: no response — Python fallback")
        return _fallback_doubt_questions(parsed_data)

    result = _extract_json(raw_response)
    if not result or "doubt_questions" not in result:
        logger.warning("Ollama Task 3: bad JSON — Python fallback")
        return _fallback_doubt_questions(parsed_data)

    questions = result.get("doubt_questions", [])
    # Validate: each must have type and question
    valid = [
        q for q in questions
        if isinstance(q, dict) and q.get("type") and q.get("question")
    ]
    if not valid:
        logger.warning("Ollama Task 3: empty valid questions — Python fallback")
        return _fallback_doubt_questions(parsed_data)

    logger.info("Ollama Task 3: %d questions generated via LLM", len(valid))
    n_projects = len(parsed_data.get("projects", []))
    n_exp      = len(parsed_data.get("experience", []))
    cap        = max(8, n_projects + n_exp)  # one per entry, min 8
    return valid[:cap]


def _fallback_doubt_questions(parsed_data: dict) -> list[dict]:
    """Python rule-based fallback for Task 3 — includes project/experience specific questions."""
    return [] # Disabled to avoid duplicate generation; doubt_engine.py handles fallback now.
    questions = []

    def _q(type_, question, context="General"):
        return {"type": type_, "question": question, "context": context, "required": False}

    # ── Project-specific questions ────────────────────────────────────────────
    candidate_skills = parsed_data.get("skills", [])
    # Use meaningful skills as examples (skip single-char like "c", "r")
    example_skills = [s for s in candidate_skills if len(s) > 1 and s.lower() not in {"c","r","go"}]
    example_techs  = ", ".join(example_skills[:4]) if example_skills else "Python, SQL, frameworks used"

    # Build project list from both projects AND experience entries
    # (experience entries become projects after reclassification)
    COMP_SIGS = {"baja","abaja","competition","hackathon","contest","robocon","olympiad","sae"}
    PROJ_SIGS = {"dbms","alumni","web creation","platform","portal","website","system","bot",
                 "dashboard","capstone","thesis","project","module","application"}

    all_project_candidates = []
    for proj in parsed_data.get("projects", []):
        all_project_candidates.append({
            "title":    proj.get("title","") or proj.get("company","") or "Project",
            "type":     proj.get("project_type",""),
            "desc":     proj.get("description",""),
            "techs":    proj.get("technologies",[]),
        })
    # Also pull from experience — these may be competitions/academic projects
    for exp in parsed_data.get("experience", []):
        company = exp.get("company","") or ""
        role    = exp.get("role","") or ""
        combined = (company + " " + role).lower()
        if any(k in combined for k in COMP_SIGS):
            all_project_candidates.append({
                "title": company or role,
                "type":  "competition",
                "desc":  " ".join(exp.get("responsibilities",[])),
                "techs": exp.get("technologies",[]),
            })
        elif any(k in combined for k in PROJ_SIGS):
            all_project_candidates.append({
                "title": company or role,
                "type":  "academic",
                "desc":  " ".join(exp.get("responsibilities",[])),
                "techs": exp.get("technologies",[]),
            })

    for proj in all_project_candidates:
        raw_title = proj["title"].strip()
        p_type    = proj["type"].lower()
        desc      = proj["desc"].strip()
        techs     = proj["techs"]

        if not raw_title or len(raw_title) < 2:
            continue
            
        # Ignore titles that are just dates or garbage symbols
        import re as _re2
        if _re2.match(r"^[\d\W]+$", raw_title):
            continue

        # ── Plan A: extract real project name from description ────────────────
        # e.g. "Worked for a dbms project (Alumni Connect Platform)" → "Alumni Connect Platform"
        # e.g. "Developed the ABAJA autonomous vehicle system" → use as-is
        real_name = None

        # Pattern 1: name in parentheses — "(Alumni Connect Platform)"
        paren_m = _re2.search(r"\(([A-Z][A-Za-z0-9 &\-]{3,40})\)", desc)
        if paren_m:
            candidate_name = paren_m.group(1).strip()
            # Only use if it looks like a proper name (not just "2025" or "B.Tech")
            if not _re2.match(r"^\d+$", candidate_name) and len(candidate_name) > 4:
                real_name = candidate_name

        # Pattern 2: "for a/the <ProjectName> project/platform/system"
        if not real_name:
            phrase_m = _re2.search(
                r"(?:for\s+(?:a|the|an)\s+)([A-Z][A-Za-z0-9 &\-]{3,35})"
                r"(?:\s+(?:project|platform|system|tool|app|module|website))",
                desc, _re2.IGNORECASE
            )
            if phrase_m:
                real_name = phrase_m.group(1).strip().title()

        # Use real name if found, otherwise fall back to raw_title
        display_title = real_name if real_name else raw_title

        # Ask for tech stack if missing or only single generic skill
        if not techs or (len(techs) <= 1 and (not techs or techs[0].lower() in {"c","python","sql"})):
            questions.append(_q("project_tech",
                f"For your '{display_title}' project: What specific problem did this solve, what features did you implement, and what technologies did you use? "
                f"(Your skills include: {example_techs})",
                f"Projects — {display_title}"))

        # Ask about outcome for competition projects
        if p_type == "competition" or any(k in raw_title.lower() for k in {"baja","abaja","hackathon","competition","robocon"}):
            questions.append(_q("project_tech",
                f"For '{display_title}': What was your specific contribution — what features or components did you build, and what technologies were used?",
                f"Projects — {display_title}"))

        # Ask for more detail if description is vague
        elif len(desc.split()) < 10:
            questions.append(_q("project_tech",
                f"Can you briefly describe what you built for '{display_title}', what problem it solved, and the core technologies used?",
                f"Projects — {display_title}"))

    # ── Experience-specific questions ─────────────────────────────────────────
    for exp in parsed_data.get("experience", []):
        role    = exp.get("role", "")
        company = exp.get("company", "")
        resps   = exp.get("responsibilities", [])
        exp_type = (exp.get("experience_type") or "").lower()

        if exp_type == "internship" and company and company.lower() not in {"unknown", ""}:
            questions.append(_q("experience_clarification",
                f"For your internship at {company} as {role}: "
                f"What specific projects did you build or contribute to, what were your core responsibilities, and what technologies did you use?",
                f"Experience — {company}"))
        elif resps and len(" ".join(resps).split()) < 8:
            questions.append(_q("experience_clarification",
                f"For '{role}': Can you describe in more detail what you worked on and what you achieved?",
                f"Experience — {role}"))

    # ── Removed duplicates: Hackathons/Workshops/Certs are handled centrally by doubt_engine.py

    return questions


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR — called from app.py /analyze route
# ─────────────────────────────────────────────────────────────────────────────
# Architecture:
#   Parsing        → smart_parser.py        (fast, deterministic)
#   Classification → smart_parser.py        (Python signal words)
#   Doubt questions→ Ollama phi3 Task 3     (natural language, role-aware)
# ─────────────────────────────────────────────────────────────────────────────

def run_ollama_pipeline(raw_text: str, target_role: str, parsed_context: dict = None) -> dict:
    """
    Runs ONLY Task 3 — doubt question generation via Ollama phi3.

    Parsing and classification are always handled by smart_parser.py unless parsed_context is provided.
    Ollama is used only for generating context-aware clarification questions.

    Returns:
    {
        "parsed":           None,        # always — smart_parser handles parsing
        "used_ollama":      False,       # always — smart_parser handles parsing
        "doubt_questions":  list[dict],  # Ollama or Python fallback
        "ollama_available": bool,
    }
    """
    available = ollama_is_available()
    if not available:
        logger.info("Ollama not available — Python doubt question fallback")
        return {
            "parsed":           None,
            "used_ollama":      False,
            "doubt_questions":  [],
            "ollama_available": False,
        }

    # Parse resume first to give Ollama real context for question generation
    try:
        from core.smart_parser import parse_resume as _smart_parse, _reclassify_experience_entries
        if not parsed_context:
            parsed_context = _smart_parse(raw_text)

        # Reclassify experience → projects so titles are correct
        # e.g. company="ABAJA", role="Automation Team Member" → project title="ABAJA"
        exp_dicts = parsed_context.get("experience", [])
        proj_dicts = parsed_context.get("projects", [])

        # Handle both dict and object formats from smart_parse
        def _to_dict(e):
            if isinstance(e, dict): return e
            return {
                "role":             getattr(e, "role", ""),
                "company":          getattr(e, "company", ""),
                "duration":         getattr(e, "duration", ""),
                "responsibilities": getattr(e, "responsibilities", []),
                "technologies":     getattr(e, "technologies", []),
            }
        def _proj_to_dict(p):
            if isinstance(p, dict): return p
            return {
                "title":        getattr(p, "title", ""),
                "description":  getattr(p, "description", ""),
                "technologies": getattr(p, "technologies", []),
                "project_type": getattr(p, "project_type", ""),
            }

        exp_dicts  = [_to_dict(e)  for e in exp_dicts]
        proj_dicts = [_proj_to_dict(p) for p in proj_dicts]

        reclassified_exp, reclassified_proj = _reclassify_experience_entries(exp_dicts, proj_dicts)
        parsed_context["experience"] = reclassified_exp
        parsed_context["projects"]   = reclassified_proj

    except Exception as ex:
        logger.warning("smart_parse for Ollama context failed: %s", ex)
        parsed_context = {
            "name": "", "skills": [], "experience": [], "projects": [],
            "education": [], "github": "", "linkedin": "", "certifications": [],
        }

    logger.info("Ollama Task 3: generating doubt questions for role=%s", target_role)
    doubt_questions = generate_doubt_questions(parsed_context, target_role)

    return {
        "parsed":           None,
        "used_ollama":      False,
        "doubt_questions":  doubt_questions,
        "ollama_available": True,
    }