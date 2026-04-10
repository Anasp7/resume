"""
core/smart_parser.py
=====================
Zero-LLM resume parser — 100% Python regex + heuristics.
Drop-in replacement for core/smart_parser.py.

v3 changes:
  - Experience → Project reclassification (_reclassify_experience_entries)
  - Handles resumes with NO projects section (moves project-like experience entries)
  - ABAJA / SAE Baja → competition project
  - DBMS / Alumni Connect / WEB CREATION → academic project
  - Improved section splitter for uppercase headers
  - Better contact extraction (LinkedIn, GitHub)
"""
from __future__ import annotations
import re
import logging
from typing import Optional

logger = logging.getLogger("smart_resume.smart_parser")

# ─────────────────────────────────────────────────────────────────────────────
# SKILL VOCABULARY SETS
# ─────────────────────────────────────────────────────────────────────────────

LANG_SET = {
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go", "rust",
    "kotlin", "swift", "r", "scala", "ruby", "php", "bash", "dart", "matlab",
    "lua", "perl", "fortran", "vhdl", "verilog", "assembly", "asm", "groovy",
    "haskell", "elixir", "clojure",
}
FWK_SET = {
    "react", "vue", "angular", "nextjs", "svelte", "gatsby", "nuxt", "astro",
    "bootstrap", "tailwind", "htmx", "jquery",
    "django", "flask", "fastapi", "spring", "express", "nestjs", "gin", "fiber",
    "laravel", "rails", "phoenix", "actix",
    "flutter", "reactnative", "xamarin", "ionic", "swiftui", "jetpack",
    "pytorch", "tensorflow", "keras", "sklearn", "scikit", "xgboost", "lightgbm",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly", "opencv", "pillow",
    "langchain", "llamaindex", "huggingface", "transformers",
    "electron", "tauri", "unity", "unreal", "godot", "gtk", "qt", "tkinter", "kivy",
    "ros", "ros2", "moveit", "rviz", "gazebo", "pcl",
}
DB_SET = {
    "sql", "postgresql", "mysql", "sqlite", "mongodb", "redis", "firebase",
    "cassandra", "dynamodb", "elasticsearch", "mariadb", "oracle", "neo4j",
    "supabase", "prisma",
}
TOOL_SET = {
    "docker", "kubernetes", "git", "github", "linux", "aws", "gcp", "azure",
    "jenkins", "terraform", "grafana", "postman", "ansible", "vagrant", "helm",
    "nginx", "apache", "cmake", "arduino", "raspberry", "simulink", "openocd",
    "freertos", "rtos", "can", "modbus", "uart", "spi", "i2c", "esp32", "stm32",
    "labview", "proteus", "keil", "autocad", "solidworks", "catia", "fusion360",
    "kicad", "altium", "ltspice",
    "hadoop", "spark", "airflow", "kafka", "mlflow", "dvc", "wandb",
    "jupyter", "colab", "tableau", "powerbi", "excel",
    "figma", "jira", "notion", "trello", "vscode", "vim", "latex", "obsidian",
}
ALWAYS_KEEP = {"c", "r", "go"}

LABEL_WORDS = {
    "languages", "language", "softwares", "software", "tools", "tool",
    "frameworks", "framework", "libraries", "library", "others", "other",
    "technical", "technologies", "technology", "stack", "skills", "skill",
    "databases", "database", "and", "the", "a", "an", "of", "for", "in",
    "is", "to", "platforms", "platform", "using", "with", "project",
    "management", "time", "teamwork", "leadership", "communication",
    "proficient", "familiar", "experience", "exposure", "knowledge",
    "understanding", "basic", "advanced", "intermediate", "beginner",
    "expert", "strong", "dbms",
}
SOFT_SKILLS = {
    "teamwork", "time management", "leadership", "project management",
    "communication", "problem solving", "critical thinking", "interpersonal",
    "attention to detail", "adaptability", "creativity", "collaboration",
    "work ethic", "self motivated", "multitasking", "flexibility",
    "presentation", "negotiation", "decision making",
}

KNOWN_SECTIONS = {
    "objective", "technical skills", "skills", "experience", "work experience",
    "projects", "education", "certifications", "achievements", "awards",
    "publications", "languages", "summary", "profile", "internships",
    "extracurriculars", "activities", "interests", "hobbies", "volunteer",
    "positions of responsibility", "por", "technical expertise", "key skills",
    "core competencies", "tools & technologies", "tools and technologies",
    "academic projects", "personal projects", "mini projects",
    "scholastic achievements", "co-curricular", "extra curricular", "declaration",
    "about me", "career objective", "professional summary", "work history",
    "professional experience", "academic background", "academic details",
    "courses", "coursework", "relevant coursework", "training", "workshops",
}

DEGREE_PAT = re.compile(
    r"bachelor|b\.?[\s-]*tech|b\.?[\s-]*e\.?\b|master|m\.?[\s-]*tech|m\.?[\s-]*e\.?\b|"
    r"phd|ph\.d|diploma|b\.sc|m\.sc|b\.com|mba|plus two|10\+2|hsc|sslc|"
    r"higher secondary|secondary|12th|10th|class 12|class 10|ssc|cbse|icse",
    re.I,
)
GPA_PAT    = re.compile(r"(?:cgpa|gpa|grade|percentage)\s*[:/]?\s*([\d.]+\s*%?)", re.I)
YEAR_PAT   = re.compile(r"\b(?:Expected\s*)?(20\d\d|19\d\d)\b", re.I)
PHONE_PAT  = re.compile(r"(?:\+\d{1,3}[\s-]?)?\(?\d{3,5}\)?[\s.-]?\d{3,5}[\s.-]?\d{3,5}")
EMAIL_PAT  = re.compile(r"[\w.+-]+@[\w.-]+\.\w{2,}")
DURATION_PAT = re.compile(
    r"(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*)?"
    r"(\b20\d\d|\b19\d\d)\s*[-–—to]+\s*"
    r"(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*)?"
    r"(\b20\d\d|\b19\d\d|present|current|now|\b\d{4})",
    re.I,
)

# ─────────────────────────────────────────────────────────────────────────────
# RECLASSIFICATION SIGNALS
# ─────────────────────────────────────────────────────────────────────────────

RECLASSIFY_AS_COMPETITION = {
    "baja", "abaja", "a baja", "sae baja", "hackathon", "ideathon",
    "datathon", "competition", "contest", "challenge", "olympiad",
    "smart india", "robocon", "robowars",
}

KEEP_AS_EXPERIENCE_SIGNALS = {
    "intern", "internship", "trainee", "apprentice",
    "full time", "fulltime", "full-time", "part time", "part-time",
    "employee", "staff", "associate", "fellowship", "research assistant",
}

RECLASSIFY_AS_PROJECT_SIGNALS = {
    "dbms", "alumni connect", "alumni", "connect platform", "web creation",
    "web project", "college project", "university project", "semester project",
    "academic project", "mini project", "major project", "final year project",
    "capstone", "coursework", "assignment", "platform", "portal", "application",
    "website", "system", "tool", "bot", "dashboard", "module", "pipeline",
    "simulation", "prototype", "demo", "thesis",
}


def _reclassify_experience_entries(
    experience: list[dict],
    projects: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Redistribute experience entries to projects where appropriate.
    Called at end of parse_resume().

    Precedence:
      1. Competition signals (baja, hackathon, etc.) → projects (competition)
      2. Employment signals (intern, full-time, etc.) → keep in experience
      3. Project signals (dbms, platform, etc.)       → projects (academic)
      4. Ambiguous                                     → keep in experience
    """
    final_exp  = []
    extra_proj = []

    for entry in experience:
        all_text = (
            (entry.get("role", "") or "") + " " +
            (entry.get("company", "") or "") + " " +
            " ".join(entry.get("responsibilities", []))
        ).lower()

        if any(sig in all_text for sig in RECLASSIFY_AS_COMPETITION):
            desc = " ".join(entry.get("responsibilities", []))
            if not desc:
                desc = (entry.get("role", "") + " " + entry.get("company", "")).strip()
            # Use company as title (e.g. "ABAJA") — it's the project/competition name
            # Role (e.g. "Automation Team Member") becomes part of description
            proj_title = entry.get("company") or entry.get("role") or "Competition"
            proj_desc  = entry.get("role", "") + (" — " + desc.strip() if desc.strip() else "")
            extra_proj.append({
                "title":        proj_title,
                "description":  proj_desc.strip(" — "),
                "technologies": entry.get("technologies", []),
                "project_type": "competition",
            })
            logger.debug("Reclassified '%s' → competition project '%s'",
                         entry.get("role"), proj_title)
            continue

        if any(sig in all_text for sig in KEEP_AS_EXPERIENCE_SIGNALS):
            final_exp.append(entry)
            continue

        if any(sig in all_text for sig in RECLASSIFY_AS_PROJECT_SIGNALS):
            desc = " ".join(entry.get("responsibilities", []))
            if not desc:
                desc = (entry.get("role", "") + " at " + entry.get("company", "")).strip(" at")

            # Plan A: extract real project name from description
            # e.g. "Worked for a dbms project (Alumni Connect Platform)" → "Alumni Connect Platform"
            real_title = None
            paren_m = re.search(r"\(([A-Z][A-Za-z0-9 &\-]{3,40})\)", desc)
            if paren_m:
                candidate = paren_m.group(1).strip()
                if not re.match(r"^\d+$", candidate) and len(candidate) > 4:
                    real_title = candidate
            if not real_title:
                phrase_m = re.search(
                    r"(?:for\s+(?:a|the|an)\s+)([A-Z][A-Za-z0-9 &\-]{3,35})"
                    r"(?:\s+(?:project|platform|system|tool|app|module|website))",
                    desc, re.IGNORECASE
                )
                if phrase_m:
                    real_title = phrase_m.group(1).strip().title()

            # Use extracted real name, fall back to company/role
            proj_title = real_title or entry.get("company") or entry.get("role") or "Project"
            proj_desc  = entry.get("role", "") + (" — " + desc.strip() if desc.strip() else "")
            extra_proj.append({
                "title":        proj_title,
                "description":  proj_desc.strip(" — "),
                "technologies": entry.get("technologies", []),
                "project_type": "academic",
            })
            logger.debug("Reclassified '%s' → academic project '%s'",
                         entry.get("role"), proj_title)
            continue

        final_exp.append(entry)

    return final_exp, extra_proj + projects


# ─────────────────────────────────────────────────────────────────────────────
# SECTION SPLITTER
# ─────────────────────────────────────────────────────────────────────────────

def _split_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current = "__header__"
    buf: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        ll = stripped.lower().rstrip(":")
        is_header = ll in KNOWN_SECTIONS

        # Heuristic: ALL-CAPS line with no digits/pipes → likely a section header
        _prev = buf[-1].strip() if buf else ""
        _prev_is_year_range = bool(re.match(r"^\d{4}\s*[-–·]\s*\d{0,4}\s*$", _prev))
        if (not is_header and stripped.isupper() and 3 < len(stripped) < 45
                and not re.search(r"\d", stripped)
                and "|" not in stripped
                and ":" not in stripped
                and not _prev_is_year_range
                and not stripped.startswith(("•", "-", "*", "·"))):
            is_header = True

        if is_header:
            sections[current] = "\n".join(buf).strip()
            current = ll
            buf = []
        else:
            buf.append(line)

    sections[current] = "\n".join(buf).strip()
    return sections


# ─────────────────────────────────────────────────────────────────────────────
# CONTACT EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def _extract_contact(sections: dict, raw_text: str) -> dict:
    header_text = sections.get("__header__", "") + "\n" + raw_text[:600]

    phone_m = PHONE_PAT.search(header_text)
    email_m = EMAIL_PAT.search(header_text)

    name = ""
    INSTITUTION_KEYWORDS = {"school", "college", "university", "academy", "institute", "hhs", "ghss", "vhs", "higher secondary"}
    for line in header_text.splitlines():
        l = line.strip()
        # Skip empty lines, lines with emails/numbers, URLs, or known institution keywords
        if l and not re.search(r"[@\d]", l) and not l.startswith("http"):
            l_lower = l.lower()
            if any(k in l_lower for k in INSTITUTION_KEYWORDS):
                continue
            if 2 <= len(l.split()) <= 4 and len(l) < 50:
                name = l
                break

    loc_m = re.search(r"\b([A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s]+)\b", header_text)
    location = loc_m.group(1).strip() if loc_m else ""

    linkedin = ""
    github   = ""
    for url in re.findall(
        r"https?://\S+|linkedin\.com/\S+|github\.com/\S+",
        header_text + "\n" + raw_text,
        re.I,
    ):
        if "linkedin" in url.lower() and not linkedin:
            linkedin = url.rstrip(".,;)")
        if "github" in url.lower() and not github:
            github = url.rstrip(".,;)")

    return {
        "name":     name,
        "phone":    phone_m.group().strip() if phone_m else "",
        "email":    (email_m.group() + "m" if email_m and email_m.group().endswith(".co") else (email_m.group() if email_m else "")),
        "location": location,
        "linkedin": linkedin,
        "github":   github,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SKILL EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def _extract_skills(sections: dict, raw_text: str) -> list[str]:
    skill_text = (sections.get("technical skills", "") + "\n" +
                  sections.get("skills", "") + "\n" +
                  sections.get("key skills", "") + "\n" +
                  sections.get("core competencies", ""))

    candidates: list[str] = []

    for line in skill_text.splitlines():
        val = re.sub(r"^[A-Za-z &()/]+[:/]\s*", "", line).strip()
        candidates += re.split(r"[,.|/;\s\u2022\u25e6]+", val)

    # Also pull from tech lines in experience/projects
    for line in raw_text.splitlines():
        if re.search(r"\|\s*tech\s*:", line, re.I):
            tech = re.sub(r".*tech\s*:\s*", "", line, flags=re.I)
            candidates += re.split(r"[,\s]+", tech)

    for section_name in ("experience", "work experience", "projects", "internships"):
        sec_text = sections.get(section_name, "")
        for line in sec_text.splitlines():
            for word in re.split(r"[,\s()]+", line):
                w = word.strip().lower().rstrip(".,;:")
                if w in LANG_SET | FWK_SET | DB_SET | TOOL_SET:
                    candidates.append(w)

    classified: set[str] = set()
    langs, fwks, dbs, tools = [], [], [], []

    for s in candidates:
        s = s.strip().lower().strip("()[]•-.,;:")
        if not s or s in LABEL_WORDS or s in SOFT_SKILLS:
            continue
        if not (s in ALWAYS_KEEP or (1 < len(s) < 30)):
            continue
        if s.replace(" ", "").isdigit():
            continue
        if re.match(r"^\d+[./]\d+$", s):
            continue
        if s in classified:
            continue
        if   s in LANG_SET:  langs.append(s);  classified.add(s)
        elif s in FWK_SET:   fwks.append(s);   classified.add(s)
        elif s in DB_SET:    dbs.append(s);    classified.add(s)
        elif s in TOOL_SET:  tools.append(s);  classified.add(s)

    other = []
    for s in candidates:
        s = s.strip().lower().strip("()[]•-.,;:")
        if not s or s in LABEL_WORDS or s in SOFT_SKILLS or s in classified:
            continue
        if not (s in ALWAYS_KEEP or (1 < len(s) < 20)):
            continue
        if s.isdigit() or re.match(r"^\d", s):
            continue
        if re.match(r"^[a-z][a-z0-9+#._-]{1,18}$", s):
            other.append(s)
            classified.add(s)

    # ── Collect soft skills from the skills section ───────────────────────────
    # Soft skills appear explicitly in the skills section (e.g. "Teamwork", "Leadership")
    # They were excluded from tech extraction but should be included in the skills list
    soft_found = []
    skill_section_text = (
        sections.get("skills", "") + "\n" +
        sections.get("technical skills", "") + "\n" +
        sections.get("key skills", "")
    ).lower()

    # Multi-word soft skills first
    MULTI_SOFT = [
        "problem solving", "time management", "project management",
        "critical thinking", "decision making", "attention to detail",
        "self motivated", "work ethic", "team player",
    ]
    for ms in MULTI_SOFT:
        if ms in skill_section_text and ms not in classified:
            soft_found.append(ms)
            classified.add(ms)

    # Single-word soft skills
    for line in skill_section_text.splitlines():
        line = line.strip()
        # Strip label prefix like "Soft Skills:" or "Others:"
        line = re.sub(r"^[a-z &/]+[:/]\s*", "", line)
        for token in re.split(r"[,•|\n]+", line):
            t = token.strip().lower().strip("()[]•-.,;:")
            if t and t in SOFT_SKILLS and t not in classified:
                soft_found.append(t)
                classified.add(t)

    return langs + fwks + dbs + tools + other + soft_found


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIENCE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def _extract_experience(sections: dict) -> list[dict]:
    exp_text = (sections.get("experience", "") or
                sections.get("work experience", "") or
                sections.get("professional experience", "") or
                sections.get("internships", ""))
    if not exp_text:
        return []

    blocks: list[list[str]] = []
    current_block: list[str] = []
    _EXP_YEAR_RE = re.compile(r"\b(20\d\d|19\d\d)\b")

    for line in exp_text.splitlines():
        stripped = line.strip()
        clean_nb = stripped.lstrip("•-* ")
        _has_year     = bool(_EXP_YEAR_RE.search(stripped))
        _is_pipe      = "|" in stripped and not stripped.startswith("•")
        is_role_header = _is_pipe or (_has_year and clean_nb)
        if is_role_header and current_block:
            blocks.append(current_block)
            current_block = [stripped]
        elif not stripped and current_block:
            blocks.append(current_block)
            current_block = []
        elif stripped:
            current_block.append(stripped)

    if current_block:
        blocks.append(current_block)

    experience = []
    for block in blocks:
        if not block:
            continue
        header = block[0].strip()
        parts  = [p.strip() for p in header.split("|")]

        role    = parts[0].lstrip("•-* ").strip() if parts else ""
        company = parts[1].strip() if len(parts) > 1 else ""

        duration = ""
        if len(parts) > 2:
            duration = parts[2]
        if not duration:
            dm = DURATION_PAT.search(header)
            if dm:
                duration = dm.group()
                if not company:
                    before_date = header[:dm.start()].strip().lstrip("•-*· ")
                    company = before_date.strip() if before_date else ""
                role = ""
            else:
                ym = YEAR_PAT.search(header)
                if ym:
                    duration = ym.group()
                    if not company:
                        before_yr = header[:ym.start()].strip().lstrip("•-*· ")
                        company = before_yr.strip() if before_yr else ""
                    role = ""

        company = DURATION_PAT.sub("", company).strip().rstrip("|, ")

        # Try to detect role from second line if not parsed from header
        _role_line_used = None
        if not role:
            for _bl in block[1:]:
                _bc = _bl.lstrip("•-* ").strip()
                if not _bc:
                    continue
                _is_role = (
                    len(_bc.split()) <= 5 and
                    not DURATION_PAT.search(_bc) and
                    not YEAR_PAT.search(_bc) and
                    not _bc.startswith((
                        "Worked", "Built", "Developed", "Implemented",
                        "Designed", "Created", "Contributed", "Assisted",
                    ))
                )
                if _is_role:
                    role = _bc
                    _role_line_used = _bc
                break

        if not role:
            role = company

        responsibilities = []
        technologies = []
        for line in block[1:]:
            line_clean = line.lstrip("•-*◦▸ ").strip()
            if not line_clean:
                continue
            if _role_line_used and line_clean == _role_line_used:
                continue
            if re.match(r"(?i)tech(nologies)?\s*:", line_clean):
                tech_str = re.sub(r"(?i)tech(nologies)?\s*:\s*", "", line_clean)
                technologies = [t.strip().lower() for t in re.split(r"[,\s]+", tech_str) if t.strip()]
            elif not DURATION_PAT.search(line_clean):
                responsibilities.append(line_clean)

        _all_text = (role + " " + company + " " + " ".join(responsibilities)).lower()
        if any(k in _all_text for k in {"intern", "internship", "trainee", "apprentice", "summer intern"}):
            exp_type = "internship"
        elif any(k in _all_text for k in {"workshop", "bootcamp", "training", "certificate program", "seminar"}):
            exp_type = "workshop"
        elif any(k in _all_text for k in {"hackathon", "competition", "contest", "challenge", "olympiad", "ideathon"}):
            exp_type = "competition"
        elif any(k in _all_text for k in {"freelance", "freelancer", "contract", "consultant", "part-time", "part time"}):
            exp_type = "freelance"
        elif any(k in _all_text for k in {"full time", "fulltime", "full-time", "employee", "associate", "engineer", "developer", "analyst"}):
            exp_type = "job"
        else:
            exp_type = ""

        if role or responsibilities:
            experience.append({
                "role":             role,
                "company":          company,
                "duration":         duration,
                "responsibilities": responsibilities,
                "technologies":     technologies,
                "experience_type":  exp_type,
            })

    return experience


# ─────────────────────────────────────────────────────────────────────────────
# PROJECT EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def _extract_projects(sections: dict) -> list[dict]:
    proj_text = (sections.get("projects", "") or
                 sections.get("academic projects", "") or
                 sections.get("personal projects", "") or
                 sections.get("mini projects", ""))
    if not proj_text:
        return []

    blocks: list[list[str]] = []
    current_block: list[str] = []

    for line in proj_text.splitlines():
        stripped = line.strip()
        is_proj_header = (
            stripped and
            not stripped.startswith(("•", "-", "*")) and
            not stripped.startswith(tuple("0123456789")) and
            ("|" in stripped or (not DURATION_PAT.search(stripped) and len(stripped) < 80))
        )
        if is_proj_header and current_block:
            blocks.append(current_block)
            current_block = [stripped]
        elif not stripped and current_block:
            blocks.append(current_block)
            current_block = []
        elif stripped:
            current_block.append(stripped)

    if current_block:
        blocks.append(current_block)

    _EDU_LEAK = re.compile(
        r"bachelor|b\.?\s*tech|master|phd|diploma|plus two|10\+2|"
        r"college|university|school|institute|hss|engineering|technology",
        re.I,
    )

    projects = []
    for block in blocks:
        if not block:
            continue
        header = block[0]
        tech_m = re.search(r"\|\s*tech\s*:\s*(.+)$", header, re.I)
        tech   = (
            [t.strip().lower() for t in re.split(r"[,\s]+", tech_m.group(1)) if t.strip()]
            if tech_m else []
        )
        title = re.sub(r"\s*\|.*$", "", header).strip()

        desc_lines = []
        for line in block[1:]:
            line = line.lstrip("•-*◦▸ ").strip()
            if line:
                desc_lines.append(line)

        if title and len(title) > 2 and not _EDU_LEAK.search(title):
            _all = (title + " " + " ".join(desc_lines)).lower()
            if any(k in _all for k in {"hackathon", "competition", "contest", "challenge", "winner", "finalist", "ideathon", "datathon", "olympiad"}):
                proj_type = "competition"
            elif any(k in _all for k in {"open source", "open-source", "github", "npm", "pypi", "contribution", "pull request"}):
                proj_type = "open-source"
            elif any(k in _all for k in {"freelance", "client", "paid", "commissioned"}):
                proj_type = "freelance"
            elif any(k in _all for k in {"minor project", "major project", "final year", "semester", "academic", "coursework", "assignment"}):
                proj_type = "academic"
            else:
                proj_type = "personal"

            projects.append({
                "title":        title,
                "technologies": tech,
                "description":  " ".join(desc_lines),
                "project_type": proj_type,
            })

    return projects


# ─────────────────────────────────────────────────────────────────────────────
# EDUCATION EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def _extract_education(sections: dict) -> list[dict]:
    edu_text = (sections.get("education", "") or
                sections.get("academic background", "") or
                sections.get("academic details", ""))
    if not edu_text:
        return []

    lines   = [l.strip() for l in edu_text.splitlines() if l.strip()]
    entries = []
    i       = 0
    processed: set[int] = set()
    _EDU_YEAR_RE = re.compile(r"^(\d{4})\s*[-–·]?\s*(\d{0,4}|present|current|now)?\s*$", re.I)

    while i < len(lines):
        if i in processed:
            i += 1
            continue
        line = lines[i]
        clean_line = line.lstrip("•-* ").strip()

        # Handle "2022 - 2023" style year range lines (common in some resume formats)
        yr_range_m = _EDU_YEAR_RE.match(clean_line)
        if yr_range_m:
            end_val = (yr_range_m.group(2) or "").lower()
            is_ongoing = end_val in {"present", "current", "now", ""}
            grad_yr = int(yr_range_m.group(2)) if (yr_range_m.group(2) and yr_range_m.group(2).isdigit()) else int(yr_range_m.group(1))
            
            processed.add(i)
            inst, degree, gpa = "", "", None
            for j in range(i + 1, min(i + 6, len(lines))):
                nxt = lines[j].lstrip("•-* ").strip()
                if not nxt:
                    break
                if _EDU_YEAR_RE.match(nxt):
                    break
                gm = GPA_PAT.search(nxt)
                if gm and not gpa:
                    gpa = gm.group(1)
                    processed.add(j)
                    continue
                if DEGREE_PAT.search(nxt) and not degree:
                    degree = nxt
                    processed.add(j)
                    continue
                if not inst and nxt:
                    inst = nxt
                    processed.add(j)
            if inst or degree:
                _grad_yr_val = f"{int(yr_range_m.group(1))} \u2013 Present" if is_ongoing else grad_yr
                entries.append({
                    "degree":          degree or "Plus Two",
                    "institution":     inst,
                    "graduation_year": _grad_yr_val,
                    "gpa":             gpa,
                })
            i += 1
            continue

        if not DEGREE_PAT.search(line):
            i += 1
            continue

        parts       = [p.strip() for p in line.split("|")]
        degree      = parts[0]
        institution = parts[1] if len(parts) > 1 else ""

        # Check previous line for institution if missing (common in some formats)
        if not institution and i > 0:
            prev_line = lines[i-1].lstrip("•-* ").strip()
            if prev_line and not DEGREE_PAT.search(prev_line) and not _EDU_YEAR_RE.match(prev_line):
                institution = prev_line

        yr  = None
        gpa = None
        for j in range(i, min(i + 4, len(lines))):
            ym = YEAR_PAT.search(lines[j])
            gm = GPA_PAT.search(lines[j])
            if ym and not yr:  yr  = int(ym.group(1))
            if gm and not gpa: gpa = gm.group(1).strip()

        entries.append({
            "degree":          degree,
            "institution":     institution,
            "graduation_year": yr,
            "gpa":             gpa,
        })
        processed.add(i)
        i += 1

    return entries


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def parse_resume(raw_text: str) -> dict:
    """
    Parse resume text into structured dict.
    Zero LLM calls. Includes experience → project reclassification.

    Returns dict with keys:
      name, phone, email, location, linkedin, github,
      skills, experience, projects, education, summary
    """
    sections  = _split_sections(raw_text)
    contact   = _extract_contact(sections, raw_text)
    skills    = _extract_skills(sections, raw_text)
    experience = _extract_experience(sections)
    projects   = _extract_projects(sections)
    education  = _extract_education(sections)

    # ── Reclassify misplaced experience entries into projects ─────────────────
    experience, projects = _reclassify_experience_entries(experience, projects)

    logger.info(
        "smart_parser: name=%r skills=%d exp=%d projects=%d edu=%d",
        contact["name"], len(skills), len(experience), len(projects), len(education),
    )

    return {
        "name":       contact["name"],
        "phone":      contact["phone"],
        "email":      contact["email"],
        "location":   contact["location"],
        "linkedin":   contact.get("linkedin", ""),
        "github":     contact.get("github", ""),
        "skills":     skills,
        "experience": experience,
        "projects":   projects,
        "education":  education,
        "summary":    (sections.get("objective", "") or
                       sections.get("summary", "") or
                       sections.get("career objective", "") or
                       sections.get("professional summary", "")),
    }