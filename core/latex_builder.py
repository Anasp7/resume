import re
import re as _re
from typing import Any, List, Dict, Optional

def _tex(s: str) -> str:
    """Escape special LaTeX characters in a string."""
    if not s: return ""
    replacements = [
        ("\\", r"\textbackslash{}"), ("&",  r"\&"), ("%",  r"\%"), ("$",  r"\$"),
        ("#",  r"\#"), ("_",  r"\_"), ("{",  r"\{"), ("}",  r"\}"),
        ("~",  r"\textasciitilde{}"), ("^",  r"\textasciicircum{}"),
    ]
    for char, repl in replacements: s = s.replace(char, repl)
    return s

def _pro_edu(text: str) -> str:
    """Standardize degree terminology for a professional look."""
    t = str(text)
    t = re.sub(r"(?i)\bB\.?Tech\b|btech", "Bachelor of Technology", t)
    t = re.sub(r"(?i)\bB\.?E\b|be(?=\b)", "Bachelor of Engineering", t)
    t = re.sub(r"(?i)\bM\.?Tech\b|mtech", "Master of Technology", t)
    t = re.sub(r"(?i)\bCSE\b", "Computer Science and Engineering", t)
    return t

def build_latex_resume(data: Dict[str, Any], resume_text: str = "") -> str:
    """
    MASTER JEROME 2.0: Clean-Slate Conditional LaTeX Generator.
    Matches the 'Anas' structure with 'Jerome' typography.
    """
    def _get(obj, key, default=None):
        if isinstance(obj, dict): return obj.get(key, default)
        return getattr(obj, key, default)

    # 1. METADATA & HEADER
    name     = _tex(_get(data, "name", "Candidate"))
    phone    = _tex(_get(data, "phone", ""))
    email    = _tex(_get(data, "email", ""))
    linkedin = _tex(_get(data, "linkedin", ""))
    github   = _tex(_get(data, "github", ""))

    # 2. CONTENT SCRAPER (1:1 SYNC WITH PREVIEW)
    def _extract_section(text: str, headers: List[str]) -> List[str]:
        if not text: return []
        h_pat = "|".join(headers)
        m_start = _re.search(rf"(?i)(?:{h_pat})\b\s*\n+", text)
        if not m_start: return []
        remaining = text[m_start.end():]
        m_next = _re.search(rf"\n[A-Z\s]{{5,}}(?:\n+|\:)", remaining)
        section_raw = remaining[:m_next.start()].strip() if m_next else remaining.strip()
        bullets = []
        for line in section_raw.split("\n"):
            line_str = line.strip()
            if line_str: bullets.append(line_str)
        return bullets

    # Scrape Sections
    pro_summary = _extract_section(resume_text, ["OBJECTIVE", "SUMMARY"])
    pro_summary = pro_summary[0] if pro_summary else _get(data, "summary", "")
    
    raw_exp     = _extract_section(resume_text, ["EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE", "EMPLOYMENT HISTORY"])
    raw_proj    = _extract_section(resume_text, ["PROJECTS", "FEATURING PROJECTS", "KEY PROJECTS", "ACADEMIC PROJECTS", "PERSONAL PROJECTS", "TECHNICAL PROJECTS"])
    raw_skills  = _extract_section(resume_text, ["SKILLS", "TECHNICAL SKILLS", "SOFT SKILLS", "SKILLSET", "CORE COMPETENCIES"])
    raw_edu     = _extract_section(resume_text, ["EDUCATION", "ACADEMIC BACKGROUND", "QUALIFICATIONS"])
    raw_cert    = _extract_section(resume_text, ["CERTIFICATIONS", "LICENSES & CERTIFICATIONS"])

    # 3. PREAMBLE
    latex = r"""\documentclass[10pt,a4paper]{article}
\usepackage[a4paper, top=0.4in, bottom=0.4in, left=0.4in, right=0.4in]{geometry}
\usepackage{enumitem}
\usepackage{hyperref}
\usepackage{fontawesome5}
\usepackage{charter}
\usepackage{titlesec}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}

\setlist[itemize]{noitemsep, topsep=0pt, leftmargin=15pt, parsep=2pt}
\hypersetup{colorlinks=true, urlcolor=black, linkcolor=black}
\pagestyle{empty}

\titleformat{\section}{\large\bfseries\uppercase}{}{0em}{}[\titlerule]
\titlespacing{\section}{0pt}{12pt}{6pt}

\begin{document}

\begin{center}
    {\huge\bfseries """ + name + r"""} \\ \vspace{2pt}
    {\small
    \faEnvelope\ \href{mailto:""" + email + r"""}{""" + email + r"""} \quad 
    """ + (r"\faPhone\ " + phone if phone else "") + r""" \quad 
    """ + (r"\faLinkedin\ \href{https://" + linkedin.lstrip("https://") + r"}{" + linkedin.split('/')[-1] + r"}" if linkedin else "") + r""" \quad 
    """ + (r"\faGithub\ \href{https://" + github.lstrip("https://") + r"}{" + github.split('/')[-1] + r"}" if github else "") + r"""
    }
\end{center}
\vspace{-12pt}
"""

    # 4. SUMMARY/OBJECTIVE
    summary_latex = ""
    if pro_summary:
        summary_latex = r"\section{Summary}" + "\n" + r"{\small " + _tex(str(pro_summary)) + r"}" + "\n"

    # 5. EXPERIENCE — only render if the backend found real experience entries
    exp_latex = ""
    json_exp = data.get("experience", [])
    has_real_exp = isinstance(json_exp, list) and any(
        (e.get("role") or e.get("company")) for e in json_exp
    )
    if has_real_exp and raw_exp:
        exp_latex += r"\section{Experience}" + "\n"
        has_open_itemize = False
        has_items = False
        
        for line in raw_exp:
            is_bullet = line.startswith(("-", "*", "•", "·", "▸", "◦"))
            # Date-only lines on their own (like July 2024 - Oct 2024) shouldn't be treated as new role titles
            is_date_only = bool(_re.match(r"^(january|february|march|april|may|june|july|august|september|october|november|december|expected|\d{2,4})\b.*?(?:-|–|to).*?\d{4}.*$", line.strip().lower()))
            
            if not is_bullet and not is_date_only and ("|" in line or " — " in line or " at " in line or (len(line) < 80 and len(line) > 5)):
                if has_open_itemize:
                    if not has_items:
                        exp_latex += r"  \item \vspace{-4pt}" + "\n"
                    exp_latex += r"\end{itemize}" + "\n"
                
                parts = _re.split(r"\||—| at ", line)
                rl = parts[0].strip()
                co = parts[1].strip() if len(parts) > 1 else ""
                dt = (parts[2].strip() if len(parts) > 2 else "").replace("Duration:", "").strip()
                
                exp_latex += r"\noindent\textbf{" + _tex(rl) + r"} " + (r"| \textit{" + _tex(co) + r"}" if co else "") + r" \hfill \textit{" + _tex(dt) + r"} \\" + "\n"
                exp_latex += r"\begin{itemize}" + "\n"
                has_open_itemize = True
                has_items = False
            else:
                line_clean = _re.sub(r"^[\*•\-·▸◦\s]+", "", line)
                if line_clean:
                    if not has_open_itemize:
                        exp_latex += r"\begin{itemize}" + "\n"
                        has_open_itemize = True
                    exp_latex += r"  \item " + _tex(line_clean) + "\n"
                    has_items = True
                    
        if has_open_itemize:
            if not has_items:
                exp_latex += r"  \item \vspace{-4pt}" + "\n"
            exp_latex += r"\end{itemize}\vspace{2pt}" + "\n"
        
    elif has_real_exp and json_exp:
        # FALLBACK: If LLM failed to format experience in text, generate directly from JSON data
        exp_latex += r"\section{Experience}" + "\n"
        for idx, e in enumerate(json_exp):
            rl = e.get("role", "")
            co = e.get("company", "")
            dt = e.get("duration", "")
            
            # Skip empty entries completely
            if not rl and not co:
                continue
                
            rl_str = r"\noindent\textbf{" + _tex(rl) + r"}" if rl else ""
            co_str = r" | \textit{" + _tex(co) + r"}" if co else ""
            
            if not rl:
                head = r"\noindent\textbf{" + _tex(co) + r"}"
            else:
                head = rl_str + co_str
                
            dt_str = r" \hfill \textit{" + _tex(dt) + r"}" if dt else ""
            exp_latex += head + dt_str + r" \\" + "\n"
            
            resp = e.get("responsibilities", [])
            if resp:
                exp_latex += r"\begin{itemize}" + "\n"
                for r in resp:
                    exp_latex += r"  \item " + _tex(str(r)) + "\n"
                exp_latex += r"\end{itemize}" + ("" if idx == len(json_exp)-1 else r"\vspace{2pt}") + "\n"
            else:
                exp_latex += (r"\vspace{2pt}" if idx < len(json_exp)-1 else "") + "\n"

    # 6. PROJECTS
    proj_latex = ""
    json_projects = data.get("projects", [])
    if raw_proj:
        proj_latex += r"\section{Projects}" + "\n"
        # Group bullets by project
        p_groups = []
        cur_p = None
        for line in raw_proj:
            is_bullet = line.startswith(("-", "*", "•", "·", "▸", "◦"))
            if not is_bullet and line.strip():
                title = line
                tech = ""
                # Handle "Project Title | Tech: React, Python" format
                if " | " in line or " - " in line:
                    parts = _re.split(r"\||—|-", line)
                    title = parts[0].strip()
                    for part in parts[1:]:
                        if "tech" in part.lower():
                            tech = part.split(":", 1)[-1].strip()
                        elif not tech:
                            tech = part.strip()
                
                cur_p = {"title": title, "tech": tech, "bullets": []}
                p_groups.append(cur_p)
            elif cur_p:
                if "technologies:" in line.lower() or "tech:" in line.lower():
                    cur_p["tech"] = line.split(":", 1)[-1].strip()
                else:
                    cur_p["bullets"].append(_re.sub(r"^[\*•\-·▸◦\s]+", "", line))
        
        for p in p_groups:
            t_str = p["tech"]
            
            # Auto-inject technologies from verified JSON if the LLM forgot them in the header
            if not t_str:
                p_title_lower = p["title"].lower().strip()
                for jp in json_projects:
                    j_title = jp.get("title", "").lower().strip()
                    if len(j_title) > 3 and (j_title in p_title_lower or p_title_lower in j_title):
                        tech_list = jp.get("technologies", [])
                        if tech_list:
                            t_str = ", ".join(tech_list[:4])
                        break
                        
            title_clean = _tex(p["title"])
            if t_str:
                proj_latex += r"\noindent\textbf{" + title_clean + r"} \hfill \textit{" + _tex(t_str) + r"} \\" + "\n"
            else:
                proj_latex += r"\noindent\textbf{" + title_clean + r"} \\" + "\n"
                
            if p["bullets"]:
                proj_latex += r"\begin{itemize}" + "\n"
                for b in p["bullets"]: proj_latex += r"  \item " + _tex(b) + "\n"
                proj_latex += r"\end{itemize}\vspace{1mm}" + "\n"

    elif json_projects:
        # FALLBACK: If LLM failed to format projects in text, generate directly from JSON data
        proj_latex += r"\section{Projects}" + "\n"
        for p in json_projects:
            title = _tex(p.get("title", "Project") or "Project")
            tech_raw = p.get("technologies", [])
            tech = _tex(", ".join(tech_raw)) if tech_raw else ""
            
            if tech:
                proj_latex += r"\noindent\textbf{" + title + r"} \hfill \textit{" + tech + r"} \\" + "\n"
            else:
                proj_latex += r"\noindent\textbf{" + title + r"} \\" + "\n"
                
            desc = p.get("description", "").strip()
            metrics = p.get("metrics", [])
            bullets = []
            if desc:
                b_parts = _re.split(r"(?<=\s)[\*\-•](?=\s)|^\s*[\*\-•]\s*", desc)
                for x in b_parts:
                    cl = _re.sub(r"^[\*•\-·▸◦\s]+", "", x).strip()
                    if len(cl) > 5: bullets.append(cl)
            bullets.extend([m.strip() for m in metrics if m.strip()])
            
            if bullets:
                proj_latex += r"\begin{itemize}" + "\n"
                for b in bullets:
                    proj_latex += r"  \item " + _tex(b) + "\n"
                proj_latex += r"\end{itemize}\vspace{1mm}" + "\n"

    # 7. EDUCATION (ABOVE SKILLS)
    edu_latex = ""
    json_edu = data.get("education")
    if isinstance(json_edu, list) and json_edu:
        edu_latex = r"\section{Education}" + "\n"

        # ── Year-range helper ──────────────────────────────────────────────────
        # Returns True if yr string already contains a range (e.g. "2023-2027",
        # "2023 – 2027", "2023 to 2027", "2023 - Present").
        def _is_range(yr_str: str) -> bool:
            s = yr_str.strip()
            return bool(
                _re.search(r"20\d\d\s*[-–—to]\s*(20\d\d|[Pp]resent)", s)
            )

        # Smart Heuristic: Find 12th/Higher-Secondary completion year ONLY.
        # Used to infer whether B.Tech is still ongoing (single end-year case).
        school_year = None
        for ed in json_edu:
            _dl = (ed.get("degree") or "").lower()
            _lv = (ed.get("level") or "").lower()
            _is_school_entry = (
                any(k in _dl for k in {"plus two", "hsc", "sslc", "higher secondary",
                                       "class 12", "class xii", "class12",
                                       "secondary", "cbse", "icse", "state board"})
                or _lv in {"class12", "class10"}
            )
            if _is_school_entry:
                m = _re.search(r"20\d\d", str(ed.get("graduation_year") or ed.get("year") or ""))
                if m:
                    school_year = int(m.group())

        for ed in json_edu:
            inst = ed.get("institution") or ""
            deg  = ed.get("degree") or ""
            fld  = ed.get("field") or ""
            if fld and fld.lower() not in deg.lower():
                deg = f"{deg} in {fld}"

            gpa  = str(ed.get("gpa") or "")
            yr   = str(ed.get("graduation_year") or ed.get("year") or "")

            # ── RULE 1: If yr is already a range → keep it exactly as-is ─────
            # e.g. "2023-2027", "2023 – Present" → do NOT modify
            if not _is_range(yr):
                # ── RULE 2: Single-year end date + B.Tech + school_year known
                # → infer "StartYear – Present" only if start hasn't been stored
                _is_btech = any(k in deg.lower() for k in
                                {"b.tech", "btech", "bachelor", "b.e",
                                 "be", "engineering"})
                _deg_l = deg.lower()
                _is_school_deg = any(k in _deg_l for k in
                                     {"plus two", "hsc", "sslc", "higher secondary",
                                      "class 12", "class xii", "secondary",
                                      "cbse", "icse", "class12"})

                if _is_btech and not _is_school_deg and school_year and "present" not in yr.lower():
                    m_yr = _re.search(r"20\d\d", yr)
                    if m_yr:
                        btech_end = int(m_yr.group())
                        btech_start = school_year + 1  # typical: pass 12th → join B.Tech
                        if (btech_end - school_year) <= 5:  # sanity: 4-yr deg ±1
                            yr = f"{btech_start}\u2013{btech_end}"

            edu_latex += r"\noindent\textbf{" + _tex(inst) + r"} \hfill \textit{" + _tex(yr) + r"} \\" + "\n"

            deg_str = _tex(_pro_edu(deg))
            if gpa and gpa.lower() not in {"none", "null", ""}:
                # Determine correct label: school/Plus Two = Percentage, college = CGPA
                _deg_l = deg.lower()
                _is_school = any(k in _deg_l for k in
                                 {"plus two", "hsc", "sslc", "10th", "12th",
                                  "secondary", "cbse", "icse", "class12", "class 12"})
                if "cgpa" in gpa.lower() or "gpa" in gpa.lower():
                    _label = ""   # label already in value
                elif _is_school:
                    _label = "Percentage: "
                else:
                    _label = "CGPA: "
                deg_str += r", " + _tex(_label + gpa)
            if deg_str:
                edu_latex += r"\textit{" + deg_str + r"} \\" + "\n"
            edu_latex += r"\vspace{1mm}" + "\n"
    elif raw_edu:
        # Fallback to legacy scraping if JSON structure is missing
        edu_latex = r"\section{Education}" + "\n"
        current_ed = ""
        for line in raw_edu:
            l_lower = line.lower()
            if any(x in l_lower for x in ["college", "university", "institute", "school"]):
                if current_ed: current_ed += r"\vspace{1mm}" + "\n"
                parts = _re.split(r"\||—| at |,", line)
                inst = parts[0].strip()
                dt   = parts[-1].strip() if len(parts) > 1 else ""
                deg  = ""
                for p in parts:
                    pl = p.lower()
                    if any(x in pl for x in ["college", "university", "institute", "school"]): inst = p.strip()
                    elif any(x in pl for x in ["expected", "202", "201"]): dt = p.strip()
                    elif any(x in pl for x in ["btech", "b.tech", "degree", "bachelor", "master"]): deg = p.strip()
                
                edu_latex += r"\noindent\textbf{" + _tex(inst) + r"} \hfill \textit{" + _tex(dt) + r"} \\" + "\n"
                if deg: edu_latex += r"\textit{" + _tex(_pro_edu(deg)) + r"} \\" + "\n"
            elif any(x in l_lower for x in ["btech", "b.tech", "degree", "bachelor", "master"]):
                edu_latex += r"\textit{" + _tex(_pro_edu(line)) + r"} \\" + "\n"
            elif "cgpa" in l_lower or "percentage" in l_lower:
                edu_latex += _tex(line) + r" \\" + "\n"
            else:
                edu_latex += _tex(line) + r" \\" + "\n"

    # 8. SKILLS
    skills_latex = ""
    json_skills = data.get("skills")
    
    if isinstance(json_skills, dict) and any(json_skills.values()):
        skills_latex = r"\section{Skills}" + "\n"
        skills_latex += r"\begin{tabular}{@{}p{4.5cm}p{12.5cm}@{}}" + "\n"
        for cat, items in json_skills.items():
            if not items: continue
            val = ", ".join(items) if isinstance(items, list) else str(items)
            skills_latex += r"\textbf{" + _tex(cat) + r"} & " + _tex(val) + r" \\" + "\n"
        skills_latex += r"\end{tabular}" + "\n"
    elif raw_skills:
        skills_latex = r"\section{Skills}" + "\n"
        skills_latex += r"\begin{tabular}{@{}p{4.5cm}p{12.5cm}@{}}" + "\n"
        for s in raw_skills:
            if ":" in s:
                cat, val = s.split(":", 1)
                skills_latex += r"\textbf{" + _tex(cat.strip()) + r"} & " + _tex(val.strip()) + r" \\" + "\n"
            else:
                skills_latex += r"\multicolumn{2}{@{}l}{" + _tex(s) + r"} \\" + "\n"
        skills_latex += r"\end{tabular}" + "\n"

    # 9. CERTIFICATIONS
    cert_latex = ""
    if raw_cert:
        cert_latex = r"\section{Certifications}" + "\n"
        cert_latex += r"\begin{itemize}" + "\n"
        for c in raw_cert:
            cert_latex += r"  \item " + _tex(c) + "\n"
        cert_latex += r"\end{itemize}\vspace{2pt}" + "\n"

    # ASSEMBLY
    doc = latex + summary_latex + exp_latex + proj_latex + edu_latex + skills_latex + cert_latex + r"\end{document}"
    return doc
