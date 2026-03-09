"""
Smart Resume — Resume Exporter v3
====================================
Pure Python — NO Node.js required.
  - DOCX via python-docx  (already in requirements.txt)
  - PDF  via reportlab    (pip install reportlab)

ALL content parsed from LLM-generated text.
"""
from __future__ import annotations
import logging, os, re
from io import BytesIO
from typing import Optional

logger = logging.getLogger("smart_resume.exporter")

ALL_SECTIONS = ["OBJECTIVE","TECHNICAL SKILLS","EXPERIENCE","PROJECTS","EDUCATION","CERTIFICATIONS"]
ACCENT = (26, 60, 110)   # #1A3C6E as RGB


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def export_resume(resume_text: str, parsed_resume, fmt: str = "docx") -> bytes:
    d = _build_resume_data(resume_text, parsed_resume)
    if fmt == "docx": return _export_docx(d)
    if fmt == "pdf":  return _export_pdf(d)
    raise ValueError(f"Unknown format: {fmt}")


# ─────────────────────────────────────────────────────────────────────────────
# DATA BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _extract_section(text: str, name: str) -> str:
    others = [s for s in ALL_SECTIONS if s != name]
    stop   = "|".join(re.escape(s) for s in others)
    pat    = r"(?:^|\n)" + re.escape(name) + r"\s*\n(.*?)(?=\n(?:" + stop + r")\s*\n|\Z)"
    m      = re.search(pat, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""

def _build_resume_data(resume_text: str, pr) -> dict:
    name     = pr.name or "Candidate"
    linkedin = getattr(pr, "linkedin", None) or ""
    github   = getattr(pr, "github",   None) or ""
    cp       = [p for p in [pr.phone or "", pr.email or "", pr.location or ""] if p.strip()]
    if linkedin: cp.append(linkedin)
    if github:   cp.append(github)
    contact  = "  |  ".join(cp)

    objective = _extract_section(resume_text, "OBJECTIVE")
    if not objective:
        m = re.search(r"OBJECTIVE\s*\n+(.*?)(?=\n[A-Z]{3}|\Z)", resume_text, re.DOTALL|re.IGNORECASE)
        objective = m.group(1).strip() if m else (pr.summary or "")

    sb = _extract_section(resume_text, "TECHNICAL SKILLS")
    skills = {}
    if sb:
        for line in sb.splitlines():
            line = line.strip()
            if ":" in line:
                cat, _, vals = line.partition(":")
                items = [v.strip() for v in vals.split(",") if v.strip()]
                if items:  # only add if there are actual values
                    skills[cat.strip()] = items
    if not skills:
        skills = _auto_categorize(pr.skills or [])
    # Remove any category with empty list (LLM sometimes writes "Frameworks : ")
    skills = {k: v for k, v in skills.items() if v}

    eb = _extract_section(resume_text, "EXPERIENCE")
    experience = _parse_exp(eb) if eb else []
    if not experience:
        experience = [{"role":e.role,"company":e.company,"duration":e.duration or "","responsibilities":e.responsibilities} for e in pr.experience]

    pb = _extract_section(resume_text, "PROJECTS")
    projects = _parse_proj(pb) if pb else []
    if not projects:
        projects = [{"title":p.title,"description":p.description,"technologies":p.technologies,"metrics":getattr(p,"metrics",[])} for p in pr.projects]

    from datetime import datetime as _dte
    _cur_yr = _dte.now().year
    def _edu_year(e):
        if not e.graduation_year: return ""
        yr = int(str(e.graduation_year)[:4])
        if yr > _cur_yr: return f"Expected {yr}"
        if yr >= 2023 and e.gpa:  return str(yr)          # has CGPA → graduated
        if yr >= 2023:            return f"{yr} - Present"  # ongoing
        return str(yr)
    education = [{
        "degree":          e.degree,
        "institution":     e.institution,
        "graduation_year": _edu_year(e),
        "gpa":             f"CGPA: {e.gpa}" if e.gpa else "",
    } for e in pr.education]
    certifications = [{"name":c.name,"issuer":c.issuer or "","year":c.year or ""} for c in pr.certifications]

    return {"name":name,"contact":contact,"objective":objective,"skills":skills,"experience":experience,"projects":projects,"education":education,"certifications":certifications}

def _parse_exp(block: str) -> list:
    exps, cur = [], None
    for raw in block.splitlines():
        line = raw.strip()
        if not line: continue
        is_bul = line.startswith(("-","•","*"))
        if not is_bul and ("|" in line or re.match(r"^[A-Z][A-Za-z /]+(?:\||-|\d)", line)):
            if cur: exps.append(cur)
            parts = [p.strip() for p in line.split("|")]
            cur = {"role":parts[0],"company":parts[1] if len(parts)>1 else "","duration":parts[2] if len(parts)>2 else "","responsibilities":[]}
        elif is_bul:
            if cur is None: cur={"role":"","company":"","duration":"","responsibilities":[]}
            t = line.lstrip("-•* ").strip()
            if t: cur["responsibilities"].append(t)
    if cur: exps.append(cur)
    return [e for e in exps if e.get("role") or e.get("responsibilities")]

def _parse_proj(block: str) -> list:
    projs, cur = [], None
    for raw in block.splitlines():
        line = raw.strip()
        if not line: continue
        is_bul = line.startswith(("-","•","*"))
        if not is_bul:
            if cur: projs.append(cur)
            if "|" in line:
                parts  = [p.strip() for p in line.split("|",1)]
                title  = parts[0]
                tech_s = re.sub(r"(?i)^tech\s*:\s*","",parts[1]) if len(parts)>1 else ""
                techs  = [t.strip() for t in tech_s.split(",") if t.strip()]
            else:
                title, techs = line, []
            cur = {"title":title,"description":"","technologies":techs,"metrics":[]}
        else:
            if cur is None: cur={"title":"","description":"","technologies":[],"metrics":[]}
            t = line.lstrip("-•* ").strip()
            if not cur["description"]: cur["description"] = t
            elif t: cur["metrics"].append(t)
    if cur: projs.append(cur)
    return [p for p in projs if p.get("title") or p.get("description")]

def _auto_categorize(skills_list: list) -> dict:
    LANGS = {"python","java","javascript","typescript","c","c++","c#","go","rust","kotlin","swift","r","scala","ruby","php","bash","sql","dart","matlab"}
    FWKS  = {"react","vue","angular","nextjs","django","flask","fastapi","spring","pytorch","tensorflow","keras","ros","ros2","express","flutter","langchain"}
    DBS   = {"postgresql","mysql","sqlite","mongodb","redis","firebase","cassandra","dynamodb","supabase","elasticsearch"}
    TOOLS = {"docker","kubernetes","git","github","linux","aws","gcp","azure","jenkins","terraform","grafana","postman"}
    cats  = {"Languages":[],"Frameworks":[],"Databases":[],"Tools":[],"Other":[]}
    for s in skills_list:
        sl = s.lower()
        if sl in LANGS:       cats["Languages"].append(s)
        elif sl in FWKS:      cats["Frameworks"].append(s)
        elif sl in DBS:       cats["Databases"].append(s)
        elif sl in TOOLS:     cats["Tools"].append(s)
        else:                 cats["Other"].append(s)
    return {k:v for k,v in cats.items() if v}


# ─────────────────────────────────────────────────────────────────────────────
# DOCX EXPORT — pure python-docx, NO Node.js
# ─────────────────────────────────────────────────────────────────────────────

def _export_docx(d: dict) -> bytes:
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor, Inches, Cm
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
    except ImportError:
        raise RuntimeError("python-docx not installed. Run: pip install python-docx")

    doc = Document()

    # Narrow margins
    for sec in doc.sections:
        sec.top_margin    = Cm(1.5)
        sec.bottom_margin = Cm(1.5)
        sec.left_margin   = Cm(2.0)
        sec.right_margin  = Cm(2.0)

    ACCENT_COLOR = RGBColor(*ACCENT)
    DARK         = RGBColor(17, 17, 17)
    GREY         = RGBColor(80, 80, 80)

    def _set_run(run, size, bold=False, color=None, italic=False):
        run.bold    = bold
        run.italic  = italic
        run.font.size = Pt(size)
        run.font.color.rgb = color or DARK

    def _add_hr(para, color=ACCENT_COLOR):
        """Add a bottom border to a paragraph (acts as a horizontal rule)."""
        pPr = para._p.get_or_add_pPr()
        pBdr = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"),   "single")
        bottom.set(qn("w:sz"),    "6")
        bottom.set(qn("w:space"), "1")
        bottom.set(qn("w:color"), f"{color.rgb:06X}" if hasattr(color, 'rgb') else "1A3C6E")
        pBdr.append(bottom)
        pPr.append(pBdr)

    def _section_heading(text):
        p   = doc.add_paragraph()
        run = p.add_run(text.upper())
        _set_run(run, 11, bold=True, color=ACCENT_COLOR)
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after  = Pt(2)
        _add_hr(p)
        return p

    def _bullet(text):
        p   = doc.add_paragraph(style="List Bullet")
        run = p.add_run(text)
        _set_run(run, 10)
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after  = Pt(1)
        return p

    # ── Name ─────────────────────────────────────────────────────────────────
    name_p = doc.add_paragraph()
    name_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    nr = name_p.add_run(d.get("name",""))
    _set_run(nr, 20, bold=True, color=ACCENT_COLOR)
    name_p.paragraph_format.space_after = Pt(2)

    # ── Contact ───────────────────────────────────────────────────────────────
    if d.get("contact"):
        cp = doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cr = cp.add_run(d["contact"])
        _set_run(cr, 9, color=GREY)
        cp.paragraph_format.space_after = Pt(4)

    # ── Objective ─────────────────────────────────────────────────────────────
    obj = d.get("objective","")
    if obj and obj.strip():
        _section_heading("Objective")
        op = doc.add_paragraph()
        or_ = op.add_run(obj)
        _set_run(or_, 10)
        op.paragraph_format.space_after = Pt(4)

    # ── Technical Skills ──────────────────────────────────────────────────────
    skills = d.get("skills", {})
    if skills:
        _section_heading("Technical Skills")
        for cat, items in skills.items():
            if not items: continue
            p  = doc.add_paragraph()
            br = p.add_run(f"{cat}: ")
            _set_run(br, 10, bold=True)
            vr = p.add_run(", ".join(items) if isinstance(items, list) else str(items))
            _set_run(vr, 10)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after  = Pt(2)

    # ── Experience ────────────────────────────────────────────────────────────
    experience = d.get("experience", [])
    if experience:
        _section_heading("Experience")
        for exp in experience:
            p  = doc.add_paragraph()
            rr = p.add_run(exp.get("role",""))
            _set_run(rr, 10, bold=True)
            if exp.get("company"):
                cr = p.add_run(f"  |  {exp['company']}")
                _set_run(cr, 10, color=GREY)
            if exp.get("duration"):
                dr = p.add_run(f"  |  {exp['duration']}")
                _set_run(dr, 10, italic=True, color=GREY)
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after  = Pt(1)
            for resp in exp.get("responsibilities", []):
                if resp and resp.strip():
                    _bullet(resp.strip())

    # ── Projects ──────────────────────────────────────────────────────────────
    projects = d.get("projects", [])
    if projects:
        _section_heading("Projects")
        for proj in projects:
            p  = doc.add_paragraph()
            tr = p.add_run(proj.get("title",""))
            _set_run(tr, 10, bold=True)
            techs = proj.get("technologies", [])
            if techs:
                techstr = p.add_run("  |  Tech: " + ", ".join(techs))
                _set_run(techstr, 10, italic=True, color=GREY)
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after  = Pt(1)
            if proj.get("description","").strip():
                _bullet(proj["description"].strip())
            for m in proj.get("metrics", []):
                if m and m.strip(): _bullet(m.strip())

    # ── Education ─────────────────────────────────────────────────────────────
    education = d.get("education", [])
    if education:
        _section_heading("Education")
        for edu in education:
            main  = "  |  ".join(filter(None,[edu.get("degree",""),edu.get("institution","")]))
            sub_p = []
            if edu.get("graduation_year"): sub_p.append(str(edu["graduation_year"]))
            if edu.get("gpa"):             sub_p.append(f"GPA: {edu['gpa']}")
            p  = doc.add_paragraph()
            mr = p.add_run(main)
            _set_run(mr, 10, bold=True)
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after  = Pt(1)
            if sub_p:
                sp = doc.add_paragraph()
                sr = sp.add_run("  |  ".join(sub_p))
                _set_run(sr, 10, color=GREY)
                sp.paragraph_format.space_after = Pt(2)

    # ── Certifications ────────────────────────────────────────────────────────
    certs = d.get("certifications", [])
    if certs:
        _section_heading("Certifications")
        for cert in certs:
            parts = [cert.get("name",""), cert.get("issuer","")]
            if cert.get("year"): parts.append(str(cert["year"]))
            _bullet("  |  ".join(p for p in parts if p))

    buf = BytesIO()
    doc.save(buf)
    docx_bytes = buf.getvalue()
    logger.info("DOCX generated (python-docx): %d bytes", len(docx_bytes))
    return docx_bytes


# ─────────────────────────────────────────────────────────────────────────────
# PDF EXPORT — reportlab
# ─────────────────────────────────────────────────────────────────────────────

def _export_pdf(d: dict) -> bytes:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, KeepTogether
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
    except ImportError:
        raise RuntimeError(
            "reportlab not installed.\n"
            "Run:  pip install reportlab\n"
            "Then restart python main.py"
        )

    AC    = colors.HexColor("#1A3C6E")
    BLACK = colors.HexColor("#111111")
    GREY  = colors.HexColor("#555555")

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.75*inch,  bottomMargin=0.75*inch)

    N  = ParagraphStyle("N",  fontName="Helvetica-Bold", fontSize=18, textColor=AC,    alignment=TA_CENTER, spaceAfter=4)
    C  = ParagraphStyle("C",  fontName="Helvetica",      fontSize=9,  textColor=GREY,  alignment=TA_CENTER, spaceAfter=8)
    SH = ParagraphStyle("SH", fontName="Helvetica-Bold", fontSize=11, textColor=AC,    spaceBefore=12, spaceAfter=3)
    B  = ParagraphStyle("B",  fontName="Helvetica",      fontSize=10, textColor=BLACK, spaceBefore=2, spaceAfter=2, leading=14)
    BL = ParagraphStyle("BL", fontName="Helvetica",      fontSize=10, textColor=BLACK, leftIndent=14, firstLineIndent=-8, spaceBefore=1, spaceAfter=1, leading=13)

    def div():     return HRFlowable(width="100%", thickness=0.8, color=AC, spaceAfter=4)
    def sh(t):     return [Paragraph(t.upper(), SH), div()]
    def safe(t):   return str(t).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    def bul(t):    return Paragraph(f"• {safe(t)}", BL)

    story = []
    story.append(Paragraph(safe(d.get("name","")), N))
    if d.get("contact"):
        story.append(Paragraph(safe(d["contact"]), C))
    story.append(div())

    obj = d.get("objective","")
    if obj and obj.strip():
        story += sh("Objective")
        story.append(Paragraph(safe(obj), B))
        story.append(Spacer(1,6))

    skills = d.get("skills",{})
    non_empty = {k:v for k,v in skills.items() if v and (v if isinstance(v,str) else len(v)>0)}
    if non_empty:
        story += sh("Technical Skills")
        for cat, items in non_empty.items():
            iv = ", ".join(items) if isinstance(items,list) else str(items)
            if iv.strip():
                story.append(Paragraph(f"<b>{safe(cat)}:</b> {safe(iv)}", B))
        story.append(Spacer(1,4))

    exp_list = d.get("experience", [])
    if exp_list:
        story += sh("Experience")
    for exp in exp_list:
        role = safe(exp.get("role",""))
        co   = safe(exp.get("company",""))
        dur  = safe(exp.get("duration",""))
        hdr  = f"<b>{role}</b>"
        if co:  hdr += f"  |  {co}"
        if dur: hdr += f'  |  <i><font color="#888888">{dur}</font></i>'
        block = [Paragraph(hdr, B)]
        for r in exp.get("responsibilities",[]):
            if r and r.strip(): block.append(bul(r.strip()))
        block.append(Spacer(1,4))
        story.append(KeepTogether(block))

    proj_list = d.get("projects", [])
    if proj_list:
        story += sh("Projects")
    for proj in proj_list:
        title = safe(proj.get("title",""))
        techs = proj.get("technologies",[])
        ts    = ("  |  Tech: " + safe(", ".join(techs))) if techs else ""
        hdr   = f'<b>{title}</b><i><font color="#555555">{ts}</font></i>'
        block = [Paragraph(hdr, B)]
        if proj.get("description","").strip(): block.append(bul(proj["description"].strip()))
        for m in proj.get("metrics",[]): 
            if m and m.strip(): block.append(bul(m.strip()))
        block.append(Spacer(1,4))
        story.append(KeepTogether(block))

    education = d.get("education",[])
    if education:
        story += sh("Education")
        for edu in education:
            main = "  |  ".join(filter(None,[edu.get("degree",""),edu.get("institution","")]))
            subs = []
            if edu.get("graduation_year"): subs.append(str(edu["graduation_year"]))
            if edu.get("gpa"):             subs.append(f"GPA: {edu['gpa']}")
            story.append(Paragraph(f"<b>{safe(main)}</b>", B))
            if subs: story.append(Paragraph(f'<font color="#555555">{safe("  |  ".join(s for s in subs if s))}</font>', B))
            story.append(Spacer(1,4))

    certs = d.get("certifications",[])
    if certs:
        story += sh("Certifications")
        for cert in certs:
            parts = [cert.get("name",""), cert.get("issuer","")]
            if cert.get("year"): parts.append(str(cert["year"]))
            story.append(bul("  |  ".join(p for p in parts if p)))

    doc.build(story)
    pdf_bytes = buf.getvalue()
    logger.info("PDF generated: %d bytes", len(pdf_bytes))
    return pdf_bytes