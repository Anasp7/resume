"""
Professional Resume Templates System
100+ structured resume templates with adaptive formatting
"""
from typing import Dict, List, Any, Optional
import re
from dataclasses import dataclass
from enum import Enum


class TemplateStyle(Enum):
    MODERN = "modern"
    CLASSIC = "classic"
    CREATIVE = "creative"
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    MINIMAL = "minimal"


class TemplateColor(Enum):
    BLUE = "blue"
    GRAY = "gray"
    BLACK = "black"
    GREEN = "green"
    PURPLE = "purple"
    RED = "red"


@dataclass
class ResumeTemplate:
    """Professional resume template structure"""
    name: str
    style: TemplateStyle
    color: TemplateColor
    sections: List[str]
    layout: str
    description: str
    best_for: List[str]
    font_family: str
    font_size: str
    margins: str
    line_spacing: str


class ResumeTemplateManager:
    """Manages 100+ professional resume templates"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.current_template = None
    
    def _initialize_templates(self) -> Dict[str, ResumeTemplate]:
        """Initialize 100+ professional resume templates"""
        templates = {}
        
        # Modern Templates (20)
        for i in range(1, 21):
            templates[f"modern_{i}"] = ResumeTemplate(
                name=f"Modern Professional {i}",
                style=TemplateStyle.MODERN,
                color=TemplateColor.BLUE,
                sections=["header", "summary", "experience", "education", "skills", "projects"],
                layout="two_column",
                description=f"Clean modern design with sidebar layout",
                best_for=["tech", "business", "marketing"],
                font_family="Calibri",
                font_size="11pt",
                margins="0.5in",
                line_spacing="1.15"
            )
        
        # Classic Templates (20)
        for i in range(1, 21):
            templates[f"classic_{i}"] = ResumeTemplate(
                name=f"Classic Professional {i}",
                style=TemplateStyle.CLASSIC,
                color=TemplateColor.BLACK,
                sections=["header", "summary", "experience", "education", "skills"],
                layout="single_column",
                description="Traditional chronological format",
                best_for=["executive", "finance", "legal"],
                font_family="Times New Roman",
                font_size="12pt",
                margins="1in",
                line_spacing="1.5"
            )
        
        # Technical Templates (20)
        for i in range(1, 21):
            templates[f"technical_{i}"] = ResumeTemplate(
                name=f"Technical Professional {i}",
                style=TemplateStyle.TECHNICAL,
                color=TemplateColor.GRAY,
                sections=["header", "technical_summary", "experience", "technical_skills", "projects", "education", "certifications"],
                layout="two_column",
                description="Optimized for technical roles with skills emphasis",
                best_for=["software", "engineering", "IT"],
                font_family="Consolas",
                font_size="10pt",
                margins="0.75in",
                line_spacing="1.0"
            )
        
        # Creative Templates (20)
        for i in range(1, 21):
            templates[f"creative_{i}"] = ResumeTemplate(
                name=f"Creative Professional {i}",
                style=TemplateStyle.CREATIVE,
                color=TemplateColor.PURPLE,
                sections=["header", "portfolio", "experience", "skills", "education"],
                layout="creative",
                description="Visually appealing design for creative roles",
                best_for=["design", "marketing", "media"],
                font_family="Arial",
                font_size="11pt",
                margins="0.75in",
                line_spacing="1.2"
            )
        
        # Executive Templates (10)
        for i in range(1, 11):
            templates[f"executive_{i}"] = ResumeTemplate(
                name=f"Executive Professional {i}",
                style=TemplateStyle.EXECUTIVE,
                color=TemplateColor.BLACK,
                sections=["header", "executive_summary", "leadership_experience", "achievements", "education", "board_positions"],
                layout="executive",
                description="Premium format for senior executives",
                best_for=["c-level", "senior_management", "directors"],
                font_family="Georgia",
                font_size="12pt",
                margins="0.75in",
                line_spacing="1.3"
            )
        
        # Academic Templates (10)
        for i in range(1, 11):
            templates[f"academic_{i}"] = ResumeTemplate(
                name=f"Academic Professional {i}",
                style=TemplateStyle.ACADEMIC,
                color=TemplateColor.BLUE,
                sections=["header", "research_interests", "publications", "teaching_experience", "education", "grants"],
                layout="academic",
                description="CV format for academic and research positions",
                best_for=["professors", "researchers", "scientists"],
                font_family="Times New Roman",
                font_size="11pt",
                margins="1in",
                line_spacing="1.5"
            )
        
        return templates
    
    def get_best_template(self, resume_data: Dict[str, Any], target_role: str = "") -> ResumeTemplate:
        """Select the best template based on resume data and target role"""
        
        # Pull data from either original or optimized format
        skills = resume_data.get('skills', {})
        experience = resume_data.get('experience', []) or resume_data.get('work_experience', [])
        
        # Determine role category
        role_category = self._categorize_role(target_role, skills, experience)
        
        # Select template based on category
        if role_category in ["software", "engineering", "IT", "data", "technical"]:
            template_list = [t for t in self.templates.values() if t.style == TemplateStyle.TECHNICAL]
        elif role_category in ["executive", "management", "director"]:
            template_list = [t for t in self.templates.values() if t.style == TemplateStyle.EXECUTIVE]
        elif role_category in ["design", "marketing", "creative"]:
            template_list = [t for t in self.templates.values() if t.style == TemplateStyle.CREATIVE]
        elif role_category in ["professor", "research", "academic", "scientist"]:
            template_list = [t for t in self.templates.values() if t.style == TemplateStyle.ACADEMIC]
        else:
            template_list = [t for t in self.templates.values() if t.style == TemplateStyle.MODERN]
        
        # Return the first matching template
        return template_list[0] if template_list else self.templates["modern_1"]
    
    def _categorize_role(self, target_role: str, skills: Dict, experience: List) -> str:
        """Categorize the role for template selection"""
        role_lower = target_role.lower()
        
        # Technical roles
        if any(keyword in role_lower for keyword in ["software", "engineer", "developer", "programmer", "data", "analyst", "technical"]):
            return "technical"
        
        # Executive roles
        if any(keyword in role_lower for keyword in ["executive", "manager", "director", "vp", "president", "ceo", "cto"]):
            return "executive"
        
        # Creative roles
        if any(keyword in role_lower for keyword in ["design", "creative", "artist", "writer", "marketing", "content"]):
            return "creative"
        
        # Academic roles
        if any(keyword in role_lower for keyword in ["professor", "research", "academic", "scientist", "phd"]):
            return "academic"
        
        return "business"
    
    def format_resume(self, resume_data: Dict[str, Any], template: ResumeTemplate, target_role: str = "") -> Dict[str, str]:
        """Format resume data using the specified template, returning both text and HTML"""
        
        # Generate text version (existing logic)
        if template.style == TemplateStyle.MODERN:
            text_version = self._format_modern_resume(resume_data, template, target_role)
        elif template.style == TemplateStyle.TECHNICAL:
            text_version = self._format_technical_resume(resume_data, template, target_role)
        elif template.style == TemplateStyle.CLASSIC:
            text_version = self._format_classic_resume(resume_data, template, target_role)
        elif template.style == TemplateStyle.CREATIVE:
            text_version = self._format_creative_resume(resume_data, template, target_role)
        elif template.style == TemplateStyle.EXECUTIVE:
            text_version = self._format_executive_resume(resume_data, template, target_role)
        elif template.style == TemplateStyle.ACADEMIC:
            text_version = self._format_academic_resume(resume_data, template, target_role)
        else:
            text_version = self._format_modern_resume(resume_data, template, target_role)

        # Generate professional HTML version
        html_version = self._format_as_html(resume_data, template, target_role)
        
        return {
            "text": text_version,
            "html": html_version
        }

    def _format_as_html(self, resume_data: Dict[str, Any], template: ResumeTemplate, target_role: str) -> str:
        """Format resume as professional HTML.

        The optimise endpoint already builds a fully normalised dict with:
          name, email, phone, location, linkedin, title, summary,
          experience=[{title, company, dates, bullets}],
          education=[{degree, school, year}],
          skills=[str, ...],
          projects=[{name, description}]

        Pass it straight through.  Only use the fallback path if the caller
        passed raw structured_data (legacy / batch paths).
        """
        # If data is already normalised (has 'name' set), use it directly.
        if resume_data.get('name') and resume_data.get('name') not in ('YOUR NAME', ''):
            data = resume_data
        elif ('experience' in resume_data or 'skills' in resume_data) and 'name' in resume_data:
            data = resume_data
        else:
            # Legacy fallback: build data from structured_data / original_text
            structured_data = resume_data.get('structured_data', {})
            original_text   = resume_data.get('original_text', '')

            # Extract name from raw text
            candidate_name = ''
            for line in original_text.split('\n')[:8]:
                line = line.strip()
                if (line and len(line) < 55
                        and not any(c.isdigit() for c in line)
                        and '@' not in line and 'http' not in line
                        and '/' not in line):
                    candidate_name = line
                    break
            if not candidate_name:
                candidate_name = self._extract_name_from_text(original_text)

            raw_skills = structured_data.get('skills', {})
            if isinstance(raw_skills, dict):
                skills_list = [s for v in raw_skills.values()
                               if isinstance(v, list) for s in v]
            else:
                skills_list = list(raw_skills) if raw_skills else []

            data = {
                'name':     candidate_name,
                'email':    structured_data.get('personal_info', {}).get('email',
                            self._extract_email_from_text(original_text)),
                'phone':    structured_data.get('personal_info', {}).get('phone',
                            self._extract_phone_from_text(original_text)),
                'location': structured_data.get('personal_info', {}).get('location', ''),
                'linkedin': structured_data.get('personal_info', {}).get('linkedin',
                            self._extract_linkedin_from_text(original_text)),
                'title':    target_role.upper(),
                'summary':  self._generate_summary_from_data(
                                structured_data.get('work_experience', []),
                                structured_data.get('skills', {}),
                                target_role),
                'experience': [
                    {
                        'title':   exp.get('position', exp.get('title', '')),
                        'company': exp.get('company', ''),
                        'dates':   exp.get('dates', ''),
                        'bullets': (exp.get('description', [])
                                    if isinstance(exp.get('description'), list)
                                    else [exp.get('description', '')]),
                    }
                    for exp in structured_data.get('work_experience', [])
                ],
                'education': [
                    {
                        'degree': edu.get('degree', ''),
                        'school': edu.get('institution', ''),
                        'year':   edu.get('year', edu.get('end_date', '')),
                    }
                    for edu in structured_data.get('education', [])
                ],
                'skills':   skills_list,
                'projects': [
                    {'name': p.get('name', ''), 'description': p.get('description', '')}
                    for p in structured_data.get('projects', [])
                ],
            }

        primary_color = (
            '#007bff' if template.color == TemplateColor.BLUE else
            '#2c3e50' if template.color == TemplateColor.BLACK else
            '#d32f2f' if template.color == TemplateColor.RED else
            '#2e7d32' if template.color == TemplateColor.GREEN else
            '#6f42c1' if template.color == TemplateColor.PURPLE else '#333'
        )

        if template.style == TemplateStyle.MODERN:
            return self._generate_modern_html(data, template, primary_color)
        elif template.style == TemplateStyle.TECHNICAL:
            return self._generate_technical_html(data, template, primary_color)
        elif template.style == TemplateStyle.CLASSIC:
            return self._generate_classic_html(data, template, primary_color)
        elif template.style == TemplateStyle.CREATIVE:
            return self._generate_creative_html(data, template, primary_color)
        elif template.style == TemplateStyle.EXECUTIVE:
            return self._generate_executive_html(data, template, primary_color)
        elif template.style == TemplateStyle.ACADEMIC:
            return self._generate_academic_html(data, template, primary_color)
        else:
            return self._generate_modern_html(data, template, primary_color)

    def _generate_modern_html(self, data: Dict[str, Any], template: ResumeTemplate, color: str) -> str:
        """Modern two-column layout"""
        exp_html = ''
        for exp in data.get('experience', []):
            bullets = ''.join(
                f'<li>{b}</li>' for b in exp.get('bullets', []) if b
            )
            exp_html += f"""
            <div style="margin-top:15px;">
                <div style="display:flex;justify-content:space-between;font-weight:600;font-size:15px;color:#333;">
                    <span>{exp.get('title','')}</span>
                    <span style="color:{color};font-size:13px;">{exp.get('dates','')}</span>
                </div>
                <div style="font-style:italic;font-size:14px;color:#666;margin-bottom:5px;">{exp.get('company','')}</div>
                <ul style="margin:0;padding-left:18px;font-size:13.5px;color:#444;line-height:1.5;">{bullets}</ul>
            </div>"""

        # Skills: handle flat list OR category dict
        skills_html = ''
        skills_data = data.get('skills', [])
        if isinstance(skills_data, dict):
            for cat, items in skills_data.items():
                if items:
                    skills_html += f"""
                    <div style="margin-bottom:12px;">
                        <div style="font-weight:700;font-size:12px;color:{color};text-transform:uppercase;margin-bottom:4px;">{cat}</div>
                        <div style="font-size:13px;color:#555;">{', '.join(items)}</div>
                    </div>"""
        elif isinstance(skills_data, list) and skills_data:
            skills_html = f'<div style="font-size:13px;color:#555;line-height:1.8;">{", ".join(skills_data)}</div>'

        edu_html = ''
        for edu in data.get('education', []):
            edu_html += f"""
            <div style="margin-bottom:10px;">
                <div style="font-weight:600;font-size:14px;color:#333;">{edu.get('degree','')}</div>
                <div style="font-size:13px;color:#666;">{edu.get('school','')}</div>
                <div style="font-size:12px;color:{color};">{edu.get('year','')}</div>
            </div>"""

        proj_html = ''
        projects = data.get('projects', [])
        if projects:
            proj_html = f'<div style="margin-top:20px;"><h2 style="font-size:16px;color:{color};border-bottom:2px solid {color}22;padding-bottom:4px;text-transform:uppercase;">Projects</h2>'
            for p in projects:
                name = p.get('name', '') if isinstance(p, dict) else str(p)
                desc = p.get('description', '') if isinstance(p, dict) else ''
                tech = p.get('tech_stack', [])
                tech_html = f'<div style="font-size:12px;color:{color};margin-top:4px;font-weight:600;">Technologies: {", ".join(tech)}</div>' if tech else ''
                proj_html += f"""
                <div style="margin-top:10px;">
                    <div style="font-weight:600;font-size:14px;color:#333;">{name}</div>
                    <div style="font-size:13px;color:#555;margin-top:2px;">{desc}</div>
                    {tech_html}
                </div>"""
            proj_html += '</div>'

        safe_name = data.get('name', '') or ''
        initial = safe_name[0] if safe_name else '?'

        html = f"""
        <div class="resume-paper" style="font-family:'Inter',sans-serif;color:#333;max-width:800px;margin:0 auto;background:white;padding:0;display:flex;min-height:1000px;box-shadow:0 0 20px rgba(0,0,0,0.05);">
            <!-- Sidebar -->
            <div style="width:260px;background:#f8f9fa;padding:40px 25px;border-right:1px solid #eee;">
                <div style="text-align:center;margin-bottom:30px;">
                    <div style="width:80px;height:80px;background:{color};color:white;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:32px;font-weight:700;margin:0 auto 15px;">{initial}</div>
                    <h1 style="font-size:20px;margin:0;color:#222;line-height:1.2;">{safe_name}</h1>
                    <p style="font-size:13px;color:{color};font-weight:600;margin-top:5px;text-transform:uppercase;">{data.get('title','')}</p>
                </div>

                <div style="margin-bottom:30px;">
                    <h2 style="font-size:14px;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:15px;border-bottom:1px solid #ddd;padding-bottom:5px;">Contact</h2>
                    <div style="font-size:13px;color:#555;margin-bottom:8px;">📧 {data.get('email','')}</div>
                    <div style="font-size:13px;color:#555;margin-bottom:8px;">📱 {data.get('phone','')}</div>
                    {f'<div style="font-size:13px;color:#555;margin-bottom:8px;">📍 {data.get("location")}</div>' if data.get('location') else ''}
                    {f'<div style="font-size:13px;color:#555;">🔗 {data.get("linkedin")}</div>' if data.get('linkedin') else ''}
                </div>

                <div style="margin-bottom:30px;">
                    <h2 style="font-size:14px;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:15px;border-bottom:1px solid #ddd;padding-bottom:5px;">Education</h2>
                    {edu_html}
                </div>

                <div>
                    <h2 style="font-size:14px;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:15px;border-bottom:1px solid #ddd;padding-bottom:5px;">Skills</h2>
                    {skills_html}
                </div>
            </div>

            <!-- Main Content -->
            <div style="flex:1;padding:40px 35px;">
                <section style="margin-bottom:30px;">
                    <h2 style="font-size:16px;color:{color};border-bottom:2px solid {color}22;padding-bottom:4px;text-transform:uppercase;">Summary</h2>
                    <p style="font-size:14px;line-height:1.6;color:#444;margin-top:10px;">{data.get('summary','')}</p>
                </section>

                <section style="margin-bottom:30px;">
                    <h2 style="font-size:16px;color:{color};border-bottom:2px solid {color}22;padding-bottom:4px;text-transform:uppercase;">Experience</h2>
                    {exp_html}
                </section>

                {proj_html}
            </div>
        </div>
        """
        return html

    def _generate_technical_html(self, data: Dict[str, Any], template: ResumeTemplate, color: str) -> str:
        """Technical skill-dense layout"""
        exp_html = ''
        for exp in data.get('experience', []):
            bullets = ''.join(f'<li>{b}</li>' for b in exp.get('bullets', []) if b)
            exp_html += f"""
            <div style="margin-bottom:15px;">
                <div style="display:flex;justify-content:space-between;align-items:baseline;">
                    <strong style="font-size:15px;">{exp.get('company','')}</strong>
                    <span style="font-size:12px;color:#666;">{exp.get('dates','')}</span>
                </div>
                <div style="font-size:14px;font-weight:500;color:{color};">{exp.get('title','')}</div>
                <ul style="margin:5px 0 0 0;padding-left:15px;font-size:13px;color:#333;">{bullets}</ul>
            </div>"""

        # Skills: handle flat list OR category dict
        skills_data = data.get('skills', [])
        skills_cat_html = ''
        if isinstance(skills_data, dict):
            for cat, items in skills_data.items():
                if items:
                    skills_cat_html += f"""
                <div style="margin-bottom:5px;">
                    <span style="font-weight:700;width:100px;display:inline-block;font-size:12px;">{cat.upper()}:</span>
                    <span style="font-size:13px;">{', '.join(items)}</span>
                </div>"""
        elif isinstance(skills_data, list) and skills_data:
            # Wrap flat list as a single-category block
            skills_cat_html = f'<div style="font-size:13px;line-height:1.8;">{"<br>".join(skills_data)}</div>'

        safe_name = data.get('name', '') or ''
        html = f"""
        <div class="resume-paper" style="font-family:'Consolas','Monaco',monospace;background:white;color:#1a1a1a;padding:40px;max-width:800px;margin:0 auto;box-shadow:0 0 10px rgba(0,0,0,0.1);">
            <header style="border-bottom:3px solid #333;padding-bottom:10px;margin-bottom:20px;">
                <h1 style="font-size:28px;margin:0;color:#000;">{safe_name.upper()}</h1>
                <div style="font-size:12px;margin-top:5px;color:#555;">
                    {data.get('email','')} | {data.get('phone','')} | {data.get('location','')} | {data.get('linkedin','')}
                </div>
            </header>

            <section style="margin-bottom:20px;">
                <div style="background:#f4f4f4;padding:5px 10px;font-weight:bold;border-left:5px solid {color};margin-bottom:10px;">SUMMARY</div>
                <p style="font-size:13px;line-height:1.4;">{data.get('summary','')}</p>
            </section>

            <section style="margin-bottom:20px;">
                <div style="background:#f4f4f4;padding:5px 10px;font-weight:bold;border-left:5px solid {color};margin-bottom:10px;">SKILLS &amp; TECHNOLOGIES</div>
                <div style="padding-left:10px;">{skills_cat_html}</div>
            </section>

            <section style="margin-bottom:20px;">
                <div style="background:#f4f4f4;padding:5px 10px;font-weight:bold;border-left:5px solid {color};margin-bottom:10px;">EXPERIENCE</div>
                {exp_html}
            </section>

            <section>
                <div style="background:#f4f4f4;padding:5px 10px;font-weight:bold;border-left:5px solid {color};margin-bottom:10px;">PROJECTS</div>
                {''.join([f'<div style="margin-bottom:10px;"><strong style="font-size:14px;">{p.get("name","")}</strong><br/><span style="font-size:12px;color:#444;">{p.get("description","")}</span>{f"<br/><span style=\'font-size:11px;color:{color};font-weight:600;\'>TECH: " + ", ".join(p.get("tech_stack",[])) + "</span>" if p.get("tech_stack") else ""}</div>' for p in data.get('projects',[])])}
            </section>

            <section>
                <div style="background:#f4f4f4;padding:5px 10px;font-weight:bold;border-left:5px solid {color};margin-bottom:10px;">EDUCATION</div>
                {''.join([f'<div style="margin-bottom:5px;font-size:13px;"><strong>{e.get("school","")}</strong> - {e.get("degree","")} ({e.get("year","")})</div>' for e in data.get('education',[])])}
            </section>
        </div>
        """
        return html

    def _generate_classic_html(self, data: Dict[str, Any], template: ResumeTemplate, color: str) -> str:
        """Classic centered layout"""
        return f"""
        <div class="resume-paper" style="font-family: 'Times New Roman', serif; padding: 50px; background: white; max-width: 800px; margin: 0 auto; line-height: 1.5;">
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="font-size: 24pt; margin: 0; font-weight: normal;">{data.get('name').upper()}</h1>
                <div style="font-size: 11pt; margin-top: 5px;">
                    {data.get('email')} &bull; {data.get('phone')} &bull; {data.get('location')}
                </div>
                <div style="font-size: 11pt;">{data.get('linkedin')}</div>
            </div>

            <div style="border-top: 1px solid #000; margin-top: 10px;"></div>

            <h2 style="font-size: 12pt; text-align: center; text-transform: uppercase; margin: 15px 0 10px;">Professional Summary</h2>
            <p style="font-size: 11pt; text-align: justify; margin: 0;">{data.get('summary')}</p>

            <h2 style="font-size: 12pt; text-align: center; text-transform: uppercase; margin: 20px 0 10px;">Experience</h2>
            {"".join([f'''
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; font-weight: bold; font-size: 11pt;">
                    <span>{exp.get('company').upper()}</span>
                    <span>{exp.get('dates')}</span>
                </div>
                <div style="font-style: italic; font-size: 11pt;">{exp.get('title')}</div>
                <ul style="margin: 5px 0 0 0; padding-left: 20px; font-size: 11pt;">
                    {"".join([f'<li>{b}</li>' for b in exp.get('bullets', [])])}
                </ul>
            </div>
            ''' for exp in data.get('experience', [])])}

            <h2 style="font-size: 12pt; text-align: center; text-transform: uppercase; margin: 20px 0 10px;">Education</h2>
            {"".join([f'''
            <div style="display: flex; justify-content: space-between; font-size: 11pt; margin-bottom: 5px;">
                <span><strong>{edu.get('school')}</strong>, {edu.get('degree')}</span>
                <span>{edu.get('year')}</span>
            </div>
            ''' for edu in data.get('education', [])])}
        </div>
        """

    def _generate_creative_html(self, data: Dict[str, Any], template: ResumeTemplate, color: str) -> str:
        """Creative modern layout with colors and bold design"""
        exp_html = ''
        for exp in data.get('experience', []):
            bullets = ''.join(
                f'<li style="margin-bottom:5px;">{b}</li>'
                for b in exp.get('bullets', []) if b
            )
            exp_html += f"""
            <div style="margin-bottom:25px;position:relative;padding-left:20px;border-left:2px solid {color}44;">
                <div style="position:absolute;left:-6px;top:0;width:10px;height:10px;border-radius:50%;background:{color};"></div>
                <div style="font-weight:700;font-size:16px;color:#1a1a1a;">{exp.get('title','')}</div>
                <div style="color:{color};font-weight:600;font-size:14px;margin-bottom:8px;">{exp.get('company','')} | {exp.get('dates','')}</div>
                <ul style="margin:0;padding-left:15px;font-size:13.5px;color:#555;line-height:1.6;">{bullets}</ul>
            </div>"""

        # Skills: handle flat list OR category dict
        skills_data = data.get('skills', [])
        skills_tags = ''
        if isinstance(skills_data, dict):
            for cat, items in skills_data.items():
                for item in items:
                    skills_tags += f'<span style="background:{color}11;color:{color};padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;display:inline-block;margin:0 5px 8px 0;">{item}</span>'
        elif isinstance(skills_data, list):
            for item in skills_data:
                skills_tags += f'<span style="background:{color}11;color:{color};padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;display:inline-block;margin:0 5px 8px 0;">{item}</span>'

        safe_name = data.get('name', '') or ''
        html = f"""
        <div class="resume-paper" style="font-family:'Poppins',sans-serif;background:#fff;max-width:800px;margin:0 auto;min-height:1000px;display:flex;flex-direction:column;overflow:hidden;box-shadow:0 10px 30px rgba(0,0,0,0.1);">
            <div style="background:{color};color:white;padding:50px 40px;text-align:left;">
                <h1 style="font-size:42px;margin:0;font-weight:800;letter-spacing:-1px;line-height:1;">{safe_name}</h1>
                <p style="font-size:18px;margin:10px 0 0 0;opacity:0.9;text-transform:uppercase;letter-spacing:2px;">{data.get('title','')}</p>
                <div style="margin-top:25px;display:flex;gap:20px;font-size:13px;opacity:0.9;">
                    <span>📧 {data.get('email','')}</span>
                    <span>📱 {data.get('phone','')}</span>
                    {f'<span>📍 {data.get("location")}</span>' if data.get('location') else ''}
                </div>
            </div>

            <div style="display:flex;flex:1;">
                <div style="flex:2;padding:40px;">
                    <section style="margin-bottom:40px;">
                        <h2 style="font-size:20px;color:#1a1a1a;font-weight:800;margin-bottom:20px;position:relative;">
                            ABOUT ME
                            <span style="position:absolute;bottom:-8px;left:0;width:40px;height:4px;background:{color};"></span>
                        </h2>
                        <p style="font-size:14px;line-height:1.8;color:#666;">{data.get('summary','')}</p>
                    </section>

                    <section style="margin-bottom:40px;">
                        <h2 style="font-size:20px;color:#1a1a1a;font-weight:800;margin-bottom:25px;position:relative;">
                            PROJECTS
                            <span style="position:absolute;bottom:-8px;left:0;width:40px;height:4px;background:{color};"></span>
                        </h2>
                        {''.join([f'''
                        <div style="margin-bottom:20px;">
                            <div style="font-weight:700;font-size:15px;color:#1a1a1a;">{p.get('name','')}</div>
                            <div style="font-size:13px;color:#666;margin-top:4px;">{p.get('description','')}</div>
                            {f"<div style='margin-top:6px;display:flex;flex-wrap:wrap;gap:5px;'>" + "".join([f"<span style='background:{color}22;color:{color};font-size:10px;padding:2px 8px;border-radius:10px;font-weight:600;'>{t}</span>" for t in p.get('tech_stack',[])]) + "</div>" if p.get('tech_stack') else ""}
                        </div>''' for p in data.get('projects',[])])}
                    </section>

                    <section>
                        <h2 style="font-size:20px;color:#1a1a1a;font-weight:800;margin-bottom:25px;position:relative;">
                            EXPERIENCE
                            <span style="position:absolute;bottom:-8px;left:0;width:40px;height:4px;background:{color};"></span>
                        </h2>
                        {exp_html}
                    </section>
                </div>

                <div style="flex:1;padding:40px;background:#fafafa;">
                    <section style="margin-bottom:40px;">
                        <h2 style="font-size:18px;color:#1a1a1a;font-weight:800;margin-bottom:20px;">SKILLS</h2>
                        <div style="display:flex;flex-wrap:wrap;">{skills_tags}</div>
                    </section>

                    <section style="margin-bottom:40px;">
                        <h2 style="font-size:18px;color:#1a1a1a;font-weight:800;margin-bottom:20px;">EDUCATION</h2>
                        {''.join([f'''
                        <div style="margin-bottom:15px;">
                            <div style="font-weight:700;font-size:14px;color:#1a1a1a;">{edu.get('degree','')}</div>
                            <div style="font-size:13px;color:#666;">{edu.get('school','')}</div>
                            <div style="font-size:12px;color:{color};font-weight:600;">{edu.get('year','')}</div>
                        </div>''' for edu in data.get('education',[])])}
                    </section>
                </div>
            </div>
        </div>
        """
        return html

    def _generate_executive_html(self, data: Dict[str, Any], template: ResumeTemplate, color: str) -> str:
        """Sleek executive layout"""
        return f"""
        <div class="resume-paper" style="font-family: 'Lato', sans-serif; background: #fff; max-width: 800px; margin: 0 auto; padding: 50px; color: #333; box-shadow: 0 0 20px rgba(0,0,0,0.05);">
            <header style="display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 1px solid #ddd; padding-bottom: 30px; margin-bottom: 30px;">
                <div>
                    <h1 style="font-size: 34px; margin: 0; color: {color}; font-weight: 300; letter-spacing: 1px;">{data.get('name').upper()}</h1>
                    <p style="font-size: 16px; margin: 5px 0 0 0; color: #666; letter-spacing: 2px;">{data.get('title')}</p>
                </div>
                <div style="text-align: right; font-size: 13px; color: #777; line-height: 1.6;">
                    {data.get('email')}<br/>
                    {data.get('phone')}<br/>
                    {data.get('location')}
                </div>
            </header>

            <section style="margin-bottom: 35px;">
                <h2 style="font-size: 14px; color: {color}; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 15px;">Executive Summary</h2>
                <p style="font-size: 14.5px; line-height: 1.7; color: #444; border-left: 3px solid {color}44; padding-left: 20px;">{data.get('summary')}</p>
            </section>

            <section style="margin-bottom: 35px;">
                <h2 style="font-size: 14px; color: {color}; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 15px;">Key Projects</h2>
                {"".join([f'''
                <div style="margin-bottom: 20px;">
                    <div style="font-weight: 700; font-size: 14.5px;">{p.get('name')}</div>
                    <div style="font-size: 13.5px; color: #555; margin-top: 4px;">{p.get('description')}</div>
                    {f"<div style='font-size: 12px; color: {color}; font-weight: 600; margin-top: 4px;'>Tech Stack: " + ", ".join(p.get('tech_stack', [])) + "</div>" if p.get('tech_stack') else ""}
                </div>
                ''' for p in data.get('projects', [])])}
            </section>

            <section style="margin-bottom: 35px;">
                <h2 style="font-size: 14px; color: {color}; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 15px;">Strategic Experience</h2>
                {"".join([f'''
                <div style="margin-bottom: 25px;">
                    <div style="display: flex; justify-content: space-between; font-weight: 700; font-size: 15px;">
                        <span>{exp.get('title')} | {exp.get('company')}</span>
                        <span style="color: #888;">{exp.get('dates')}</span>
                </div>
                <ul style="margin: 10px 0 0 0; padding-left: 18px; font-size: 14px; color: #555;">
                    {"".join([f'<li style="margin-bottom: 6px;">{b}</li>' for b in exp.get('bullets', [])])}
                </ul>
            </div>
            ''' for exp in data.get('experience', [])])}
            </section>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 40px;">
                <section>
                    <h2 style="font-size: 13px; color: {color}; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 15px;">Key Expertise</h2>
                    <ul style="padding-left: 15px; font-size: 13px; line-height: 1.8;">
                        {"".join([f'<li><strong>{k}:</strong> {", ".join(v[:3])}</li>' for k,v in data.get('skills', {}).items() if v])}
                    </ul>
                </section>
                <section>
                    <h2 style="font-size: 13px; color: {color}; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 15px;">Academic Background</h2>
                    {"".join([f'<div style="margin-bottom: 10px; font-size: 13px;"><strong>{e.get("degree")}</strong><br/>{e.get("school")}, {e.get("year")}</div>' for e in data.get('education', [])])}
                </section>
            </div>
        </div>
        """

    def _generate_academic_html(self, data: Dict[str, Any], template: ResumeTemplate, color: str) -> str:
        """Formal academic CV layout"""
        return self._generate_classic_html(data, template, color)

    def _format_modern_resume(self, resume_data: Dict[str, Any], template: ResumeTemplate, target_role: str) -> str:
        return html
    
    def _format_modern_resume(self, resume_data: Dict[str, Any], template: ResumeTemplate, target_role: str) -> str:
        """Format modern two-column resume"""
        
        # Extract real data from uploaded resume
        original_text = resume_data.get('original_text', '')
        structured_data = resume_data.get('structured_data', {})
        personal_info = structured_data.get('personal_info', {})
        
        # Parse real name from resume text
        name = self._extract_name_from_text(original_text)
        email = personal_info.get('email', self._extract_email_from_text(original_text))
        phone = personal_info.get('phone', self._extract_phone_from_text(original_text))
        linkedin = personal_info.get('linkedin', self._extract_linkedin_from_text(original_text))
        
        # Extract real experience
        work_experience = structured_data.get('work_experience', [])
        education = structured_data.get('education', [])
        skills = structured_data.get('skills', {})
        projects = structured_data.get('projects', [])
        
        # Generate professional summary based on real data
        summary = self._generate_summary_from_data(work_experience, skills, target_role)
        
        resume = f"""
╔══════════════════════════════════════════════════════════════╗
║                    {name.upper():^57} ║
╚══════════════════════════════════════════════════════════════╝

📧 {email} | 📱 {phone} | 💼 {linkedin if linkedin else 'LinkedIn: linkedin.com/in/yourprofile'}

═══════════════════════════════════════════════════════════════

📋 PROFESSIONAL SUMMARY
{summary}

═══════════════════════════════════════════════════════════════

💼 WORK EXPERIENCE
"""
        
        # Add real work experience
        for exp in work_experience[:3]:  # Limit to top 3
            company = exp.get('company', '').replace('|', '').strip()
            position = exp.get('position', '').replace('|', '').strip()
            if company and position:
                resume += f"""
🏢 {position}
   {company}
   • {self._extract_achievements_from_text(exp.get('description', []))}
"""
        
        resume += """
═══════════════════════════════════════════════════════════════

🎓 EDUCATION
"""
        
        # Add real education
        for edu in education[:2]:  # Limit to top 2
            institution = edu.get('institution', '').replace('|', '').strip()
            degree = edu.get('degree', 'Degree')
            if institution:
                resume += f"""
📚 {degree}
   {institution}
"""
        
        resume += """
═══════════════════════════════════════════════════════════════

🛠️ TECHNICAL SKILLS
"""
        
        # Add real skills
        for category, skill_list in skills.items():
            if skill_list:
                resume += f"\n{category.title()}: {', '.join(skill_list[:8])}"  # Limit skills
        
        resume += """
═══════════════════════════════════════════════════════════════

🚀 PROJECTS
"""
        
        # Add real projects
        for project in projects[:3]:  # Limit to top 3
            proj_name = project.get('name', '').replace('-', '').strip()
            proj_desc = project.get('description', '')
            if proj_name:
                resume += f"""
📊 {proj_name}
   {proj_desc}
"""
        
        return resume.strip()
    
    def _format_technical_resume(self, resume_data: Dict[str, Any], template: ResumeTemplate, target_role: str) -> str:
        """Format technical resume with emphasis on skills"""
        
        original_text = resume_data.get('original_text', '')
        structured_data = resume_data.get('structured_data', {})
        personal_info = structured_data.get('personal_info', {})
        
        name = self._extract_name_from_text(original_text)
        email = self._extract_email_from_text(original_text)
        phone = self._extract_phone_from_text(original_text)
        github = personal_info.get('github', self._extract_github_from_text(original_text))
        
        skills = structured_data.get('skills', {})
        work_experience = structured_data.get('work_experience', [])
        projects = structured_data.get('projects', [])
        
        resume = f"""
┌─────────────────────────────────────────────────────────────────┐
│                        {name.upper():^57} │
└─────────────────────────────────────────────────────────────────┘

📧 {email} | 📱 {phone} | 💻 {github if github else 'GitHub: github.com/yourusername'}

═══════════════════════════════════════════════════════════════

🔧 TECHNICAL SKILLS
"""
        
        # Emphasize technical skills
        for category, skill_list in skills.items():
            if skill_list:
                resume += f"\n{category.upper()}: {', '.join(skill_list)}"
        
        resume += f"""

═══════════════════════════════════════════════════════════════

💻 PROFESSIONAL EXPERIENCE
"""
        
        for exp in work_experience[:3]:
            company = exp.get('company', '').replace('|', '').strip()
            position = exp.get('position', '').replace('|', '').strip()
            if company and position:
                resume += f"""
{position.upper()}
{company}
• {self._extract_achievements_from_text(exp.get('description', []))}
"""
        
        resume += f"""

═══════════════════════════════════════════════════════════════

🚀 TECHNICAL PROJECTS
"""
        
        for project in projects[:4]:  # More emphasis on projects
            proj_name = project.get('name', '').replace('-', '').strip()
            proj_desc = project.get('description', '')
            if proj_name:
                resume += f"""
{proj_name.upper()}
{proj_desc}
Technologies: {', '.join(project.get('technologies', skills.get('technical', [])[:5]))}
"""
        
        return resume.strip()
    
    def _format_classic_resume(self, resume_data: Dict[str, Any], template: ResumeTemplate, target_role: str) -> str:
        """Format classic chronological resume"""
        
        original_text = resume_data.get('original_text', '')
        structured_data = resume_data.get('structured_data', {})
        personal_info = structured_data.get('personal_info', {})
        
        name = self._extract_name_from_text(original_text)
        email = self._extract_email_from_text(original_text)
        phone = self._extract_phone_from_text(original_text)
        
        work_experience = structured_data.get('work_experience', [])
        education = structured_data.get('education', [])
        skills = structured_data.get('skills', {})
        
        resume = f"""
{'='*60}
{name.upper()}
{'='*60}

Contact Information:
Email: {email}
Phone: {phone}

PROFESSIONAL SUMMARY
{self._generate_summary_from_data(work_experience, skills, target_role)}

PROFESSIONAL EXPERIENCE
"""
        
        for exp in work_experience:
            company = exp.get('company', '').replace('|', '').strip()
            position = exp.get('position', '').replace('|', '').strip()
            if company and position:
                resume += f"""
{position}
{company}
• {self._extract_achievements_from_text(exp.get('description', []))}
"""
        
        resume += """

EDUCATION
"""
        
        for edu in education:
            institution = edu.get('institution', '').replace('|', '').strip()
            degree = edu.get('degree', 'Degree')
            if institution:
                resume += f"""
{degree}
{institution}
"""
        
        resume += """

SKILLS
"""
        
        for category, skill_list in skills.items():
            if skill_list:
                resume += f"{category.title()}: {', '.join(skill_list)}\n"
        
        return resume.strip()
    
    def _format_creative_resume(self, resume_data: Dict[str, Any], template: ResumeTemplate, target_role: str) -> str:
        """Format creative resume"""
        
        original_text = resume_data.get('original_text', '')
        structured_data = resume_data.get('structured_data', {})
        
        name = self._extract_name_from_text(original_text)
        email = self._extract_email_from_text(original_text)
        phone = self._extract_phone_from_text(original_text)
        
        work_experience = structured_data.get('work_experience', [])
        skills = structured_data.get('skills', {})
        projects = structured_data.get('projects', [])
        
        resume = f"""
✨ {name.upper()} ✨
{'~'*50}
📧 {email} | 📱 {phone}

🎨 CREATIVE PROFILE
{self._generate_summary_from_data(work_experience, skills, target_role)}

💼 CREATIVE EXPERIENCE
"""
        
        for exp in work_experience[:3]:
            company = exp.get('company', '').replace('|', '').strip()
            position = exp.get('position', '').replace('|', '').strip()
            if company and position:
                resume += f"""
🌟 {position}
   {company}
   • {self._extract_achievements_from_text(exp.get('description', []))}
"""
        
        resume += """

🎨 CREATIVE SKILLS
"""
        
        for category, skill_list in skills.items():
            if skill_list:
                resume += f"• {category.title()}: {', '.join(skill_list)}\n"
        
        return resume.strip()
    
    def _format_executive_resume(self, resume_data: Dict[str, Any], template: ResumeTemplate, target_role: str) -> str:
        """Format executive resume"""
        
        original_text = resume_data.get('original_text', '')
        structured_data = resume_data.get('structured_data', {})
        
        name = self._extract_name_from_text(original_text)
        email = self._extract_email_from_text(original_text)
        phone = self._extract_phone_from_text(original_text)
        
        work_experience = structured_data.get('work_experience', [])
        skills = structured_data.get('skills', {})
        
        resume = f"""
═══════════════════════════════════════════════════════════════
                    EXECUTIVE PROFILE
                    {name.upper()}
═══════════════════════════════════════════════════════════════

EXECUTIVE CONTACT
📧 {email} | 📱 {phone}

EXECUTIVE SUMMARY
{self._generate_executive_summary(work_experience, skills, target_role)}

LEADERSHIP EXPERIENCE
"""
        
        for exp in work_experience:
            company = exp.get('company', '').replace('|', '').strip()
            position = exp.get('position', '').replace('|', '').strip()
            if company and position:
                resume += f"""
🏢 {position.upper()}
   {company}
   • {self._extract_achievements_from_text(exp.get('description', []))}
"""
        
        return resume.strip()
    
    def _format_academic_resume(self, resume_data: Dict[str, Any], template: ResumeTemplate, target_role: str) -> str:
        """Format academic CV"""
        
        original_text = resume_data.get('original_text', '')
        structured_data = resume_data.get('structured_data', {})
        
        name = self._extract_name_from_text(original_text)
        email = self._extract_email_from_text(original_text)
        
        education = structured_data.get('education', [])
        skills = structured_data.get('skills', {})
        
        resume = f"""
ACADEMIC CURRICULUM VITAE
{'='*50}
{name.upper()}
{'='*50}

CONTACT INFORMATION
📧 {email}

ACADEMIC BACKGROUND
"""
        
        for edu in education:
            institution = edu.get('institution', '').replace('|', '').strip()
            degree = edu.get('degree', 'Degree')
            if institution:
                resume += f"""
🎓 {degree}
   {institution}
"""
        
        resume += """

RESEARCH INTERESTS
"""
        
        for category, skill_list in skills.items():
            if skill_list:
                resume += f"• {category.title()}: {', '.join(skill_list)}\n"
        
        return resume.strip()
    
    def _extract_name_from_text(self, text: str) -> str:
        """Extract name from resume text"""
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and len(line) < 50 and not any(char.isdigit() for char in line):
                if '@' not in line and 'http' not in line:
                    return line
        return "YOUR NAME"
    
    def _extract_email_from_text(self, text: str) -> str:
        """Extract email from resume text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return emails[0] if emails else "your.email@example.com"
    
    def _extract_phone_from_text(self, text: str) -> str:
        """Extract phone from resume text"""
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        return phones[0] if phones else "(555) 123-4567"
    
    def _extract_linkedin_from_text(self, text: str) -> str:
        """Extract LinkedIn from resume text"""
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedins = re.findall(linkedin_pattern, text)
        return linkedins[0] if linkedins else ""
    
    def _extract_github_from_text(self, text: str) -> str:
        """Extract GitHub from resume text"""
        github_pattern = r'github\.com/[\w-]+'
        githubs = re.findall(github_pattern, text)
        return githubs[0] if githubs else ""
    
    def _extract_achievements_from_text(self, descriptions: List[str]) -> str:
        """Extract achievements from description list"""
        if not descriptions:
            return "• Professional experience and achievements"
        
        achievements = []
        for desc in descriptions[:3]:  # Limit to 3 achievements
            desc = desc.strip().replace('-', '').replace('•', '').strip()
            if desc and len(desc) > 10:
                achievements.append(f"• {desc}")
        
        return '\n'.join(achievements) if achievements else "• Professional experience and achievements"
    
    def _generate_summary_from_data(self, work_experience: List, skills: Dict, target_role: str) -> str:
        """Generate professional summary from real data"""
        
        # Extract real skills
        all_skills = []
        for skill_list in skills.values():
            all_skills.extend(skill_list)
        
        # Determine experience level
        exp_count = len(work_experience)
        exp_level = "Experienced" if exp_count > 0 else "Entry-level"
        
        # Generate summary based on real data
        if all_skills:
            skill_summary = f"with expertise in {', '.join(all_skills[:5])}"
        else:
            skill_summary = "with diverse professional skills"
        
        summary = f"{exp_level} professional {skill_summary} "
        summary += f"seeking {target_role} position. "
        summary += f"Proven track record of delivering results and driving success."
        
        return summary
    
    def _generate_executive_summary(self, work_experience: List, skills: Dict, target_role: str) -> str:
        """Generate executive summary"""
        
        exp_count = len(work_experience)
        leadership_level = "Senior" if exp_count > 2 else "Executive"
        
        summary = f"{leadership_level} professional with extensive experience in strategic leadership "
        summary += f"and business development. Proven track record of driving organizational growth "
        summary += f"and leading high-performing teams. Seeking {target_role} opportunities."
        
        return summary
    
    def get_template_list(self) -> List[Dict[str, Any]]:
        """Get list of all available templates"""
        return [
            {
                "id": template_id,
                "name": template.name,
                "style": template.style.value,
                "color": template.color.value,
                "description": template.description,
                "best_for": template.best_for
            }
            for template_id, template in self.templates.items()
        ]
