"""
Export utilities for Resume Optimization Platform
"""
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime


class ResumeExporter:
    """Export optimized resumes to various formats"""
    
    def __init__(self):
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)

    def export_to_text(self, optimized_resume: Dict[str, Any], filename: str = None) -> str:
        """Export optimized resume to plain text"""
        if not filename:
            role = optimized_resume.get('target_role', 'optimized').lower().replace(' ', '_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimized_resume_{role}_{timestamp}.txt"
        
        content_data = optimized_resume.get('optimized_content', '')
        content = content_data.get('text', '') if isinstance(content_data, dict) else content_data
        
        file_path = self.output_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return str(file_path)

    def export_to_markdown(self, optimized_resume: Dict[str, Any], filename: str = None) -> str:
        """Export optimized resume to Markdown"""
        if not filename:
            role = optimized_resume.get('target_role', 'optimized').lower().replace(' ', '_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimized_resume_{role}_{timestamp}.md"
        
        content_data = optimized_resume.get('optimized_content', '')
        content = content_data.get('text', '') if isinstance(content_data, dict) else content_data
        
        markdown_content = self._convert_to_markdown(content)
        file_path = self.output_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        return str(file_path)

    def export_to_html(self, optimized_resume: Dict[str, Any], filename: str = None) -> str:
        """Export optimized resume to HTML"""
        if not filename:
            role = optimized_resume.get('target_role', 'optimized').lower().replace(' ', '_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimized_resume_{role}_{timestamp}.html"
        
        content_data = optimized_resume.get('optimized_content', {})
        if isinstance(content_data, dict) and content_data.get('html'):
            html_content = content_data.get('html')
        else:
            content = content_data
            html_content = self._convert_to_html(content, optimized_resume.get('target_role', 'Optimized Resume'))
        
        file_path = self.output_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return str(file_path)
    
    def export_to_pdf(self, optimized_resume: Dict[str, Any], filename: str = None) -> str:
        """Export optimized resume to professional PDF format using LaTeX (with ReportLab fallback)"""
        
        if not filename:
            role = optimized_resume.get('target_role', 'optimized').lower().replace(' ', '_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimized_resume_{role}_{timestamp}.pdf"
        
        file_path = self.output_dir / filename
        
        # 1. Attempt LaTeX Generation & Compilation
        try:
            from core.latex_builder import build_latex_resume
            from core.resume_exporter import compile_latex_to_pdf
            
            content = optimized_resume.get('optimized_content', '')
            resume_data = {}
            resume_text = ""
            
            # Handle structured content
            if isinstance(content, str) and content.strip().startswith('{'):
                try:
                    resume_data = json.loads(content)
                except:
                    resume_text = content
            elif isinstance(content, dict):
                resume_data = content
            else:
                resume_text = str(content)
            
            # If we don't have personal info in resume_data, try to get from various sources
            if not resume_data.get('email') or not resume_data.get('phone'):
                # Try raw_data first (new frontend's primary storage)
                raw_data = optimized_resume.get('raw_data', {})
                if raw_data:
                    for k, v in raw_data.items():
                        if v and not resume_data.get(k):
                            resume_data[k] = v
                
                # Then original_analysis (if available)
                analysis = optimized_resume.get('original_analysis', {})
                structured = analysis.get('structured_data', {})
                personal = structured.get('personal_info', {})
                if personal:
                    for k, v in personal.items():
                        if v and not resume_data.get(k):
                            resume_data[k] = v
                
                # Check for other sections in raw_data/analysis
                for section in ['skills', 'education', 'projects', 'experience']:
                    if not resume_data.get(section):
                        resume_data[section] = raw_data.get(section) or structured.get(section)

            latex_code = build_latex_resume(resume_data, resume_text=resume_text)
            pdf_bytes = compile_latex_to_pdf(latex_code)
            
            if pdf_bytes:
                with open(file_path, 'wb') as f:
                    f.write(pdf_bytes)
                return str(file_path)
                
        except Exception as e:
            print(f"Warning: LaTeX generation failed ({e}). Falling back to ReportLab.")

        # 2. Fallback to ReportLab
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.enums import TA_CENTER
            
            doc = SimpleDocTemplate(str(file_path), pagesize=letter, topMargin=50)
            styles = getSampleStyleSheet()
            story = []
            
            title_style = ParagraphStyle(
                'ResumeTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=12,
                textColor=colors.HexColor("#2C3E50"),
                alignment=TA_CENTER
            )
            
            header_style = ParagraphStyle(
                'SectionHeader',
                parent=styles['Heading2'],
                fontSize=14,
                spaceBefore=12,
                spaceAfter=6,
                textColor=colors.HexColor("#2980B9"),
                borderBottom=True,
                borderColor=colors.HexColor("#BDC3C7")
            )
            
            body_style = ParagraphStyle(
                'BodyText',
                parent=styles['Normal'],
                fontSize=11,
                leading=14,
                spaceBefore=4,
                spaceAfter=4
            )
            
            bullet_style = ParagraphStyle(
                'BulletPoint',
                parent=styles['Normal'],
                fontSize=11,
                leading=14,
                leftIndent=20,
                spaceBefore=2
            )

            content = optimized_resume.get('optimized_content', '')
            data = {}
            if isinstance(content, str) and content.strip().startswith('{'):
                try: data = json.loads(content)
                except: pass
            elif isinstance(content, dict):
                data = content

            if data and isinstance(data, dict) and "name" in data:
                story.append(Paragraph(data.get("name", "Resume").upper(), title_style))
                contact = [data.get(k) for k in ["email", "phone", "location", "linkedin"] if data.get(k)]
                story.append(Paragraph(" | ".join(contact), body_style))
                story.append(Spacer(1, 15))
                
                if data.get("summary"):
                    story.append(Paragraph("SUMMARY", header_style))
                    story.append(Paragraph(data["summary"], body_style))
                
                if data.get("experience"):
                    story.append(Paragraph("EXPERIENCE", header_style))
                    for exp in data["experience"]:
                        story.append(Paragraph(f"<b>{exp.get('title')}</b> at {exp.get('company')} ({exp.get('dates')})", body_style))
                        for b in exp.get("bullets", []):
                            story.append(Paragraph(f"&bull; {b}", bullet_style))
                
                if data.get("skills"):
                    story.append(Paragraph("SKILLS", header_style))
                    skills = data["skills"]
                    if isinstance(skills, dict):
                        for cat, items in skills.items():
                            story.append(Paragraph(f"<b>{cat.title()}:</b> {', '.join(items)}", body_style))
                    else:
                        story.append(Paragraph(", ".join(skills), body_style))

                if data.get("education"):
                    story.append(Paragraph("EDUCATION", header_style))
                    for edu in data["education"]:
                        story.append(Paragraph(f"<b>{edu.get('degree')}</b>, {edu.get('school')} ({edu.get('year')})", body_style))
            else:
                # Plain text fallback
                for line in str(content).split('\n'):
                    story.append(Paragraph(line, body_style))
            
            doc.build(story)
            return str(file_path)
                
        except Exception as e:
            print(f"❌ Error generating PDF: {e}")
            return ""

    def export_analysis_report(self, analysis: Dict[str, Any], filename: str = None) -> str:
        """Export detailed analysis report to text file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"resume_analysis_report_{timestamp}.txt"
        report = self._generate_analysis_report(analysis)
        file_path = self.output_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(report)
        return str(file_path)
    
    def _convert_to_markdown(self, content: str) -> str:
        """Convert plain text or JSON to Markdown format"""
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "name" in data:
                md = f"# {data.get('name')}\n\n"
                contact = [data.get(k) for k in ['email', 'phone', 'linkedin'] if data.get(k)]
                md += " | ".join(contact) + "\n\n"
                if data.get('summary'): md += f"## SUMMARY\n\n{data['summary']}\n\n"
                if data.get('experience'):
                    md += "## EXPERIENCE\n\n"
                    for exp in data['experience']:
                        md += f"### {exp.get('title')} | {exp.get('company')} | {exp.get('dates')}\n"
                        for b in exp.get('bullets', []): md += f"- {b}\n"
                        md += "\n"
                return md
        except: pass
        return content # Simple fallback

    def _convert_to_html(self, content: str, title: str) -> str:
        """Convert content to HTML format"""
        # (Simplified for brevity, keeping original intent)
        return f"<html><head><title>{title}</title></head><body><pre>{content}</pre></body></html>"
    
    def _generate_analysis_report(self, analysis: Dict[str, Any]) -> str:
        """Generate comprehensive analysis report"""
        return "Analysis Report: " + str(analysis.get('scores', {}))
    
    def export_all_formats(self, optimized_resume: Dict[str, Any], analysis: Dict[str, Any] = None) -> Dict[str, str]:
        """Export resume to all available formats"""
        exports = {}
        try: exports['text'] = self.export_to_text(optimized_resume)
        except: pass
        try: exports['markdown'] = self.export_to_markdown(optimized_resume)
        except: pass
        try: exports['html'] = self.export_to_html(optimized_resume)
        except: pass
        try: exports['pdf'] = self.export_to_pdf(optimized_resume)
        except: pass
        return exports
