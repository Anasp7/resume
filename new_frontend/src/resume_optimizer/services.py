"""
Core services for Resume Optimization Platform
"""
import re
import spacy
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path

from .models import (
    ResumeAnalysis, StructuredResume, ResumeFault, 
    FaultSeverity, FaultCategory, PersonalInfo, 
    WorkExperience, Education, Project, OptimizedResume,
    JobRoleRequirements
)
from ..llm_client import call_llm


class ResumeAnalyzer:
    """Analyzes resumes and identifies faults"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Using basic pattern matching.")
            self.nlp = None
    
    def analyze_resume(self, resume_text: str) -> ResumeAnalysis:
        """Complete resume analysis"""
        structured_data = self._extract_structured_data(resume_text)
        faults = self._identify_faults(resume_text, structured_data)
        
        # Calculate scores
        ats_score = self._calculate_ats_score(structured_data)
        keyword_score = self._calculate_keyword_score(structured_data)
        formatting_score = self._calculate_formatting_score(resume_text)
        overall_score = (ats_score + keyword_score + formatting_score) / 3
        
        return ResumeAnalysis(
            original_text=resume_text,
            structured_data=structured_data,
            faults=faults,
            ats_score=ats_score,
            keyword_score=keyword_score,
            formatting_score=formatting_score,
            overall_score=overall_score
        )
    
    def _extract_structured_data(self, text: str) -> StructuredResume:
        """Convert unstructured resume text to structured data"""
        structured = StructuredResume()
        
        # Extract personal information
        structured.personal_info = self._extract_personal_info(text)
        
        # Extract sections
        structured.summary = self._extract_summary(text)
        structured.work_experience = self._extract_work_experience(text)
        structured.education = self._extract_education(text)
        structured.projects = self._extract_projects(text)
        structured.skills = self._extract_skills(text)
        structured.certifications = self._extract_certifications(text)
        
        return structured
    
    def _extract_personal_info(self, text: str) -> PersonalInfo:
        """Extract personal information from resume"""
        personal_info = PersonalInfo()
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            personal_info.email = emails[0]
        
        # Phone pattern
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        if phones:
            personal_info.phone = phones[0]
        
        # LinkedIn pattern
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedins = re.findall(linkedin_pattern, text)
        if linkedins:
            personal_info.linkedin = linkedins[0]
        
        # GitHub pattern
        github_pattern = r'github\.com/[\w-]+'
        githubs = re.findall(github_pattern, text)
        if githubs:
            personal_info.github = githubs[0]
        
        return personal_info
    
    def _extract_work_experience(self, text: str) -> List[WorkExperience]:
        """Extract work experience from resume"""
        experiences = []
        
        # Simple pattern matching for work experience
        # This is a basic implementation - can be enhanced with NLP
        lines = text.split('\n')
        current_exp = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for company/position patterns but ensuring the line is short enough to be a title
            if len(line) < 60 and (re.search(r'(Inc|Ltd|LLC|Corporation|Company)\b', line, re.IGNORECASE) or \
               re.search(r'\b(Engineer|Developer|Manager|Analyst|Director|Intern|Student)\b', line, re.IGNORECASE)):
                if current_exp:
                    experiences.append(current_exp)
                current_exp = WorkExperience(company=line, position=line)
            elif current_exp and len(line) > 10:
                current_exp.description.append(line)
        
        if current_exp:
            experiences.append(current_exp)
        
        return experiences
    
    def _extract_education(self, text: str) -> List[Education]:
        """Extract education information"""
        education = []
        
        # Look for education keywords
        education_keywords = ['university', 'college', 'bachelor', 'master', 'phd', 'btech', 'mtech']
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword.lower() in line.lower() for keyword in education_keywords):
                education.append(Education(institution=line, degree="Degree"))
        
        return education
    
    def _extract_projects(self, text: str) -> List[Project]:
        """Extract project information"""
        projects = []
        
        # Look for project-related keywords
        project_keywords = ['project', 'developed', 'built', 'created', 'implemented']
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword.lower() in line.lower() for keyword in project_keywords):
                projects.append(Project(name=line, description=line))
        
        return projects
    
    def _extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract skills and categorize them"""
        skills = {
            'technical': [],
            'soft': [],
            'tools': [],
            'languages': []
        }
        
        # Technical skills
        tech_skills = ['python', 'java', 'javascript', 'react', 'nodejs', 'sql', 'aws', 'docker']
        for skill in tech_skills:
            if skill.lower() in text.lower():
                skills['technical'].append(skill)
        
        return skills
    
    def _extract_summary(self, text: str) -> Optional[str]:
        """Extract professional summary"""
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if len(line) > 50 and 'summary' not in line.lower():
                return line
        return None
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Extract certifications"""
        cert_keywords = ['certified', 'certification', 'aws certified', 'google certified']
        certifications = []
        
        for keyword in cert_keywords:
            if keyword.lower() in text.lower():
                certifications.append(keyword)
        
        return certifications
    
    def _identify_faults(self, text: str, structured: StructuredResume) -> List[ResumeFault]:
        """Identify faults in the resume"""
        faults = []
        
        # Check for missing contact information
        if not structured.personal_info.email:
            faults.append(ResumeFault(
                category=FaultCategory.CONTENT,
                severity=FaultSeverity.HIGH,
                description="Missing email address",
                suggestion="Add a professional email address"
            ))
        
        if not structured.personal_info.phone:
            faults.append(ResumeFault(
                category=FaultCategory.CONTENT,
                severity=FaultSeverity.MEDIUM,
                description="Missing phone number",
                suggestion="Add a phone number for contact"
            ))
        
        # Check for missing summary
        if not structured.summary:
            faults.append(ResumeFault(
                category=FaultCategory.STRUCTURE,
                severity=FaultSeverity.MEDIUM,
                description="Missing professional summary",
                suggestion="Add a 2-3 sentence professional summary"
            ))
        
        # Check formatting issues
        if len(text) > 10000:  # Too long
            faults.append(ResumeFault(
                category=FaultCategory.FORMATTING,
                severity=FaultSeverity.MEDIUM,
                description="Resume is too long",
                suggestion="Keep resume to 1-2 pages maximum"
            ))
        
        # Check for ATS optimization
        if len(structured.skills.get('technical', [])) < 5:
            faults.append(ResumeFault(
                category=FaultCategory.ATS_OPTIMIZATION,
                severity=FaultSeverity.HIGH,
                description="Insufficient technical keywords",
                suggestion="Add more relevant technical skills"
            ))
        
        return faults
    
    def _calculate_ats_score(self, structured: StructuredResume) -> float:
        """Calculate ATS optimization score"""
        score = 0.0
        
        # Contact info (20 points)
        if structured.personal_info.email:
            score += 10
        if structured.personal_info.phone:
            score += 10
        
        # Skills (30 points)
        tech_skills = len(structured.skills.get('technical', []))
        score += min(tech_skills * 3, 30)
        
        # Experience (30 points)
        if structured.work_experience:
            score += 30
        
        # Education (20 points)
        if structured.education:
            score += 20
        
        return min(score, 100.0)
    
    def _calculate_keyword_score(self, structured: StructuredResume) -> float:
        """Calculate keyword optimization score"""
        total_skills = sum(len(skills) for skills in structured.skills.values())
        return min(total_skills * 5, 100.0)
    
    def _calculate_formatting_score(self, text: str) -> float:
        """Calculate formatting score"""
        score = 100.0
        
        # Deduct points for common formatting issues
        if len(text) > 10000:
            score -= 20
        if text.count('\t') > 10:  # Too many tabs
            score -= 10
        if len(text.split('\n')) > 100:  # Too many lines
            score -= 10
        
        return max(score, 0.0)


class StructuredDataConverter:
    """Converts unstructured resume data to structured format"""
    
    def __init__(self):
        self.analyzer = ResumeAnalyzer()
    
    def convert_to_structured(self, resume_text: str) -> StructuredResume:
        """Convert unstructured text to structured resume data"""
        return self.analyzer._extract_structured_data(resume_text)
    
    def convert_to_json(self, structured: StructuredResume) -> Dict[str, Any]:
        """Convert structured resume to JSON format"""
        return {
            'personal_info': {
                'name': structured.personal_info.name,
                'email': structured.personal_info.email,
                'phone': structured.personal_info.phone,
                'linkedin': structured.personal_info.linkedin,
                'github': structured.personal_info.github,
                'location': structured.personal_info.location
            },
            'summary': structured.summary,
            'work_experience': [
                {
                    'company': exp.company,
                    'position': exp.position,
                    'start_date': exp.start_date,
                    'end_date': exp.end_date,
                    'description': exp.description,
                    'achievements': exp.achievements
                }
                for exp in structured.work_experience
            ],
            'education': [
                {
                    'institution': edu.institution,
                    'degree': edu.degree,
                    'field': edu.field,
                    'start_date': edu.start_date,
                    'end_date': edu.end_date,
                    'gpa': edu.gpa
                }
                for edu in structured.education
            ],
            'projects': [
                {
                    'name': proj.name,
                    'description': proj.description,
                    'technologies': proj.technologies,
                    'duration': proj.duration,
                    'achievements': proj.achievements
                }
                for proj in structured.projects
            ],
            'skills': structured.skills,
            'certifications': structured.certifications,
            'languages': structured.languages,
            'awards': structured.awards
        }


class ResumeOptimizer:
    """Optimizes resumes for specific job roles"""
    
    def __init__(self):
        self.analyzer = ResumeAnalyzer()
        self.converter = StructuredDataConverter()
    
    def optimize_for_role(self, resume_text: str, job_role: str, 
                         job_requirements: Optional[JobRoleRequirements] = None,
                         user_data: Optional[Dict[str, Any]] = None) -> OptimizedResume:
        """Optimize resume for a specific job role"""
        
        # Analyze original resume
        analysis = self.analyzer.analyze_resume(resume_text)
        
        # Generate optimized content
        optimized_content = self._generate_optimized_content(analysis, job_role, job_requirements, user_data=user_data)
        
        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(analysis, job_requirements)
        
        return OptimizedResume(
            target_role=job_role,
            original_analysis=analysis,
            optimized_content=optimized_content,
            optimization_score=optimization_score,
            job_requirements=job_requirements
        )
    
    def _generate_optimized_content(self, analysis: ResumeAnalysis, 
                                  job_role: str, 
                                  requirements: Optional[JobRoleRequirements],
                                  user_data: Optional[Dict[str, Any]] = None) -> str:
        """Generate optimized resume content using ResumeForge-X prompt"""
        
        # Load the ResumeForge-X prompt template
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / "resumeforge_x_prompt.txt"
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_template = f.read()
        except FileNotFoundError:
            # Fallback to a basic version of ResumeForge-X if file missing
            prompt_template = "Optimize resume for {JOB_TITLE} at {COMPANY_NAME}. Keywords: {JOB_KEYWORDS}\n\nRESUME:\n{resume_text}"

        # Prepare context variables
        job_title = job_role
        company_name = "Target Company" # Default if not provided
        job_keywords = []
        
        if requirements:
            job_keywords.extend(requirements.required_skills)
            job_keywords.extend(requirements.preferred_skills)
            job_keywords.extend(requirements.industry_keywords)
        
        resume_text_to_use = analysis.original_text
        if user_data:
            # Construct a structured text version of user data to guide the LLM
            user_info_parts = []
            if user_data.get('experience') and isinstance(user_data['experience'], list):
                user_info_parts.append("CONFIRMED EXPERIENCE:")
                for exp in user_data['experience']:
                    if not isinstance(exp, dict): continue
                    user_info_parts.append(f"- {exp.get('title', 'Unknown')} at {exp.get('company', 'Unknown')}")
                    bullets = exp.get('bullets', [])
                    if isinstance(bullets, list):
                        for bullet in bullets:
                            user_info_parts.append(f"  * {bullet}")
            
            if user_data.get('skills') and isinstance(user_data['skills'], list):
                user_info_parts.append(f"CONFIRMED SKILLS: {', '.join(user_data['skills'])}")
            
            if user_data.get('projects') and isinstance(user_data['projects'], list):
                user_info_parts.append("CONFIRMED PROJECTS:")
                for proj in user_data['projects']:
                    user_info_parts.append(f"- {proj}")

            if user_data.get('clarifications') and isinstance(user_data['clarifications'], list):
                user_info_parts.append("ADDITIONAL PROJECT/EXPERIENCE DETAILS:")
                raw_achievements = []
                for clar in user_data['clarifications']:
                    q = clar.get('question', '')
                    a = clar.get('answer', '')
                    user_info_parts.append(f"Q: {q}")
                    user_info_parts.append(f"A: {a}")
                    
                    # Extract raw achievements for explicit professionalization
                    for line in a.split('\n'):
                        if line.strip().startswith('-'):
                            raw_achievements.append(line.strip()[1:].strip())
                
                if raw_achievements:
                    user_info_parts.append("\nUSER-PROVIDED ACHIEVEMENTS (RAW) - TRANSFORM INTO EXPERT BULLETS:")
                    for achievement in raw_achievements:
                        user_info_parts.append(f"- {achievement}")
            
            user_info_text = "\n".join(user_info_parts)
            # Prepend user data to the prompt context
            resume_text_to_use = f"{user_info_text}\n\nINSTRUCTIONS FOR USER DATA:\n1. Transform any 'USER-PROVIDED ACHIEVEMENTS (RAW)' into professional, high-impact resume bullet points using the Google XYZ formula (Accomplished [X] as measured by [Y], by doing [Z]).\n2. Incorporate these into the most relevant project or experience entry.\n3. Ensure any 'CONFIRMED SKILLS' are explicitly included in the final Skills section.\n\nORIGINAL RESUME TEXT:\n{analysis.original_text}"

        # Format the prompt
        try:
            prompt = prompt_template.format(
                JOB_TITLE=job_title,
                COMPANY_NAME=company_name,
                JOB_KEYWORDS=", ".join(job_keywords) if job_keywords else "standard industry keywords",
                resume_text=resume_text_to_use
            )
        except KeyError as e:
            # Handle cases where formatting might fail if template wasn't escaped correctly
            print(f"Warning: Prompt formatting failed ({e}). Using simple fallback.")
            prompt = f"Optimize for {job_title}\n\n{analysis.original_text}"
        
        # Call LLM (now returns JSON string for ResumeForge-X)
        return call_llm(prompt)

    
    def _format_requirements(self, requirements: JobRoleRequirements) -> str:
        """Format job requirements for prompt"""
        if not requirements:
            return "No specific requirements provided"
        
        return f"""
        Required Skills: {', '.join(requirements.required_skills)}
        Preferred Skills: {', '.join(requirements.preferred_skills)}
        Experience Level: {requirements.experience_level}
        Key Responsibilities: {', '.join(requirements.key_responsibilities)}
        Industry Keywords: {', '.join(requirements.industry_keywords)}
        """
    
    def _calculate_optimization_score(self, analysis: ResumeAnalysis, 
                                    requirements: Optional[JobRoleRequirements]) -> float:
        """Calculate optimization score based on analysis and requirements"""
        base_score = analysis.overall_score
        
        if requirements:
            # Check skill alignment
            tech_skills = analysis.structured_data.skills.get('technical', [])
            required_matches = len(set(tech_skills) & set(requirements.required_skills))
            skill_bonus = (required_matches / len(requirements.required_skills)) * 20
            
            base_score += skill_bonus
        
        return min(base_score, 100.0)
