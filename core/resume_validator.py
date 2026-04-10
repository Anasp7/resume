#!/usr/bin/env python3
"""
Resume Template & Structure Validator
=====================================

Reference templates and schema validation for:
1. Correct section mapping (Experience vs Projects)
2. Skill structure (Technical vs Soft, separately)
3. Missing education data (CGPA/percentage)
4. Template-based doubt generation
5. Experience classification (internship/workshop/job vs project)
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# ──────────────────────────────────────────────────────────────────────────
# ENUMS: Resume classification
# ──────────────────────────────────────────────────────────────────────────

class ExperienceType(Enum):
    """Classify experience entries correctly"""
    INTERNSHIP = "internship"              # 1-6 months training
    WORKSHOP = "workshop"                  # 1-3 days training/bootcamp
    JOB = "job"                            # Full-time/part-time employment
    FREELANCE = "freelance"                # Independent work
    UNKNOWN = "unknown"                    # Couldn't classify


class SectionType(Enum):
    """Resume sections with strict definitions"""
    CONTACT = "contact"
    SUMMARY = "summary"
    EXPERIENCE = "experience"              # Jobs, internships, workshops
    PROJECTS = "projects"                  # Standalone projects/competitions
    TECHNICAL_SKILLS = "technical_skills" # Languages, frameworks, tools
    SOFT_SKILLS = "soft_skills"            # Communication, leadership (SEPARATE)
    EDUCATION = "education"                # Degree, CGPA, percentage
    CERTIFICATIONS = "certifications"      # Courses, credentials
    AWARDS = "awards"                      # Recognition, competitions


class ResumeTemplate(Enum):
    """Template types based on content"""
    FRESHER_ACADEMIC = "fresher-academic"           # No job exp, focused on education
    FRESHER_WITH_PROJECTS = "fresher-with-projects" # No job exp, with projects
    INTERN_FOCUSED = "intern-focused"               # 1-2 internships + projects
    JUNIOR_DEVELOPER = "junior-developer"           # 0-2 yrs job exp
    MID_LEVEL = "mid-level"                         # 2-5 yrs job exp
    SENIOR = "senior"                               # 5+ yrs job exp
    PORTFOLIO = "portfolio"                         # Heavy on projects (bootcamp grad)
    ACADEMIC = "academic"                           # Heavy on education (PhD, research)


# ──────────────────────────────────────────────────────────────────────────
# REFERENCE TEMPLATES: What a good resume should contain
# ──────────────────────────────────────────────────────────────────────────

GOOD_RESUME_TEMPLATES: Dict[ResumeTemplate, Dict] = {
    
    ResumeTemplate.FRESHER_ACADEMIC: {
        "name": "Fresher - Academic Focused",
        "description": "For recent graduates with no job experience",
        "sections": {
            "contact": {"required": True, "fields": ["name", "email", "phone", "location"]},
            "summary": {"required": False, "fields": ["professional_summary"]},
            "education": {
                "required": True,
                "fields": ["institution", "degree", "cgpa_or_percentage", "graduation_year", "relevant_coursework"]
            },
            "projects": {
                "required": True,
                "count": "3-5",
                "fields": ["title", "description", "technologies", "github_link"]
            },
            "technical_skills": {
                "required": True,
                "fields": ["languages", "frameworks", "databases", "tools"]
            },
            "soft_skills": {
                "required": False,
                "fields": ["communication", "leadership", "teamwork"]
            },
            "certifications": {"required": False, "fields": ["name", "issuer", "date"]},
            "experience": {"required": False, "count": "0", "note": "No job experience expected"},
        },
        "missing_indicators": [
            "❌ CGPA/percentage missing from education",
            "❌ No projects listed",
            "❌ No GitHub links for projects",
            "❌ No technologies mentioned",
        ],
        "success_indicators": [
            "✓ Education with CGPA/percentage",
            "✓ 3-5 projects with descriptions",
            "✓ Technical skills organized by category",
            "✓ Contact information complete",
        ]
    },

    ResumeTemplate.FRESHER_WITH_PROJECTS: {
        "name": "Fresher - Projects Emphasized",
        "description": "For graduates/bootcamp students with portfolio focus",
        "sections": {
            "contact": {"required": True, "fields": ["name", "email", "phone", "portfolio_link"]},
            "summary": {"required": True, "fields": ["professional_summary", "key_achievements"]},
            "projects": {
                "required": True,
                "count": "5-8",
                "fields": ["title", "description", "role", "technologies", "github_link", "live_link", "impact"]
            },
            "technical_skills": {
                "required": True,
                "fields": ["languages", "frameworks", "databases", "tools", "proficiency_level"]
            },
            "soft_skills": {
                "required": True,
                "fields": ["collaboration", "problem_solving", "communication"]
            },
            "education": {
                "required": True,
                "fields": ["institution", "degree", "graduation_year"]
            },
            "certifications": {"required": True, "fields": ["bootcamp", "specialty_courses"]},
        },
        "missing_indicators": [
            "❌ Projects < 3",
            "❌ No GitHub or live links",
            "❌ No impact metrics in projects",
            "❌ Soft skills missing",
        ],
        "success_indicators": [
            "✓ 5+ projects with links",
            "✓ Each project has tech stack",
            "✓ Project impact/results mentioned",
            "✓ Portfolio link in contact",
        ]
    },

    ResumeTemplate.INTERN_FOCUSED: {
        "name": "Intern - Experience Building",
        "description": "For students/fresh grads with internship experience",
        "sections": {
            "contact": {"required": True, "fields": ["name", "email", "phone", "linkedin"]},
            "summary": {"required": True, "fields": ["career_objective", "skills_summary"]},
            "experience": {
                "required": True,
                "count": "1-2",
                "fields": ["company", "position", "duration", "responsibilities", "achievements"],
                "note": "Internships should clearly state 'Internship' in title"
            },
            "projects": {
                "required": True,
                "count": "2-4",
                "fields": ["title", "description", "technologies", "outcome"]
            },
            "education": {
                "required": True,
                "fields": ["institution", "degree", "cgpa", "graduation_year", "relevant_courses"]
            },
            "technical_skills": {
                "required": True,
                "fields": ["primary_languages", "frameworks", "tools"]
            },
            "soft_skills": {"required": False, "fields": ["teamwork", "communication"]},
            "certifications": {"required": False, "fields": ["online_courses"]},
        },
        "missing_indicators": [
            "❌ No internship experience",
            "❌ Internship duration not specified",
            "❌ No measurable achievements",
            "❌ < 2 projects",
        ],
        "success_indicators": [
            "✓ Internship title clear (avoid 'experience' for interns)",
            "✓ Duration clearly stated",
            "✓ 3+ achievements with metrics",
            "✓ 2-4 relevant projects",
        ]
    },

    ResumeTemplate.JUNIOR_DEVELOPER: {
        "name": "Junior Developer - Experience Focused",
        "description": "1-2 years of job experience",
        "sections": {
            "contact": {"required": True, "fields": ["name", "email", "phone", "linkedin", "github"]},
            "summary": {"required": True, "fields": ["professional_summary", "tech_stack", "key_accomplishments"]},
            "experience": {
                "required": True,
                "count": "1-2 jobs",
                "fields": ["company", "job_title", "employment_type", "duration", "key_achievements", "technologies_used"]
            },
            "projects": {
                "required": True,
                "count": "2-3",
                "fields": ["title", "description", "technologies", "GitHub_link"]
            },
            "technical_skills": {
                "required": True,
                "fields": ["primary_languages", "web_frameworks", "databases", "tools", "methodologies"]
            },
            "soft_skills": {"required": True, "fields": ["teamwork", "communication", "problem_solving"]},
            "education": {
                "required": True,
                "fields": ["degree", "institution", "graduation_year"]
            },
            "certifications": {"required": False, "fields": ["relevant_courses", "online_learning"]},
        },
        "missing_indicators": [
            "❌ Job duration unclear",
            "❌ No measurable achievements (use metrics!)",
            "❌ Technologies not specified per job",
            "❌ Soft skills missing",
        ],
        "success_indicators": [
            "✓ Clear job titles and durations",
            "✓ 3+ achievements with quantified results",
            "✓ Tech stack per role specified",
            "✓ GitHub/portfolio links present",
        ]
    },
}


# ──────────────────────────────────────────────────────────────────────────
# CLASSIFIERS: Differentiate experience from projects
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """Result of experience/project classification"""
    section_type: SectionType
    confidence: float  # 0-1
    reason: str
    recommendations: List[str]


class ResumeContentClassifier:
    """Classify resume sections and entries"""
    
    # Keywords for experience types
    INTERNSHIP_KEYWORDS = {
        "internship", "intern", "summer intern", "winter intern",
        "internship program", "intern role", "training period"
    }
    
    WORKSHOP_KEYWORDS = {
        "workshop", "bootcamp", "training", "course", "seminar",
        "bootcamp graduate", "training program", "short-term"
    }
    
    JOB_KEYWORDS = {
        "developer", "engineer", "manager", "lead", "analyst",
        "full-time", "part-time", "employee", "associate", "consultant",
        "full time", "part time"
    }
    
    PROJECT_KEYWORDS = {
        "project", "developed", "built", "created", "designed",
        "side project", "portfolio project", "personal project",
        "github", "competition", "hackathon", "contest"
    }
    
    EDUCATION_KEYWORDS = {
        "b.tech", "b.s.", "b.a.", "m.tech", "m.s.", "m.a.",
        "bachelor", "master", "phd", "diploma", "degree",
        "cgpa", "gpa", "percentage", "percentage", "marks"
    }
    
    @staticmethod
    def classify_experience_entry(text: str) -> Tuple[ExperienceType, float]:
        """
        Classify an experience entry.
        
        Returns: (ExperienceType, confidence: 0-1)
        """
        text_lower = text.lower()
        
        # Check for internship
        if any(kw in text_lower for kw in ResumeContentClassifier.INTERNSHIP_KEYWORDS):
            return ExperienceType.INTERNSHIP, 0.95
        
        # Check for workshop/bootcamp
        if any(kw in text_lower for kw in ResumeContentClassifier.WORKSHOP_KEYWORDS):
            return ExperienceType.WORKSHOP, 0.90
        
        # Check for job
        if any(kw in text_lower for kw in ResumeContentClassifier.JOB_KEYWORDS):
            return ExperienceType.JOB, 0.85
        
        return ExperienceType.UNKNOWN, 0.0
    
    @staticmethod
    def classify_section(section_text: str, section_header: Optional[str] = None) -> ClassificationResult:
        """
        Classify a resume section entry as experience or project.
        """
        text_lower = section_text.lower()
        
        # Check for project indicators
        project_score = sum(1 for kw in ResumeContentClassifier.PROJECT_KEYWORDS if kw in text_lower)
        
        # Check for experience indicators
        exp_score = sum(1 for kw in ResumeContentClassifier.JOB_KEYWORDS if kw in text_lower)
        
        # Duration indicates job (full-time, 1 year, etc.)
        if any(word in text_lower for word in ["months", "years", "duration", "employment"]):
            exp_score += 2
        
        # Company/organization indicates job
        if any(word in text_lower for word in ["company", "organization", "employer", "worked at"]):
            exp_score += 2
        
        # GitHub/live link indicates project
        if any(word in text_lower for word in ["github", "live link", "deployed", "demo"]):
            project_score += 2
        
        # Determine classification
        if project_score > exp_score:
            return ClassificationResult(
                section_type=SectionType.PROJECTS,
                confidence=min(project_score / (project_score + exp_score), 1.0),
                reason=f"Project indicators found: {project_score}, Experience indicators: {exp_score}",
                recommendations=[
                    "Move this to PROJECTS section (not EXPERIENCE)",
                    "Include GitHub link and live demo if available",
                    "Specify role and technologies used"
                ]
            )
        elif exp_score > project_score:
            return ClassificationResult(
                section_type=SectionType.EXPERIENCE,
                confidence=min(exp_score / (exp_score + project_score), 1.0),
                reason=f"Experience indicators found: {exp_score}, Project indicators: {project_score}",
                recommendations=[
                    "Keep in EXPERIENCE section",
                    "Specify employment type (job/internship/workshop)",
                    "Include duration and company",
                    "Add 3-5 measurable achievements"
                ]
            )
        else:
            return ClassificationResult(
                section_type=SectionType.EXPERIENCE,  # Default to experience if tied
                confidence=0.5,
                reason="Unclear whether this is experience or project",
                recommendations=[
                    "Clarify: Is this a job/internship or a project?",
                    "If job: specify company, duration, achievements",
                    "If project: add GitHub link, impact metrics"
                ]
            )


# ──────────────────────────────────────────────────────────────────────────
# TEMPLATE SELECTOR: Based on resume content
# ──────────────────────────────────────────────────────────────────────────

class TemplateSelector:
    """Select best template based on resume content"""
    
    @staticmethod
    def select_template(
        years_of_experience: int,
        has_job_experience: bool,
        num_projects: int,
        total_word_count: int,
        focus_area: str = None  # "academic", "portfolio", "professional"
    ) -> Tuple[ResumeTemplate, float]:
        """
        Select appropriate template.
        
        Returns: (ResumeTemplate, match_score: 0-100)
        """
        
        # Decision tree
        if years_of_experience == 0:
            if num_projects >= 5:
                return ResumeTemplate.FRESHER_WITH_PROJECTS, 95
            else:
                return ResumeTemplate.FRESHER_ACADEMIC, 90
        
        elif years_of_experience < 1:
            # Internship/bootcamp
            if num_projects >= 3:
                return ResumeTemplate.FRESHER_WITH_PROJECTS, 90
            else:
                return ResumeTemplate.INTERN_FOCUSED, 85
        
        elif years_of_experience < 2:
            return ResumeTemplate.JUNIOR_DEVELOPER, 90
        
        else:
            return ResumeTemplate.MID_LEVEL, 85


# ──────────────────────────────────────────────────────────────────────────
# VALIDATION ENGINE: Find missing data
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class MissingData:
    """Missing or incomplete data in resume"""
    section: SectionType
    field: str
    severity: str  # "critical", "important", "nice_to_have"
    template_reference: str
    question_for_user: str


class ResumeValidator:
    """Validate resume against template and identify gaps"""
    
    @staticmethod
    def validate_against_template(
        resume_data: Dict,
        template: ResumeTemplate
    ) -> Tuple[List[MissingData], float]:
        """
        Validate resume against template.
        
        Returns: (list of missing data, completeness score: 0-100)
        """
        
        template_spec = GOOD_RESUME_TEMPLATES[template]
        missing_items = []
        
        # Check each section
        for section, spec in template_spec["sections"].items():
            if spec.get("required", False):
                if section not in resume_data or not resume_data[section]:
                    missing_items.append(MissingData(
                        section=SectionType[section.upper()],
                        field=section,
                        severity="critical",
                        template_reference=f"Required in {template.value}",
                        question_for_user=f"❌ MISSING: {section.upper()}. Do you have {section} to add?"
                    ))
            
            # Check specific fields if provided
            if isinstance(spec, dict) and "fields" in spec:
                for field in spec.get("fields", []):
                    if field not in str(resume_data.get(section, {})).lower():
                        # Only flag if it's a critical field
                        if spec.get("required"):
                            missing_items.append(MissingData(
                                section=SectionType[section.upper()],
                                field=field,
                                severity="important",
                                template_reference=f"{field} expected in {template.value}",
                                question_for_user=f"⚠️  {section}: Is {field} available? (e.g., CGPA, GitHub link, duration)"
                            ))
        
        # Calculate completeness
        completeness = max(0, 100 - len(missing_items) * 10)
        
        return missing_items, completeness


if __name__ == "__main__":
    # Demo
    print("=" * 70)
    print("RESUME TEMPLATE & STRUCTURE VALIDATOR")
    print("=" * 70)
    
    # Example 1: Classify entry
    print("\n[EXAMPLE 1] Classify entry as Experience or Project")
    print("─" * 70)
    
    text1 = "SDE Intern at Google, 6 months, built payment system"
    result1 = ResumeContentClassifier.classify_section(text1)
    print(f"Text: {text1}")
    print(f"Classification: {result1.section_type.value}")
    print(f"Confidence: {result1.confidence:.0%}")
    print(f"Reason: {result1.reason}")
    print(f"Recommendations:")
    for rec in result1.recommendations:
        print(f"  • {rec}")
    
    # Example 2: Classify experience type
    print("\n[EXAMPLE 2] Classify experience type")
    print("─" * 70)
    
    text2 = "Summer Internship at Microsoft"
    exp_type, confidence = ResumeContentClassifier.classify_experience_entry(text2)
    print(f"Text: {text2}")
    print(f"Type: {exp_type.value}")
    print(f"Confidence: {confidence:.0%}")
    
    # Example 3: Select template
    print("\n[EXAMPLE 3] Select template based on profile")
    print("─" * 70)
    
    template, score = TemplateSelector.select_template(
        years_of_experience=0,
        has_job_experience=False,
        num_projects=4,
        total_word_count=300
    )
    print(f"Profile: Fresher with 4 projects")
    print(f"Recommended template: {template.value}")
    print(f"Match score: {score}/100")
    
    # Example 4: Show template spec
    print("\n[EXAMPLE 4] Template requirements")
    print("─" * 70)
    
    template_spec = GOOD_RESUME_TEMPLATES[ResumeTemplate.FRESHER_WITH_PROJECTS]
    print(f"Template: {template_spec['name']}")
    print(f"Description: {template_spec['description']}")
    print(f"\nSuccess Indicators:")
    for indicator in template_spec['success_indicators']:
        print(f"  {indicator}")
    print(f"\nMissing Indicators:")
    for indicator in template_spec['missing_indicators']:
        print(f"  {indicator}")
    
    print("\n" + "=" * 70)
