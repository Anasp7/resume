"""
Data models for Resume Optimization Platform
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class FaultSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FaultCategory(Enum):
    FORMATTING = "formatting"
    CONTENT = "content"
    STRUCTURE = "structure"
    ATS_OPTIMIZATION = "ats_optimization"
    KEYWORD_OPTIMIZATION = "keyword_optimization"


@dataclass
class ResumeFault:
    """Represents a fault found in resume analysis"""
    category: FaultCategory
    severity: FaultSeverity
    description: str
    location: Optional[str] = None
    suggestion: Optional[str] = None
    line_number: Optional[int] = None


@dataclass
class PersonalInfo:
    """Personal information extracted from resume"""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    location: Optional[str] = None


@dataclass
class WorkExperience:
    """Work experience entry"""
    company: str
    position: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)


@dataclass
class Education:
    """Education entry"""
    institution: str
    degree: str
    field: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    gpa: Optional[str] = None


@dataclass
class Project:
    """Project entry"""
    name: str
    description: str
    technologies: List[str] = field(default_factory=list)
    duration: Optional[str] = None
    achievements: List[str] = field(default_factory=list)


@dataclass
class StructuredResume:
    """Structured representation of resume data"""
    personal_info: PersonalInfo = field(default_factory=PersonalInfo)
    summary: Optional[str] = None
    work_experience: List[WorkExperience] = field(default_factory=list)
    education: List[Education] = field(default_factory=list)
    projects: List[Project] = field(default_factory=list)
    skills: Dict[str, List[str]] = field(default_factory=dict)
    certifications: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    awards: List[str] = field(default_factory=list)


@dataclass
class ResumeAnalysis:
    """Complete analysis of a resume"""
    original_text: str
    structured_data: StructuredResume
    faults: List[ResumeFault] = field(default_factory=list)
    ats_score: float = 0.0
    keyword_score: float = 0.0
    formatting_score: float = 0.0
    overall_score: float = 0.0
    analysis_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class JobRoleRequirements:
    """Requirements for a specific job role"""
    role: str
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    experience_level: str = "entry"
    key_responsibilities: List[str] = field(default_factory=list)
    industry_keywords: List[str] = field(default_factory=list)


@dataclass
class OptimizedResume:
    """Optimized resume for a specific job role"""
    target_role: str
    original_analysis: ResumeAnalysis
    optimized_content: str
    improvements_made: List[str] = field(default_factory=list)
    optimization_score: float = 0.0
    generation_timestamp: datetime = field(default_factory=datetime.now)
    job_requirements: Optional[JobRoleRequirements] = None
