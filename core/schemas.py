"""
Smart Resume — Core Schemas (Phase 1)
====================================
Central Pydantic models shared across every layer of the system.
Backend populates BackendPayload; every downstream module reads from it.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class ProficiencyLevel(str, Enum):
    BEGINNER     = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED     = "Advanced"
    EXPERT       = "Expert"


class TemplateType(str, Enum):
    FRESHER_ACADEMIC  = "Fresher-Academic"
    PROJECT_FOCUSED   = "Project-Focused"
    EXPERIENCE_FOCUSED = "Experience-Focused"
    SKILL_FOCUSED     = "Skill-Focused"


class Domain(str, Enum):
    BACKEND  = "Backend"
    ML       = "ML"
    DATA     = "Data"
    FRONTEND = "Frontend"
    DEVOPS   = "DevOps"
    UNKNOWN  = "Unknown"


# ─────────────────────────────────────────────
# INPUT MODELS
# ─────────────────────────────────────────────

class SkillProficiency(BaseModel):
    """User-declared proficiency for a single skill."""
    skill_name: str
    level: ProficiencyLevel

    @field_validator("skill_name")
    @classmethod
    def strip_skill(cls, v: str) -> str:
        return v.strip().lower()


class ClarificationAnswer(BaseModel):
    """One Q&A pair from the clarification round."""
    question: str
    answer: str


# ─────────────────────────────────────────────
# PARSED RESUME SECTIONS
# ─────────────────────────────────────────────

class ProjectEntry(BaseModel):
    title: str
    description: str
    technologies: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)        # verified technologies from clarifications
    metrics: list[str] = Field(default_factory=list)          # quantifiable claims
    duration: Optional[str] = None
    is_experience_mislabeled: bool = False                    # mismatch flag
    project_type: str = ""  # competition | academic | personal | open-source | freelance | (empty = unknown)


class ExperienceEntry(BaseModel):
    company: str
    role: str
    duration: Optional[str] = None
    duration_certain: bool = True
    responsibilities: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    is_project_mislabeled: bool = False                       # mismatch flag
    experience_type: str = ""  # internship | workshop | job | competition | freelance | (empty = unknown)


class EducationEntry(BaseModel):
    institution: str
    degree: str
    field: Optional[str] = None
    graduation_year: Optional[Any] = None   # int (2023), str ("2023 – Present"), or None
    gpa: Optional[str] = None

    @field_validator("gpa", mode="before")
    @classmethod
    def coerce_gpa_to_str(cls, v):
        """LLM sometimes returns GPA as float (e.g. 7.79) — coerce to string."""
        if v is None:
            return None
        return str(v)


class CertificationEntry(BaseModel):
    name: str
    issuer: Optional[str] = None
    year: Optional[int] = None


# ─────────────────────────────────────────────
# PARSED RESUME — FULL STRUCTURED EXTRACTION
# ─────────────────────────────────────────────

class ParsedResume(BaseModel):
    """Output of Section 1 — Structured Extraction."""
    raw_text: str
    # Contact info (populated by LLM parser when available)
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    summary: Optional[str] = None
    # Resume sections
    skills: list[str] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    certifications: list[CertificationEntry] = Field(default_factory=list)
    claims_with_metrics: list[str] = Field(default_factory=list)
    years_of_experience: float = 0.0
    project_count: int = 0
    experience_count: int = 0
    school_10th: Optional[str] = None
    school_12th: Optional[str] = None


# ─────────────────────────────────────────────
# JOB DESCRIPTION
# ─────────────────────────────────────────────

class ParsedJobDescription(BaseModel):
    """Structured extraction from the raw job description text."""
    raw_text: str
    target_role: str
    detected_domain: Domain = Domain.UNKNOWN
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    required_experience_years: Optional[float] = None
    key_responsibilities: list[str] = Field(default_factory=list)
    jd_provided: bool = True


# ─────────────────────────────────────────────
# BACKEND PAYLOAD — CENTRAL CONTRACT
# ─────────────────────────────────────────────

class BackendPayload(BaseModel):
    """
    The single normalized object that flows from Phase 1 (backend)
    into the Smart Resume prompt and all downstream phases.

    Backend populates every field here.
    The AI prompt reads from this — it never computes math itself.
    """

    # ── Raw inputs ──────────────────────────────
    resume_raw_text: str                          = Field(..., description="PDF/DOCX → plain text")
    job_description_raw_text: Optional[str]       = Field(None, description="Raw JD text (optional, uses dataset if null)")
    target_role: str                              = Field(..., description="Role being applied to")

    # ── Parsed structures ────────────────────────
    parsed_resume: ParsedResume
    parsed_jd: ParsedJobDescription

    # ── Similarity (backend computed) ────────────
    semantic_similarity_score: float             = Field(..., ge=0, le=100,
                                                        description="Embedding cosine sim scaled 0–100")

    # ── User declared proficiencies ──────────────
    user_proficiencies: list[SkillProficiency]   = Field(default_factory=list)

    # ── Clarification (optional round-trip) ──────
    clarification_answers: Optional[list[ClarificationAnswer]] = Field(default=None)

    # ── Backend decisions ────────────────────────
    selected_template: TemplateType
    needs_optimization: bool

    # ── Metadata ─────────────────────────────────
    session_id: Optional[str] = None
    timestamp: Optional[str] = None


# ─────────────────────────────────────────────
# Smart Resume RESPONSE STRUCTURE
# ─────────────────────────────────────────────

class SmartResumeResponse(BaseModel):
    """
    Structured output returned by Smart Resume after processing BackendPayload.
    Each section maps to Sections 1–12 of the system prompt.
    """
    session_id: Optional[str] = None

    # Sections
    structured_extraction: dict[str, Any]        = Field(default_factory=dict)
    skill_classification: dict[str, list[str]]   = Field(
        default_factory=dict,
        description="Must contain: programming_languages, frameworks_libraries, tools_platforms, databases, soft_skills"
    )
    job_match_analysis: dict[str, Any]           = Field(
        default_factory=dict,
        description="Must contain: score, verdict, matched_skills, missing_skills, evidence_bullets"
    )
    doubt_detection: dict[str, Any]              = Field(default_factory=dict)
    proficiency_consistency: dict[str, Any]      = Field(default_factory=dict)
    factual_evaluation: dict[str, Any]           = Field(default_factory=dict)
    internal_consistency: dict[str, Any]         = Field(default_factory=dict)
    resume_quality_assessment: dict[str, Any]    = Field(default_factory=dict)
    template_selection: dict[str, Any]           = Field(default_factory=dict)
    final_resume: Optional[str]                  = None
    latex_template: Optional[str]                = None
    career_improvement_plan: dict[str, Any]      = Field(default_factory=dict)
    skill_gap_analysis: dict[str, Any]           = Field(default_factory=dict)

    # Flags
    clarification_required: bool                 = False
    clarification_questions: list[str]           = Field(default_factory=list)
    mismatch_corrections: list[str]              = Field(default_factory=list)  # project↔experience fixes

    # STEP X — Missing profile / academic info questions
    missing_profile_detection: list[dict[str, Any]] = Field(
        default_factory=list,
        description="STEP X: Questions about missing GitHub, LinkedIn, CGPA, certs, etc.",
    )

    # STEP Y — Applied change log after profile answers are verified
    profile_change_log: list[str]                = Field(
        default_factory=list,
        description="STEP Y: Human-readable log of changes applied from profile answers.",
    )