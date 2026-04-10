"""
Smart Resume — LLM Configuration & Routing
===========================================

Controls:
1. Which LLM to use for each task
2. Fallback strategy
3. Dynamic rule application
4. Optional JD handling
"""

import os
from enum import Enum
from typing import Literal

class LLMEngine(str, Enum):
    """Available LLM engines"""
    OLLAMA = "ollama"      # Local Phi-3 - primary for heavy lifting
    GROQ = "groq"          # API - primary for formatting
    HYBRID = "hybrid"      # Try one, fallback to other


class TaskType(str, Enum):
    """Types of tasks and their preferred engines"""
    # Heavy lifting - prefer Ollama (local, free, no rate limits)
    PARSE_RESUME = "parse_resume"
    CLASSIFY_SKILLS = "classify_skills"
    GENERATE_DOUBTS = "generate_doubts"
    CAREER_ANALYSIS = "career_analysis"
    GROWTH_PLANNING = "growth_planning"
    
    # Formatting - prefer Groq (fast, good at structure)
    FORMAT_RESUME = "format_resume"
    STRUCTURE_DATA = "structure_data"
    VERIFY_CONTENT = "verify_content"
    
    # Can use either
    EVALUATE_RESUME = "evaluate_resume"


# ─────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────

OLLAMA_ENABLED = os.getenv("OLLAMA_ENABLED", "true").lower() == "true"
GROQ_ENABLED = os.getenv("GROQ_API_KEY", "").strip() != ""
JD_REQUIRED = os.getenv("JD_REQUIRED", "false").lower() == "true"  # JD is OPTIONAL
USE_LOCAL_LLM_FIRST = os.getenv("LOCAL_LLM_FIRST", "true").lower() == "true"

# Task routing configuration
TASK_ROUTING = {
    TaskType.PARSE_RESUME: {
        "prefer_engine": LLMEngine.OLLAMA if OLLAMA_ENABLED else LLMEngine.GROQ,
        "require_model": "parsing_model",
        "timeout": 120,
        "description": "Parse resume into structured data"
    },
    TaskType.CLASSIFY_SKILLS: {
        "prefer_engine": LLMEngine.OLLAMA if OLLAMA_ENABLED else LLMEngine.GROQ,
        "require_model": "classification_model",
        "timeout": 60,
        "description": "Classify and categorize skills"
    },
    TaskType.GENERATE_DOUBTS: {
        "prefer_engine": LLMEngine.OLLAMA if OLLAMA_ENABLED else LLMEngine.GROQ,
        "require_model": "doubt_model",
        "timeout": 90,
        "description": "Generate doubt/clarification questions"
    },
    TaskType.CAREER_ANALYSIS: {
        "prefer_engine": LLMEngine.OLLAMA if OLLAMA_ENABLED else LLMEngine.GROQ,
        "require_model": "career_model",
        "timeout": 90,
        "description": "Analyze career trajectory and recommendations"
    },
    TaskType.GROWTH_PLANNING: {
        "prefer_engine": LLMEngine.OLLAMA if OLLAMA_ENABLED else LLMEngine.GROQ,
        "require_model": "growth_model",
        "timeout": 120,
        "description": "Generate week-by-week learning plan"
    },
    TaskType.FORMAT_RESUME: {
        "prefer_engine": LLMEngine.GROQ if GROQ_ENABLED else LLMEngine.OLLAMA,
        "require_model": "formatting_model",
        "timeout": 60,
        "description": "Format resume for ATS"
    },
    TaskType.STRUCTURE_DATA: {
        "prefer_engine": LLMEngine.GROQ if GROQ_ENABLED else LLMEngine.OLLAMA,
        "require_model": "structuring_model",
        "timeout": 30,
        "description": "Structure resume data into templates"
    },
    TaskType.VERIFY_CONTENT: {
        "prefer_engine": LLMEngine.GROQ if GROQ_ENABLED else LLMEngine.OLLAMA,
        "require_model": "verification_model",
        "timeout": 60,
        "description": "Verify resume content for hallucinations"
    },
}

# ─────────────────────────────────────────────────────────────────────────
# JD HANDLING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────

class JDMode(str, Enum):
    """How to handle Job Description"""
    REQUIRED = "required"      # Must be provided
    OPTIONAL = "optional"      # Nice to have, but can work without
    INFERRED = "inferred"      # Generate synthetic JD from role name


# Default: JD is OPTIONAL - system works with or without it
JD_MODE = JDMode.OPTIONAL

# When JD is missing, infer skills for this role
DEFAULT_ROLE_SKILLS = {
    "ml engineer": [
        "python", "tensorflow", "pytorch", "sklearn",
        "sql", "data preprocessing", "neural networks",
        "model deployment", "cuda"
    ],
    "backend engineer": [
        "python", "node.js", "java", "golang",
        "fastapi", "django", "postgresql", "microservices",
        "docker", "kubernetes", "aws", "gcp"
    ],
    "frontend engineer": [
        "javascript", "typescript", "react", "vue",
        "html", "css", "tailwindcss", "webpack",
        "jest", "cypress", "git", "figma"
    ],
    "data scientist": [
        "python", "sql", "pandas", "numpy",
        "tableau", "power bi", "statistics",
        "data visualization", "experimental design"
    ],
    "devops engineer": [
        "kubernetes", "docker", "terraform", "ansible",
        "ci/cd", "jenkins", "aws", "linux",
        "networking", "monitoring", "grafana"
    ],
}


# ─────────────────────────────────────────────────────────────────────────
# DYNAMIC RULES APPLICATION
# ─────────────────────────────────────────────────────────────────────────

class DynamicRules:
    """Apply dynamic rules based on context"""
    
    @staticmethod
    def get_resume_rules(template_type: str) -> dict:
        """Get resume formatting rules for specific template"""
        rules = {
            "fresher-academic": {
                "sections": ["OBJECTIVE", "TECHNICAL SKILLS", "PROJECTS", "EXPERIENCE", "EDUCATION"],
                "max_pages": 1,
                "highlight_projects": True,
                "emphasis": "educational achievements and projects",
            },
            "experience-focused": {
                "sections": ["OBJECTIVE", "EXPERIENCE", "TECHNICAL SKILLS", "PROJECTS", "EDUCATION"],
                "max_pages": 2,
                "highlight_projects": False,
                "emphasis": "professional experience and impact",
            },
            "project-focused": {
                "sections": ["OBJECTIVE", "PROJECTS", "TECHNICAL SKILLS", "EXPERIENCE", "EDUCATION"],
                "max_pages": 1,
                "highlight_projects": True,
                "emphasis": "portfolio and technical projects",
            },
            "skill-focused": {
                "sections": ["OBJECTIVE", "TECHNICAL SKILLS", "EXPERIENCE", "PROJECTS", "EDUCATION"],
                "max_pages": 1,
                "highlight_projects": False,
                "emphasis": "technical depth and breadth",
            },
        }
        return rules.get(template_type.lower(), rules["fresher-academic"])
    
    @staticmethod
    def get_skill_categorization_rules(experience_years: float) -> dict:
        """Get skill categorization rules based on experience level"""
        if experience_years < 1:
            return {
                "level": "fresher",
                "classification_depth": "basic",
                "focus": "fundamental skills and learning projects",
                "ats_emphasis": "high (limited experience)",
            }
        elif experience_years < 3:
            return {
                "level": "junior",
                "classification_depth": "moderate",
                "focus": "practical application and contributions",
                "ats_emphasis": "high (show impact)",
            }
        elif experience_years < 6:
            return {
                "level": "mid-level",
                "classification_depth": "detailed",
                "focus": "technical depth and complex projects",
                "ats_emphasis": "moderate (achievements expected)",
            }
        else:
            return {
                "level": "senior",
                "classification_depth": "comprehensive",
                "focus": "leadership, architecture, impact at scale",
                "ats_emphasis": "low (experience speaks)",
            }
    
    @staticmethod
    def get_optimization_rules(similarity_score: float, has_jd: bool) -> dict:
        """Get resume optimization rules based on fit and JD availability"""
        if not has_jd:
            return {
                "mode": "generic-optimization",
                "strategy": "enhance general appeal",
                "emphasis": "versatile skills and broad impact",
                "auto_target_role": True,
            }
        
        if similarity_score >= 75:
            return {
                "mode": "targeted-enhancement",
                "strategy": "emphasize aligned skills",
                "rewrite_intensity": "low (already good fit)",
                "metrics_required": True,
            }
        elif similarity_score >= 50:
            return {
                "mode": "gap-filling",
                "strategy": "highlight transferable skills, add metrics",
                "rewrite_intensity": "medium",
                "metrics_required": True,
            }
        else:
            return {
                "mode": "major-restructuring",
                "strategy": "reposition skills, reframe experience",
                "rewrite_intensity": "high",
                "metrics_required": True,
            }


# ─────────────────────────────────────────────────────────────────────────
# LLM SELECTION LOGIC
# ─────────────────────────────────────────────────────────────────────────

def select_llm_engine(task: TaskType, prefer_local: bool = True) -> Literal["ollama", "groq", "hybrid"]:
    """
    Select which LLM to use for given task.
    
    Args:
        task: Type of task
        prefer_local: If True, prefer Ollama; if False, prefer Groq
    
    Returns:
        "ollama" / "groq" / "hybrid"
    """
    config = TASK_ROUTING.get(task, {})
    prefer_engine = config.get("prefer_engine", LLMEngine.HYBRID)
    
    if prefer_engine == LLMEngine.OLLAMA:
        return "ollama" if OLLAMA_ENABLED else ("groq" if GROQ_ENABLED else "hybrid")
    elif prefer_engine == LLMEngine.GROQ:
        return "groq" if GROQ_ENABLED else ("ollama" if OLLAMA_ENABLED else "hybrid")
    else:
        if prefer_local and OLLAMA_ENABLED:
            return "ollama"
        elif GROQ_ENABLED:
            return "groq"
        elif OLLAMA_ENABLED:
            return "ollama"
        else:
            return "hybrid"


def infer_jd_skills(target_role: str) -> list[str]:
    """
    When JD is not provided, infer typical skills for the role.
    
    Args:
        target_role: Job title/role
    
    Returns:
        List of typical skills for this role
    """
    role_lower = target_role.lower()
    
    for role_key, skills in DEFAULT_ROLE_SKILLS.items():
        if role_key in role_lower:
            return skills
    
    # Default: return generic skills
    return ["problem solving", "communication", "collaboration"]


if __name__ == "__main__":
    print("="*70)
    print("LLM CONFIGURATION")
    print("="*70)
    print(f"\n[Status]")
    print(f"  Ollama Enabled: {OLLAMA_ENABLED}")
    print(f"  Groq Enabled: {GROQ_ENABLED}")
    print(f"  Local LLM First: {USE_LOCAL_LLM_FIRST}")
    print(f"  JD Required: {JD_REQUIRED}")
    print(f"  JD Mode: {JD_MODE.value}")
    print(f"\n[Dynamic Rules Available]")
    print(f"  ✓ Resume rules (by template)")
    print(f"  ✓ Skill categorization (by experience level)")
    print(f"  ✓ Optimization strategy (by fit score)")
    print(f"\n[Task Routing]")
    for task in TaskType:
        config = TASK_ROUTING.get(task)
        if config:
            print(f"  • {task.value}: {config.get('prefer_engine').value}")
