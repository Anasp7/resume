"""
Smart Resume — Career Analysis & Growth Plan Module
====================================================
Handles:
1. Career trajectory prediction
2. Current role fit assessment
3. Growth plan generation
4. Skill gap categorization
5. Learning roadmap creation
"""

from __future__ import annotations
from typing import Optional, Any
import json
from core.schemas import ParsedResume, ParsedJobDescription, Domain


DOMAIN_TRAJECTORIES = {
    Domain.BACKEND: {
        "current_roles": ["Backend Developer", "API Developer", "Server-Side Engineer"],
        "next_roles": ["Senior Backend Engineer", "Staff Engineer", "Tech Lead", "Solutions Architect"],
        "trajectory_timeline": {
            "senior": "2-3 years with leadership experience",
            "staff": "5-7 years with mentoring and architecture",
            "lead": "3-5 years with team management",
        },
    },
    Domain.FRONTEND: {
        "current_roles": ["Frontend Developer", "UI Developer", "Web Developer"],
        "next_roles": ["Senior Frontend Engineer", "Tech Lead", "Product Engineer", "Design Systems Lead"],
        "trajectory_timeline": {
            "senior": "2-3 years with complex state management",
            "lead": "3-5 years with mentoring and architecture",
            "product": "4-6 years with product sense",
        },
    },
    Domain.ML: {
        "current_roles": ["ML Engineer", "Data Scientist", "ML Researcher"],
        "next_roles": ["Senior ML Engineer", "ML Architect", "ML Platform Engineer", "AI Research Engineer"],
        "trajectory_timeline": {
            "senior": "3-4 years with production models",
            "architect": "5-7 years with end-to-end systems",
            "research": "4+ years with novel contributions",
        },
    },
    Domain.DATA: {
        "current_roles": ["Data Analyst", "Data Engineer", "Analytics Engineer"],
        "next_roles": ["Senior Data Engineer", "Analytics Lead", "Data Architect", "BI Engineer"],
        "trajectory_timeline": {
            "senior": "3-4 years with pipeline architecture",
            "architect": "5-7 years with data governance",
            "lead": "4-6 years with team management",
        },
    },
    Domain.DEVOPS: {
        "current_roles": ["DevOps Engineer", "Site Reliability Engineer", "Infrastructure Engineer"],
        "next_roles": ["Senior DevOps/SRE", "DevOps Architect", "Infrastructure Lead", "Platform Engineer"],
        "trajectory_timeline": {
            "senior": "2-3 years with infrastructure design",
            "architect": "4-6 years with multi-region systems",
            "platform": "5-7 years with internal platform ownership",
        },
    },
}

SKILL_LEARN_TIME_ESTIMATES = {
    # Beginner skills (can learn in weeks)
    "python": "2-3 weeks",
    "javascript": "2-3 weeks",
    "sql": "1-2 weeks",
    "git": "1 week",
    "html": "1 week",
    "css": "1 week",
    
    # Intermediate skills (weeks to months)
    "fastapi": "3-4 weeks",
    "django": "4-6 weeks",
    "react": "4-6 weeks",
    "postgresql": "2-3 weeks",
    "docker": "2-3 weeks",
    "kubernetes": "4-6 weeks",
    "pytorch": "4-6 weeks",
    "tensorflow": "4-6 weeks",
    "aws": "4-6 weeks",
    "gcp": "4-6 weeks",
    "azure": "4-6 weeks",
    
    # Advanced skills (months)
    "ml": "8-12 weeks",
    "nlp": "8-12 weeks",
    "computer vision": "8-12 weeks",
    "reinforcement learning": "10-16 weeks",
    "distributed systems": "12+ weeks",
    "microservices": "8-12 weeks",
}

RESOURCE_MAPPINGS = {
    "python": [
        "Python.org official tutorial",
        "Codecademy Python course",
        "freeCodeCamp Python on YouTube"
    ],
    "fastapi": [
        "FastAPI official documentation",
        "full-stack-fastapi-postgresql by tiangolo",
        "Udemy FastAPI course by José Salvatierra"
    ],
    "pytorch": [
        "Fast.ai Practical Deep Learning course",
        "PyTorch official tutorials",
        "Kaggle Learn: PyTorch"
    ],
    "tensorflow": [
        "TensorFlow official tutorials",
        "Google TensorFlow Datasets",
        "Coursera TensorFlow specialization"
    ],
    "react": [
        "React official documentation",
        "freeCodeCamp React courseware",
        "Scrimba React course"
    ],
    "kubernetes": [
        "Kubernetes official documentation",
        "Linux Academy Kubernetes course",
        "Minikube local playground"
    ],
    "aws": [
        "AWS free tier with hands-on labs",
        "Coursera AWS fundamentals",
        "freeCodeCamp AWS course on YouTube"
    ],
    "nlp": [
        "Fast.ai NLP course",
        "Hugging Face course 'NLP with Transformers'",
        "Stanford CS224N on YouTube"
    ],
}


def assess_role_fit(
    resume: ParsedResume,
    jd: ParsedJobDescription,
    required_skills: set,
    matched_skills: list,
    domain: Domain,
) -> dict[str, Any]:
    """
    Assess how well candidate fits the target role.
    
    Returns:
    {
        "verdict": "Strong/Moderate/Weak",
        "score": 0-100,
        "reasoning": "Specific to this candidate",
        "strengths": ["strength 1", "strength 2"],
        "critical_gaps": ["gap 1", "gap 2"]
    }
    """
    # Calculate coverage: percentage of required skills matched
    skill_coverage = len(matched_skills) / max(len(required_skills), 1) * 100
    
    # Experience alignment
    exp_alignment_score = 0
    if jd.required_experience_years:
        if resume.years_of_experience >= jd.required_experience_years:
            exp_alignment_score = 100
        elif resume.years_of_experience >= jd.required_experience_years * 0.7:
            exp_alignment_score = 70
        else:
            exp_alignment_score = (resume.years_of_experience / jd.required_experience_years) * 60
    else:
        exp_alignment_score = min(100, resume.years_of_experience * 10)
    
    # Project complexity (proxy for capability)
    project_evidence = len(resume.projects) > 0
    project_score = 40 if project_evidence else 0
    
    # Combined score
    combined_score = (skill_coverage * 0.5) + (exp_alignment_score * 0.35) + (project_score * 0.15)
    
    # Determine verdict
    if combined_score >= 75:
        verdict = "Strong"
    elif combined_score >= 50:
        verdict = "Moderate"
    else:
        verdict = "Weak"
    
    # Extract strengths
    strengths = []
    if skill_coverage >= 70:
        strengths.append(f"Strong technical skill coverage ({len(matched_skills)}/{len(required_skills)} required skills)")
    if resume.years_of_experience >= (jd.required_experience_years or 2):
        strengths.append(f"Meets experience requirement ({resume.years_of_experience:.1f} years)")
    if len(resume.projects) >= 3:
        strengths.append(f"Solid project portfolio ({len(resume.projects)} projects)")
    if resume.education and any("CS" in (e.degree or "") for e in resume.education):
        strengths.append("Computer Science background")
    
    # Extract critical gaps
    gaps = []
    if skill_coverage < 50:
        missing = required_skills - set(m.lower() for m in matched_skills)
        gaps.append(f"Missing critical skills: {', '.join(list(missing)[:3])}")
    if resume.years_of_experience < (jd.required_experience_years or 2) * 0.5:
        gaps.append(f"Insufficient experience ({resume.years_of_experience:.1f} years, need {jd.required_experience_years or 2})")
    if not resume.projects or all(not p.description for p in resume.projects):
        gaps.append("Limited project evidence/portfolio")
    
    return {
        "verdict": verdict,
        "score": int(combined_score),
        "reasoning": f"{verdict} fit for {jd.target_role}: {len(matched_skills)} of {len(required_skills)} required skills + {resume.years_of_experience:.1f}yrs exp",
        "strengths": strengths[:2],  # Top 2
        "critical_gaps": gaps[:2],  # Top 2
    }


def predict_career_trajectory(
    resume: ParsedResume,
    jd: ParsedJobDescription,
    domain: Domain,
    matched_skills: list,
) -> dict[str, Any]:
    """
    Predict likely next roles and career progression.
    
    Returns:
    {
        "next_role_1": "title + reason",
        "next_role_2": "title + reason",
        "next_role_3": "title + reason",
        "timeline_to_target": "X months"
    }
    """
    trajectory = DOMAIN_TRAJECTORIES.get(domain, {})
    current_level = "junior" if resume.years_of_experience < 2 else ("mid" if resume.years_of_experience < 5 else "senior")
    next_roles = trajectory.get("next_roles", [])
    
    if not next_roles:
        next_roles = ["Senior " + jd.target_role, "Staff " + jd.target_role, "Tech Lead"]
    
    # Determine timeline based on experience gap
    if resume.years_of_experience >= 3:
        timeline = "6-12 months"
    elif resume.years_of_experience >= 2:
        timeline = "12-18 months"
    else:
        timeline = "18-24 months"
    
    return {
        "current_level": current_level,
        "next_role_1": f"{next_roles[0]} — Natural progression with mastery of current skills",
        "next_role_2": f"{next_roles[1] if len(next_roles) > 1 else 'Staff Engineer'} — Requires architectural experience",
        "next_role_3": f"{next_roles[2] if len(next_roles) > 2 else 'Tech Lead'} — Requires team leadership",
        "timeline_to_target": timeline,
    }


def categorize_skill_gaps(
    candidate_skills: set,
    required_skills: set,
    preferred_skills: set,
) -> dict[str, list]:
    """
    Categorize missing skills into 3 tiers.
    
    Returns:
    {
        "tier_1_critical": [...],
        "tier_2_important": [...],
        "tier_3_nice_to_have": [...]
    }
    """
    candidate_lower = {s.lower() for s in candidate_skills}
    required_lower = {s.lower() for s in required_skills}
    preferred_lower = {s.lower() for s in preferred_skills}
    
    tier_1 = []
    tier_2 = []
    tier_3 = []
    
    # Tier 1: Missing required skills
    for skill in required_lower - candidate_lower:
        tier_1.append({
            "skill": skill.title(),
            "gap_severity": "High",
            "why": f"Required for {len([s for s in required_skills if s.lower() == skill])} key responsibilities",
            "learn_time": SKILL_LEARN_TIME_ESTIMATES.get(skill, "3-4 weeks"),
        })
    
    # Tier 2: Weak required skills (optional expansion)
    # This would require more detailed evidence tracking
    
    # Tier 3: Missing preferred skills
    for skill in preferred_lower - candidate_lower:
        tier_3.append({
            "skill": skill.title(),
            "gap_severity": "Low",
            "why": f"Preferred skill for competitive advantage",
            "learn_time": SKILL_LEARN_TIME_ESTIMATES.get(skill, "4-6 weeks"),
        })
    
    return {
        "tier_1_critical": tier_1[:5],  # Top 5
        "tier_2_important": tier_2,
        "tier_3_nice_to_have": tier_3[:3],  # Top 3
    }


def generate_growth_plan(
    resume: ParsedResume,
    jd: ParsedJobDescription,
    candidate_skills: set,
    missing_skills: list,
    domain: Domain,
) -> dict[str, Any]:
    """
    Generate week-by-week learning roadmap.
    
    Returns:
    {
        "immediate_skills": [...],
        "suggested_projects": [...],
        "week_by_week_roadmap": [...],
        "total_duration": "X weeks",
        "success_criteria": [...]
    }
    """
    # Select top 3 skills to learn
    immediate_skills = []
    for skill in missing_skills[:3]:
        resources = RESOURCE_MAPPINGS.get(skill.lower(), [
            f"{skill} official documentation",
            f"Udemy {skill} course",
            f"freeCodeCamp {skill} tutorial",
        ])
        immediate_skills.append({
            "skill": skill,
            "why": f"Critical for {jd.target_role} role",
            "resources": resources[:2],
            "estimated_time": SKILL_LEARN_TIME_ESTIMATES.get(skill.lower(), "4-6 weeks"),
        })
    
    # Suggest projects that use missing skills
    suggested_projects = []
    if "python" in [s.lower() for s in missing_skills]:
        suggested_projects.append({
            "title": "Build REST API with FastAPI",
            "scope": "Develop backend service with request validation and database integration",
            "technologies": ["Python", "FastAPI", "PostgreSQL", "Docker"],
            "difficulty": "Intermediate",
            "estimated_hours": "40-60",
        })
    
    if "pytorch" in [s.lower() for s in missing_skills]:
        suggested_projects.append({
            "title": "Image Classification Model with PyTorch",
            "scope": "Train CNN on CIFAR-10 dataset, achieve 85%+ accuracy",
            "technologies": ["PyTorch", "Python", "Torchvision", "Matplotlib"],
            "difficulty": "Intermediate",
            "estimated_hours": "30-50",
        })
    
    if "kubernetes" in [s.lower() for s in missing_skills]:
        suggested_projects.append({
            "title": "Deploy Multi-Service App on Kubernetes",
            "scope": "Containerize 3 services, deploy to local Minikube, set up monitoring",
            "technologies": ["Kubernetes", "Docker", "Docker Compose", "Prometheus"],
            "difficulty": "Advanced",
            "estimated_hours": "50-80",
        })
    
    # Generate week-by-week roadmap (8 weeks typical)
    week_by_week = []
    focus_skills = immediate_skills
    for week in range(1, 9):
        skill_idx = min(week - 1, len(focus_skills) - 1)
        current_skill = focus_skills[skill_idx] if focus_skills else {"skill": "General"}
        
        week_by_week.append({
            "week": week,
            "focus": f"Master {current_skill.get('skill', 'General')} fundamentals",
            "task": f"Complete official tutorials + build mini project",
            "resource": current_skill.get('resources', ['Official documentation'])[0] if current_skill else "Docs",
            "expected_outcome": f"Hands-on mastery of {current_skill.get('skill', 'topic')}",
        })
    
    success_criteria = [
        f"Build 2-3 projects using {', '.join([s['skill'] for s in immediate_skills][:2])}",
        f"Deploy working project to production (GitHub + live URL)",
        f"Earn certification in {immediate_skills[0]['skill'] if immediate_skills else 'target skill'}",
    ]
    
    return {
        "immediate_skills": immediate_skills,
        "suggested_projects": suggested_projects,
        "week_by_week_roadmap": week_by_week,
        "total_duration": "8-12 weeks",
        "success_criteria": success_criteria,
    }


if __name__ == "__main__":
    print("Career analysis module loaded.")
