"""
Smart Resume — Hybrid LLM Engine (Ollama Phi-3 + Groq)
=======================================================

Dual-engine approach:
- Ollama Phi-3 (local, free) for: parsing, skill classification, doubt generation, career recommendations
- Groq API (free tier) for: structuring, formatting, verification
- Graceful fallback when one fails
- JD is OPTIONAL - works even without job description

Run locally: ollama pull phi3
Then: ollama serve
"""

from __future__ import annotations
import os
import json
import logging
from typing import Optional, Any
import httpx

logger = logging.getLogger("smart_resume.hybrid_llm")

# ─────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")
OLLAMA_TIMEOUT = 120  # seconds for local LLM

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_TIMEOUT = 30  # seconds


# ─────────────────────────────────────────────────────────────────────────
# OLLAMA LOCAL LLM (Phi-3)
# ─────────────────────────────────────────────────────────────────────────

def call_ollama(prompt: str, system: str = "", temperature: float = 0.3) -> Optional[str]:
    """
    Call local Ollama Phi-3 model.
    
    Args:
        prompt: User message
        system: System message (instructions)
        temperature: 0.0-1.0 (lower = more deterministic)
    
    Returns:
        Response text or None if failed
    """
    try:
        import requests
        
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"{system}\n\n{prompt}" if system else prompt,
            "stream": False,
            "temperature": temperature,
        }
        
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            logger.warning(f"Ollama returned {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        logger.warning(f"Ollama call failed: {e}")
        return None


def call_ollama_json(prompt: str, system: str = "") -> Optional[dict]:
    """
    Call Ollama and expect JSON response.
    
    Returns:
        Parsed JSON dict or None if failed
    """
    response_text = call_ollama(prompt, system, temperature=0.1)
    if not response_text:
        return None
    
    try:
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            return json.loads(json_match.group())
        else:
            return json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Ollama JSON response: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────
# GROQ API (Free tier - fallback)
# ─────────────────────────────────────────────────────────────────────────

def call_groq(prompt: str, system: str = "") -> Optional[str]:
    """
    Call Groq API (fallback when Ollama unavailable).
    
    Returns:
        Response text or None if failed
    """
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set, cannot call Groq")
        return None
    
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        messages = [
            {"role": "system", "content": system} if system else None,
            {"role": "user", "content": prompt}
        ]
        messages = [m for m in messages if m]
        
        payload = {
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1500,
        }
        
        with httpx.Client(timeout=GROQ_TIMEOUT) as client:
            response = client.post(GROQ_API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        else:
            logger.warning(f"Groq returned {response.status_code}")
            return None
            
    except Exception as e:
        logger.warning(f"Groq call failed: {e}")
        return None


def call_groq_json(prompt: str, system: str = "") -> Optional[dict]:
    """Call Groq and expect JSON response."""
    response_text = call_groq(prompt, system)
    if not response_text:
        return None
    
    try:
        import re
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            return json.loads(json_match.group())
        else:
            return json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Groq JSON response: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────
# HYBRID INTERFACE (Try Ollama first, fallback to Groq)
# ─────────────────────────────────────────────────────────────────────────

def call_llm_hybrid(prompt: str, system: str = "", prefer_ollama: bool = True) -> Optional[str]:
    """
    Call LLM with fallback strategy.
    
    Args:
        prompt: User prompt
        system: System/instruction message
        prefer_ollama: If True, try Ollama first; if False, try Groq first
    
    Returns:
        Response text
    """
    if prefer_ollama:
        # Try Ollama first
        result = call_ollama(prompt, system)
        if result:
            logger.info("✓ Using Ollama Phi-3")
            return result
        
        # Fallback to Groq
        logger.info("Ollama unavailable, falling back to Groq...")
        result = call_groq(prompt, system)
        if result:
            logger.info("✓ Using Groq")
            return result
    else:
        # Try Groq first
        result = call_groq(prompt, system)
        if result:
            logger.info("✓ Using Groq")
            return result
        
        # Fallback to Ollama
        logger.info("Groq unavailable, falling back to Ollama...")
        result = call_ollama(prompt, system)
        if result:
            logger.info("✓ Using Ollama Phi-3")
            return result
    
    logger.error("Both Ollama and Groq unavailable!")
    return None


def call_llm_json_hybrid(prompt: str, system: str = "", prefer_ollama: bool = True) -> Optional[dict]:
    """Call LLM hybrid and expect JSON response."""
    response_text = call_llm_hybrid(prompt, system, prefer_ollama)
    if not response_text:
        return None
    
    try:
        import re
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            return json.loads(json_match.group())
        else:
            return json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────
# CORE PARSING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────

def parse_resume_hybrid(resume_text: str) -> dict[str, Any]:
    """
    Parse resume using hybrid LLM (Ollama Phi-3 primary).
    
    Returns structured resume data with:
    - Contact info (name, email, phone, location)
    - Skills (extracted and categorized)
    - Experience (roles, companies, responsibilities)
    - Projects (titles, descriptions, tech stack)
    - Education (degree, institution, year, GPA)
    - Certifications
    """
    prompt = f"""Parse this resume and extract structured data.

CRITICAL: Resolve any text hyphenation artifacts (e.g., join 'Athena-' and 'sius' into 'Athanasius'). Clean all broken sentences.

RESUME:
{resume_text}

Return JSON with this structure (extract ALL info):
{{
  "name": "Full name",
  "email": "email@address.com",
  "phone": "+country-code-number",
  "location": "City, Country",
  "linkedin": "linkedin URL or null",
  "github": "github URL or null",
  "summary": "Professional summary (2 sentences, NO specific CGPA or institution names)",
  "skills": ["skill1", "skill2", ...],
  
  "experience": [
    {{
      "company": "Company Name",
      "role": "Job Title",
      "duration": "Start - End (e.g., July 2023 - Present)",
      "responsibilities": ["Detailed bullet 1", "Detailed bullet 2"],
      "technologies": ["tech1", "tech2"]
    }}
  ],
  
  "projects": [
    {{
      "title": "Project Name",
      "description": "What was built and why",
      "technologies": ["tech1", "tech2"],
      "metrics": ["Key achievements or stats"]
    }}
  ],
  
  "education": [
    {{
      "institution": "Full University/College Name (e.g. Mar Athanasius College of Engineering)",
      "degree": "Degree (e.g. Bachelor of Technology)",
      "field": "Field (e.g. Computer Science)",
      "graduation_year": "Year or 'Present'",
      "gpa": "CGPA value (e.g. 8.93/10.0)"
    }}
  ],
  
  "certifications": [
    {{
      "name": "Certification Name",
      "issuer": "Issuer",
      "year": 2024
    }}
  ]
}}

Extract ALL information. Be thorough. DO NOT miss projects or internships. Return ONLY valid JSON."""

    result = call_llm_json_hybrid(prompt, prefer_ollama=True)
    if result:
        logger.info(f"✓ Parsed resume: {result.get('name', 'Unknown')}")
        return result
    else:
        logger.error("Failed to parse resume")
        return {}


def classify_skills_hybrid(skills: list[str], jd_skills: Optional[list[str]] = None) -> dict[str, Any]:
    """
    Classify skills using hybrid LLM (Ollama Phi-3 primary).
    
    Categorizes into: Languages, Frameworks, Databases, Tools, Concepts
    Can compare against JD skills if provided (JD optional).
    """
    jd_context = f"\nJD Required Skills: {', '.join(jd_skills)}" if jd_skills else "\n[No JD provided - classify based on industry standards]"
    
    prompt = f"""Classify these skills into categories and match against typical requirements.
    
Candidate Skills: {', '.join(skills)}
{jd_context}

Return JSON:
{{
  "profile": "Junior/Middle/Senior level based on skills",
  "languages": ["python", "java", ...],
  "frameworks": ["fastapi", "react", ...],
  "databases": ["postgresql", "mongodb", ...],
  "tools": ["docker", "git", ...],
  "concepts": ["microservices", "ml", ...],
  "strengths": ["what they're good at"],
  "gaps": ["what's missing for typical role"]
}}

Return ONLY valid JSON."""

    result = call_llm_json_hybrid(prompt, prefer_ollama=True)
    if result:
        logger.info(f"✓ Classified skills: {len(result.get('languages', []))} languages")
        return result
    else:
        return {"languages": skills, "frameworks": [], "databases": [], "tools": [], "concepts": []}


def generate_doubt_questions_hybrid(
    resume: dict,
    target_role: str,
    jd_skills: Optional[list[str]] = None
) -> list[dict]:
    """
    Generate doubt questions using hybrid LLM (Ollama Phi-3 primary).
    
    Identifies ambiguities in resume and asks for clarification.
    Works with or without JD.
    """
    jd_context = f"Target role: {target_role}\nRequired skills: {', '.join(jd_skills) if jd_skills else 'Infer from role'}"
    
    prompt = f"""Generate 3-5 clarification questions for this resume.

CANDIDATE:
Name: {resume.get('name', 'Unknown')}
Skills: {', '.join(resume.get('skills', []))}
Experience: {len(resume.get('experience', []))} roles
Target: {target_role}
{jd_context}

Identify ambiguities or gaps. Ask specific, answerable questions.
CRITICAL: DO NOT ask questions implying soft skills (like project management, communication, leadership). ONLY ask technical questions about projects, what they built, and the tools/technologies used.

Return JSON:
{{
  "questions": [
    {{
      "question": "Specific question",
      "category": "experience/skills/clarity",
      "reason": "Why this matters for {{target_role}}"
    }}
  ]
}}

Return ONLY valid JSON."""

    result = call_llm_json_hybrid(prompt, prefer_ollama=True)
    if result:
        questions = result.get("questions", [])
        logger.info(f"✓ Generated {len(questions)} doubt questions")
        return questions
    else:
        return []


def generate_career_recommendation_hybrid(
    resume: dict,
    target_role: str,
    jd_skills: Optional[list[str]] = None,
    matched_skills: Optional[list[str]] = None
) -> dict[str, Any]:
    """
    Generate career recommendations using hybrid LLM (Ollama Phi-3 primary).
    
    - Career trajectory (next 3 roles)
    - Timeline to target role
    - Immediate actions
    - Growth areas
    """
    jd_context = f"Required: {', '.join(jd_skills)}\nMatched: {', '.join(matched_skills or [])}" if jd_skills else "No JD - infer from industry"
    
    prompt = f"""Analyze career trajectory for this candidate.

CANDIDATE:
Name: {resume.get('name', 'Unknown')}
Current Skills: {', '.join(resume.get('skills', []))}
Years of Exp: {resume.get('years_of_exp', 0)}
Target Role: {target_role}
Skill Match: {jd_context}

Provide:
1. Next 3 likely career roles (progression)
2. Timeline to target role (months)
3. Immediate actions (next 30 days)
4. Key strengths for THIS role
5. Critical gaps to address

Return JSON:
{{
  "current_level": "junior/mid/senior",
  "next_roles": ["role1", "role2", "role3"],
  "timeline_months": 12,
  "immediate_actions": ["action1", "action2", "action3"],
  "strengths": ["strength1", "strength2"],
  "gaps": ["gap1", "gap2"],
  "growth_areas": ["area1", "area2"]
}}

Return ONLY valid JSON."""

    result = call_llm_json_hybrid(prompt, prefer_ollama=True)
    if result:
        logger.info(f"✓ Generated career recommendations for {target_role}")
        return result
    else:
        return {"next_roles": [], "timeline_months": 12}


def generate_growth_plan_hybrid(
    skills_to_learn: list[str],
    current_skills: list[str],
    target_role: str,
    weeks_available: int = 12
) -> dict[str, Any]:
    """
    Generate week-by-week growth plan using hybrid LLM (Ollama Phi-3 primary).
    
    Creates actionable learning roadmap with specific resources.
    """
    prompt = f"""Create a {weeks_available}-week learning plan.

Current Skills: {', '.join(current_skills)}
Skills to Learn: {', '.join(skills_to_learn)}
Goal: {target_role}
Available: {weeks_available} weeks at 10-15 hrs/week

Generate week-by-week roadmap:
- Each week: focus + task + specific resource
- Only teach skills not already known
- Include hands-on projects
- Use free resources (Fast.ai, Kaggle, freeCodeCamp, docs)

Return JSON:
{{
  "total_weeks": {weeks_available},
  "roadmap": [
    {{
      "week": 1,
      "skill": "skill name",
      "focus": "learning goal",
      "task": "specific task",
      "resource": "named course/doc/tutorial",
      "difficulty": "Beginner/Intermediate/Advanced"
    }}
  ],
  "projects": [
    {{
      "week": "weeks X-Y",
      "title": "project name",
      "goal": "what to build",
      "technologies": ["tech1", "tech2"]
    }}
  ],
  "success_criteria": ["can do X", "understand Y", "built Z"]
}}

Return ONLY valid JSON."""

    result = call_llm_json_hybrid(prompt, prefer_ollama=True)
    if result:
        logger.info(f"✓ Generated {weeks_available}-week growth plan")
        return result
    else:
        return {"roadmap": [], "projects": [], "success_criteria": []}


# ─────────────────────────────────────────────────────────────────────────
# STRUCTURED RESUME OUTPUT (Groq for formatting)
# ─────────────────────────────────────────────────────────────────────────

def format_resume_for_ats(
    parsed_data: dict,
    target_role: str,
    template: str = "standard"
) -> str:
    """
    Format parsed resume data into ATS-optimized text using Groq.
    
    Returns plain-text resume with proper sections.
    """
    prompt = f"""Format this resume data into ATS-ready text.

DATA:
{json.dumps(parsed_data, indent=2)}

Template: {template}
Target Role: {target_role}

Structure:
Name | Contact Info
OBJECTIVE (2 sentences mentioning target role)
TECHNICAL SKILLS (by category: Languages | Frameworks | Tools)
EXPERIENCE (role | company | duration, 3-4 bullets per job)
PROJECTS (project name | tech, 1-2 bullets per project)
EDUCATION (degree | institution | year | CGPA if > 7)
CERTIFICATIONS (if any)

Rules:
- Plain text only (no bold, no colors, no special chars)
- Action verbs at start of bullets
- Quantifiable results
- No generic phrases ("team player", "eager to learn")
- ATS-friendly formatting

Return formatted resume text ONLY (no JSON)."""

    result = call_llm_hybrid(prompt, prefer_ollama=False)  # Use Groq for formatting
    if result:
        logger.info("✓ Formatted resume for ATS")
        return result
    else:
        logger.warning("Failed to format resume")
        return ""



# ─────────────────────────────────────────────────────────────────────────
# OLLAMA-FIRST WRAPPER — drop-in for call_llm_json (accepts api_key)
# ─────────────────────────────────────────────────────────────────────────

def call_ollama_check() -> bool:
    """Quick check if Ollama is reachable."""
    try:
        import requests
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def call_llm_json_with_fallback(
    prompt: str,
    api_key: str,
    *,
    max_tokens: int = 600,
    system_msg: str = "You are a resume analysis assistant. Respond only with valid JSON.",
    smart: bool = False,
    prefer_ollama: bool = True,
) -> dict:
    """
    Ollama-first JSON call with Groq fallback.

    Replaces call_llm_json() from llm_client.py for tasks where Ollama is
    preferred (doubt generation, virtual JD, skill classification).

    - prefer_ollama=True  → try Ollama Phi-3 first, Groq if Ollama unavailable
    - prefer_ollama=False → Groq first (for resume generation where quality matters)

    Falls back gracefully — never raises.
    """
    import re as _re, json as _json

    # ── Try Ollama first if preferred ────────────────────────────────────
    if prefer_ollama:
        result = call_ollama_json(prompt, system=system_msg)
        if result:
            logger.info("✓ call_llm_json_with_fallback — used Ollama Phi-3")
            return result
        logger.info("Ollama unavailable or bad JSON — falling back to Groq")

    # ── Groq fallback via llm_client ─────────────────────────────────────
    try:
        from core.llm_client import call_llm_json as _groq_json
        data = _groq_json(
            prompt, api_key,
            max_tokens=max_tokens,
            system_msg=system_msg,
            smart=smart,
        )
        logger.info("✓ call_llm_json_with_fallback — used Groq")
        return data
    except Exception as e:
        logger.warning("Groq fallback also failed: %s", e)
        return {}


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    
    # Test configuration
    print("="*70)
    print("SMART RESUME — HYBRID LLM ENGINE")
    print("="*70)
    print(f"\n[Config]")
    print(f"  Ollama URL: {OLLAMA_BASE_URL}")
    print(f"  Ollama Model: {OLLAMA_MODEL}")
    print(f"  Groq API Key: {'SET' if GROQ_API_KEY else 'NOT SET'}")
    print(f"\n[Available Functions]")
    print(f"  ✓ parse_resume_hybrid()")
    print(f"  ✓ classify_skills_hybrid()")
    print(f"  ✓ generate_doubt_questions_hybrid()")
    print(f"  ✓ generate_career_recommendation_hybrid()")
    print(f"  ✓ generate_growth_plan_hybrid()")
    print(f"  ✓ format_resume_for_ats()")
    print(f"  ✓ call_llm_hybrid() [fallback interface]")
    print(f"\nTo use: Import functions and call with your data")
