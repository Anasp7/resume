"""
Smart Resume — Prompt Builder v7 (Enhanced)
=============================================
Improvements in Phase 2:
1. Candidate-specific career roadmap with trajectory prediction
2. Skill gap with 3-tier severity categorization (Critical/Important/Nice-to-have)
3. Structured growth plan with week-by-week roadmap
4. ATS optimization rules explicitly stated
5. Verified information integration with no hallucinations (Strict Integrity)
6. Career insights analysis (current fit, next roles, timeline)
7. Resume improvements tracking (what changed and why)
8. Improved output formatting for LLM consumption
"""

from __future__ import annotations
import json
import re
from core.schemas import BackendPayload
from core.similarity import compute_proficiency_evidence_scores
from core.doubt_engine import _detect_domain
from core.improved_prompt_system import build_improved_system_prompt, get_ats_optimization_rules


def get_integrity_and_classification_rules() -> str:
    """Consolidated rules for project/exp classification and factual integrity."""
    return """
=== EXPERIENCE vs PROJECT CLASSIFICATION (STRICT) ===
You MUST redistribute items between EXPERIENCE and PROJECTS based on these rules:
1. EXPERIENCE = Professional roles, Internships, Industrial Training, and Workshops.
   - RULE: Any 'Workshop' attended MUST be placed in the EXPERIENCE section, not Projects. 
2. PROJECTS = Technical builds, competitions, and Hackathons.
   - Types: Personal Project, Academic Project, Open Source, Hackathon, Coding Competition (e.g. LeetCode, CodeChef).
   - RULE: All Hackathons and Coding Competitions MUST go in the PROJECTS section.
   - RULE: If the original resume has no "Projects" section, you MUST create one for these items.

=== CERTIFICATIONS RULE ===
- If the user explicitly provides an empty string, "nil", "none", or "no" for Certifications in the clarification answers, you MUST omit the Certifications section entirely. DO NOT invent certifications. DO NOT put workshops here.

=== GRAMMAR & TENSE RULES ===
1. If an item marked as "PRESENT" or "Ongoing", use PRESENT TENSE (e.g., "Developing", "Implementing").
2. If an item has an end date or is clearly completed, MUST use PAST TENSE (e.g., "Developed", "Implemented").
3. DO NOT use weak phrases like "Assisted in", "Worked on", "Helped with". Use STRONG action verbs ("Architected", "Optimized", "Engineered").

=== SKILL ENRICHMENT from VERIFIED ANSWERS ===
1. DYNAMIC EXTRACTION: You MUST scan every word of the "VERIFIED CANDIDATE ANSWERS" for mentioned technical skills (languages, frameworks, tools, libraries, or methods). 
2. If a user mentions a specific technology in their answer that is NOT in the "SKILLS" list, you MUST include it in the "categorized_skills" JSON and "Technical Skills" section of the resume. 
3. EXCLUSION: If a user explicitly says "I didn't use [X]" or "[X] is not used", do NOT include that skill.

=== STRICT FACTUAL INTEGRITY & HALLUCINATION PREVENTION (REINFORCED) ===
1. VERIFIED ANSWERS: These are the absolute source of truth. If they contain a metric or tool, use it. If they don't, DO NOT invent one.
2. ZERO FABRICATION: Never invent technologies, metrics, percentages, or features to "fill in gaps". If data is vague, stay vague but professional.
3. GOOGLE XYZ FORMULA: Only use "Accomplished [X] as measured by [Y], by doing [Z]" if BOTH an outcome (X) AND a metric (Y) are provided. 
4. DO NOT write "Improving performance by X%" or "Reduced costs by Y" unless that specific number appears in the source text. 
5. NO PRONOUNS: Use third-person action verbs only. (e.g., "Architected...", "Optimized...").
6. PRUNE GENERIC STATEMENTS: Remove "I am a...", "I have...", "Seeking an opportunity...".
"""


def build_phase2_analysis_prompt(payload: BackendPayload) -> str:
    """
    Build Phase 2A Analysis prompt (Optimized for Groq 8B).
    Focuses on Career trajectory, gaps, and roadmap.
    """
    pr   = payload.parsed_resume
    pjd  = payload.parsed_jd
    sim  = payload.semantic_similarity_score
    tmpl = payload.selected_template
    jd_provided = getattr(pjd, "jd_provided", bool(pjd.raw_text.strip()))

    prof_scores = compute_proficiency_evidence_scores(pr, payload.user_proficiencies)
    domain      = _detect_domain(pjd.target_role)

    # ── Candidate profile logic ──────────────────────────────────────────────
    ALWAYS_KEEP = {"c", "r", "go"}
    safe_skills = [s for s in pr.skills if len(s) > 1 or s in ALWAYS_KEEP]

    def match_techs(text: str) -> list:
        tl = text.lower()
        return [s for s in safe_skills if re.search(r"\\b" + re.escape(s) + r"\\b", tl)]

    candidate_has = set(s.lower() for s in pr.skills)
    
    resume_data = {
        "name":     pr.name or _extract_name(pr.raw_text),
        "skills":   pr.skills,
        "experience": [
            {
                "role":             e.role,
                "company":          e.company,
                "duration":         e.duration,
                "responsibilities": e.responsibilities[:5],
                "technologies":     match_techs(" ".join(e.responsibilities)),
            }
            for e in pr.experience
        ],
        "projects": [
            {
                "title":        p.title,
                "description":  p.description,
                "technologies": match_techs(p.description),
            }
            for p in pr.projects
        ]
    }

    # ── Instructions ─────────────────────────────────────────────────────────
    if jd_provided:
        jd_block = f"Target Role: {pjd.target_role}\\nDomain: {domain}\\nRequired Skills: {json.dumps(pjd.required_skills)}"
        match_instruction = "matched_skills: ONLY skills in BOTH JD and resume. missing_skills: skills in JD but NOT in resume."
    else:
        jd_block = f"Target Role: {pjd.target_role} (Infer requirements)"
        match_instruction = "Infer standard skills for this role."

    # Analysis sections
    career_trajectory_instruction = "Analyze current_fit (verdict/score), trajectory_prediction (next_roles), and immediate_actions."
    skill_gap_instruction = "Categorize missing skills into Tier 1 (Critical), 2 (Important), 3 (Nice-to-have)."
    growth_plan_instruction = "Generate immediate_skills_to_learn and 12-week week_by_week_roadmap."

    # ── Build final prompt ───────────────────────────────────────────────────
    answered_block = ""
    if payload.clarification_answers:
        answered_block = "\\nVERIFIED ANSWERS (MUST prioritize):\\n" + "\\n".join(
            f"Q: {a.question}\\nA: {a.answer}" for a in payload.clarification_answers
        )

    common_rules = get_integrity_and_classification_rules()

    prompt = f"""You are Smart Resume — professional resume evaluator and career analyst v7.
Return a single JSON object. No markdown. No text outside JSON.

{jd_block}

=== CANDIDATE RESUME ===
{json.dumps(resume_data, indent=2)}
{answered_block}

{common_rules}

=== EVALUATION INSTRUCTIONS ===
1. {match_instruction}
2. {career_trajectory_instruction}
3. {skill_gap_instruction}
4. {growth_plan_instruction}

factual_evaluation:
- technical_depth, logical_consistency, metric_realism, skill_alignment

=== OUTPUT SCHEMA ===
{{
  "job_match_analysis": {{ "domain": "", "matched_skills": [], "missing_skills": [], "verdict": "Strong/Moderate/Weak" }},
  "career_improvement_plan": {{ "current_fit": {{ "verdict": "", "score": 0, "reasoning": "" }}, "next_roles": [], "roadmap": [] }},
  "skill_gap_analysis": {{ "tier_1_critical": [], "tier_2_important": [], "tier_3_nice_to_have": [] }},
  "factual_evaluation": {{ "technical_depth": "", "metric_realism": "", "confidence": "" }}
}}
"""
    return prompt


def build_phase2b_generation_prompt(payload: BackendPayload) -> str:
    """
    Build Phase 2B Generation prompt (Optimized for Gemini 2.5 Flash).
    Focuses on high-fidelity copywriting, categorical skill classification, and LaTeX generation.
    """
    pr   = payload.parsed_resume
    pjd  = payload.parsed_jd
    tmpl = payload.selected_template
    
    # ── Candidate profile logic ──────────────────────────────────────────────
    contact_info = {
        "name":     pr.name or _extract_name(pr.raw_text),
        "email":    pr.email or "",
        "phone":    pr.phone or "",
        "location": pr.location or "",
        "linkedin": getattr(pr, "linkedin", ""),
        "github":   getattr(pr, "github", ""),
    }
    
    # Use FULL data for generation
    resume_data = {
        "skills":   pr.skills,
        "experience": [
            {
                "role":             e.role,
                "company":          e.company,
                "duration":         e.duration,
                "responsibilities": e.responsibilities,
            }
            for e in pr.experience
        ],
        "projects": [
            {
                "title":        p.title,
                "description":  p.description,
                "technologies": p.technologies,
            }
            for p in pr.projects
        ],
        "education": [
            {
                "institution":  e.institution,
                "degree":       e.degree,
                "gpa":          e.gpa,
                "year":         e.graduation_year,
            }
            for e in pr.education
        ],
        "school_10th": getattr(pr, "school_10th", None),
        "school_12th": getattr(pr, "school_12th", None),
    }

    # ── Clarification context ─────────────────────────────────────────────────
    answered_block = ""
    if payload.clarification_answers:
        answered_block = "\\nVERIFIED CANDIDATE ANSWERS (USE THESE TO ENRICH CONTENT):\\n" + "\\n".join(
            f"Q: {a.question}\\nA: {a.answer}" for a in payload.clarification_answers
        )

    common_rules = get_integrity_and_classification_rules()
    ats_rules    = get_ats_optimization_rules()

    prompt = f"""You are Smart Resume — professional resume writer and career coach.
Return a single JSON object. No markdown. No text outside JSON.

Target Role: {pjd.target_role}

=== CONTACT INFO ===
{json.dumps(contact_info, indent=2)}

=== CANDIDATE DATA ===
{json.dumps(resume_data, indent=2)}
{answered_block}

{common_rules}
{ats_rules}

=== GENERATION INSTRUCTIONS (STRICT) ===
1. GOLDEN RULE: Use the Google XYZ Formula (Accomplished [X] as measured by [Y], by doing [Z]) for EVERY bullet point in Experience and Projects. If a metric is in the verified answers, it MUST be used.
2. SUMMARY: Begin `final_resume_text` with a `SUMMARY` header. Write a 2-3 line professional summary STRICTLY targeted at: {pjd.target_role}.
   - FORBIDDEN openers: "Highly motivated", "results-oriented", "passionate", "detail-oriented", any generic adjective.
   - FORBIDDEN content: Do NOT mention degree name, college name, or any education detail.
   - REQUIRED: Open with the candidate's strongest concrete skill relevant to the target role. End with a clear value proposition for that role.
3. SKILL ENRICHMENT: Scan answers and place ALL mentioned technical skills into the 'categorized_skills'.
4. CLASSIFICATION: Follow Section Classification rules strictly:
   - Workshops/Internships -> Experience.
   - Hackathons/Competitions -> Projects.
5. HEADER FORMATTING:
   - For Projects: ALWAYS put the tech stack on the project header line exactly formatted as `Project Name | Tech: Tech1, Tech2`.
   - For Experience: ALWAYS put the role, company, and dates on ONE single line exactly formatted as `Role | Company Name | Date`. NEVER put dates on a separate line!
6. SKILLS QUALITY: In 'categorized_skills', ONLY include specific tool/technology/library names (e.g. Python, Flask, MySQL, Docker). NEVER include vague phrases like "Backend Development", "Automation", "Autonomous Systems", "Web Technologies", "DBMS", or any category-description as a skill value.
7. WRITING STYLE: Third-person, past tense for finished work, present for ongoing. No pronouns. Professional action verbs only.

=== OUTPUT SCHEMA ===
{{
  "final_resume_text": "...",
  "categorized_skills": {{
      "Programming Languages": [],
      "Web Development": [],
      "Databases & Backend": [],
      "AI & Tools": [],
      "DevOps & Others": [],
      "Soft Skills": []
  }}
}}
"""
    return prompt


def _extract_name(raw_text: str) -> str:
    for line in raw_text.splitlines():
        line = line.strip()
        if (line and len(line.split()) >= 2 and len(line.split()) <= 6 
            and not re.search(r"[@\\d\\+\\-\\|\\\\/@]", line)):
            return line
    return "Candidate"