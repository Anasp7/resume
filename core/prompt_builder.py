"""
Smart Resume — Prompt Builder v6
==================================
Phase 2 → 90%+ upgrades:
1. Candidate-specific career roadmap (uses actual skills gap, not generic)
2. Skill gap with current evidence + specific resource per skill
3. Proficiency evidence from raw_text not just structured fields
4. Matched skills verified against resume content — not LLM guesswork
5. Cleaner JSON template with stricter instructions
"""

from __future__ import annotations
import json
import re
from core.schemas import BackendPayload
from core.similarity import compute_proficiency_evidence_scores
from core.doubt_engine import _detect_domain


def build_phase2_prompt(payload: BackendPayload) -> str:
    pr   = payload.parsed_resume
    pjd  = payload.parsed_jd
    sim  = payload.semantic_similarity_score
    tmpl = payload.selected_template
    jd_provided = getattr(pjd, "jd_provided", bool(pjd.raw_text.strip()))

    prof_scores = compute_proficiency_evidence_scores(payload.user_proficiencies, pr)
    domain      = _detect_domain(pjd.target_role)

    # ── Candidate profile summary for context ─────────────────────────────────
    ALWAYS_KEEP = {"c", "r", "go"}
    safe_skills = [s for s in pr.skills if len(s) > 1 or s in ALWAYS_KEEP]

    def match_techs(text: str) -> list:
        tl = text.lower()
        return [s for s in safe_skills if re.search(r"\b" + re.escape(s) + r"\b", tl)]

    # Candidate already has these skills (for roadmap context)
    candidate_has = set(s.lower() for s in pr.skills)

    resume_data = {
        "name":     pr.name or _extract_name(pr.raw_text),
        "email":    pr.email or "",
        "phone":    pr.phone or "",
        "location": pr.location or "",
        "summary":  pr.summary or "",
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
                "description":  p.description[:400],
                "technologies": match_techs(p.description),
            }
            for p in pr.projects
        ],
        "education":       [{"institution": e.institution, "degree": e.degree,
                             "graduation_year": e.graduation_year, "gpa": e.gpa}
                            for e in pr.education],
        "certifications":  [c.name for c in pr.certifications],
        "years_of_experience": pr.years_of_experience,
    }

    # ── Highlight-First Pre-Ranking ──────────────────────────────────────────
    # Pre-compute top 3 role-relevant items for each section
    # Backend ranks by: JD skill overlap + evidence density in description
    jd_skill_set = set(s.lower() for s in pjd.required_skills + pjd.preferred_skills)

    def _relevance_score(text: str) -> int:
        """Count how many JD skills appear in this text."""
        tl = text.lower()
        return sum(1 for s in jd_skill_set if re.search(r"\b" + re.escape(s) + r"\b", tl))

    # Rank projects by JD relevance
    ranked_projects = sorted(
        pr.projects,
        key=lambda p: _relevance_score(p.description + " " + " ".join(p.technologies)),
        reverse=True
    )
    top_projects = ranked_projects[:3]

    # Rank experience by JD relevance
    ranked_exp = sorted(
        pr.experience,
        key=lambda e: _relevance_score(" ".join(e.responsibilities) + " " + e.role),
        reverse=True
    )
    top_exp = ranked_exp[:2]

    # Top matching skills
    top_skills = sorted(
        [s for s in pr.skills if s.lower() in jd_skill_set],
        key=lambda s: 1, reverse=False
    ) or pr.skills[:8]

    highlight_block = f"""=== HIGHLIGHT-FIRST (pre-ranked by backend — use this order) ===
Top role-relevant projects (use these first in resume):
{chr(10).join(f"  {i+1}. {p.title} — {p.description[:80]}" for i, p in enumerate(top_projects))}
Top role-relevant experience:
{chr(10).join(f"  {i+1}. {e.role} at {e.company}" for i, e in enumerate(top_exp)) if top_exp else "  None"}
Top matching skills to lead with: {", ".join(top_skills[:8])}
"""

    # ── JD context block ──────────────────────────────────────────────────────
    if jd_provided:
        jd_keywords = ", ".join(filter(None, pjd.required_skills + pjd.preferred_skills))
        if not jd_keywords:
            jd_words = re.findall(r"[A-Za-z][A-Za-z0-9+#.]{2,}", pjd.raw_text)
            stop = {"the","and","for","with","you","are","our","have","will","this",
                    "that","from","your","they","must","also","role","work"}
            jd_keywords = ", ".join(dict.fromkeys(
                w.lower() for w in jd_words if w.lower() not in stop
            ))[:300]

        # Pre-compute what's missing for targeted roadmap
        required_lower = {s.lower() for s in pjd.required_skills}
        missing_skills = sorted(required_lower - candidate_has)
        already_has    = sorted(required_lower & candidate_has)

        jd_block = f"""=== JOB DESCRIPTION PROVIDED ===
Target Role        : {pjd.target_role}
Domain             : {domain}
Required Skills    : {json.dumps(pjd.required_skills)}
Preferred Skills   : {json.dumps(pjd.preferred_skills)}
Required Exp Years : {pjd.required_experience_years or "Not specified"}
Raw JD             : {pjd.raw_text[:700]}
Similarity Score   : {sim:.1f}/100

CANDIDATE SKILL ANALYSIS (pre-computed):
Already has from JD requirements : {json.dumps(already_has)}
Missing from JD requirements     : {json.dumps(missing_skills)}"""

        match_instruction = f"""job_match_analysis:
- domain: detect from JD — never hardcode
- matched_skills: ONLY skills that appear BOTH in JD requirements AND in candidate resume
  Cross-check against candidate skills list above — do not guess
- missing_skills: skills from JD required list that are ABSENT from candidate profile
  Note: ros2 satisfies ros, python3 satisfies python — don't list satisfied equivalents
- alignment_score_reasoning: specific explanation of {sim:.1f}/100 for THIS candidate vs THIS role
  Mention exact skills matched and exact gaps
- verdict: determined by backend formula — use Strong/Moderate/Weak based on actual overlap"""

        skill_gap_instruction = f"""skill_gap_analysis (3 tiers — BE SPECIFIC TO THIS CANDIDATE):
Use the pre-computed missing skills above. For each missing skill:
- tier_1_critical: required JD skills with ZERO evidence in resume → will get screened out
- tier_2_important: required JD skills with PARTIAL evidence → hiring manager differentiator
- tier_3_nice_to_have: preferred JD skills missing from resume

For each item:
  skill: exact skill name
  gap_severity: High (critical) / Medium (important) / Low (nice to have)
  current_evidence: what the candidate currently has that's RELATED (be honest — "None" if truly absent)
  what_to_do: specific actionable step — name an actual project, course, or tool

career_improvement_plan (CANDIDATE-SPECIFIC — not generic):
Context: This candidate already has: {json.dumps(already_has)}
They are missing: {json.dumps(missing_skills)}
Target role: {pjd.target_role}

- missing_skills_to_learn: only the {json.dumps(missing_skills)} list — already_has skills NOT included
  For each: skill, why_it_matters (specific to {pjd.target_role}), resource (name actual course/tool/dataset)
  
- suggested_projects: 2-3 projects that would DIRECTLY fill the gap
  Must use technologies from missing_skills list
  Must be buildable by someone with: {json.dumps(already_has)}
  Be specific: "Build a sentiment classifier using HuggingFace on a Kaggle dataset" not "ML project"

- learning_roadmap: week-by-week plan starting from candidate's CURRENT level
  Week 1 should assume candidate already knows: {json.dumps(already_has)}
  Name SPECIFIC resources: "fast.ai Practical Deep Learning", "Kaggle ML course", "CS50 AI" etc
  Each week: exact focus area + specific task to complete"""

    else:
        required_lower = set()
        missing_skills = []
        already_has    = sorted(candidate_has)

        jd_block = f"""=== NO JD PROVIDED — ROLE-BASED EVALUATION ===
Target Role    : {pjd.target_role}
Domain         : {domain}
Candidate Has  : {json.dumps(sorted(candidate_has))}
Use your knowledge of what {pjd.target_role} roles require in 2025.
Infer standard required/preferred skills for this role."""

        match_instruction = f"""job_match_analysis:
- domain: infer from "{pjd.target_role}"
- matched_skills: candidate skills relevant to {pjd.target_role} based on industry standards
- missing_skills: skills typically required for {pjd.target_role} absent from candidate profile
- alignment_score_reasoning: how well candidate fits {pjd.target_role} based on their skills
- verdict: Strong / Moderate / Weak"""

        skill_gap_instruction = f"""skill_gap_analysis (3 tiers — SPECIFIC TO THIS CANDIDATE):
Candidate already has: {json.dumps(sorted(candidate_has))}
Identify gaps for {pjd.target_role} based on industry standards.

For each gap item:
  skill: exact name
  gap_severity: High / Medium / Low
  current_evidence: what candidate has that's related (check their skills list — be honest)
  what_to_do: specific step with actual resource name

career_improvement_plan (CANDIDATE-SPECIFIC):
Context: Candidate has {json.dumps(sorted(candidate_has))} and is targeting {pjd.target_role}

- missing_skills_to_learn: top missing skills for {pjd.target_role}
  SKIP any skill already in candidate's list: {json.dumps(sorted(candidate_has))}
  For each: skill, why_it_matters for this role, specific FREE resource to learn it

- suggested_projects: 2-3 SPECIFIC projects buildable by someone with {json.dumps(sorted(candidate_has))}
  Not generic — name the dataset, framework, deployment target
  Example: "Deploy a ROS2 navigation node on Raspberry Pi that avoids obstacles using LIDAR"

- learning_roadmap: week-by-week starting from candidate's ACTUAL current level
  Do NOT teach skills they already have: {json.dumps(sorted(candidate_has))}
  Week 1 = first real gap, not basics they already know
  Name specific courses, tools, datasets for each week"""

    # ── Clarification context ─────────────────────────────────────────────────
    # NOTE: Doubt questions are generated in evaluator.py — not here.
    # prompt_builder just passes clarification answers into the prompt context.
    answered_block = ""
    if payload.clarification_answers:
        answered_block = "\nVERIFIED CANDIDATE ANSWERS (use ONLY these to add confirmed facts):\n" + "\n".join(
            f"  Q: {a.question}\n  A: {a.answer}"
            for a in payload.clarification_answers
        )
    optimization_allowed = True  # evaluator controls blocking — prompt always allows

    # ── Contact ───────────────────────────────────────────────────────────────
    name         = pr.name or _extract_name(pr.raw_text)
    contact_line = " | ".join(filter(None, [pr.phone or "", pr.email or "", pr.location or ""]))

    # ── Resume format rules ───────────────────────────────────────────────────
    resume_instruction = f"""RESUME FORMATTING RULES:
1. Section heading: OBJECTIVE (not Summary/Profile)
2. OBJECTIVE: 2 sentences — who they are + what they seek for {pjd.target_role}. Existing content only.
3. TECHNICAL SKILLS: skill names only — NEVER sentences. Group by category.
   Example: Languages: Python, C, SQL | Frameworks: ROS2
4. EXPERIENCE bullets: past-tense action verb. No generic filler like "collaborated with team".
5. PROJECTS: include ALL projects. Format: Project Name | Tech Stack
6. EDUCATION: include GPA if present. Never write "Not Specified".
7. CERTIFICATIONS: omit section if none exist.
8. NEVER include: Note sections, soft skill paragraphs, "eager to learn" text.
9. NEVER fabricate metrics, tools, or responsibilities.
10. Use clarification answers (if any) to add ONLY confirmed facts."""

    integrity_block = ""
    if answered_block:
        integrity_block = f"""
=== VERIFIED CANDIDATE ANSWERS ===
{answered_block}
RULE: Use these answers ONLY to add confirmed facts. Never embellish or invent.
"""

    prompt = f"""You are Smart Resume — professional resume evaluator and ATS formatter.
Return a single JSON object. No markdown. No extra text outside JSON.

{jd_block}

{highlight_block}

=== BACKEND FACTS ===
Template           : {tmpl.value if hasattr(tmpl, 'value') else str(tmpl)}
Proficiency Scores : {json.dumps(prof_scores)}

=== CANDIDATE RESUME ===
{json.dumps(resume_data, indent=2)}

=== EXPERIENCE vs PROJECT CLASSIFICATION ===
EXPERIENCE = employed role at company/org (internship, job, team member)
PROJECT    = something candidate BUILT (app, model, website, tool, system)
Misclassified items → record in mismatch_corrections.

{integrity_block}

=== EVALUATION INSTRUCTIONS ===
{match_instruction}

{skill_gap_instruction}

proficiency_consistency:
- For each declared proficiency, check evidence in resume content (not just skills list)
- Look at project descriptions, experience bullets, responsibilities for usage evidence
- evidence_level: 0=none, 1=mentioned once, 2=used in project/role, 3=central to work
- Flag gaps between declared level and evidence level

factual_evaluation:
- technical_depth: Deep (complex systems built), Moderate (standard work), Surface (listed only)
- logical_consistency: do dates, roles, and claims align without contradiction?
- metric_realism: are any numbers/percentages realistic and traceable?
- skill_alignment: do claimed skills appear in actual project/experience work?

resume_quality_assessment:
- Be specific to THIS resume — not generic feedback
- improvements: list 3-5 specific actionable changes (e.g. "Add tech stack to BAJA project description")

=== RETURN EXACT JSON ===
{{
  "structured_extraction": {{
    "summary": "2-3 sentence profile from actual resume content",
    "skills": {json.dumps(pr.skills)},
    "experience": [
      {{"role":"","company":"","duration":"","responsibilities":[],"technologies":[]}}
    ],
    "projects": [
      {{"title":"","description":"","technologies":[]}}
    ],
    "education": [{{"institution":"","degree":"","graduation_year":null,"gpa":null}}],
    "mismatch_corrections": []
  }},
  "skill_classification": {{
    "programming_languages": [],
    "frameworks_libraries": [],
    "tools_platforms": [],
    "databases": [],
    "core_cs_concepts": []
  }},
  "job_match_analysis": {{
    "domain": "",
    "alignment_score_reasoning": "specific explanation mentioning exact skills and gaps",
    "matched_skills": [],
    "missing_skills": [],
    "alignment_gaps": [],
    "verdict": "Strong / Moderate / Weak"
  }},
  "skill_gap_analysis": {{
    "tier_1_critical": [
      {{"skill":"","gap_severity":"High","current_evidence":"exact evidence or None","what_to_do":"specific step with resource name"}}
    ],
    "tier_2_important": [
      {{"skill":"","gap_severity":"Medium","current_evidence":"","what_to_do":""}}
    ],
    "tier_3_nice_to_have": [
      {{"skill":"","gap_severity":"Low","current_evidence":"","what_to_do":""}}
    ]
  }},
  "doubt_detection": {{
    "clarification_required": false,
    "issues": []
  }},
  "proficiency_consistency": {{
    "analysis": [
      {{"skill":"","declared_level":"","evidence_level":0,"aligned":true,"reasoning":"specific to this resume"}}
    ],
    "overall_assessment": ""
  }},
  "factual_evaluation": {{
    "technical_depth": "Deep / Moderate / Surface",
    "logical_consistency": "Consistent / Minor Issues / Inconsistent",
    "metric_realism": "Realistic / Unverifiable / No Metrics",
    "skill_alignment": "Strong / Partial / Weak",
    "confidence_narrative": "1 sentence specific to this candidate"
  }},
  "internal_consistency": {{
    "timeline_alignment": "OK / Issues Found",
    "skill_usage_coverage": "High / Medium / Low",
    "claim_density": "Appropriate / High / Low",
    "cross_section_coherence": "Coherent / Minor Issues / Inconsistent",
    "flags": []
  }},
  "resume_quality_assessment": {{
    "structure_clarity": "High / Medium / Low",
    "ats_compatibility": "High / Medium / Low",
    "role_relevance": "High / Medium / Low",
    "redundancy_level": "Low / Medium / High",
    "overall_quality": "Strong / Acceptable / Needs Work",
    "improvements": ["specific actionable improvement 1", "specific actionable improvement 2"]
  }},
  "template_selection": {{
    "selected": "{tmpl.value if hasattr(tmpl, "value") else str(tmpl)}",
    "justification": "1 sentence"
  }},
  "final_resume": null,
  "career_improvement_plan": {{
    "target_domain": "",
    "missing_skills_to_learn": [
      {{"skill":"","reason":"why it matters for this specific role","resource":"specific course/tool/dataset name"}}
    ],
    "suggested_projects": [
      {{"title":"specific project name","description":"what to build, which dataset/tool, expected outcome"}}
    ],
    "learning_roadmap": [
      {{"week":"Week 1","focus":"specific skill","task":"exact thing to do","resource":"specific resource name"}}
    ]
  }}
}}

=== ABSOLUTE RULES ===
1. final_resume: always null — generated separately
2. verdict: exactly Strong, Moderate, or Weak — no other values
3. technical_depth: exactly Deep, Moderate, or Surface
4. skill_gap what_to_do: MUST name a specific resource (course, tool, platform) — never say "learn X"
5. career_improvement_plan: MUST be specific to this candidate — never give generic advice
6. matched_skills: ONLY skills that appear in BOTH JD requirements AND candidate resume
7. learning_roadmap: starts from candidate's actual level — never re-teach skills they already have
8. All fields must reference THIS candidate's actual content — zero generic statements
"""
    return prompt


def _extract_name(raw_text: str) -> str:
    for line in raw_text.splitlines():
        line = line.strip()
        if (line
                and len(line.split()) >= 2
                and len(line.split()) <= 6
                and not re.search(r"[@\d\+\-\|\\/@]", line)):
            return line
    return "Candidate"