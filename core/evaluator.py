"""
Smart Resume — Evaluator Engine v4
=====================================
Two-call strategy:
  Call 1: Full evaluation JSON (no resume — focused)
  Call 2: Dedicated resume generation (plain text, full attention)
"""

from __future__ import annotations
import json
import logging
import re
import time
from typing import Optional

import httpx

from core.schemas import BackendPayload, SmartResumeResponse
from core.llm_client import call_llm, call_llm_text, call_llm_json
from core.prompt_builder import build_phase2_prompt
from core.llm_validator import validate_and_fix, needs_retry

logger = logging.getLogger("smart_resume.evaluator")

# ── LLM Provider config ───────────────────────────────────────────────────────
# Primary: Groq (fast, free tier — rate limits apply)
# Fallback: Together AI (also free, different quota)
# Both use OpenAI-compatible API format
GROQ_API_URL     = "https://api.groq.com/openai/v1/chat/completions"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

GROQ_MODEL       = "llama-3.3-70b-versatile"
GROQ_MODEL_FAST  = "llama-3.1-8b-instant"
TOGETHER_MODEL   = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"  # free on Together

MODEL            = GROQ_MODEL   # default, may be overridden per-call
MAX_RETRIES      = 2

import os as _os

def _get_all_groq_keys(primary_key: str) -> list:
    keys = [k for k in [primary_key] + [
        _os.getenv(f"GROQ_API_KEY_{i}", "") for i in range(2, 6)
    ] if k and k.strip()]
    return keys

def _get_together_key() -> str:
    return _os.getenv("TOGETHER_API_KEY", "").strip()


# ── Main entry ────────────────────────────────────────────────────────────────

def evaluate(payload: BackendPayload, api_key: str) -> SmartResumeResponse:
    from core.doubt_engine import generate_doubt_questions, all_required_answered

    # Step 1: Skip JD re-parsing if skills already populated from app.py
    payload = _upgrade_jd_parsing(payload, api_key)

    # Step 2: LLM-powered doubt detection
    # CRITICAL: If user has already submitted clarification answers, skip re-generating
    # questions entirely — treat as answered. Re-generating produces different question text
    # which never matches the stored answers, permanently blocking resume generation.
    # clarification_answers = None  → first run, generate doubt questions
    # clarification_answers = [...]  → user answered, skip doubt gate
    # clarification_answers = []     → user clicked "Skip", proceed without answers
    if payload.clarification_answers is not None:
        # User either answered or explicitly skipped — proceed directly
        doubt_issues      = []
        required_answered = True
        unanswered        = []
        logger.info("Clarification answers set (%d answers) — skipping doubt gate",
                    len(payload.clarification_answers))
    else:
        doubt_issues = generate_doubt_questions(
            payload.parsed_resume,
            payload.user_proficiencies,
            payload.parsed_jd.target_role,
            api_key=api_key,
        )
        required_answered, unanswered = all_required_answered(
            doubt_issues, payload.clarification_answers
        )

    # Step 2b: Backend internal consistency check
    consistency_flags = _check_internal_consistency(payload)

    # Step 3: Build prompt and run two-call strategy
    from core.similarity import compute_weighted_score
    pr  = payload.parsed_resume
    pjd = payload.parsed_jd
    prompt = build_phase2_prompt(payload)
    data   = _call_with_retry(prompt, api_key, payload.needs_optimization, payload)

    # Step 4: Override verdict with deterministic weighted score (uses LLM matched_skills)
    jma     = data.get("job_match_analysis", {})
    matched = jma.get("matched_skills", [])
    weighted = compute_weighted_score(
        similarity_score          = payload.semantic_similarity_score,
        matched_skills            = matched,
        required_skills           = pjd.required_skills,
        years_of_experience       = pr.years_of_experience,
        required_experience_years = pjd.required_experience_years,
    )
    jma["verdict"]                   = weighted["verdict"]
    jma["weighted_score"]            = weighted["score"]
    jma["score_breakdown"]           = weighted["breakdown"]
    data["job_match_analysis"]       = jma

    # Step 5: Inject backend consistency flags
    if consistency_flags:
        ic = data.get("internal_consistency", {})
        existing_flags = ic.get("flags", [])
        ic["flags"] = existing_flags + consistency_flags
        if any("overlap" in f.lower() or "conflict" in f.lower() for f in consistency_flags):
            ic["timeline_alignment"] = "Issues Found"
        data["internal_consistency"] = ic

    # Step 6: Build response
    # Build response directly from dict
    result = SmartResumeResponse(
        session_id                = payload.session_id,
        structured_extraction     = data.get("structured_extraction", {}),
        skill_classification      = data.get("skill_classification", {}),
        job_match_analysis        = data.get("job_match_analysis", {}),
        doubt_detection           = data.get("doubt_detection", {}),
        proficiency_consistency   = data.get("proficiency_consistency", {}),
        factual_evaluation        = data.get("factual_evaluation", {}),
        internal_consistency      = data.get("internal_consistency", {}),
        resume_quality_assessment = data.get("resume_quality_assessment", {}),
        template_selection        = data.get("template_selection", {}),
        final_resume              = data.get("final_resume"),
        career_improvement_plan   = data.get("career_improvement_plan", {}),
        skill_gap_analysis        = data.get("skill_gap_analysis", {}),
        clarification_required    = data.get("clarification_required", False),
        clarification_questions   = data.get("clarification_questions", []),
        mismatch_corrections      = data.get("mismatch_corrections", []),
    )

    # Step 7: Override clarification with deterministic results
    if unanswered:
        result.clarification_required  = True
        result.clarification_questions = [i.question for i in unanswered]
        if not required_answered:
            result.final_resume = None

    return result


# ── Internal consistency checker ──────────────────────────────────────────────

def _check_internal_consistency(payload) -> list[str]:
    """
    Backend-enforced consistency checks that LLM cannot override.
    Returns list of flag strings to inject into internal_consistency.flags.
    """
    flags = []
    pr  = payload.parsed_resume

    # 1. Graduation year vs years_of_experience cross-check
    grad_year = None
    for edu in pr.education:
        if edu.graduation_year:
            try:
                yr = int(str(edu.graduation_year)[:4])
                if 2000 <= yr <= 2030:
                    grad_year = yr
                    break
            except (ValueError, TypeError):
                pass

    if grad_year and pr.years_of_experience > 0:
        from datetime import datetime
        current_year = datetime.now().year
        max_possible_exp = current_year - grad_year
        # Allow 2 years buffer for internships before graduation
        if pr.years_of_experience > max_possible_exp + 2:
            flags.append(
                f"Experience claim ({pr.years_of_experience:.1f} years) exceeds "
                f"what is possible since graduation year {grad_year} "
                f"({max_possible_exp} years max)"
            )

    # 2. Skill count vs evidence density
    total_skills    = len(pr.skills)
    evidence_text   = " ".join(
        " ".join(e.responsibilities) for e in pr.experience
    ) + " " + " ".join(p.description for p in pr.projects)
    evidenced_count = sum(
        1 for s in pr.skills
        if re.search(r"(?<![a-z0-9])" + re.escape(s.lower()) + r"(?![a-z0-9])", evidence_text.lower())
    )
    if total_skills > 5:
        coverage_pct = evidenced_count / total_skills
        if coverage_pct < 0.4:
            flags.append(
                f"Low skill evidence coverage: only {evidenced_count}/{total_skills} "
                f"listed skills ({coverage_pct:.0%}) appear in project or experience descriptions"
            )

    # 3. Future graduation date with work experience claims
    if grad_year:
        from datetime import datetime
        current_year = datetime.now().year
        if grad_year > current_year and pr.years_of_experience > 1:
            flags.append(
                f"Candidate has future graduation ({grad_year}) but claims "
                f"{pr.years_of_experience:.1f} years of experience — verify dates"
            )

    # 4. Project count vs project descriptions — detect empty shells
    thin_projects = [
        p.title for p in pr.projects
        if len(p.description.strip().split()) < 5
    ]
    if thin_projects:
        flags.append(
            f"Thin project descriptions (under 5 words): {', '.join(thin_projects[:3])}"
        )

    return flags


# ── JD upgrade ────────────────────────────────────────────────────────────────

def _upgrade_jd_parsing(payload: BackendPayload, api_key: str) -> BackendPayload:
    from core.jd_parser_llm import parse_jd_with_llm
    from core.schemas import Domain

    if payload.parsed_jd.required_skills:
        logger.info("JD already has %d skills — skipping re-parse",
                    len(payload.parsed_jd.required_skills))
        return payload

    raw_jd = payload.parsed_jd.raw_text
    if len(raw_jd.strip()) < 20:
        return payload

    try:
        llm_jd = parse_jd_with_llm(raw_jd, api_key)
        if not llm_jd:
            return payload

        pjd = payload.parsed_jd
        domain_map = {
            "ml": Domain.ML, "backend": Domain.BACKEND,
            "frontend": Domain.FRONTEND, "devops": Domain.DEVOPS,
            "data": Domain.DATA, "unknown": Domain.UNKNOWN,
        }
        domain = domain_map.get(llm_jd.get("domain", "unknown").lower(), Domain.UNKNOWN)

        req_skills  = [s.lower() for s in llm_jd.get("required_skills", [])]
        pref_skills = [s.lower() for s in llm_jd.get("preferred_skills", [])]

        updated_jd = pjd.model_copy(update={
            "detected_domain":           domain,
            "required_skills":           req_skills  or pjd.required_skills,
            "preferred_skills":          pref_skills or pjd.preferred_skills,
            "key_responsibilities":      llm_jd.get("key_responsibilities", pjd.key_responsibilities),
            "required_experience_years": llm_jd.get("required_experience_years", pjd.required_experience_years),
        })
        return payload.model_copy(update={"parsed_jd": updated_jd})

    except Exception as e:
        logger.warning("JD upgrade failed: %s", e)
        return payload


# ── Two-call strategy ─────────────────────────────────────────────────────────

def _call_with_retry(prompt: str, api_key: str, needs_opt: bool, payload: "BackendPayload") -> dict:
    """
    Call 1: Evaluation JSON only (final_resume = null).
    Call 2: Dedicated plain-text resume generation — payload passed directly, no regex extraction.
    """
    # Call 1 — evaluation only
    eval_note = "\n\nIMPORTANT: Set final_resume to null. Resume will be generated separately."
    data = call_llm_json(prompt + eval_note, api_key, max_tokens=2000)
    data, fixes = validate_and_fix(data, needs_optimization=False)

    if fixes:
        logger.info("Validator applied %d fixes", len(fixes))

    # Call 2 — dedicated resume generation (payload passed directly — no fragile regex extraction)
    if needs_opt:
        logger.info("Running dedicated resume generation...")
        try:
            resume_text = _generate_resume(payload, api_key)
        except Exception as _gen_err:
            logger.warning("LLM resume gen raised: %s — falling back to rule-based", _gen_err)
            resume_text = None
        if resume_text:
            from core.resume_verifier import verify_resume
            # Build verification dict from payload directly — no regex needed
            pr = payload.parsed_resume
            resume_data_for_verify = {
                "skills":     pr.skills,
                "experience": [e.model_dump() for e in pr.experience],
                "projects":   [p.model_dump() for p in pr.projects],
                "summary":    pr.summary or "",
            }
            # Pass full ClarificationAnswer objects so verifier can extract denied techs
            verified_text, issues = verify_resume(resume_text, resume_data_for_verify, payload.clarification_answers or [])
            if issues:
                logger.info("Verifier fixed %d issues in generated resume", len(issues))
            data["final_resume"]         = verified_text
            data["resume_verify_issues"] = issues
            logger.info("Resume generated and verified: %d chars", len(verified_text))
        else:
            logger.warning("Resume generation returned None — final_resume will be null")
            data["final_resume"] = None
            data["resume_verify_issues"] = ["Resume generation failed — try again in 30 seconds"]

    return data


# ── Dedicated resume generator ────────────────────────────────────────────────

def _generate_resume_rule_based(payload: "BackendPayload") -> str:
    """
    Zero-LLM resume generator — produces clean ATS resume from structured data.
    Used as instant fallback when Groq is rate-limited.
    Applies: strong action verbs, per-entry tech locking, denial filtering.
    """
    pr  = payload.parsed_resume
    pjd = payload.parsed_jd

    name     = pr.name or "Candidate"
    phone    = pr.phone or ""
    email    = pr.email or ""
    location = pr.location or ""
    linkedin = getattr(pr, "linkedin", None) or ""
    github   = getattr(pr, "github",   None) or ""
    role     = pjd.target_role if pjd else "Software Engineer"

    # Denied techs from answers
    NIL_WORDS = {"nil","none","no","nothing","not used","didn't use","did not use",
                 "not applicable","n/a","na","i didn't","i did not"}
    denied = set()
    confirmed = set()
    for ca in (payload.clarification_answers or []):
        ans = (ca.answer if hasattr(ca, "answer") else str(ca)).strip().lower()
        if ans in NIL_WORDS or len(ans) <= 3:
            continue
        # Extract denials
        for pat in [r"(\w[\w+#]*)\s+is\s+not\s+used", r"(\w[\w+#]*)\s+not\s+used",
                    r"didn.t\s+use\s+(\w[\w+#]*)", r"did\s+not\s+use\s+(\w[\w+#]*)"]:
            for m in re.finditer(pat, ans):
                denied.add(m.group(1).strip())
        # Extract confirmed (positive)
        for sk in (pr.skills or []):
            import re as _re2
            if _re2.search(r"(?<![a-zA-Z0-9])" + _re2.escape(sk.lower()) + r"(?![a-zA-Z0-9])", ans)                and sk.lower() not in denied:
                confirmed.add(sk)

    skills = [s for s in (pr.skills or []) if s.lower() not in denied]
    confirmed_extra = confirmed - denied

    # Action verb map
    VERBS = {
        "develop": "Developed", "build": "Built", "create": "Created",
        "implement": "Implemented", "design": "Designed", "work": "Contributed to",
        "aim": "Working on", "simulate": "Simulated", "test": "Tested",
        "engineer": "Engineered", "manage": "Managed", "lead": "Led",
        "use": "Utilized", "write": "Wrote", "deploy": "Deployed",
    }

    def strengthen_bullet(text: str) -> str:
        t = text.strip().rstrip(".")
        first_word = t.split()[0].lower() if t.split() else ""
        for trigger, verb in VERBS.items():
            if first_word.startswith(trigger):
                rest = " ".join(t.split()[1:])
                return f"{verb} {rest}."
        # If no match, capitalize first letter
        return t[0].upper() + t[1:] + ("." if not t.endswith(".") else "")

    def lock_tech(text: str, allowed: set, denied_set: set) -> str:
        """Remove tech names not in allowed set from text."""
        # Simple: remove denied terms
        for term in denied_set:
            text = re.sub(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])",
                          "", text, flags=re.IGNORECASE)
        return re.sub(r"\s{2,}", " ", text).strip().rstrip(",").strip()

    lines = []

    # Header
    contact_parts = [p for p in [phone, email, location] if p]
    if linkedin: contact_parts.append(linkedin)
    if github:   contact_parts.append(github)
    lines.append(name)
    if contact_parts:
        lines.append("  |  ".join(contact_parts))
    lines.append("")

    # Objective — 2 sentences, rule-based
    top_skills = [s for s in skills if s.lower() not in {"teamwork","leadership","time management","project management"}][:3]
    skills_str = ", ".join(top_skills) if top_skills else "software development"
    lines.append("OBJECTIVE")
    lines.append(
        f"B.Tech Computer Science student with hands-on experience in {skills_str}, "
        f"focused on practical implementation and problem-solving. "
        f"Seeking to apply technical skills in a {role} role to deliver efficient, data-driven solutions."
    )
    lines.append("")

    # Technical Skills
    LANG_SET  = {"python","java","c","c++","c#","javascript","typescript","go","rust","sql","r","matlab","bash","dart","scala","swift","kotlin"}
    FWK_SET   = {"ros2","ros","react","vue","django","flask","fastapi","pytorch","tensorflow","keras","angular","spring","flutter","langchain","nextjs","express"}
    DB_SET    = {"sql","postgresql","mysql","sqlite","mongodb","redis","firebase","cassandra","dynamodb","elasticsearch"}
    TOOL_SET  = {"docker","kubernetes","git","github","linux","aws","gcp","azure","jenkins","terraform","grafana","postman"}

    langs = [s for s in skills if s.lower() in LANG_SET and s.lower() != "sql"]
    fwks  = [s for s in skills if s.lower() in FWK_SET]
    dbs   = [s for s in skills if s.lower() in DB_SET]
    tools = [s for s in skills if s.lower() in TOOL_SET]
    other = [s for s in skills if s.lower() not in LANG_SET|FWK_SET|DB_SET|TOOL_SET
             and s.lower() not in {"teamwork","leadership","time management","project management"}]

    lines.append("TECHNICAL SKILLS")
    if langs:  lines.append(f"Languages  : {', '.join(langs)}")
    if fwks:   lines.append(f"Frameworks : {', '.join(fwks)}")
    if dbs:    lines.append(f"Databases  : {', '.join(dbs)}")
    if tools:  lines.append(f"Tools      : {', '.join(tools)}")
    if other:  lines.append(f"Other      : {', '.join(other)}")
    lines.append("")

    # Experience
    if pr.experience:
        lines.append("EXPERIENCE")
        for exp in pr.experience:
            duration = exp.duration or "Present"
            lines.append(f"{exp.role} | {exp.company} | {duration}")
            locked = set(exp.technologies or [])
            for sk in skills:
                if sk.lower() in " ".join(exp.responsibilities).lower():
                    locked.add(sk)
            locked.update(confirmed_extra)
            locked -= denied
            for r in exp.responsibilities:
                bullet = strengthen_bullet(r)
                bullet = lock_tech(bullet, locked, denied)
                if bullet:
                    lines.append(f"- {bullet}")
        lines.append("")

    # Projects
    if pr.projects:
        lines.append("PROJECTS")
        for proj in pr.projects:
            locked = set(proj.technologies or [])
            for sk in skills:
                if sk.lower() in (proj.description or "").lower():
                    locked.add(sk)
            locked.update(confirmed_extra)
            locked -= denied
            tech_str = ", ".join(sorted(locked)) if locked else ""
            title_line = proj.title
            if tech_str:
                title_line += f" | Tech: {tech_str}"
            lines.append(title_line)
            if proj.description:
                desc = strengthen_bullet(proj.description)
                desc = lock_tech(desc, locked, denied)
                if desc:
                    lines.append(f"- {desc}")
        lines.append("")

    # Education
    if pr.education:
        lines.append("EDUCATION")
        for edu in pr.education:
            parts = [edu.degree, edu.institution]
            if edu.graduation_year:
                yr = str(edu.graduation_year)
                parts.append(f"Expected: {yr}" if int(edu.graduation_year) > 2025 else yr)
            if edu.gpa:
                gpa_raw = str(edu.gpa).upper().replace("CGPA:", "").replace("GPA:", "").strip()
                if gpa_raw:
                    parts.append(f"CGPA: {gpa_raw}")
            lines.append(" | ".join(filter(None, parts)))
        lines.append("")

    # Certifications
    if pr.certifications:
        lines.append("CERTIFICATIONS")
        for c in pr.certifications:
            parts = [c.name, c.issuer or ""]
            if c.year: parts.append(str(c.year))
            lines.append("- " + " | ".join(p for p in parts if p))

    return "\n".join(lines)


def _generate_resume(payload: "BackendPayload", api_key: str) -> Optional[str]:
    """
    Micro-rewrite architecture — zero hallucination by design.

    Structure is built 100% deterministically in Python from parsed facts.
    LLM is called only for:
      1. Rewriting individual bullets (one at a time, tightly scoped)
      2. Writing the objective (given only verified facts + target role)

    The LLM never sees the full resume. It never controls structure.
    It cannot add a section, invent a skill, or fabricate a project.
    """
    from core.llm_client import call_llm, call_llm_text, call_llm_json

    pr  = payload.parsed_resume
    pjd = payload.parsed_jd

    # ── Contact ─────────────────────────────────────────────────────────────
    name     = pr.name or "Candidate"
    email    = pr.email or ""
    phone    = pr.phone or ""
    location = pr.location or ""
    linkedin = getattr(pr, "linkedin", None) or ""
    github   = getattr(pr, "github",   None) or ""
    contact_parts = [p for p in [phone, email, location, linkedin, github] if p]
    contact_line  = "  |  ".join(contact_parts)

    target_role = pjd.target_role or "Software Engineer"
    jd_required = ", ".join(pjd.required_skills[:10]) if pjd.required_skills else ""
    skills      = pr.skills or []
    logger.info("Skills from parser: %s", skills)

    # ── Skills section (100% deterministic) ─────────────────────────────────
    LANG_TERMS = {"python","java","javascript","typescript","c","c++","c#","go","rust",
                  "kotlin","swift","r","scala","ruby","php","bash","dart","matlab"}
    FWK_TERMS  = {"ros2","ros","react","vue","angular","nextjs","django","flask","fastapi",
                  "spring","pytorch","tensorflow","keras","express","flutter","langchain"}
    DB_TERMS   = {"sql","postgresql","mysql","sqlite","mongodb","redis","firebase",
                  "cassandra","dynamodb","elasticsearch"}
    TOOL_TERMS = {"docker","kubernetes","git","github","linux","aws","gcp","azure",
                  "jenkins","terraform","grafana","postman"}
    classified  = set()
    langs, fwks, dbs, tools = [], [], [], []
    for s in skills:
        sl = s.lower()
        if   sl in LANG_TERMS: langs.append(s);  classified.add(sl)
        elif sl in FWK_TERMS:  fwks.append(s);   classified.add(sl)
        elif sl in DB_TERMS:   dbs.append(s);    classified.add(sl)
        elif sl in TOOL_TERMS: tools.append(s);  classified.add(sl)
    # Unclassified → languages bucket
    langs += [s for s in skills if s.lower() not in classified]
    skill_lines = []
    if langs:  skill_lines.append(f"Languages  : {', '.join(langs)}")
    if fwks:   skill_lines.append(f"Frameworks : {', '.join(fwks)}")
    if dbs:    skill_lines.append(f"Databases  : {', '.join(dbs)}")
    if tools:  skill_lines.append(f"Tools      : {', '.join(tools)}")
    skills_block = "\n".join(skill_lines) if skill_lines else f"Languages  : {', '.join(skills)}"

    # ── Parse answers ────────────────────────────────────────────────────────
    NIL_WORDS = {"nil","none","no","nothing","not used","didn't use","did not use",
                 "not applicable","n/a","na","not mentioned","not related","never used"}
    confirmed_by_entry = {}   # {entry_name_lower: set_of_techs}
    confirmed_global   = set()
    global_denied      = set()
    answers_summary    = []   # human-readable for LLM context

    KNOWN_TECH = {s.lower() for s in skills} | {
        "ros2","ros","gazebo","python","sql","c","git","linux","docker",
        "pytorch","tensorflow","opencv","flask","django","fastapi","react",
        "mongodb","postgresql","redis","aws","gcp","azure","kubernetes",
        "node.js","express","spring","matlab","arduino","raspberry pi",
    }
    all_entry_names = (
        [e.role.lower()    for e in pr.experience] +
        [e.company.lower() for e in pr.experience] +
        [p.title.lower()   for p in pr.projects]
    )

    def _wb(skill, txt):
        """Word-boundary safe match — prevents 'c' matching 'connect'."""
        pat = r"(?<![a-zA-Z0-9])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9])"
        return bool(re.search(pat, txt.lower()))

    if payload.clarification_answers:
        try:
            from core.parser import KNOWN_TECH_TERMS as _KTT
        except Exception:
            _KTT = KNOWN_TECH

        for ca in payload.clarification_answers:
            ans = ca.answer.strip()
            if not ans or ans.lower() in NIL_WORDS or len(ans) <= 3:
                answers_summary.append(f"Q: {ca.question}\nA: [candidate had nothing to add]")
                continue

            # Detect denials
            denied_here = set()
            for pat in [
                r"(\w[\w+#]*)\s+is\s+not\s+used",
                r"(\w[\w+#]*)\s+not\s+used",
                r"didn['']t\s+use\s+(\w[\w+#]*)",
                r"did\s+not\s+use\s+(\w[\w+#]*)",
                r"(\w[\w+#]*)\s+(?:is\s+)?only\s+for\s+(?:data structures|lab|practice|course)",
                r"(\w[\w+#]*)\s+for\s+data\s+structures",
            ]:
                for m in re.finditer(pat, ans.lower()):
                    g = m.group(1).strip()
                    if g: denied_here.add(g)
            global_denied.update(denied_here)

            # Detect confirmed techs
            found = set()
            for term in _KTT:
                if _wb(term, ans):
                    denied_ctx = any(re.search(dp, ans.lower()) for dp in [
                        r"not.*" + re.escape(term),
                        re.escape(term) + r".*not used",
                        r"didn.t.*" + re.escape(term),
                    ])
                    if not denied_ctx:
                        found.add(term)
            found -= global_denied

            # Map to entry
            q_lower = ca.question.lower()
            best_entry = None
            for name in all_entry_names:
                if name and len(name) > 2 and name in q_lower:
                    if best_entry is None or len(name) > len(best_entry):
                        best_entry = name
            if best_entry and found:
                confirmed_by_entry.setdefault(best_entry, set()).update(found)
            elif found:
                confirmed_global.update(found)

            answers_summary.append(f"Q: {ca.question}\nA: {ans}")

    answers_context = "\n\n".join(answers_summary) if answers_summary else "None."

    # ── Helper: get allowed tech for an entry ───────────────────────────────
    def _get_allowed(entry_names_lower: set, context_text: str) -> set:
        allowed = set()
        # From global skills — word-boundary match in context
        for sk in skills:
            if _wb(sk, context_text):
                allowed.add(sk)
        # From confirmed answers for this entry
        for en in entry_names_lower:
            allowed.update(confirmed_by_entry.get(en, set()))
        # From global confirmed — only if in context
        for sk in confirmed_global:
            if _wb(sk, context_text):
                allowed.add(sk)
        allowed -= global_denied
        return allowed

    # ── Helper: call LLM for ONE bullet rewrite ──────────────────────────────
    COMMON_HALLUCINATIONS = {
        "flask","django","fastapi","spring","express","react","angular","vue",
        "pytorch","tensorflow","scikit-learn","opencv","docker","kubernetes",
        "aws","gcp","azure","redis","mongodb","next.js","nextjs","laravel",
        "node.js","nodejs","bootstrap","tailwind","jquery","hadoop","spark",
        "kafka","elasticsearch","cassandra","firebase","supabase",
    }

    # ── Pre-extract structured facts from every answer ──────────────────────
    # Build: {entry_alias_lower: [list of clean fact strings]}
    # This runs ONCE and gives _rewrite_bullet precise facts, not raw Q&A.
    # ── Confidence scoring for answer facts ─────────────────────────────────
    _HEDGE_PATTERNS = [
        r"\bi (might|may|think|believe|guess) (have )?(used|worked|done|tried)",
        r"\bnot (sure|certain|remember)",
        r"\bprobably\b", r"\bpossibly\b", r"\bmaybe\b",
        r"\bkind of\b", r"\bsomething like\b",
        r"\bonly (for|during) (lab|practice|course|class|assignment)",
        r"\bjust (for|to) (learn|try|test|practice)",
    ]
    def _get_confidence(sentence: str) -> str:
        sl = sentence.lower()
        if any(re.search(p, sl) for p in _HEDGE_PATTERNS):
            return "LOW"
        if re.search(r"\b(used|built|developed|implemented|designed|created|wrote|deployed)\b", sl):
            return "HIGH"
        return "MEDIUM"

    _entry_fact_map = {}   # {alias: ["fact 1", "fact 2", ...]}
    _global_facts   = []   # facts not tied to a specific entry

    # Aliases: every way an entry might be referenced in a question
    _entry_aliases = {}    # {alias_lower: canonical_entry_name}
    for exp in pr.experience:
        for token in [exp.role, exp.company] + exp.company.split():
            t_lower = token.lower().strip()
            if len(t_lower) > 2:
                _entry_aliases[t_lower] = exp.company
    for proj in pr.projects:
        for token in [proj.title] + proj.title.split():
            t_lower = token.lower().strip()
            if len(t_lower) > 2:
                _entry_aliases[t_lower] = proj.title

    def _extract_facts_from_answer(question: str, answer: str) -> tuple[str | None, list[str]]:
        """
        Returns (matched_entry_name_or_None, [fact_string, ...]).
        Facts are clean sentences extracted from the answer.
        """
        q_lower = question.lower()
        ans     = answer.strip()

        # Find which entry this question is about
        matched_entry = None
        best_match_len = 0
        for alias, canonical in _entry_aliases.items():
            if alias in q_lower and len(alias) > best_match_len:
                matched_entry  = canonical
                best_match_len = len(alias)

        # Extract facts with confidence scoring
        raw_facts = re.split(r"[.;\n]+", ans)
        facts = []
        for f in raw_facts:
            f = f.strip()
            if not f or len(f) < 5:
                continue
            if f.lower() in NIL_WORDS:
                continue
            if re.search(r"\b(not used|didn.t use|never used|not applicable|only for)\b", f.lower()):
                continue
            # Short comma-separated lists → split into individual facts
            if "," in f and len(f) < 60:
                for sp in [p.strip() for p in f.split(",") if len(p.strip()) > 3]:
                    facts.append((sp, _get_confidence(sp)))
            else:
                facts.append((f, _get_confidence(f)))

        return matched_entry, facts

    for ca in (payload.clarification_answers or []):
        ans = ca.answer.strip()
        if not ans or ans.lower() in NIL_WORDS or len(ans) <= 3:
            continue
        entry, fact_tuples = _extract_facts_from_answer(ca.question, ans)
        if not fact_tuples:
            continue
        if entry:
            _entry_fact_map.setdefault(entry.lower(), []).extend(fact_tuples)
        else:
            _global_facts.extend(fact_tuples)

    def _get_entry_answers(entry_name_lower: str, extra_keys: set = None) -> str:
        """
        Return structured facts for this entry as a clean list.
        Matches on entry name, all aliases, and any extra_keys (e.g. parent exp names).
        Falls back to global facts if no entry-specific facts found.
        """
        search_keys = {entry_name_lower} | (extra_keys or set())

        # Expand via aliases
        for alias, canonical in _entry_aliases.items():
            if canonical.lower() in search_keys or any(k in alias for k in search_keys):
                search_keys.add(canonical.lower())
                search_keys.add(alias)

        raw = []
        for key in search_keys:
            raw += _entry_fact_map.get(key, [])
        raw += _global_facts

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for item in raw:
            # Support both plain strings (legacy) and (fact, conf) tuples
            fact = item[0] if isinstance(item, tuple) else item
            conf = item[1] if isinstance(item, tuple) else "MEDIUM"
            fl   = fact.lower().strip()
            if fl not in seen:
                seen.add(fl)
                unique.append((fact, conf))

        if not unique:
            entry_words = [w for w in entry_name_lower.split() if len(w) > 3]
            for item in _global_facts:
                fact = item[0] if isinstance(item, tuple) else item
                conf = item[1] if isinstance(item, tuple) else "MEDIUM"
                if any(w in fact.lower() for w in entry_words):
                    unique.append((fact, conf))

        if not unique:
            return ""

        # Sort: HIGH first, then MEDIUM, then LOW — LOW facts labelled as uncertain
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        unique.sort(key=lambda x: order.get(x[1], 1))

        lines = []
        for fact, conf in unique[:6]:  # max 6 facts
            if conf == "LOW":
                lines.append(f"- [UNCERTAIN — use only if no better fact] {fact}")
            elif conf == "HIGH":
                lines.append(f"- [CONFIRMED] {fact}")
            else:
                lines.append(f"- {fact}")
        return "\n".join(lines)

    def _rewrite_bullet(original: str, allowed: set, is_ongoing: bool,
                        role_context: str, entry_name: str,
                        extra_keys: set = None) -> str:
        """Single bullet rewrite — used by verifier feedback loop only."""
        allowed_clean = {t for t in allowed
                         if t.lower() not in COMMON_HALLUCINATIONS
                         or t.lower() in {s.lower() for s in skills}}
        allowed_str = ", ".join(sorted(allowed_clean)) if allowed_clean else "none — omit all tech names"
        tense_rule  = ("Present progressive (Developing, Building, Working on, Implementing)"
                       if is_ongoing else
                       "Past tense (Developed, Built, Implemented, Designed)")

        # Get facts for this entry — also search parent experience entries (extra_keys)
        entry_answers = _get_entry_answers(entry_name.lower(), extra_keys=extra_keys)
        extra_facts   = f"\nCANDIDATE ADDED THESE FACTS ABOUT THIS ENTRY:\n{entry_answers}" if entry_answers else ""

        # Build the enriched original — merge answer facts directly into original text
        # This makes them first-class facts, not optional hints the LLM can ignore
        enriched_original = original
        facts_block = ""
        if entry_answers:
            facts_block = f"""
VERIFIED FACTS FROM CANDIDATE (treat these as equally real as ORIGINAL):
{entry_answers}
INSTRUCTION: Your bullet MUST incorporate the most specific, relevant detail from above.
The original text alone is incomplete — the candidate added these facts to fill the gaps."""

        prompt = f"""You are a resume writer. Rewrite this ONE bullet into a single strong professional sentence.

ORIGINAL: {original}{facts_block}

CONSTRAINTS:
- Entry: {entry_name}
- Target role: {target_role}
- Tense: {tense_rule}
- ALLOWED TECH (never name anything outside this): {allowed_str}

HOW TO WRITE IT:
1. Start from ORIGINAL facts. If VERIFIED ADDITIONAL FACTS exist, add the most relevant specific detail.
2. Use the strongest action verb for the tense.
3. Name the actual system/platform (e.g. "Alumni Connect Platform", "BAJA autonomous vehicle").
4. Never add tools, libraries, or achievements not in ORIGINAL or VERIFIED FACTS.
5. Max 22 words. Output the sentence only — no dash, no prefix."""

        result = call_llm_text(prompt, api_key, max_tokens=80)
        if not result or len(result.strip()) < 5:
            return original

        bullet = result.strip().lstrip("-•* ").strip()
        # Hard enforce: strip hallucinated tech
        for word in COMMON_HALLUCINATIONS:
            if word.lower() not in {t.lower() for t in allowed_clean}:
                bullet = re.sub(
                    r"(?<![a-zA-Z0-9])" + re.escape(word) + r"(?![a-zA-Z0-9])",
                    "", bullet, flags=re.IGNORECASE
                ).strip(" ,.")
        # Clean up double spaces/commas left by stripping
        bullet = re.sub(r"  +", " ", bullet)
        bullet = re.sub(r" ,", ",", bullet).strip(" ,.")
        return bullet if bullet else original

    # ── Batch bullet rewriter — 1 LLM call for ALL bullets ──────────────────
    def _rewrite_all_bullets(entries: list[dict]) -> dict[str, str]:
        """
        entries: list of {
          "id": str,             # unique key to match result back
          "original": str,       # raw bullet text
          "allowed": set,        # allowed tech ceiling
          "is_ongoing": bool,
          "entry_name": str,
          "extra_keys": set
        }
        Returns: {id: rewritten_bullet}
        Falls back to original for any entry that fails.
        """
        if not entries:
            return {}

        # Build numbered task list — each entry is one numbered item
        task_lines = []
        id_map     = {}
        for i, e in enumerate(entries):
            eid = str(i + 1)
            id_map[eid] = e

            allowed_clean = {t for t in e["allowed"]
                             if t.lower() not in COMMON_HALLUCINATIONS
                             or t.lower() in {s.lower() for s in skills}}
            allowed_str   = ", ".join(sorted(allowed_clean)) if allowed_clean else "none"
            tense_rule    = "present progressive" if e["is_ongoing"] else "past tense"
            entry_answers = _get_entry_answers(e["entry_name"].lower(), extra_keys=e.get("extra_keys"))

            task_lines.append(f"[{eid}] Entry: {e['entry_name']} | Tense: {tense_rule} | Allowed tech: {allowed_str}")
            task_lines.append(f"    Original: {e['original']}")
            if entry_answers:
                task_lines.append(f"    Verified facts: {entry_answers.replace(chr(10), ' | ')}")
            task_lines.append("")

        task_block   = "\n".join(task_lines)
        batch_prompt = f"""You are a resume writer. Rewrite each numbered bullet below into ONE strong professional sentence.

TARGET ROLE: {target_role}

BULLETS TO REWRITE:
{task_block}

RULES (apply to ALL bullets):
1. Keep every fact from Original. Add specific detail from Verified facts if provided — this is mandatory.
2. Use the tense specified. Never switch tense.
3. Never add any tech outside Allowed tech. Never invent tools, frameworks, or outcomes.
4. Name the actual system/platform if mentioned in Original (e.g. "Alumni Connect Platform").
5. Max 22 words per bullet.
6. No weak verbs: never "Worked for", "Helped with", "Was involved in".

OUTPUT FORMAT — respond with ONLY this, nothing else:
[1] rewritten sentence here
[2] rewritten sentence here
...and so on for every number."""

        try:
            raw = call_llm_text(batch_prompt, api_key, max_tokens=60 * len(entries) + 100)
        except Exception as _be:
            logger.warning("Batch rewrite LLM failed (%s) — applying rule-based beautification", _be)
            raw = ""
        results = {}

        if raw:
            for line in raw.strip().splitlines():
                m = re.match(r"\[(\d+)\]\s*(.+)", line.strip())
                if m:
                    eid, rewritten = m.group(1), m.group(2).strip()
                    if eid in id_map:
                        bullet = rewritten.lstrip("-•* ").strip()
                        # Enforce: strip hallucinated tech
                        for word in COMMON_HALLUCINATIONS:
                            e = id_map[eid]
                            allowed_lower = {t.lower() for t in e["allowed"]}
                            if word.lower() not in allowed_lower:
                                bullet = re.sub(
                                    r"(?<![a-zA-Z0-9])" + re.escape(word) + r"(?![a-zA-Z0-9])",
                                    "", bullet, flags=re.IGNORECASE
                                ).strip(" ,.")
                        bullet = re.sub(r"  +", " ", bullet).strip(" ,.")
                        results[eid] = bullet if bullet else id_map[eid]["original"]

        # Fallback: apply rule-based beautification for any missing entries
        WEAK_STARTS = {
            "worked for": "Developed backend for",
            "worked on":  "Built",
            "worked as":  "Contributed as",
            "helped":     "Assisted in developing",
            "aim to develop": "Developing",
            "aim to build":   "Building",
            "aimed to":   "Developed",
            "trying to":  "Working to",
        }
        TENSE_MAP = {
            "developed": "Developing",
            "built":     "Building",
            "created":   "Creating",
            "implemented":"Implementing",
            "designed":  "Designing",
        }
        for eid, e in id_map.items():
            if eid not in results or not results[eid]:
                bullet = e["original"]
                b_lower = bullet.lower().strip()
                # Fix weak starts
                for weak, strong in WEAK_STARTS.items():
                    if b_lower.startswith(weak):
                        bullet = strong + bullet[len(weak):]
                        break
                # Fix tense for ongoing
                if e["is_ongoing"]:
                    words = bullet.split()
                    if words and words[0].lower() in TENSE_MAP:
                        bullet = TENSE_MAP[words[0].lower()] + " " + " ".join(words[1:])
                # Capitalise first letter
                bullet = bullet[0].upper() + bullet[1:] if bullet else bullet
                results[eid] = bullet
                logger.info("Rule-based fallback for %s: %s", e["entry_name"], bullet[:50])

        return results

    # ── Gather ALL bullets → single batch LLM call ─────────────────────────
    _batch_entries = []
    _exp_meta      = []
    _proj_meta     = []

    for exp in pr.experience:
        is_ongoing  = "present" in (exp.duration or "").lower()
        entry_names = {exp.role.lower(), exp.company.lower()}
        context     = exp.role + " " + exp.company + " " + " ".join(exp.responsibilities)
        allowed     = _get_allowed(entry_names, context)
        original    = " ".join(exp.responsibilities) if exp.responsibilities else f"Contributed as {exp.role} at {exp.company}."
        header      = f"{exp.role} | {exp.company}" + (f" | {exp.duration}" if exp.duration else "")
        _batch_entries.append({"id": f"exp_{len(_exp_meta)}", "original": original,
                                "allowed": allowed, "is_ongoing": is_ongoing,
                                "entry_name": exp.company, "extra_keys": entry_names})
        _exp_meta.append(header)

    for proj in pr.projects:
        proj_context = (proj.title + " " + (proj.description or "")).lower()
        entry_names  = {proj.title.lower()}
        for exp in pr.experience:
            exp_full   = (exp.company + " " + " ".join(exp.responsibilities)).lower()
            proj_words = [w for w in proj.title.lower().split() if len(w) > 3]
            if any(w in exp_full for w in proj_words) or exp.company.lower() in proj_context:
                entry_names.update({exp.role.lower(), exp.company.lower()})
        allowed    = _get_allowed(entry_names, proj.title + " " + (proj.description or ""))
        is_ongoing = (
            any("present" in (e.duration or "").lower() for e in pr.experience
                if e.company.lower() in entry_names or e.role.lower() in entry_names)
            or "aim to" in (proj.description or "").lower()
            or "working on" in (proj.description or "").lower()
        )
        tech_str = ", ".join(sorted(
            t for t in allowed
            if t not in COMMON_HALLUCINATIONS or t.lower() in {s.lower() for s in skills}
        ))
        header   = f"{proj.title} | Tech: {tech_str}" if tech_str else proj.title
        original = proj.description or f"Built {proj.title}."
        _batch_entries.append({"id": f"proj_{len(_proj_meta)}", "original": original,
                                "allowed": allowed, "is_ongoing": is_ongoing,
                                "entry_name": proj.title, "extra_keys": entry_names})
        _proj_meta.append(header)

    # ONE LLM call for all bullets
    batch_results = _rewrite_all_bullets(_batch_entries)

    # ── Assemble EXPERIENCE section ───────────────────────────────────────────
    exp_section_lines = []
    for i, header in enumerate(_exp_meta):
        eid    = f"exp_{i}"
        bullet = batch_results.get(eid, _batch_entries[i]["original"])
        if bullet == _batch_entries[i]["original"] and _batch_entries[i]["is_ongoing"]:
            bullet = re.sub(r"^Aim to", "Working to", bullet, flags=re.IGNORECASE)
            bullet = re.sub(r"^Worked for", "Developing backend for", bullet, flags=re.IGNORECASE)
        exp_section_lines += [header, f"- {bullet}", ""]

    # ── Assemble PROJECTS section ─────────────────────────────────────────────
    proj_section_lines = []
    exp_count = len(_exp_meta)
    for i, header in enumerate(_proj_meta):
        eid    = f"proj_{i}"
        bullet = batch_results.get(eid, _batch_entries[exp_count + i]["original"])
        # Ensure tense on fallback too
        if bullet == _batch_entries[exp_count + i]["original"]:
            if any("present" in (e.duration or "").lower() for e in pr.experience
                   if e.company.lower() in _batch_entries[exp_count+i].get("extra_keys",set())
                   or e.role.lower() in _batch_entries[exp_count+i].get("extra_keys",set())):
                bullet = re.sub(r"^Aim to develop", "Developing", bullet, flags=re.IGNORECASE)
                bullet = re.sub(r"^Worked for", "Developed backend for", bullet, flags=re.IGNORECASE)
        proj_section_lines += [header, f"- {bullet}", ""]

        # ── Build EDUCATION section (100% deterministic) ─────────────────────────
    from datetime import datetime as _dt
    _cur_yr   = _dt.now().year
    edu_lines = []
    sorted_edu = sorted(pr.education, key=lambda e: int(str(e.graduation_year)[:4]) if e.graduation_year else 0, reverse=True)
    for edu in sorted_edu:
        parts = [edu.degree, edu.institution]
        is_degree  = edu.degree and any(w in edu.degree.lower() for w in
                     ("bachelor","b.tech","b.e","b.eng","master","m.tech","phd","diploma"))
        is_school  = edu.degree and any(w in edu.degree.lower() for w in
                     ("plus two","10+2","hsc","sslc","secondary","higher secondary","12th","10th","class 12","class 10"))
        if edu.graduation_year:
            yr = int(str(edu.graduation_year)[:4])
            if yr > _cur_yr:
                parts.append(f"Expected: {yr}")
            elif is_degree and yr >= 2022:
                parts.append(f"{yr} - Present")
            elif is_school:
                # Show range: graduated 2022 → "2021 - 2022" (2-year course)
                parts.append(f"{yr - 1} - {yr}")
            else:
                parts.append(str(yr))
        elif is_degree:
            # No year parsed — infer ongoing
            parts.append(f"2023 - Present")
        if edu.gpa:
            gpa_raw = str(edu.gpa).strip()
            # Strip any label prefix: "CGPA: 7.79" → "7.79", "GPA:7.79" → "7.79"
            gpa_raw = re.sub(r"(?i)^(cgpa|gpa)\s*[:/]\s*", "", gpa_raw).strip()
            # Strip bare word with no colon: "CGPA 7.79" → "7.79"
            gpa_raw = re.sub(r"(?i)^(cgpa|gpa)\s+", "", gpa_raw).strip()
            if gpa_raw and re.search(r"\d", gpa_raw):
                parts.append(f"CGPA: {gpa_raw}")
        edu_lines.append(" | ".join(filter(None, parts)))

    # ── Write OBJECTIVE via LLM (tightly scoped) ─────────────────────────────
    # Summarise candidate's ACTUAL work for the LLM — no hallucination possible
    actual_work_summary = []
    for exp in pr.experience:
        actual_work_summary.append(f"- {exp.role} at {exp.company}: {' '.join(exp.responsibilities[:2])}")
    for proj in pr.projects:
        actual_work_summary.append(f"- Project '{proj.title}': {(proj.description or '')[:100]}")
    actual_work_text = "\n".join(actual_work_summary)

    # Collect all extracted facts for objective context
    all_extracted = []
    for _ek, _ef in _entry_fact_map.items():
        for _f in _ef:
            all_extracted.append(f"[{_ek}] {_f}")
    all_extracted.extend(_global_facts)
    extracted_facts_str = "\n".join(f"- {f}" for f in all_extracted[:10]) if all_extracted else "None."

    obj_prompt = f"""Write a 2-sentence resume objective for this candidate.

TARGET ROLE: {target_role}
JD KEYWORDS: {jd_required or "not specified"}

CANDIDATE'S ACTUAL WORK:
{actual_work_text}

CANDIDATE'S OWN WORDS (from clarification answers — verified facts):
{extracted_facts_str}

RULES:
1. Sentence 1 (max 18 words): What has this candidate ACTUALLY DONE?
   - Degree level + domain they actually worked in (from ACTUAL WORK only).
   - Add one specific verified detail from CANDIDATE'S OWN WORDS if relevant.
   - NEVER claim experience in a domain not present in ACTUAL WORK.
   - NO filler: no "passionate", "hardworking", "eager learner", "detail-oriented".
2. Sentence 2 (max 18 words): Connect their real skills to {target_role}.
   - Use "seeking" or "aiming to apply" — honest aspiration, not false claim.
   - Name 1-2 actual skills relevant to {target_role}.
3. Output 2 sentences only. Nothing else."""

    # ── Objective: try LLM first, fall through to deterministic ──────────────
    # Build deterministic objective first — used as fallback OR if LLM is unavailable
    _exp_companies  = [e.company for e in pr.experience]
    _proj_titles    = [p.title   for p in pr.projects]
    _work_areas     = (_exp_companies + _proj_titles)[:2]
    _work_str       = " and ".join(_work_areas) if _work_areas else "software development"
    _jd_words       = set((target_role + " " + (jd_required or "")).lower().split())
    _lang_skills    = [s for s in skills if s.lower() in {"python","c","c++","java","javascript",
                        "typescript","go","rust","r","kotlin","swift","bash","dart","matlab"}]
    _all_top        = ([s for s in skills if s.lower() in _jd_words] or skills)[:3]
    _top_str        = ", ".join(_all_top) if _all_top else "programming"
    _lang_str       = " and ".join(_lang_skills[:2]) if _lang_skills else _top_str

    _det_obj = (
        f"B.Tech Computer Science student with hands-on {_lang_str} experience "
        f"in {_work_str}.\n"
        f"Seeking a {target_role} role to apply {_top_str} skills in real-world engineering projects."
    )

    # Try LLM — if it returns 2 clean sentences, use them; else use deterministic
    obj_text = _det_obj  # default
    try:
        obj_raw = call_llm(obj_prompt, api_key, json_mode=False, max_tokens=120,
                           system_msg="You are a resume writer. Output exactly 2 sentences only. No headers. No bullet points.")
        logger.info("Objective LLM raw: %r", (obj_raw or "")[:120])

        SECTION_HEADERS = {"OBJECTIVE","TECHNICAL SKILLS","EXPERIENCE","PROJECTS",
                           "EDUCATION","CERTIFICATIONS","SKILLS","PROFILE","SUMMARY"}
        obj_sentences = []
        for _line in (obj_raw or "").splitlines():
            _line = _line.strip().lstrip("-•*123456789. ")
            if not _line or _line.upper() in SECTION_HEADERS: continue
            if re.search(r"^(Languages|Frameworks|Databases|Tools|Bachelor|B\.Tech|CGPA)\s*[:/]",
                         _line, re.I): continue
            obj_sentences.append(_line)
            if len(obj_sentences) == 2: break

        if obj_sentences:
            obj_text = "\n".join(obj_sentences[:2])
            logger.info("Using LLM objective")
        else:
            logger.warning("LLM objective was empty/invalid — using deterministic")
    except Exception as _obj_err:
        logger.warning("Objective LLM failed (%s) — using deterministic objective", _obj_err)

    # Strip any domain not evidenced
    DOMAIN_GUARDS = {
        "data analysis":    ["data","analys","pandas","numpy","statistics","powerbi","tableau"],
        "machine learning": ["ml","neural","model","train","dataset","sklearn","classification"],
        "web development":  ["html","css","javascript","frontend","http","rest"],
        "cloud":            ["aws","gcp","azure","cloud","s3","lambda"],
    }
    all_work_lower = (actual_work_text + " " + answers_context).lower()
    for domain, evidence in DOMAIN_GUARDS.items():
        if domain in obj_text.lower() and not any(w in all_work_lower for w in evidence):
            obj_text = re.sub(r"\s*(and|in|with|for)?\s*" + re.escape(domain) + r"[\w\s]*",
                              "", obj_text, flags=re.IGNORECASE).strip().rstrip(",. ")

    # ── Assemble final resume (pure Python) ──────────────────────────────────
    sections = [
        name,
        contact_line,
        "",
        "OBJECTIVE",
        obj_text.strip(),
        "",
        "TECHNICAL SKILLS",
        skills_block,
        "",
        "EXPERIENCE",
        *exp_section_lines,
        "PROJECTS",
        *proj_section_lines,
        "EDUCATION",
        *edu_lines,
    ]
    if pr.certifications:
        sections += ["", "CERTIFICATIONS"]
        for cert in pr.certifications:
            line = cert.name
            if cert.issuer: line += f" | {cert.issuer}"
            if cert.year:   line += f" | {cert.year}"
            sections.append(f"- {line}")

    gen_text = "\n".join(sections).strip()

    # ── Remove duplicate TECHNICAL SKILLS (safety net) ──────────────────────
    ts_positions = [i for i, l in enumerate(gen_text.splitlines())
                    if " ".join(l.strip().upper().split()) == "TECHNICAL SKILLS"]
    if len(ts_positions) > 1:
        # Keep only the FIRST occurrence — remove all subsequent ones + their content
        keep_lines = gen_text.splitlines()
        for pos in reversed(ts_positions[1:]):
            # Find end of this duplicate section
            end = pos + 1
            while end < len(keep_lines):
                norm = " ".join(keep_lines[end].strip().upper().split())
                if norm in {"EXPERIENCE","PROJECTS","EDUCATION","CERTIFICATIONS","OBJECTIVE"}:
                    break
                end += 1
            del keep_lines[pos:end]
        gen_text = "\n".join(keep_lines)
        logger.warning("Removed %d duplicate TECHNICAL SKILLS section(s)", len(ts_positions)-1)

    # ── VERIFIER FEEDBACK LOOP ───────────────────────────────────────────────
    # Run verifier. For each flagged bullet, attempt one targeted rewrite.
    # Only touches lines the verifier flags — leaves everything else untouched.
    try:
        from core.resume_verifier import verify_resume
        pr_dict = {
            "skills":       skills,
            "summary":      "",
            "experience":   [
                {"role": e.role, "company": e.company,
                 "responsibilities": e.responsibilities}
                for e in pr.experience
            ],
            "projects":     [
                {"title": p.title, "description": p.description or "",
                 "technologies": p.technologies or []}
                for p in pr.projects
            ],
        }
        verified_text, issues = verify_resume(
            gen_text, pr_dict, payload.clarification_answers or []
        )

        if issues:
            logger.info("Verifier flagged %d issues — attempting targeted rewrites", len(issues))
            v_lines = verified_text.splitlines()
            _in_section = ""
            _cur_entry  = ""

            for i, line in enumerate(v_lines):
                norm = " ".join(line.strip().upper().split())
                if norm in {"EXPERIENCE","PROJECTS","EDUCATION","OBJECTIVE","TECHNICAL SKILLS"}:
                    _in_section = norm
                    continue

                if not line.strip().startswith("-"):
                    _cur_entry = line.strip()
                    continue

                # Bullet line — check if original bullet text appears in any issue
                bullet_text = line.lstrip("- ").strip()
                is_flagged  = any(bullet_text[:30].lower() in iss.lower() for iss in issues)

                if is_flagged and _in_section in {"EXPERIENCE","PROJECTS"}:
                    # Find allowed tech and is_ongoing for this entry
                    _allowed   = set(skills)
                    _ongoing   = False
                    _entry_key = _cur_entry.lower()
                    for exp in pr.experience:
                        if exp.company.lower() in _entry_key or exp.role.lower() in _entry_key:
                            _ongoing = "present" in (exp.duration or "").lower()
                            _allowed = _get_allowed(
                                {exp.role.lower(), exp.company.lower()},
                                exp.role + " " + exp.company + " " + " ".join(exp.responsibilities)
                            )
                    for proj in pr.projects:
                        if any(w in _entry_key for w in proj.title.lower().split() if len(w) > 3):
                            _allowed = _get_allowed({proj.title.lower()},
                                                    proj.title + " " + (proj.description or ""))

                    rewritten = _rewrite_bullet(
                        bullet_text, _allowed, _ongoing, target_role, _cur_entry
                    )
                    if rewritten and rewritten != bullet_text:
                        v_lines[i] = f"- {rewritten}"
                        logger.info("Verifier rewrite: '%s' → '%s'", bullet_text[:40], rewritten[:40])

            gen_text = "\n".join(v_lines)
        else:
            gen_text = verified_text

    except Exception as _ve:
        logger.warning("Verifier feedback loop error (non-fatal): %s", _ve)
        # Do not modify gen_text — use what we already have

    logger.info("_generate_resume (micro-rewrite): %d chars, %d exp, %d proj",
                len(gen_text), len(pr.experience), len(pr.projects))
    return gen_text