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
import math
import logging
from typing import List, Dict, Any, Optional
_re = re
import time

def _pro_edu(text: str) -> str:
    """Standardize degree terminology for a professional look."""
    t = str(text)
    t = re.sub(r"(?i)\bB\.?Tech\b|btech", "Bachelor of Technology", t)
    t = re.sub(r"(?i)\bB\.?E\b|be(?=\b)", "Bachelor of Engineering", t)
    t = re.sub(r"(?i)\bM\.?Tech\b|mtech", "Master of Technology", t)
    t = re.sub(r"(?i)\bCSE\b", "Computer Science and Engineering", t)
    return t

import httpx

from core.schemas import BackendPayload, SmartResumeResponse
from core.llm_client import call_llm, call_llm_text, call_llm_json, call_llm_json_gemini
from core.prompt_builder import build_phase2_analysis_prompt, build_phase2b_generation_prompt, get_integrity_and_classification_rules
from core.llm_validator import validate_and_fix, needs_retry
from core.growth.recommender import generate_detailed_growth_plan
from core.latex_builder import build_latex_resume

logger = logging.getLogger("smart_resume.evaluator")

GROQ_API_URL    = "https://api.groq.com/openai/v1/chat/completions"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
GROQ_MODEL      = "llama-3.1-8b-instant"
GROQ_MODEL_FAST = "llama-3.1-8b-instant"
TOGETHER_MODEL  = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
MODEL           = GROQ_MODEL
MAX_RETRIES     = 2

import os as _os

def _get_all_groq_keys(primary_key: str) -> list:
    keys = [k for k in [primary_key] + [
        _os.getenv(f"GROQ_API_KEY_{i}", "") for i in range(2, 6)
    ] if k and k.strip()]
    return keys

def _get_together_key() -> str:
    return _os.getenv("TOGETHER_API_KEY", "").strip()

def _clean_email(email: str) -> str:
    if not email: return ""
    email = email.strip().lower()
    providers = ["gmail", "outlook", "hotmail", "yahoo", "icloud"]
    for p in providers:
        if f"@{p}.co" in email:
            return str(email).replace(f"@{p}.co", f"@{p}.com")
    return email


# ── Main entry ────────────────────────────────────────────────────────────────

def evaluate(payload: BackendPayload, api_key: str) -> SmartResumeResponse:
    from core.doubt_engine import generate_doubt_questions, all_required_answered

    # Step 1: Upgrade JD parsing if needed
    payload = _upgrade_jd_parsing(payload, api_key)

    # Step 2: Doubt detection
    if payload.clarification_answers is not None:
        doubt_issues      = []
        required_answered = True
        unanswered        = []
        logger.info("Clarification answers set (%d) — skipping doubt gate",
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

    # Step 2b: Backend consistency check
    consistency_flags = _check_internal_consistency(payload)

    # ── STEP X: Missing profile information detection ─────────────────────────
    from core.doubt_engine import detect_missing_profile_info

    def generate_all_doubts(payload: BackendPayload) -> list:
        """
        Consolidated entry point for Step X (Doubt Detection).
        Attempts local LLM (Ollama) with 3s timeout, then falls back to Gemini.
        """
        from core.ollama_pipeline import generate_doubt_questions as generate_llm_doubts, ollama_is_available
        
        # Base issues always detected via rule-based logic
        issues = detect_missing_profile_info(payload.parsed_resume, 
                                             target_role=payload.parsed_jd.target_role if payload.parsed_jd else "",
                                             raw_resume_text=payload.parsed_resume.raw_text)
        
        # LLM-based technical/factual doubts
        llm_doubts = []
        try:
            # Fast 3s check for Ollama heartbeat
            if ollama_is_available():
                logger.info("Ollama available — attempting local technical doubt generation")
                llm_doubts = generate_llm_doubts(payload)
            
            if not llm_doubts:
                # Fallback to Gemini for high-quality questions if Ollama is down/slow
                logger.info("Ollama unreachable or timed out — falling back to Gemini for thoughts")
                llm_doubts = _generate_doubts_gemini(payload)
                
        except Exception as e:
            logger.warning("All LLM doubt generation failed: %s — using pure rule-based only", e)
        
        return issues + (llm_doubts or [])


    def _generate_doubts_gemini(payload: BackendPayload) -> list:
        """Fallback technical doubt generator via Gemini."""
        from core.llm_client import call_llm_json_gemini
        
        pr = payload.parsed_resume
        prompt = f"""Analyze this candidate's resume and generate 2-3 specific, probing technical questions to clarify their projects or experience.
Focus on:
1. Missing technical details (e.g. "What framework was used for X?").
2. Quantifiable achievements (e.g. "By what percentage did this optimize Y?").
3. Verification of claims (e.g. "Did this use a real-time sensor or a simulator?").

CANDIDATE DATA:
{pr.raw_text[:1500]}

OUTPUT FORMAT:
Return a JSON object with a 'doubts' key containing a list of objects with 'type', 'context', and 'question' fields.
Example: {{ "doubts": [ {{ "type": "project_metric", "context": "Autonomous Car", "question": "..." }} ] }}
"""
        try:
            api_key = os.getenv("GROQ_API_KEY", "") # llm_client will use GEMINI_API_KEY if forced
            res = call_llm_json_gemini(prompt, api_key, max_tokens=1000)
            return res.get("doubts", [])
        except Exception as e:
            logger.warning("Gemini technical doubt generator failed: %s", e)
            return []

    ollama_doubts = getattr(payload, "_ollama_doubt_questions", [])
    if payload.clarification_answers is None:
        # Instead of replacing, we will merge both pools later in _all_pool
        technical_fallback_issues = doubt_issues
    else:
        technical_fallback_issues = []

    engine_issues = detect_missing_profile_info(
        payload.parsed_resume, 
        payload.parsed_jd.target_role, 
        payload.resume_raw_text
    )

    engine_dicts = [
        {
            "type":     issue.type,
            "context":  issue.context,
            "question": issue.question,
            "required": issue.required,
        }
        for issue in engine_issues
    ]

    # ── Prioritized Merging ───────────────────────────────────────────────────
    # Priority 1: Mandatory missing contact/identity info
    # Priority 2: Technical Project/Exp Factual Integrity + Add-New sections
    # Priority 3: Other missing info
    
    # Priority 1: Mandatory missing contact/identity info + User requested additions
    mandatory_types = {"github", "linkedin", "email", "phone", "internship", "hackathons", "certifications"}
    # Factual Integrity
    technical_types = {
        "project_tech", "experience_clarification", "project_metric", "experience_metric", "thin_project",
        "additional_skills"
    }
    
    seen_norm_qs: set[str] = set()
    seen_contexts: set[str] = set()
    
    def _norm(t: str):
        return _re.sub(r"[^a-z0-9]", "", str(t).lower())

    final_questions: list[dict] = []
    
    # Combined pool: Ollama + Python Deterministic + Mandatory Links
    _all_pool = (ollama_doubts if ollama_doubts else []) + technical_fallback_issues + engine_dicts

    # 1. Add Mandatory Contact Info
    for q in engine_dicts:
        if q.get("type") in mandatory_types:
            qtext = _norm(q.get("question", ""))
            qctx = _norm(q.get("context", "")) + ":" + str(q.get("type", ""))
            if qtext not in seen_norm_qs and qctx not in seen_contexts:
                final_questions.append(q)
                seen_norm_qs.add(qtext)
                if qctx: seen_contexts.add(qctx)

    # 2. Add Technical/Factual/Add-New Questions (Limit to 4 for UX)
    _tcount = 0
    for q in _all_pool:
        _qtype = str(q.get("type", ""))
        if _qtype in technical_types and _tcount < 4:
            qtext = _norm(str(q.get("question", "")))
            qctx = _norm(str(q.get("context", ""))) + ":" + _qtype
            if qtext not in seen_norm_qs and qctx not in seen_contexts:
                final_questions.append(q)
                seen_norm_qs.add(qtext)
                if qctx: seen_contexts.add(qctx)
                _tcount += 1

    # 3. Add everything else from both sources
    for q in _all_pool:
        qtext = _norm(str(q.get("question", "")))
        _qtype = str(q.get("type", ""))
        qctx = _norm(str(q.get("context", ""))) + ":" + _qtype
        if qtext not in seen_norm_qs and qctx not in seen_contexts:
            final_questions.append(q)
            seen_norm_qs.add(qtext)
            if qctx: seen_contexts.add(qctx)

    step_x_issues = final_questions

    # ── QUESTION QUALITY FILTER ───────────────────────────────────────────────
    # Kill garbage questions that reference broken parser output (e.g. date-as-company)
    import re as _qre

    # Patterns that indicate the question references a broken parse
    _DATE_FRAGMENT = _qre.compile(r"\b\d{1,2}/(?:\d{2,4})?(?!\w)")  # matches "06/", "06/2025" without requiring \b on the slash
    _SINGLE_SLASH  = _qre.compile(r"\b\d{1,2}/\s*")                 # matches "06/ " or "06/"

    # Very generic one-word skills (parser split "Machine Learning" into "Machine")
    _STOPWORDS = {
        "machine", "learning", "deep", "critical", "thinking", "solving",
        "effective", "abilities", "team", "programming", "development",
        "fundamentals", "techniques", "problem", "vapt", "skills",
    }

    def _question_is_valid(q: dict) -> bool:
        question_text = str(q.get("question", "")).lower()
        context_text  = str(q.get("context", "")).lower()
        combined      = question_text + " " + context_text

        # Kill if the question references a date fragment as a person/company/role
        if _DATE_FRAGMENT.search(combined) or _SINGLE_SLASH.search(combined):
            logger.debug("Quality filter: killed garbage question (date-as-entity): %s", q.get("question", "")[:80])
            return False

        # Kill if context is a pure stopword (broken skill like "machine", "learning")
        ctx_words = set(context_text.split())
        if ctx_words and ctx_words.issubset(_STOPWORDS):
            logger.debug("Quality filter: killed garbage question (stopword skill): %s", q.get("question", "")[:80])
            return False

        return True

    filtered_questions = [q for q in step_x_issues if _question_is_valid(q)]
    killed = len(step_x_issues) - len(filtered_questions)
    if killed:
        logger.info("Quality filter: removed %d garbage question(s) — %d remain", killed, len(filtered_questions))

    missing_profile_questions = filtered_questions


    # ── Pre-compute Job Match (for Step 1 display) ────────────────────────────
    pr  = payload.parsed_resume
    pjd = payload.parsed_jd

    candidate_skills = {s.lower().strip() for s in (pr.skills or [])}
    required_skills  = {s.lower().strip() for s in (pjd.required_skills or [])}
    
    direct_matched = candidate_skills & required_skills
    fuzzy_matched = set()
    for cand in candidate_skills:
        for req in required_skills:
            if cand.startswith(req) or req.startswith(cand):
                fuzzy_matched.add(req)
                fuzzy_matched.add(cand)

    all_matched = sorted(direct_matched | (fuzzy_matched & candidate_skills))
    all_missing = sorted(required_skills - direct_matched - fuzzy_matched)

    from core.doubt_engine import _detect_domain
    domain = _detect_domain(pjd.target_role)
    if not domain and hasattr(pjd, "detected_domain"):
        domain = str(pjd.detected_domain).replace("Domain.", "").replace("_", " ").title()
    domain = domain or pjd.target_role.title()

    from core.similarity import compute_weighted_score, compute_proficiency_evidence_scores
    evidence_scores = compute_proficiency_evidence_scores(pr, all_matched)
    weighted = compute_weighted_score(
        similarity_score          = payload.semantic_similarity_score,
        matched_skills            = all_matched,
        required_skills           = pjd.required_skills,
        years_of_experience       = pr.years_of_experience,
        required_experience_years = pjd.required_experience_years,
        evidence_scores           = evidence_scores,
    )

    pre_jma = {
        "matched_skills":  all_matched,
        "missing_skills":  all_missing,
        "domain":          domain,
        "verdict":         weighted["verdict"],
        "weighted_score":  weighted["score"],
        "score_breakdown": weighted["breakdown"],
    }

    # ── STEP X gate: on first run (no answers yet), block resume and ask questions ──
    # clarification_answers is None = first run. [] = user skipped. [...] = answered.
    _is_first_run = payload.clarification_answers is None
    if _is_first_run and missing_profile_questions:
        logger.info("STEP X gate: first run with %d questions — blocking resume generation",
                    len(missing_profile_questions))
        # Return early with just the questions — don't waste Groq tokens on resume yet
        return SmartResumeResponse(
            session_id                = payload.session_id,
            structured_extraction     = {},
            skill_classification      = {},
            job_match_analysis        = pre_jma,
            doubt_detection           = {},
            proficiency_consistency   = {},
            factual_evaluation        = {},
            internal_consistency      = {},
            resume_quality_assessment = {},
            template_selection        = {},
            final_resume              = None,
            latex_template            = None,
            career_improvement_plan   = {},
            skill_gap_analysis        = {},
            clarification_required    = True,
            clarification_questions   = [
                q.get("question") if isinstance(q, dict) else getattr(q, "question", "")
                for q in missing_profile_questions
            ],
            mismatch_corrections      = [],
            missing_profile_detection = missing_profile_questions,
            profile_change_log        = [],
        )

    # Step 3: Build prompt and call LLM (Analysis Phase)
    pr     = payload.parsed_resume
    pjd    = payload.parsed_jd
    prompt = build_phase2_analysis_prompt(payload)
    data   = _call_with_retry(prompt, api_key, payload.needs_optimization, payload)

    # Step 4: Override job match analysis with deterministic Python computation
    jma = data.get("job_match_analysis", {})
    jma.update(pre_jma)
    data["job_match_analysis"] = jma

    # Step 5: Inject consistency flags
    if consistency_flags:
        ic = data.get("internal_consistency", {})
        ic["flags"] = ic.get("flags", []) + consistency_flags
        if any("overlap" in f.lower() or "conflict" in f.lower() for f in consistency_flags):
            ic["timeline_alignment"] = "Issues Found"
        data["internal_consistency"] = ic

    # Step 6: Build response
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
        latex_template            = data.get("latex_template"),
        career_improvement_plan   = data.get("career_improvement_plan", {}),
        skill_gap_analysis        = data.get("skill_gap_analysis", {}),
        clarification_required    = data.get("clarification_required", False),
        clarification_questions   = data.get("clarification_questions", []),
        mismatch_corrections      = data.get("mismatch_corrections", []),
        missing_profile_detection = missing_profile_questions,
        profile_change_log        = getattr(payload, "_profile_change_log", []),
    )

    # Step 7: Override clarification gate
    if unanswered:
        result.clarification_required  = True
        result.clarification_questions = [i.question for i in unanswered]
        if not required_answered:
            result.final_resume = None

    # ── Silent background data collection (zero impact) ───────────────────────
    # Fires-and-forgets in a daemon thread — never delays or breaks the response
    try:
        from core.ats_dataset import collect_sample
        _input_text  = getattr(payload.parsed_resume, "raw_text", "") or ""
        _output_text = data.get("final_resume") or ""
        _role        = getattr(payload.parsed_jd, "target_role", "") or ""
        collect_sample(_input_text, _output_text, _role, data)
    except Exception:
        pass  # silently ignored — collection must never affect the pipeline

    return result


# ── Internal consistency checker ──────────────────────────────────────────────

def _check_internal_consistency(payload) -> list[str]:
    flags = []
    pr    = payload.parsed_resume

    # Guard: ensure pr is a ParsedResume object not a list
    if not hasattr(pr, "education"):
        logger.warning("_check_internal_consistency: parsed_resume is not a ParsedResume object")
        return flags

    grad_year = None
    for edu in (pr.education or []):
        grad_val = getattr(edu, "graduation_year", None) or (edu.get("graduation_year") if isinstance(edu, dict) else None)
        if grad_val:
            try:
                yr = int(str(grad_val)[:4])
                if 2000 <= yr <= 2030:
                    grad_year = yr
                    break
            except (ValueError, TypeError):
                pass

    if grad_year and pr.years_of_experience > 0:
        from datetime import datetime
        max_possible = datetime.now().year - grad_year
        if pr.years_of_experience > max_possible + 2:
            flags.append(
                f"Experience claim ({pr.years_of_experience:.1f} years) exceeds "
                f"what is possible since graduation {grad_year} ({max_possible} years max)"
            )

    total_skills  = len(pr.skills or [])
    evidence_text = " ".join(
        " ".join(e.responsibilities or []) for e in (pr.experience or [])
    ) + " " + " ".join(
        (p.description or "") for p in (pr.projects or [])
    )
    if total_skills > 5:
        evidenced = sum(
            1 for s in (pr.skills or [])
            if re.search(r"(?<![a-z0-9])" + re.escape(s.lower()) + r"(?![a-z0-9])",
                         evidence_text.lower())
        )
        if evidenced / total_skills < 0.4:
            flags.append(
                f"Low skill evidence: only {evidenced}/{total_skills} skills "
                f"({evidenced/total_skills:.0%}) appear in project/experience descriptions"
            )

    if grad_year:
        from datetime import datetime
        if grad_year > datetime.now().year and pr.years_of_experience > 1:
            flags.append(
                f"Future graduation ({grad_year}) but claims "
                f"{pr.years_of_experience:.1f} years experience — verify dates"
            )

    thin = [
        p.title for p in (pr.projects or [])
        if len((p.description or "").strip().split()) < 5
    ]
    if thin:
        flags.append(f"Thin project descriptions (under 5 words): {', '.join(thin[:3])}")

    return flags


# ── JD upgrade ────────────────────────────────────────────────────────────────

def _upgrade_jd_parsing(payload: BackendPayload, api_key: str) -> BackendPayload:
    from core.jd_parser_llm import parse_jd_with_llm
    from core.schemas import Domain

    if payload.parsed_jd.required_skills:
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
        domain      = domain_map.get(llm_jd.get("domain","unknown").lower(), Domain.UNKNOWN)
        req_skills  = [s.lower() for s in llm_jd.get("required_skills", [])]
        pref_skills = [s.lower() for s in llm_jd.get("preferred_skills", [])]

        updated_jd = pjd.model_copy(update={
            "detected_domain":            domain,
            "required_skills":            req_skills  or pjd.required_skills,
            "preferred_skills":           pref_skills or pjd.preferred_skills,
            "key_responsibilities":       llm_jd.get("key_responsibilities", pjd.key_responsibilities),
            "required_experience_years":  llm_jd.get("required_experience_years", pjd.required_experience_years),
        })
        return payload.model_copy(update={"parsed_jd": updated_jd})

    except Exception as e:
        logger.warning("JD upgrade failed: %s", e)
        return payload


# ── Two-call strategy ─────────────────────────────────────────────────────────

def _call_with_retry(prompt: str, api_key: str, needs_opt: bool, payload: "BackendPayload") -> dict:
    """
    Call 1: Evaluation JSON (Analysis only).
    Call 2: Dedicated resume generation (Generative).
    """
    data = call_llm_json(prompt, api_key, max_tokens=1500, smart=False)
    data, fixes = validate_and_fix(data, needs_optimization=False)

    if fixes:
        logger.info("Validator applied %d fixes", len(fixes))

    if needs_opt:
        logger.info("Running dedicated resume generation via Gemini (Structured Output)...")
        try:
            resume_text, latex_text, skills_json = _generate_resume(payload, api_key)
        except Exception as e:
            logger.warning("Gemini resume gen failed: %s — trying Rule-Based fallback", e)
            resume_text = _generate_resume_rule_based(payload)
            latex_text  = None
            skills_json = {}

        if resume_text:
            from core.resume_verifier import verify_resume
            pr = payload.parsed_resume
            resume_data_for_verify = {
                "skills":     pr.skills,
                "experience": [e.model_dump() for e in pr.experience],
                "projects":   [p.model_dump() for p in pr.projects],
                "summary":    pr.summary or "",
            }
            verified_text, issues = verify_resume(
                resume_text, resume_data_for_verify, payload.clarification_answers or []
            )
            data["final_resume"]         = verified_text
            data["latex_template"]       = latex_text
            data["resume_verify_issues"] = issues
            if skills_json:
                data["skill_classification"] = skills_json
            logger.info("Resume generated: %d chars", len(verified_text))
        else:
            data["final_resume"]         = None
            data["resume_verify_issues"] = ["Resume generation failed"]

    return data


# ── LLM resume generator ──────────────────────────────────────────────────────

def _generate_resume(payload: "BackendPayload", api_key: str) -> tuple[str, str, dict]:
    """
    Call 2: Ask Gemini to generate the optimized resume as structured JSON.
    Returns (resume_text, latex_text, categorized_skills).
    """
    from core.prompt_builder import build_phase2b_generation_prompt
    
    prompt = build_phase2b_generation_prompt(payload)
    logger.info("Generating Structured Resume via Gemini 2.5 Flash...")
    
    try:
        from core.latex_builder import build_latex_resume
        res_json = call_llm_json_gemini(prompt, api_key, max_tokens=4000)
        
        resume_text = res_json.get("final_resume_text")
        skills_json = res_json.get("categorized_skills", {})
        
        if not resume_text:
            raise ValueError("Gemini returned empty resume text")
            
        # ENRICHMENT: Merge enriched skills from LLM into the data passed to the builder
        resume_data = payload.parsed_resume.model_dump()
        if skills_json and any(skills_json.values()):
            # ── Strip hallucinated/generic skill phrases from LLM output ─────
            _BANNED_SKILL_PHRASES = {
                "dbms", "automation", "autonomous systems", "backend development",
                "web technologies", "web development", "software development",
                "problem solving", "problem-solving", "data structures",
                "computer science", "information technology", "engineering",
                "algorithms", "programming", "coding", "development",
            }
            def _is_valid_skill(s: str) -> bool:
                sl = s.lower().strip()
                # Reject if it's a known bad phrase
                if sl in _BANNED_SKILL_PHRASES:
                    return False
                # Reject if it's a generic multi-word category description (>3 words)
                words = sl.split()
                if len(words) > 3:
                    return False
                return True

            cleaned_skills = {
                cat: [s for s in items if _is_valid_skill(s)]
                for cat, items in skills_json.items()
            }
            resume_data["skills"] = cleaned_skills
            
        # NEVER trust LLM-generated LaTeX. Always use the deterministic Python builder.
        latex_text = build_latex_resume(resume_data, resume_text=resume_text)
            
        return resume_text, latex_text, skills_json
        
    except Exception as e:
        logger.warning("Gemini structured generation failed: %s — falling back to rule-based", e)
        # Fallback to legacy rule-based if Gemini fails
        resume_text = _generate_resume_rule_based(payload)
        latex_text = build_latex_resume(payload.parsed_resume.model_dump(), resume_text=resume_text)
        return resume_text, latex_text, {}



# ── Rule-based fallback resume generator ─────────────────────────────────────

def _generate_resume_rule_based(payload: "BackendPayload") -> str:
    pr   = payload.parsed_resume
    pjd  = payload.parsed_jd
    role_target = pjd.target_role if pjd else "Software Engineer"

    # ── Professionalize Degree Terminology ────────────────────────────────────
    # (Moved to global scope)

    def _beautify_bullet(text: str) -> str:
        t = text.strip()
        if not t: return ""
        return t[0].upper() + t[1:] + ("." if not t.endswith(".") else "")

    name     = pr.name or "Candidate"
    phone    = pr.phone or ""
    email    = _clean_email(pr.email or "")
    location = pr.location or ""
    linkedin = getattr(pr, "linkedin", None) or ""
    github   = getattr(pr, "github",   None) or ""

    NIL_WORDS = {"nil","none","no","nothing","not used","didn't use","did not use", "not applicable","n/a","na"}
    denied = set()
    for ca in (payload.clarification_answers or []):
        ans = (ca.answer if hasattr(ca, "answer") else str(ca)).strip().lower()
        if ans in NIL_WORDS or len(ans) <= 3:
            continue
        for pat in [r"(\w[\w+#]*)\s+is\s+not\s+used",r"(\w[\w+#]*)\s+not\s+used", r"didn.t\s+use\s+(\w[\w+#]*)",r"did\s+not\s+use\s+(\w[\w+#]*)"]:
            for m in re.finditer(pat, ans):
                denied.add(m.group(1).strip().lower())

    _NEW_SKILLS = set()
    _TECH_LIST = (pr.skills or []) + ["FastAPI","Django","Flask","React","ROS2","ROS","Docker","Kubernetes","SQL","NoSQL","C#"]
    for _ca in (payload.clarification_answers or []):
        _ans = (getattr(_ca, "answer", "") or "").lower()
        for _kw in _TECH_LIST:
            if re.search(r"\b" + re.escape(_kw.lower()) + r"\b", _ans):
                _NEW_SKILLS.add(_kw)
    
    skills = [s for s in (pr.skills or []) if s.lower() not in denied]
    for _ns in _NEW_SKILLS:
        if _ns.lower() not in {x.lower() for x in skills}:
            skills.append(_ns)

    PHRASE_MAP = [
        ("aim to develop","Developing"),("aim to build","Building"),
        ("aim to create","Creating"),("aim to implement","Implementing"),
        ("aim to","Working on"),("worked for a","Developed"),
        ("worked for","Developed"),("worked on","Built"),
        ("worked as","Served as"),("worked with","Utilized"),
        ("was responsible for","Responsible for"),("was involved in","Contributed to"),
        ("helped to","Assisted in"),("helped with","Supported"),
        ("helped","Supported"),("trying to","Working to"),
        ("focusing on","Focused on"),("i developed","Developed"),
        ("i built","Built"),("i implemented","Implemented"),
    ]

    def strengthen(raw: str, ongoing: bool = False) -> str:
        t = raw.strip().rstrip(".,")
        if not t: return raw
        tl = str(t).lower()
        for phrase, repl in PHRASE_MAP:
            if tl.startswith(phrase):
                rest = t[len(phrase):].strip()
                t = f"{repl} {rest}".strip()
                break
        t = t.strip()
        if t: t = t[0].upper() + t[1:]
        return t + ("." if t and not t.endswith(".") else "")

    lines = []
    contact_parts = [p for p in [email, phone, location, linkedin, github] if p]
    lines += [name, "  |  ".join(contact_parts), ""]

    tech_skills = [s for s in skills if s.lower() not in {"teamwork","leadership","time management","project management","communication"}]
    top2 = tech_skills[:2]
    top2_str = " and ".join(top2) if top2 else "technical skills"

    from datetime import datetime as _dt
    _current_year = _dt.now().year
    _grad_year = None
    _branch = ""
    for _edu in (pr.education or []):
        _deg = (getattr(_edu, "degree", "") or "").lower()
        if any(k in _deg for k in {"b.tech","btech","bachelor","b.e","be","b.sc"}):
            grad_val = getattr(_edu, "graduation_year", None) or (_edu.get("graduation_year") if isinstance(_edu, dict) else None)
            try: _grad_year = int(str(grad_val)[:4]) if grad_val else None
            except: pass
            # Extract branch from degree string e.g. "B.Tech in Computer Science"
            _branch_match = re.search(r"(?:in|of)\s+(.+)", getattr(_edu, "degree", "") or "", re.I)
            if _branch_match:
                _branch = _branch_match.group(1).strip()

    _is_graduate = (_grad_year is not None and int(str(_grad_year)[:4]) <= _current_year)

    # Build a role-targeted, professional summary
    _degree_str = f"B.Tech {_branch}" if _branch else "B.Tech"
    _student_label = f"{'Final-year ' if not _is_graduate else ''}{_degree_str} {'graduate' if _is_graduate else 'student'}"
    _year_exp = f"{_current_year - _grad_year}+ years of" if (_is_graduate and _grad_year) else "hands-on"
    _proj_str = f"{len(pr.projects)} academic project{'s' if len(pr.projects) != 1 else ''}" if pr.projects else "academic projects"

    _summary = (
        f"{_student_label} with {_year_exp} experience in web development and backend systems. "
        f"Proficient in {top2_str}, with {_proj_str} demonstrating practical problem-solving. "
        f"Seeking a {role_target} role to deliver scalable, user-focused solutions."
    )
    lines += ["SUMMARY", _summary, ""]

    LANG_SET  = {"python","java","c","c++","c#","javascript","typescript","go","rust",
                 "sql","r","matlab","bash","dart","scala","swift","kotlin","vhdl","verilog"}
    FWK_SET   = {"ros2","ros","react","vue","angular","nextjs","django","flask","fastapi",
                 "spring","pytorch","tensorflow","keras","express","flutter","langchain",
                 "opencv","numpy","pandas","scikit","sklearn","scikit-learn",
                 "streamlit","gazebo","moveit","rviz","scipy","matplotlib","seaborn"}
    DB_SET    = {"sql","postgresql","mysql","sqlite","mongodb","redis","firebase",
                 "cassandra","dynamodb","elasticsearch"}
    TOOL_SET  = {"docker","kubernetes","git","github","linux","aws","gcp","azure",
                 "jenkins","terraform","grafana","postman","gazebo","cmake","arduino",
                 "simulink","keil","proteus","labview","autocad","solidworks",
                 "vscode","vs code","jupyter","jupyter notebook","esp32",
                 "gitlab","bitbucket","raspberry pi"}
    CORE_CS_SET = {"data structures","data structures & algorithms","data structures and algorithms",
                  "algorithms","dsa","oop","object oriented programming","oops",
                  "dbms","database management","database management systems",
                  "operating systems","os","computer networks","networking",
                  "computer architecture","discrete mathematics","system design","design patterns"}
    SOFT_SET  = {"communication","leadership","teamwork","problem solving","time management",
                 "adaptability","collaboration","creativity","critical thinking","interpersonal",
                 "decision making","team management","team managment"}
    AI_SET_KW = {"machine learning","deep learning","neural","nlp","computer vision",
                 "data science","data visualization","feature engineering","data preprocessing",
                 "data analysis","artificial intelligence"}
    WEB_SET_KW = {"html","css","react","frontend","web development","next.js","bootstrap"}

    # Acronyms that should stay uppercase / fixed-case
    _PRESERVE_CAP = {"C++","C#","SQL","HTML","CSS","API","AI","ML","OOP","OOPS","DSA",
                     "DBMS","OS","NLP","GCP","AWS","IoT","ESP32","VS Code","REST"}
    _PRESERVE_UP  = {p.upper() for p in _PRESERVE_CAP}

    def _cap_skill(s: str) -> str:
        t = s.strip()
        if t.upper() in _PRESERVE_UP:
            for p in _PRESERVE_CAP:
                if p.upper() == t.upper(): return p
        if re.search(r"[._-]", t):
            return t[0].upper() + t[1:] if t else t
        return t.title() if t else t

    _JUNK = {
        "ing","mak","think","solv","works","work","net",
        "structures","structure","system","systems","code","notebook",
        "processing","engineering","development","management","analysis",
        "algorithms","algorithm","computer","data","feature","programming",
        "software","networks","network","operating","decision","problem",
        "critical","thinking","solving","making","team","skills","skill",
        "vs","and","of","in","for","with","the","to",
    }

    def _is_junk(s: str) -> bool:
        sl = s.strip().lower()
        if " " not in sl and len(sl) <= 3 and sl not in {"c","r","go","c#","ai","ml","os"}:
            return True
        return sl in _JUNK or sl.endswith("-")

    cats = {"Languages": [], "AI & Data Science": [], "Core CS": [],
            "Frameworks & Libraries": [], "Databases": [], "Tools & Platforms": [],
            "Web Development": [], "Soft Skills": [], "Other": []}
    for s in skills:
        sl = s.lower().strip()
        out = _cap_skill(s)
        if   sl in LANG_SET:                            cats["Languages"].append(out)
        elif any(k in sl for k in AI_SET_KW):           cats["AI & Data Science"].append(out)
        elif sl in CORE_CS_SET or any(k in sl for k in CORE_CS_SET): cats["Core CS"].append(out)
        elif sl in FWK_SET:                             cats["Frameworks & Libraries"].append(out)
        elif sl in DB_SET:                              cats["Databases"].append(out)
        elif sl in TOOL_SET:                            cats["Tools & Platforms"].append(out)
        elif any(k in sl for k in WEB_SET_KW):          cats["Web Development"].append(out)
        elif any(soft in sl for soft in SOFT_SET):      cats["Soft Skills"].append(out)
        else:
            if not _is_junk(s): cats["Other"].append(out)

    lines.append("SKILLS")
    for cat, items in cats.items():
        if items: lines.append(f"  {cat}: {', '.join(items)}")
    lines.append("")

    all_projects = list(pr.projects or [])
    lines.append("PROJECTS")
    for proj in all_projects:
        tech_str = ", ".join(proj.technologies[:5]) if proj.technologies else ""
        lines.append(f"  {proj.title or 'Project'}" + (f" | Tech: {tech_str}" if tech_str else ""))
        if proj.description and proj.description.strip():
            lines.append(f"    - {strengthen(proj.description[:180])}")
    lines.append("")

    if pr.experience:
        lines.append("EXPERIENCE")
        for e in pr.experience:
            header = f"  {e.role or 'Role'}"
            if e.company:  header += f" | {e.company}"
            if e.duration: header += f" | {e.duration}"
            lines.append(header)
            ongoing = "present" in (e.duration or "").lower()
            for r in list(e.responsibilities or [])[:4]:
                lines.append(f"    - {strengthen(r, ongoing=ongoing)}")
        lines.append("")

    lines.append("EDUCATION")
    for e in (pr.education or []):
        line = f"  {_pro_edu(e.degree or 'Degree')}"
        if e.institution:     line += f", {e.institution}"
        grad_val = getattr(e, "graduation_year", None) or (e.get("graduation_year") if isinstance(e, dict) else None)
        if grad_val: line += f" ({grad_val})"
        if e.gpa:             line += f" | CGPA: {e.gpa}"
        lines.append(line)
    s12 = getattr(pr, "school_12th", None)
    if s12: lines.append(f"  Class XII: {s12}")
    lines.append("")

    if pr.certifications:
        lines.append("CERTIFICATIONS")
        for c in pr.certifications:
            lines.append(f"  - {c.name}")
        lines.append("")

    return "\n".join(list(lines))


def _extract_verified_facts(answers: Optional[List[Any]]) -> Dict[str, Any]:
    import re
    res_metrics: List[str] = []
    res_modules: List[str] = []
    res_awards: List[str] = []
    res_seniority: str = ""
    res_migrations: List[str] = []
    
    _ans_list = answers if answers is not None else []
    for ca in _ans_list:
        ans = str(getattr(ca, "answer", "") or "")
        if not ans: continue
        for m in re.findall(r"\b\d+%\b|\b\d+x\b|\b\d+\+\s*(?:users|clients|records)\b|\$\d+[kmb]?\b", ans, re.I):
            res_metrics.append(str(m))
        for mod in re.findall(r"\b(\w+[\s-](?:module|system|feature|algorithm|controller))\b", ans, re.I):
            res_modules.append(str(mod))
        if re.search(r"\b(Awarded|Ranked|Won|Selected|Certified|Credential)\b", ans, re.I):
            res_awards.append(ans)
        m_sen = re.search(r"\b(lead|senior|headed|managed|promoted to|head of)\b", ans, re.I)
        if m_sen: res_seniority = str(m_sen.group(1))
        for mig in re.findall(r"(?i)\bmigrated\s+from\s+([\w\s,]+)\s+to\s+([\w\s,]+)\b", ans):
            res_migrations.append(f"Migrated from {mig[0].strip()} to {mig[1].strip()}")
            
    return {
        "metrics": res_metrics,
        "modules": res_modules,
        "awards": res_awards,
        "seniority": res_seniority,
        "migrations": res_migrations
    }



# ── ATS Check helpers ─────────────────────────────────────────────────────────

def _ats_has_table(t: str) -> bool:
    """Detect actual ASCII-art tables (not markdown | separators in sentences)."""
    # Match lines that START with | or +--- patterns (true table rows)
    return bool(re.search(r"(?m)^\s*[\|+][-=+|]{3,}", t))

def _ats_non_ascii(t: str) -> bool:
    """Non-ASCII check — exclude the first 3 lines (name block) and raise threshold."""
    body = "\n".join(t.splitlines()[3:])
    non_ascii = re.findall(r"[^\x00-\x7F]", body)
    return len(non_ascii) > 10

def _ats_sections_present(t: str) -> bool:
    """Resume must have at least 2 of the 3 core sections."""
    tu = t.upper()
    found = sum([
        bool(re.search(r"\b(EXPERIENCE|WORK HISTORY|EMPLOYMENT)\b", tu)),
        bool(re.search(r"\b(EDUCATION|DEGREE|UNIVERSITY|COLLEGE)\b", tu)),
        bool(re.search(r"\b(PROJECTS?|PORTFOLIO)\b", tu)),
    ])
    return found < 2

ATS_CHECKS = [
    # ── Format integrity ────────────────────────────────────────────────────
    (_ats_has_table,
     "ASCII table detected — ATS parsers like Taleo strip table content", "high"),

    (lambda t: bool(re.search(r"\b(references available|references on request)\b", t, re.I)),
     "Remove 'References available' — wastes space", "medium"),

    (_ats_non_ascii,
     "Excessive non-ASCII characters in body — may cause ATS garbling", "high"),

    (lambda t: bool(re.search(r"(CURRICULUM VITAE|\bCV\b)", t, re.I)),
     "Use 'Resume' not 'CV' for US/Canada ATS systems", "medium"),

    (lambda t: bool(re.search(r"(?m)^.{121,}", t)),
     "Lines over 120 chars — some ATS parsers truncate", "low"),

    # ── Required contact info ───────────────────────────────────────────────
    (lambda t: not bool(re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", t)),
     "No email address detected — ATS cannot identify candidate", "high"),

    (lambda t: not bool(re.search(r"(\+?\d[\d\s\-\(\)]{8,}\d|\b\d{10}\b)", t)),
     "No phone number detected — ATS contact extraction will fail", "high"),

    # ── Required sections ───────────────────────────────────────────────────
    (_ats_sections_present,
     "Missing core sections — resume needs at least 2 of: Experience, Education, Projects", "high"),

    (lambda t: not bool(re.search(r"\b(SKILLS?|TECHNICAL|TECHNOLOGIES|TECH STACK)\b", t, re.I)),
     "No skills section detected — ATS keyword matching will be weak", "medium"),

    # ── Content quality ─────────────────────────────────────────────────────
    (lambda t: len(t.split()) < 150,
     "Resume is too sparse (<150 words) — ATS may reject as incomplete", "medium"),
]

def run_ats_checks(resume_text: str) -> list[dict]:
    """Run all ATS checks. Returns list of {issue, severity} dicts (issues found only)."""
    results = []
    if not resume_text:
        return results
    for check_fn, message, severity in ATS_CHECKS:
        try:
            if check_fn(resume_text):
                results.append({"issue": message, "severity": severity})
        except Exception:
            pass
    return results

def ats_score(resume_text: str) -> dict:
    """
    Compute a deterministic ATS score (0–100) from rule checks.
    Used as a label source for the ML training dataset.
    """
    if not resume_text:
        return {"score": 0, "grade": "F", "issues": []}

    issues = run_ats_checks(resume_text)
    weights = {"high": 20, "medium": 8, "low": 3}
    deductions = sum(weights.get(i["severity"], 0) for i in issues)
    score = max(0, 100 - deductions)

    if score >= 85:   grade = "A"
    elif score >= 70: grade = "B"
    elif score >= 55: grade = "C"
    elif score >= 40: grade = "D"
    else:             grade = "F"

    return {"score": score, "grade": grade, "issues": issues}