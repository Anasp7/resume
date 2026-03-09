"""
Smart Resume — LLM Output Validator
=====================================
Fix #2: Pydantic-level validation on every Groq response field.
If a field is malformed, it gets normalized or the call is retried.

Covers:
- verdict must be exactly Strong / Moderate / Weak
- technical_depth must be exactly Deep / Moderate / Surface
- logical_consistency must be exactly Consistent / Minor Issues / Inconsistent
- final_resume must be actual content, not instructions
- All list fields must actually be lists
- Numeric fields must be numeric
- Missing top-level keys get sensible defaults
"""

from __future__ import annotations
import re
import logging

logger = logging.getLogger("smart_resume.validator")

# ── Allowed values for enum-like fields ──────────────────────────────────────

VERDICT_MAP = {
    "strong":   "Strong",
    "moderate": "Moderate",
    "partial":  "Moderate",
    "weak":     "Weak",
    "poor":     "Weak",
    "low":      "Weak",
}

DEPTH_MAP = {
    "deep":     "Deep",
    "moderate": "Moderate",
    "medium":   "Moderate",
    "surface":  "Surface",
    "shallow":  "Surface",
    "basic":    "Surface",
}

CONSISTENCY_MAP = {
    "consistent":    "Consistent",
    "minor issues":  "Minor Issues",
    "minor":         "Minor Issues",
    "inconsistent":  "Inconsistent",
    "issues found":  "Minor Issues",
}

QUALITY_MAP = {
    "strong":       "Strong",
    "good":         "Strong",
    "acceptable":   "Acceptable",
    "fair":         "Acceptable",
    "needs work":   "Needs Work",
    "poor":         "Needs Work",
    "low":          "Needs Work",
}

LEVEL_MAP = {
    "high":   "High",
    "medium": "Medium",
    "low":    "Low",
}

INSTRUCTION_MARKERS = [
    "write a complete", "write the complete", "generate full",
    "use --- as section dividers", "ats optimization rules",
    "write real content", "no placeholders", "format —",
    "optimization is blocked", "return null", "optimization_blocked",
    "write these sections", "start with this exact header",
]


def _normalize_enum(value: str, mapping: dict, default: str) -> str:
    """Map a potentially verbose string to an allowed enum value."""
    if not isinstance(value, str):
        return default
    vl = value.lower().strip()
    # Direct key match
    for key, result in mapping.items():
        if key in vl:
            return result
    return default


def _is_instruction_string(value) -> bool:
    """Return True if a string looks like an instruction rather than content."""
    if not isinstance(value, str):
        return True
    if len(value.strip()) < 80:
        return True
    vl = value.lower()
    return any(marker in vl for marker in INSTRUCTION_MARKERS)


def _ensure_list(value, default=None) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        return [value]
    return default or []


def _ensure_dict(value, default=None) -> dict:
    if isinstance(value, dict):
        return value
    return default or {}


def validate_and_fix(data: dict, needs_optimization: bool) -> tuple[dict, list[str]]:
    """
    Validate and normalize all fields in the LLM response.
    Returns (fixed_data, list_of_fixes_applied).
    """
    fixes = []

    # ── Ensure all top-level keys exist ──────────────────────────────────────
    required_keys = [
        "structured_extraction", "skill_classification", "job_match_analysis",
        "doubt_detection", "proficiency_consistency", "factual_evaluation",
        "internal_consistency", "resume_quality_assessment", "template_selection",
        "final_resume", "career_improvement_plan",
    ]
    for key in required_keys:
        if key not in data:
            data[key] = {}
            fixes.append(f"missing key '{key}' — added empty default")

    # ── job_match_analysis ────────────────────────────────────────────────────
    jma = _ensure_dict(data.get("job_match_analysis"))
    data["job_match_analysis"] = jma

    raw_verdict = jma.get("verdict", "")
    fixed_verdict = _normalize_enum(raw_verdict, VERDICT_MAP, "Weak")
    if fixed_verdict != raw_verdict:
        fixes.append(f"verdict '{raw_verdict}' → '{fixed_verdict}'")
    jma["verdict"] = fixed_verdict
    jma["matched_skills"]   = _ensure_list(jma.get("matched_skills"))
    jma["missing_skills"]   = _ensure_list(jma.get("missing_skills"))
    jma["alignment_gaps"]   = _ensure_list(jma.get("alignment_gaps"))

    # ── factual_evaluation ────────────────────────────────────────────────────
    fe = _ensure_dict(data.get("factual_evaluation"))
    data["factual_evaluation"] = fe

    raw_depth = fe.get("technical_depth", "")
    fixed_depth = _normalize_enum(raw_depth, DEPTH_MAP, "Surface")
    if fixed_depth != raw_depth:
        fixes.append(f"technical_depth '{raw_depth}' → '{fixed_depth}'")
    fe["technical_depth"] = fixed_depth

    raw_cons = fe.get("logical_consistency", "")
    fixed_cons = _normalize_enum(raw_cons, CONSISTENCY_MAP, "Consistent")
    if fixed_cons != raw_cons:
        fixes.append(f"logical_consistency '{raw_cons}' → '{fixed_cons}'")
    fe["logical_consistency"] = fixed_cons

    raw_align = fe.get("skill_alignment", "")
    fixed_align = _normalize_enum(raw_align, VERDICT_MAP, "Weak")
    if fixed_align != raw_align:
        fixes.append(f"skill_alignment '{raw_align}' → '{fixed_align}'")
    fe["skill_alignment"] = fixed_align

    # ── resume_quality_assessment ─────────────────────────────────────────────
    qa = _ensure_dict(data.get("resume_quality_assessment"))
    data["resume_quality_assessment"] = qa

    for field in ["structure_clarity", "ats_compatibility", "role_relevance"]:
        raw = qa.get(field, "")
        fixed = _normalize_enum(raw, LEVEL_MAP, "Medium")
        if fixed != raw:
            fixes.append(f"{field} '{raw}' → '{fixed}'")
        qa[field] = fixed

    raw_oq = qa.get("overall_quality", "")
    fixed_oq = _normalize_enum(raw_oq, QUALITY_MAP, "Needs Work")
    if fixed_oq != raw_oq:
        fixes.append(f"overall_quality '{raw_oq}' → '{fixed_oq}'")
    qa["overall_quality"] = fixed_oq
    qa["improvements"] = _ensure_list(qa.get("improvements"))

    # ── internal_consistency ──────────────────────────────────────────────────
    ic = _ensure_dict(data.get("internal_consistency"))
    data["internal_consistency"] = ic

    raw_tl = ic.get("timeline_alignment", "")
    if isinstance(raw_tl, str) and "issue" in raw_tl.lower():
        ic["timeline_alignment"] = "Issues Found"
    elif not isinstance(raw_tl, str) or raw_tl not in ("OK", "Issues Found"):
        ic["timeline_alignment"] = "OK"

    for field in ["skill_usage_coverage", "claim_density"]:
        ic[field] = _normalize_enum(ic.get(field, ""), LEVEL_MAP, "Low")
    ic["flags"] = _ensure_list(ic.get("flags"))

    # ── structured_extraction ─────────────────────────────────────────────────
    se = _ensure_dict(data.get("structured_extraction"))
    data["structured_extraction"] = se
    se["experience"]          = _ensure_list(se.get("experience"))
    se["projects"]            = _ensure_list(se.get("projects"))
    se["education"]           = _ensure_list(se.get("education"))
    se["mismatch_corrections"]= _ensure_list(se.get("mismatch_corrections"))
    se["skills"]              = _ensure_list(se.get("skills"))

    # ── skill_classification ──────────────────────────────────────────────────
    sc = _ensure_dict(data.get("skill_classification"))
    data["skill_classification"] = sc
    for cat in ["programming_languages","frameworks_libraries","tools_platforms","databases","core_cs_concepts"]:
        sc[cat] = _ensure_list(sc.get(cat))

    # ── proficiency_consistency ───────────────────────────────────────────────
    pc = _ensure_dict(data.get("proficiency_consistency"))
    data["proficiency_consistency"] = pc
    pc["analysis"] = _ensure_list(pc.get("analysis"))
    # Normalize each analysis item
    for item in pc["analysis"]:
        if isinstance(item, dict):
            ev = item.get("evidence_level", 0)
            try:
                item["evidence_level"] = int(float(str(ev)))
            except (ValueError, TypeError):
                item["evidence_level"] = 0
                fixes.append(f"evidence_level '{ev}' → 0")
            aligned = item.get("aligned", True)
            if isinstance(aligned, str):
                item["aligned"] = aligned.lower() not in ("false", "no", "0")

    # ── doubt_detection ───────────────────────────────────────────────────────
    dd = _ensure_dict(data.get("doubt_detection"))
    data["doubt_detection"] = dd
    dd["issues"] = _ensure_list(dd.get("issues"))
    raw_cr = dd.get("clarification_required", False)
    if isinstance(raw_cr, str):
        dd["clarification_required"] = raw_cr.lower() in ("true", "yes", "1")

    # ── skill_gap_analysis — deep validation ─────────────────────────────────
    if "skill_gap_analysis" not in data:
        data["skill_gap_analysis"] = {"tier_1_critical":[],"tier_2_important":[],"tier_3_nice_to_have":[]}
    sga = _ensure_dict(data.get("skill_gap_analysis"))
    data["skill_gap_analysis"] = sga
    SEV_MAP = {"high": "High", "medium": "Medium", "low": "Low"}
    for tier in ["tier_1_critical", "tier_2_important", "tier_3_nice_to_have"]:
        raw_tier      = _ensure_list(sga.get(tier))
        validated     = []
        for item in raw_tier:
            if not isinstance(item, dict):
                continue
            skill = str(item.get("skill", "")).strip()
            if not skill or _is_instruction_string(skill):
                continue
            sev   = _normalize_enum(str(item.get("gap_severity", "")), SEV_MAP, "Medium")
            evid  = str(item.get("current_evidence", "None")).strip() or "None"
            what  = str(item.get("what_to_do", "")).strip()
            # Reject generic "learn X" responses — require a specific resource
            if not what or re.match(r"^learn\s+\w+\.?$", what, re.IGNORECASE):
                what = f"Build a project using {skill} — see official docs or free course on freeCodeCamp/Kaggle"
                fixes.append(f"skill_gap what_to_do for '{skill}' was generic — replaced with specific guidance")
            validated.append({
                "skill":            skill,
                "gap_severity":     sev,
                "current_evidence": evid,
                "what_to_do":       what,
            })
        sga[tier] = validated

    # ── career_improvement_plan — deep validation ─────────────────────────────
    cp = _ensure_dict(data.get("career_improvement_plan"))
    data["career_improvement_plan"] = cp

    # missing_skills_to_learn
    raw_skills     = _ensure_list(cp.get("missing_skills_to_learn"))
    valid_skills   = []
    GENERIC_RESOURCES = {"online course", "youtube", "documentation", "google", "internet", ""}
    for item in raw_skills:
        if not isinstance(item, dict):
            continue
        skill = str(item.get("skill", "")).strip()
        if not skill:
            continue
        reason   = str(item.get("reason", "") or item.get("why_it_matters", "")).strip()
        resource = str(item.get("resource", "")).strip()
        if resource.lower() in GENERIC_RESOURCES:
            resource = f"Search: '{skill} hands-on project tutorial GitHub'"
            fixes.append(f"career plan resource for '{skill}' was generic — improved")
        valid_skills.append({"skill": skill, "reason": reason, "resource": resource})
    cp["missing_skills_to_learn"] = valid_skills

    # suggested_projects
    raw_projects  = _ensure_list(cp.get("suggested_projects"))
    valid_projects = []
    for item in raw_projects:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        desc  = str(item.get("description", "")).strip()
        if not title or len(desc) < 15:
            continue
        valid_projects.append({"title": title, "description": desc})
    cp["suggested_projects"] = valid_projects

    # learning_roadmap — reject generic entries
    raw_roadmap   = _ensure_list(cp.get("learning_roadmap"))
    GENERIC_FOCUS = {"learn basics", "introduction", "setup", "setup environment",
                     "getting started", "overview", "fundamentals"}
    valid_roadmap = []
    for item in raw_roadmap:
        if not isinstance(item, dict):
            continue
        week     = str(item.get("week", "")).strip()
        focus    = str(item.get("focus", "")).strip()
        task     = str(item.get("task", "") or focus).strip()
        resource = str(item.get("resource", "")).strip()
        if not week or not focus:
            continue
        if focus.lower() in GENERIC_FOCUS:
            fixes.append(f"roadmap week '{week}' had generic focus '{focus}' — skipped")
            continue
        valid_roadmap.append({"week": week, "focus": focus, "task": task, "resource": resource})
    cp["learning_roadmap"] = valid_roadmap

    # ── final_resume ──────────────────────────────────────────────────────────
    final = data.get("final_resume")
    if _is_instruction_string(final):
        if needs_optimization:
            fixes.append(f"final_resume was instruction string — set to None (needs retry)")
        data["final_resume"] = None
    elif isinstance(final, str) and len(final.strip()) > 80:
        pass  # valid content
    else:
        data["final_resume"] = None

    if fixes:
        logger.info("Validator fixed %d issues: %s", len(fixes), "; ".join(fixes[:5]))

    return data, fixes


def needs_retry(data: dict, needs_optimization: bool) -> bool:
    """
    Return True if the response is so broken it should be retried.
    Only retry for critical failures, not minor normalization issues.
    """
    # If optimization was needed but final_resume is still None after validation
    if needs_optimization and data.get("final_resume") is None:
        return True
    # If structured_extraction is completely empty
    se = data.get("structured_extraction", {})
    if not se.get("summary") and not se.get("skills"):
        return True
    return False