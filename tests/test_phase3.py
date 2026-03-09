"""
Phase 3 Tests — Evaluator Logic, Verifier, Validator, Prompt Builder, Doubt Engine
Run: python -m pytest tests/test_phase3.py -v
These tests use mocks/stubs — no real Groq API calls needed.
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.resume_verifier import verify_resume, WATCHLIST_TERMS
from core.llm_validator import validate_and_fix, _normalize_enum, VERDICT_MAP
from core.parser import compute_ats_score, is_junk_line, is_real_project_title
from core.schemas import (
    ParsedResume, ParsedJobDescription, Domain, TemplateType,
    SkillProficiency, ProficiencyLevel, ProjectEntry, ExperienceEntry,
    EducationEntry, BackendPayload, ClarificationAnswer
)
from core.similarity import compute_weighted_score, compute_proficiency_evidence_scores


# ── resume_verifier ────────────────────────────────────────────────────────────

def _make_original(skills=None, experience=None, projects=None):
    return {
        "skills": skills or ["pytorch", "fastapi", "docker"],
        "experience": experience or [{"role": "Intern", "responsibilities": ["built pytorch models"], "company": "X"}],
        "projects": projects or [{"description": "trained pytorch model", "technologies": ["pytorch"]}],
        "summary": "",
    }


def test_verifier_passes_clean_resume():
    """A resume with no fabrications should be returned unchanged."""
    resume = "PyTorch Developer\nBuilt models using PyTorch and Docker.\nFastAPI backend deployed."
    cleaned, issues = verify_resume(resume, _make_original(), [])
    assert len(issues) == 0
    assert "pytorch" in cleaned.lower()


def test_verifier_removes_fabricated_skill():
    """A watchlisted skill not in original should be removed from generated resume."""
    original = _make_original(skills=["pytorch"], projects=[{"description": "trained pytorch", "technologies": ["pytorch"]}])
    # TensorFlow is NOT in original — verifier should remove it
    resume = "PyTorch | TensorFlow | Docker\nExperience: Built TensorFlow model."
    cleaned, issues = verify_resume(resume, original, [])
    assert any("tensorflow" in i.lower() for i in issues), f"Expected TensorFlow to be flagged, got: {issues}"


def test_verifier_allows_skill_in_experience():
    """Skill that appears in experience bullets should not be flagged."""
    original = _make_original(
        skills=["fastapi"],
        experience=[{"role": "Dev", "responsibilities": ["built fastapi REST service"], "company": "Y"}],
        projects=[]
    )
    resume = "Developer\nBuilt REST APIs with FastAPI.\n"
    cleaned, issues = verify_resume(resume, original, [])
    fabricated = [i for i in issues if "fastapi" in i.lower()]
    assert not fabricated, f"FastAPI should not be flagged (it's in experience): {issues}"


def test_verifier_removes_generic_phrase():
    """Phrases like 'team player' should always be removed."""
    resume = "Developer\nI am a team player with strong communication skills."
    cleaned, issues = verify_resume(resume, _make_original(), [])
    assert "team player" not in cleaned.lower()
    assert any("team" in i.lower() for i in issues)


def test_verifier_fixes_summary_to_objective():
    """SUMMARY heading should be renamed OBJECTIVE."""
    resume = "John Doe\nSUMMARY\nExperienced developer."
    cleaned, issues = verify_resume(resume, _make_original(), [])
    assert "OBJECTIVE" in cleaned
    assert "SUMMARY" not in cleaned


def test_verifier_clarification_expands_allowed():
    """Skills confirmed via clarification answers should not be flagged."""
    original = _make_original(skills=["pytorch"])  # no tensorflow
    answers = ["I used TensorFlow for image classification in college project"]
    resume = "Built TensorFlow classifier."
    cleaned, issues = verify_resume(resume, original, answers)
    # tensorflow appears in clarification answer → allowed
    fab = [i for i in issues if "tensorflow" in i.lower()]
    assert not fab, f"TensorFlow was in clarification — should be allowed: {issues}"


# ── llm_validator ──────────────────────────────────────────────────────────────

def test_normalize_verdict_variants():
    for raw, expected in [("Strong", "Strong"), ("strong", "Strong"),
                          ("Moderate", "Moderate"), ("partial", "Moderate"),
                          ("Weak", "Weak"), ("poor", "Weak"), ("LOW", "Weak")]:
        result = _normalize_enum(raw, VERDICT_MAP, "Weak")
        assert result == expected, f"normalize({raw!r}) = {result!r}, want {expected!r}"


def test_validate_and_fix_missing_keys():
    """validate_and_fix should fill missing keys with defaults, not raise."""
    data, fixes = validate_and_fix({}, needs_optimization=False)
    assert "job_match_analysis" in data
    assert "skill_gap_analysis" in data
    assert len(fixes) > 0  # at least some defaults should be applied


def test_validate_verdict_normalized():
    """validate_and_fix normalizes verdict to Strong/Moderate/Weak."""
    data = {"job_match_analysis": {"verdict": "partially matching"}}
    fixed, _ = validate_and_fix(data, needs_optimization=False)
    verdict = fixed["job_match_analysis"]["verdict"]
    assert verdict in ("Strong", "Moderate", "Weak"), f"Unexpected verdict: {verdict}"


def test_validate_list_fields_coerced():
    """Fields that should be lists but arrive as strings get wrapped."""
    data = {"skill_gap_analysis": {"tier_1_critical": "some string"}}
    fixed, _ = validate_and_fix(data, needs_optimization=False)
    tier1 = fixed["skill_gap_analysis"]["tier_1_critical"]
    assert isinstance(tier1, list), f"Expected list, got {type(tier1)}"


# ── compute_ats_score ──────────────────────────────────────────────────────────

def _sample_resume_text():
    return """John Doe
john@example.com | +91 98765 43210

EXPERIENCE
Software Engineer at Acme Corp | 2022-2024
- Built REST APIs using FastAPI and deployed with Docker
- Reduced latency by 30% through Redis caching

PROJECTS
Resume Analyzer
- Developed NLP pipeline using Python and spaCy

EDUCATION
B.Tech Computer Science | NIT Calicut | 2022 | GPA: 8.5

SKILLS
Python, FastAPI, Docker, Redis, spaCy, PostgreSQL
"""

def test_ats_score_clean_resume():
    from core.parser import split_into_sections
    text = _sample_resume_text()
    sections = split_into_sections(text)
    result = compute_ats_score(text, sections, ["python", "fastapi", "docker", "redis"], "Backend Engineer")
    assert result["score"] >= 60, f"Clean resume should score ≥60, got {result['score']}"
    assert result["verdict"] in ("ATS Ready", "Needs Minor Fixes", "High Risk of Rejection")


def test_ats_score_missing_email():
    from core.parser import split_into_sections
    text = "John Doe\nNo contact info\n\nSKILLS\nPython\n"
    sections = split_into_sections(text)
    result = compute_ats_score(text, sections, ["python"])
    assert any("email" in f.lower() or "phone" in f.lower() for f in result["flags"])


def test_ats_score_bad_phrase():
    from core.parser import split_into_sections
    text = _sample_resume_text() + "\nReferences available upon request."
    sections = split_into_sections(text)
    result = compute_ats_score(text, sections, ["python", "fastapi"])
    assert any("references" in f.lower() for f in result["flags"])


def test_ats_score_weak_verbs():
    from core.parser import split_into_sections
    text = """john@example.com | +1234567890

EXPERIENCE
Developer at Co
- Was responsible for maintaining legacy code
- Helped with team projects

SKILLS
Python
"""
    sections = split_into_sections(text)
    result = compute_ats_score(text, sections, ["python"])
    assert any("weak" in f.lower() or "responsible" in f.lower() for f in result["flags"])


# ── is_junk_line ───────────────────────────────────────────────────────────────

def test_junk_line_phone():
    assert is_junk_line("+91 98765 43210") is True

def test_junk_line_cgpa():
    assert is_junk_line("CGPA: 8.5") is True

def test_junk_line_real_content():
    assert is_junk_line("Developed a REST API using FastAPI") is False

def test_is_real_project_title():
    assert is_real_project_title("Resume Analyzer") is True
    assert is_real_project_title("CGPA: 8.5") is False
    assert is_real_project_title("") is False


# ── compute_weighted_score edge cases ─────────────────────────────────────────

def test_weighted_score_no_jd():
    """When no JD required_skills, similarity is used as skill component."""
    result = compute_weighted_score(
        similarity_score=70.0,
        matched_skills=[],
        required_skills=[],
        years_of_experience=2.0,
        required_experience_years=None,
    )
    assert result["score"] > 0
    assert result["verdict"] in ("Strong", "Moderate", "Weak")


def test_weighted_score_perfect_match():
    result = compute_weighted_score(
        similarity_score=90.0,
        matched_skills=["python", "docker", "fastapi"],
        required_skills=["python", "docker", "fastapi"],
        years_of_experience=3.0,
        required_experience_years=2.0,
    )
    assert result["score"] >= 68
    assert result["verdict"] == "Strong"


def test_weighted_score_zero_skills():
    result = compute_weighted_score(
        similarity_score=10.0,
        matched_skills=[],
        required_skills=["python", "pytorch", "kubernetes"],
        years_of_experience=0.0,
        required_experience_years=3.0,
    )
    assert result["verdict"] == "Weak"


# ── compute_proficiency_evidence_scores ───────────────────────────────────────

def _build_resume_with_projects(skills, project_desc):
    proj = ProjectEntry(title="Test Project", description=project_desc, technologies=skills)
    exp  = ExperienceEntry(company="Acme", role="Dev", responsibilities=[project_desc])
    edu  = EducationEntry(institution="NIT", degree="B.Tech")
    return ParsedResume(
        raw_text="test",
        skills=skills,
        projects=[proj],
        experience=[exp],
        education=[edu],
    )

def test_proficiency_advanced_evidenced():
    from core.schemas import SkillProficiency, ProficiencyLevel
    pr = _build_resume_with_projects(
        ["pytorch"],
        "trained pytorch model five times with pytorch pipeline using pytorch optimizer pytorch metrics"
    )
    profs = [SkillProficiency(skill_name="pytorch", level=ProficiencyLevel.ADVANCED)]
    result = compute_proficiency_evidence_scores(profs, pr)
    assert result["pytorch"]["aligned"] is True

def test_proficiency_beginner_no_evidence_ok():
    from core.schemas import SkillProficiency, ProficiencyLevel
    pr = _build_resume_with_projects(["docker"], "no mention of the skill at all")
    profs = [SkillProficiency(skill_name="docker", level=ProficiencyLevel.BEGINNER)]
    result = compute_proficiency_evidence_scores(profs, pr)
    assert result["docker"]["gap"] == 0  # beginner needs 0 evidence