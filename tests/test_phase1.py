"""
Smart Resume — Phase 1 Tests
===========================
Tests all Phase 1 components without network or heavy dependencies.
Run with: python -m pytest tests/test_phase1.py -v
Or:       python tests/test_phase1.py
"""

import sys
import os
import json
import unittest

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# PARSER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestTextCleaner(unittest.TestCase):
    def setUp(self):
        from core.parser import _clean_text
        self.clean = _clean_text

    def test_collapses_blank_lines(self):
        text = "Line 1\n\n\n\n\nLine 2"
        result = self.clean(text)
        self.assertNotIn("\n\n\n", result)

    def test_strips_trailing_whitespace(self):
        text = "Hello   \nWorld   "
        result = self.clean(text)
        for line in result.splitlines():
            self.assertEqual(line, line.rstrip())

    def test_preserves_newlines(self):
        text = "Section A\nContent line"
        result = self.clean(text)
        self.assertIn("\n", result)


class TestSectionSplitter(unittest.TestCase):
    def setUp(self):
        from core.parser import split_into_sections
        self.split = split_into_sections

    def test_detects_education_section(self):
        text = "EDUCATION\nBachelor of Science in Computer Science\nState University 2022"
        sections = self.split(text)
        self.assertIn("education", sections)

    def test_detects_skills_section(self):
        text = "TECHNICAL SKILLS\nPython, Java, React, Docker"
        sections = self.split(text)
        self.assertIn("skills", sections)

    def test_detects_experience_section(self):
        text = "WORK EXPERIENCE\nSoftware Engineer at Acme Corp\n- Built REST APIs"
        sections = self.split(text)
        self.assertIn("experience", sections)

    def test_detects_projects_section(self):
        text = "PROJECTS\nE-Commerce Platform\n- Built with React and FastAPI"
        sections = self.split(text)
        self.assertIn("projects", sections)

    def test_unrecognized_content_goes_to_other(self):
        text = "John Doe\njohn@email.com\n+1-555-0100"
        sections = self.split(text)
        self.assertIn("header", sections)


class TestMismatchDetector(unittest.TestCase):
    def setUp(self):
        from core.parser import detect_section_mismatches
        self.detect = detect_section_mismatches

    def test_detects_project_in_experience(self):
        sections = {
            "experience": "Software Engineer at Acme\n- Built and deployed a Django web app on AWS\n- github.com/user/project",
            "projects": "",
        }
        result = self.detect(sections)
        self.assertTrue(len(result["project_in_experience"]) > 0)

    def test_detects_experience_in_projects(self):
        sections = {
            "experience": "",
            "projects": "E-Commerce Platform\n- Worked as an intern at Shopify for 6 months\n- Managed a team of 3 developers",
        }
        result = self.detect(sections)
        self.assertTrue(len(result["experience_in_project"]) > 0)

    def test_clean_resume_no_mismatches(self):
        sections = {
            "experience": "Software Engineer at Acme Corp\n- Optimized SQL queries by 40%",
            "projects": "Portfolio Website\n- Built with React and deployed on Vercel",
        }
        result = self.detect(sections)
        self.assertEqual(result["project_in_experience"], [])
        self.assertEqual(result["experience_in_project"], [])


class TestInlineSkillExtractor(unittest.TestCase):
    def setUp(self):
        from core.parser import extract_inline_skills
        self.extract = extract_inline_skills

    def test_finds_known_skills(self):
        text = "I have experience with Python, FastAPI, and PostgreSQL in production."
        skills = self.extract(text)
        self.assertIn("python", skills)
        self.assertIn("fastapi", skills)
        self.assertIn("postgresql", skills)

    def test_no_false_positives(self):
        text = "I enjoy hiking and reading books on weekends."
        skills = self.extract(text)
        self.assertEqual(skills, [])

    def test_deduplicates(self):
        text = "python python python docker docker"
        skills = self.extract(text)
        self.assertEqual(skills.count("python"), 1)
        self.assertEqual(skills.count("docker"), 1)


class TestNLPPreprocessor(unittest.TestCase):
    def setUp(self):
        from core.parser import preprocess_for_embedding
        self.preprocess = preprocess_for_embedding

    def test_returns_string(self):
        result = self.preprocess("Building scalable APIs with FastAPI and PostgreSQL.")
        self.assertIsInstance(result, str)

    def test_lowercases(self):
        result = self.preprocess("Python Java Docker")
        self.assertEqual(result, result.lower())

    def test_removes_stopwords(self):
        result = self.preprocess("the and a is was were")
        # Stopwords should be reduced or eliminated
        self.assertLess(len(result.split()), 6)

    def test_not_empty_for_valid_input(self):
        result = self.preprocess("Machine learning engineer with PyTorch experience.")
        self.assertTrue(len(result) > 0)


# ─────────────────────────────────────────────────────────────────────────────
# SIMILARITY TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestJaccardSimilarity(unittest.TestCase):
    def setUp(self):
        from core.similarity import _jaccard_similarity
        self.jaccard = _jaccard_similarity

    def test_identical_texts(self):
        text = "python fastapi postgresql docker kubernetes"
        score = self.jaccard(text, text)
        self.assertAlmostEqual(score, 99.0)

    def test_completely_different(self):
        score = self.jaccard("python fastapi", "javascript react")
        self.assertLess(score, 50)

    def test_partial_overlap(self):
        score = self.jaccard("python fastapi postgresql", "python django postgresql")
        self.assertGreater(score, 30)
        self.assertLess(score, 100)

    def test_empty_strings(self):
        score = self.jaccard("", "something")
        self.assertEqual(score, 0.0)

    def test_score_in_range(self):
        score = self.jaccard("machine learning deep learning", "deep learning nlp")
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)


class TestDomainDetection(unittest.TestCase):
    def setUp(self):
        from core.similarity import _detect_domain
        self.detect = _detect_domain

    def test_detects_backend(self):
        jd = "We need a backend engineer with REST API, database, and microservices experience."
        self.assertEqual(self.detect(jd).value, "Backend")

    def test_detects_ml(self):
        jd = "Machine learning engineer with PyTorch, deep learning, and NLP experience."
        self.assertEqual(self.detect(jd).value, "ML")

    def test_detects_frontend(self):
        jd = "Frontend developer with React, TypeScript, and responsive CSS experience."
        self.assertEqual(self.detect(jd).value, "Frontend")

    def test_detects_devops(self):
        jd = "DevOps engineer with Kubernetes, Docker, Terraform, and CI/CD pipeline experience."
        self.assertEqual(self.detect(jd).value, "DevOps")

    def test_unknown_on_ambiguous(self):
        from core.schemas import Domain
        jd = "We are looking for a motivated individual to join our team."
        result = self.detect(jd)
        # Should not crash; returns some domain or Unknown
        self.assertIsInstance(result, Domain)


class TestTemplateSelection(unittest.TestCase):
    def setUp(self):
        from core.similarity import decide_template
        from core.schemas import ParsedResume, TemplateType
        self.decide = decide_template
        self.TemplateType = TemplateType
        self.ParsedResume = ParsedResume

    def _make_resume(self, yoe=0, proj=0, exp=0, skills=None):
        return self.ParsedResume(
            raw_text="",
            skills=skills or [],
            years_of_experience=yoe,
            project_count=proj,
            experience_count=exp,
        )

    def test_fresher_academic(self):
        r = self._make_resume(yoe=0, proj=3, exp=0)
        self.assertEqual(self.decide(r), self.TemplateType.FRESHER_ACADEMIC)

    def test_project_focused(self):
        r = self._make_resume(yoe=2, proj=5, exp=1)
        self.assertEqual(self.decide(r), self.TemplateType.PROJECT_FOCUSED)

    def test_experience_focused(self):
        r = self._make_resume(yoe=4, proj=1, exp=4)
        self.assertEqual(self.decide(r), self.TemplateType.EXPERIENCE_FOCUSED)

    def test_skill_focused(self):
        r = self._make_resume(yoe=1, proj=0, exp=0, skills=["python"] * 12)
        self.assertEqual(self.decide(r), self.TemplateType.SKILL_FOCUSED)


class TestOptimizationDecision(unittest.TestCase):
    def setUp(self):
        from core.similarity import decide_needs_optimization
        self.decide = decide_needs_optimization

    def test_below_threshold_needs_optimization(self):
        self.assertTrue(self.decide(50.0))
        self.assertTrue(self.decide(71.9))

    def test_at_threshold_no_optimization(self):
        self.assertFalse(self.decide(72.0))
        self.assertFalse(self.decide(90.0))


class TestProficiencyEvidenceScores(unittest.TestCase):
    def setUp(self):
        from core.similarity import compute_proficiency_evidence_scores
        from core.schemas import ParsedResume, SkillProficiency, ProficiencyLevel, ProjectEntry
        self.compute = compute_proficiency_evidence_scores
        self.SkillProficiency = SkillProficiency
        self.ProficiencyLevel = ProficiencyLevel
        self.ParsedResume = ParsedResume
        self.ProjectEntry = ProjectEntry

    def test_expert_with_no_evidence_has_gap(self):
        resume = self.ParsedResume(raw_text="", skills=[])
        profs = [self.SkillProficiency(skill_name="pytorch", level=self.ProficiencyLevel.EXPERT)]
        result = self.compute(profs, resume)
        self.assertGreater(result["pytorch"]["gap"], 0)

    def test_beginner_with_no_evidence_aligned(self):
        resume = self.ParsedResume(raw_text="", skills=[])
        profs = [self.SkillProficiency(skill_name="rust", level=self.ProficiencyLevel.BEGINNER)]
        result = self.compute(profs, resume)
        self.assertEqual(result["rust"]["gap"], 0)

    def test_advanced_with_strong_evidence_aligned(self):
        proj = self.ProjectEntry(
            title="API Service",
            description="built with python python python python python",
            technologies=["python"],
        )
        resume = self.ParsedResume(raw_text="", skills=["python"], projects=[proj])
        profs = [self.SkillProficiency(skill_name="python", level=self.ProficiencyLevel.ADVANCED)]
        result = self.compute(profs, resume)
        self.assertTrue(result["python"]["aligned"])


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemas(unittest.TestCase):
    def test_backend_payload_assembles(self):
        from core.schemas import (
            BackendPayload, ParsedResume, ParsedJobDescription,
            TemplateType, Domain,
        )
        payload = BackendPayload(
            resume_raw_text="Sample resume text",
            job_description_raw_text="Sample JD text",
            target_role="Backend Engineer",
            parsed_resume=ParsedResume(raw_text="Sample resume"),
            parsed_jd=ParsedJobDescription(
                raw_text="Sample JD",
                target_role="Backend Engineer",
                detected_domain=Domain.BACKEND,
            ),
            semantic_similarity_score=67.5,
            user_proficiencies=[],
            clarification_answers=[],
            selected_template=TemplateType.PROJECT_FOCUSED,
            needs_optimization=True,
        )
        self.assertEqual(payload.semantic_similarity_score, 67.5)
        self.assertTrue(payload.needs_optimization)

    def test_similarity_score_bounds(self):
        from core.schemas import BackendPayload, ParsedResume, ParsedJobDescription, TemplateType, Domain
        with self.assertRaises(Exception):
            BackendPayload(
                resume_raw_text="x",
                job_description_raw_text="y",
                target_role="r",
                parsed_resume=ParsedResume(raw_text="x"),
                parsed_jd=ParsedJobDescription(raw_text="y", target_role="r", detected_domain=Domain.UNKNOWN),
                semantic_similarity_score=150.0,  # invalid
                selected_template=TemplateType.FRESHER_ACADEMIC,
                needs_optimization=False,
            )

    def test_skill_proficiency_normalizes_name(self):
        from core.schemas import SkillProficiency, ProficiencyLevel
        sp = SkillProficiency(skill_name="  Python  ", level=ProficiencyLevel.ADVANCED)
        self.assertEqual(sp.skill_name, "python")


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    test_classes = [
        TestTextCleaner,
        TestSectionSplitter,
        TestMismatchDetector,
        TestInlineSkillExtractor,
        TestNLPPreprocessor,
        TestJaccardSimilarity,
        TestDomainDetection,
        TestTemplateSelection,
        TestOptimizationDecision,
        TestProficiencyEvidenceScores,
        TestSchemas,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    sys.exit(0 if result.wasSuccessful() else 1)