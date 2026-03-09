"""
Smart Resume — Phase 2 Tests
==============================
Tests prompt builder, JSON extraction, response builder.
No API key required — mocks Claude response.
Run: python tests/test_phase2.py
"""
import sys, os, json, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPromptBuilder(unittest.TestCase):
    def _make_payload(self):
        from core.schemas import (
            BackendPayload, ParsedResume, ParsedJobDescription,
            TemplateType, Domain, SkillProficiency, ProficiencyLevel,
            ProjectEntry, ExperienceEntry, EducationEntry,
        )
        return BackendPayload(
            resume_raw_text="ANAS ABDUL NAZAR P\nStudent\nPython ROS2 SQL",
            job_description_raw_text="ML Engineer with Python and TensorFlow",
            target_role="ML Engineer",
            parsed_resume=ParsedResume(
                raw_text="ANAS ABDUL NAZAR P\nStudent\nPython ROS2 SQL",
                skills=["python", "ros2", "sql"],
                experience=[ExperienceEntry(
                    company="BAJA",
                    role="Automation Team Member",
                    duration="2025-Present",
                    responsibilities=["Develop autonomous vehicle"],
                    technologies=["python", "ros2"],
                )],
                projects=[ProjectEntry(
                    title="Alumni Connect Platform",
                    description="Backend developer for DBMS project",
                    technologies=["sql", "python"],
                )],
                education=[EducationEntry(
                    institution="MA College of Engineering",
                    degree="Bachelor of Technology",
                    graduation_year=2026,
                )],
                years_of_experience=0.0,
                project_count=1,
                experience_count=1,
            ),
            parsed_jd=ParsedJobDescription(
                raw_text="ML Engineer with Python and TensorFlow",
                target_role="ML Engineer",
                detected_domain=Domain.ML,
                required_skills=["python", "tensorflow"],
            ),
            semantic_similarity_score=62.5,
            user_proficiencies=[
                SkillProficiency(skill_name="python", level=ProficiencyLevel.INTERMEDIATE)
            ],
            selected_template=TemplateType.FRESHER_ACADEMIC,
            needs_optimization=True,
            session_id="test-session-123",
        )

    def test_prompt_contains_key_sections(self):
        from core.prompt_builder import build_phase2_prompt
        payload = self._make_payload()
        prompt  = build_phase2_prompt(payload)
        self.assertIn("62.5", prompt)
        self.assertIn("Fresher-Academic", prompt)
        self.assertIn("python", prompt.lower())
        self.assertIn("structured_extraction", prompt)
        self.assertIn("career_improvement_plan", prompt)

    def test_prompt_contains_mismatch_instruction(self):
        from core.prompt_builder import build_phase2_prompt
        prompt = build_phase2_prompt(self._make_payload())
        self.assertIn("mismatch_corrections", prompt)

    def test_prompt_injects_similarity_score(self):
        from core.prompt_builder import build_phase2_prompt
        prompt = build_phase2_prompt(self._make_payload())
        self.assertIn("62.5 / 100", prompt)

    def test_prompt_length_reasonable(self):
        from core.prompt_builder import build_phase2_prompt
        prompt = build_phase2_prompt(self._make_payload())
        # Should be substantial but not absurd
        self.assertGreater(len(prompt), 2000)
        self.assertLess(len(prompt), 50000)


class TestJSONExtraction(unittest.TestCase):
    def setUp(self):
        from core.evaluator import _extract_json
        self.extract = _extract_json

    def test_clean_json(self):
        data = {"key": "value", "number": 42}
        result = self.extract(json.dumps(data))
        self.assertEqual(result["key"], "value")

    def test_json_with_fences(self):
        raw = '```json\n{"key": "value"}\n```'
        result = self.extract(raw)
        self.assertEqual(result["key"], "value")

    def test_json_with_preamble(self):
        raw = 'Here is the JSON:\n{"key": "value"}'
        result = self.extract(raw)
        self.assertEqual(result["key"], "value")

    def test_invalid_json_returns_fallback(self):
        from core.evaluator import _fallback_response
        result = self.extract("this is not json at all")
        # Should return fallback with expected keys
        self.assertIn("structured_extraction", result)
        self.assertIn("career_improvement_plan", result)

    def test_nested_json(self):
        data = {
            "structured_extraction": {"skills": ["python", "sql"]},
            "job_match_analysis": {"verdict": "Strong"},
        }
        result = self.extract(json.dumps(data))
        self.assertEqual(result["job_match_analysis"]["verdict"], "Strong")


class TestResponseBuilder(unittest.TestCase):
    def setUp(self):
        from core.evaluator import _build_response
        self.build = _build_response

    def _sample_data(self):
        return {
            "structured_extraction":     {"summary": "Good candidate", "skills": ["python"], "mismatch_corrections": ["Alumni Connect moved to experience"]},
            "skill_classification":      {"programming_languages": ["python"], "frameworks_libraries": [], "tools_platforms": [], "databases": ["sql"], "core_cs_concepts": []},
            "job_match_analysis":        {"domain": "ML", "alignment_score_reasoning": "Partial match", "matched_skills": ["python"], "missing_skills": ["tensorflow"], "alignment_gaps": [], "verdict": "Moderate"},
            "doubt_detection":           {"clarification_required": True, "issues": [{"type": "unsupported_skill", "description": "ROS2 expertise unclear", "questions": ["Describe a ROS2 project you built", "What ROS2 version did you use?"]}]},
            "proficiency_consistency":   {"analysis": [{"skill": "python", "declared_level": "Intermediate", "evidence_level": 2, "aligned": True, "reasoning": "Used in projects"}], "overall_assessment": "Generally consistent"},
            "factual_evaluation":        {"technical_depth": "Moderate", "logical_consistency": "Consistent", "metric_realism": "No Metrics", "skill_alignment": "Partial", "confidence_narrative": "Reasonable"},
            "internal_consistency":      {"timeline_alignment": "OK", "skill_usage_coverage": "Medium", "claim_density": "Appropriate", "cross_section_coherence": "Coherent", "flags": []},
            "resume_quality_assessment": {"structure_clarity": "Medium", "ats_compatibility": "High", "role_relevance": "Low", "redundancy_level": "Low", "overall_quality": "Acceptable", "improvements": ["Add metrics", "Tailor to ML roles"]},
            "template_selection":        {"selected": "Fresher-Academic", "justification": "Student with no full-time experience"},
            "final_resume":              "ANAS ABDUL NAZAR P\n---\nEDUCATION\nMA College of Engineering",
            "career_improvement_plan":   {"target_domain": "ML", "missing_skills_to_learn": [{"skill": "PyTorch", "reason": "Core ML framework", "resource": "fast.ai course"}], "suggested_projects": [{"title": "ML Pipeline", "description": "Build end-to-end training pipeline"}], "learning_roadmap": [{"week": "1-2", "focus": "Python ML libraries"}]},
        }

    def test_response_has_all_fields(self):
        from core.schemas import SmartResumeResponse
        result = self.build(self._sample_data(), "test-123")
        self.assertIsInstance(result, SmartResumeResponse)
        self.assertEqual(result.session_id, "test-123")

    def test_clarification_required_flag(self):
        result = self.build(self._sample_data(), "test-123")
        self.assertTrue(result.clarification_required)

    def test_clarification_questions_extracted(self):
        result = self.build(self._sample_data(), "test-123")
        self.assertGreater(len(result.clarification_questions), 0)
        self.assertIn("Describe a ROS2 project you built", result.clarification_questions)

    def test_mismatch_corrections_extracted(self):
        result = self.build(self._sample_data(), "test-123")
        self.assertIn("Alumni Connect moved to experience", result.mismatch_corrections)

    def test_final_resume_present(self):
        result = self.build(self._sample_data(), "test-123")
        self.assertIsNotNone(result.final_resume)
        self.assertIn("ANAS ABDUL NAZAR P", result.final_resume)

    def test_career_plan_skills(self):
        result = self.build(self._sample_data(), "test-123")
        skills = result.career_improvement_plan.get("missing_skills_to_learn", [])
        self.assertEqual(skills[0]["skill"], "PyTorch")


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestPromptBuilder, TestJSONExtraction, TestResponseBuilder]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)