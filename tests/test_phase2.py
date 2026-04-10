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
        self.assertIn("62.5/100", prompt)

    def test_prompt_length_reasonable(self):
        from core.prompt_builder import build_phase2_prompt
        prompt = build_phase2_prompt(self._make_payload())
        # Should be substantial but not absurd
        self.assertGreater(len(prompt), 2000)
        self.assertLess(len(prompt), 50000)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestPromptBuilder]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)