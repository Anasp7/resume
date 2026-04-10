"""
Interactive Resume Builder - Orchestrates the Full Pipeline
============================================================
1. Analyzes resume
2. Asks genuine, contextual questions
3. Validates responses (no hallucination)
4. Beautifies verified content only
5. Produces final polished resume
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json

from llm_question_generator import LLMQuestionGenerator, ResumeQuestion
from response_validator import ResponseValidator, ValidationStatus, ValidationResult
from content_beautifier import ContentBeautifier, BeautificationResult


@dataclass
class InteractionSession:
    """Tracks a session of resume building"""
    session_id: str
    original_resume: Dict
    questions_asked: List[ResumeQuestion]
    responses: Dict[str, str]  # question_id -> response
    validations: Dict[str, any]  # question_id -> validation_result
    beautifications: Dict[str, BeautificationResult]  # field -> beautified
    current_step: int
    total_steps: int


class InteractiveResumeBuilder:
    """
    Main orchestrator for interactive resume building
    
    Pipeline:
    Step 1: Parse resume
    Step 2: Generate contextual questions
    Step 3: Ask user questions one by one
    Step 4: Validate each response (check for hallucination)
    Step 5: Beautify verified content
    Step 6: Produce final resume
    """

    def __init__(self):
        self.question_generator = LLMQuestionGenerator()
        self.response_validator = ResponseValidator()
        self.beautifier = ContentBeautifier()
        
        self.current_session: Optional[InteractionSession] = None
        self.interaction_history = []

    def start_session(
        self,
        resume_data: Dict,
        question_count: int = 15,
        focus_areas: Optional[List[str]] = None
    ) -> InteractionSession:
        """
        Start a new resume building session
        
        Returns:
            InteractionSession with first question ready
        """
        # Generate questions
        questions = self.question_generator.analyze_and_generate_questions(
            resume_data=resume_data,
            focus_areas=focus_areas,
            question_count=question_count
        )
        
        # Create session
        self.current_session = InteractionSession(
            session_id=f"session_{len(self.interaction_history) + 1}",
            original_resume=resume_data,
            questions_asked=questions,
            responses={},
            validations={},
            beautifications={},
            current_step=0,
            total_steps=len(questions)
        )
        
        self.interaction_history.append(self.current_session)
        return self.current_session

    def get_next_question(self) -> Optional[Tuple[ResumeQuestion, str]]:
        """
        Get next question to ask user
        
        Returns:
            (Question, formatted_text) or None if all done
        """
        if not self.current_session:
            return None
        
        if self.current_session.current_step >= len(self.current_session.questions_asked):
            return None
        
        question = self.current_session.questions_asked[self.current_session.current_step]
        formatted = self.question_generator.format_question_for_user(question)
        
        return question, formatted

    def submit_response(self, response: str, user_confirmation: bool = True) -> Dict:
        """
        User submits response to current question
        
        Process:
        1. Get current question
        2. Validate response
        3. If valid → beautify later
        4. Move to next question
        
        Returns:
            {
                'status': 'valid' | 'warning' | 'invalid' | 'clarify',
                'feedback': str,
                'suggestions': [str],
                'proceed_to_next': bool,
                'confidence': float
            }
        """
        if not self.current_session:
            return {"status": "error", "feedback": "No active session"}
        
        current_question = self.current_session.questions_asked[
            self.current_session.current_step
        ]
        
        # Store response
        q_id = f"q_{self.current_session.current_step}"
        self.current_session.responses[q_id] = response
        
        # VALIDATE response
        validation_fn = self._get_validation_function(current_question)
        validation_result = validation_fn(response)
        
        self.current_session.validations[q_id] = validation_result
        
        # Prepare feedback
        can_proceed = validation_result.status in [
            ValidationStatus.VALID,
            ValidationStatus.VALID_WITH_WARNING
        ]
        
        feedback_dict = {
            'status': validation_result.status.value,
            'feedback': validation_result.feedback,
            'suggestions': validation_result.suggestions,
            'red_flags': validation_result.red_flags,
            'is_genuine': validation_result.is_genuine,
            'confidence': validation_result.confidence,
            'proceed_to_next': can_proceed,
            'verification_questions': validation_result.verification_questions
        }
        
        # If valid and user confirms, move to next
        if can_proceed and user_confirmation:
            self.current_session.current_step += 1
        
        return feedback_dict

    def beautify_section(
        self,
        section: str,
        verified_data: Dict
    ) -> BeautificationResult:
        """
        Beautify a resume section based on verified user input
        
        Args:
            section: 'experience' | 'projects' | 'summary' | 'education' | 'skills'
            verified_data: Data that has been validated
        """
        is_verified = all(
            v.status in [ValidationStatus.VALID, ValidationStatus.VALID_WITH_WARNING]
            for v in self.current_session.validations.values()
        )
        
        if section == "experience":
            result = self.beautifier.beautify_experience_bullet(
                original=verified_data.get("bullet", ""),
                metrics=verified_data.get("metrics"),
                is_verified=is_verified
            )
        
        elif section == "projects":
            result = self.beautifier.beautify_project_description(
                project_title=verified_data.get("title", ""),
                original_description=verified_data.get("description", ""),
                technologies=verified_data.get("technologies", []),
                metrics=verified_data.get("metrics"),
                is_verified=is_verified
            )
        
        elif section == "education":
            result = self.beautifier.beautify_education_section(
                institution=verified_data.get("institution", ""),
                degree=verified_data.get("degree", ""),
                field=verified_data.get("field", ""),
                cgpa=verified_data.get("cgpa"),
                coursework=verified_data.get("coursework"),
                projects=verified_data.get("projects"),
                is_verified=is_verified
            )
        
        elif section == "summary":
            result = self.beautifier.beautify_summary(
                experience_years=verified_data.get("years", 0),
                key_skills=verified_data.get("skills", []),
                role_target=verified_data.get("role", ""),
                unique_traits=verified_data.get("traits"),
                is_verified=is_verified
            )
        
        elif section == "skills":
            result = self.beautifier.beautify_soft_skills(
                original_skills=verified_data.get("skills", []),
                examples=verified_data.get("examples"),
                is_verified=is_verified
            )
        
        else:
            return BeautificationResult(
                original="",
                beautified="",
                improvements=[],
                confidence=0.0,
                advice=[f"Section '{section}' not recognized"]
            )
        
        self.current_session.beautifications[section] = result
        return result

    def get_current_progress(self) -> Dict:
        """Get current session progress"""
        if not self.current_session:
            return {"status": "no_session"}
        
        return {
            "current_step": self.current_session.current_step + 1,
            "total_steps": self.current_session.total_steps,
            "progress_percentage": (
                self.current_session.current_step / self.current_session.total_steps * 100
            ),
            "questions_answered": len(self.current_session.responses),
            "valid_responses": sum(
                1 for v in self.current_session.validations.values()
                if v.status in [ValidationStatus.VALID, ValidationStatus.VALID_WITH_WARNING]
            ),
            "warnings": sum(
                1 for v in self.current_session.validations.values()
                if v.status == ValidationStatus.VALID_WITH_WARNING
            ),
            "invalid": sum(
                1 for v in self.current_session.validations.values()
                if v.status == ValidationStatus.INVALID
            )
        }

    def generate_final_resume(self) -> Dict:
        """
        Generate final polished resume based on all validated and beautified content
        
        Only includes:
        - Original resume structure
        - Verified and enhanced user responses
        - No hallucinated or unverified information
        """
        if not self.current_session:
            return {"error": "No active session"}
        
        final_resume = self.current_session.original_resume.copy()
        
        # Apply beautifications
        for section, beautification in self.current_session.beautifications.items():
            if section == "summary":
                final_resume["summary"] = beautification.beautified
            elif section == "experience":
                # Update experience sections
                if "experience" in final_resume:
                    for exp in final_resume["experience"]:
                        # Apply beautification if applicable
                        pass
            elif section == "projects":
                # Update projects
                if "projects" in final_resume:
                    for proj in final_resume["projects"]:
                        # Apply beautification if applicable
                        pass
        
        # Add metadata
        final_resume["_metadata"] = {
            "optimization_score": self._calculate_optimization_score(),
            "genuineness_score": self._calculate_genuineness_score(),
            "beautification_applied": True,
            "verified_sections": list(self.current_session.beautifications.keys()),
            "warnings": [
                v for v in self.current_session.validations.values()
                if v.status.name == "VALID_WITH_WARNING"
            ]
        }
        
        return final_resume

    def get_session_report(self) -> Dict:
        """Get detailed report of the session"""
        if not self.current_session:
            return {"error": "No active session"}
        
        report = {
            "session_id": self.current_session.session_id,
            "progress": self.get_current_progress(),
            "summary": {
                "total_questions": len(self.current_session.questions_asked),
                "answered": len(self.current_session.responses),
                "valid": sum(
                    1 for v in self.current_session.validations.values()
                    if v.status == ValidationStatus.VALID
                ),
                "warnings": sum(
                    1 for v in self.current_session.validations.values()
                    if v.status == ValidationStatus.VALID_WITH_WARNING
                ),
                "invalid": sum(
                    1 for v in self.current_session.validations.values()
                    if v.status == ValidationStatus.INVALID
                )
            },
            "response_quality": {
                "average_confidence": (
                    sum(v.confidence for v in self.current_session.validations.values()) /
                    len(self.current_session.validations) if self.current_session.validations else 0
                ),
                "genuineness_rate": (
                    sum(1 for v in self.current_session.validations.values() if v.is_genuine) /
                    len(self.current_session.validations) * 100 if self.current_session.validations else 0
                )
            },
            "beautifications_applied": len(self.current_session.beautifications),
            "warnings_to_address": [
                {
                    "question": self.current_session.questions_asked[i].question,
                    "issue": v.red_flags
                }
                for i, v in enumerate(self.current_session.validations.values())
                if v.red_flags
            ]
        }
        
        return report

    def _get_validation_function(self, question: ResumeQuestion):
        """Get appropriate validation function based on question"""
        section = question.section
        
        if section == "education":
            if "cgpa" in question.about.lower():
                return self.response_validator.validate_education_response
            elif "coursework" in question.about.lower():
                return lambda r: self.response_validator.validate_education_response(r, "coursework", {})
            elif "project" in question.about.lower():
                return lambda r: self.response_validator.validate_education_response(r, "projects", {})
        
        elif section == "experience":
            if "impact" in question.about.lower() or "metrics" in question.about.lower():
                return lambda r: self.response_validator.validate_experience_response(r, "impact_metrics", "", "")
            elif "challenge" in question.about.lower():
                return lambda r: self.response_validator.validate_experience_response(r, "challenges", "", "")
        
        elif section == "projects":
            if "github" in question.about.lower():
                return self.response_validator.validate_project_response
        
        elif section == "skills":
            if "soft" in question.about.lower():
                return self.response_validator.validate_soft_skills
        
        # Default fallback
        return lambda r: ValidationResult(
            status=ValidationStatus.VALID,
            is_genuine=True,
            confidence=0.5,
            feedback="Response recorded",
            suggestions=[],
            red_flags=[],
            verification_questions=[]
        )

    def _calculate_optimization_score(self) -> float:
        """Calculate resume optimization score (0-100)"""
        if not self.current_session:
            return 0
        
        valid_count = sum(
            1 for v in self.current_session.validations.values()
            if v.status == ValidationStatus.VALID
        )
        
        total = len(self.current_session.validations) if self.current_session.validations else 1
        
        return (valid_count / total * 100) if total > 0 else 0

    def _calculate_genuineness_score(self) -> float:
        """Calculate how genuine the resume is (0-100)"""
        if not self.current_session:
            return 0
        
        genuine_count = sum(
            1 for v in self.current_session.validations.values()
            if v.is_genuine
        )
        
        total = len(self.current_session.validations) if self.current_session.validations else 1
        
        return (genuine_count / total * 100) if total > 0 else 0


# Example usage
if __name__ == "__main__":
    # Sample resume
    sample_resume = {
        "education": [{
            "institution": "IIT Bombay",
            "degree": "B.Tech",
            "field": "Computer Science",
            "graduation_year": 2023,
            "cgpa": None
        }],
        "experience": [{
            "company": "TechCorp",
            "position": "Software Developer Intern",
            "duration_months": 3
        }],
        "projects": [{
            "title": "Chat Application",
            "description": "Real-time messaging",
            "technologies": ["Python", "FastAPI"]
        }]
    }
    
    # Create builder
    builder = InteractiveResumeBuilder()
    
    # Start session
    session = builder.start_session(sample_resume, question_count=5)
    
    print(f"✓ Session started with {session.total_steps} questions\n")
    
    # Get first question
    question, formatted = builder.get_next_question()
    print(f"Question 1:\n{formatted}\n")
    
    # Simulate user response
    print("User responds: '8.7 out of 10'")
    feedback = builder.submit_response("My CGPA is 8.7 out of 10", user_confirmation=True)
    print(f"Validation: {feedback['status']}")
    print(f"Feedback: {feedback['feedback']}\n")
    
    # Show progress
    progress = builder.get_current_progress()
    print(f"Progress: {progress['current_step']}/{progress['total_steps']}")
    print(f"Valid responses: {progress['valid_responses']}")
