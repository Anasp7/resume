"""
Response Validator - Checks for Hallucination & Inconsistency
==============================================================
Validates user responses to ensure they're genuine and consistent.
Prevents hallucination, inflated claims, and suspicious data.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import re


class ValidationStatus(Enum):
    """Status of validation"""
    VALID = "valid"                    # Response is genuine and can be used
    VALID_WITH_WARNING = "warning"     # Valid but suspicious, ask to verify
    INVALID = "invalid"                # Response is inconsistent or hallucinated
    NEEDS_CLARIFICATION = "clarify"    # Response needs more details


@dataclass
class ValidationResult:
    """Result of validating a response"""
    status: ValidationStatus
    is_genuine: bool                   # Is this a genuine answer?
    confidence: float                  # 0-1 confidence level
    feedback: str                      # What to tell user
    suggestions: List[str]             # How to fix/improve
    red_flags: List[str]               # Any suspicious claims
    verification_questions: List[str]  # Questions to verify authenticity


class ResponseValidator:
    """
    Validates user responses for:
    - Consistency with resume data
    - Plausibility (not hallucinated)
    - Authenticity (not inflated)
    - Data format correctness
    """

    def validate_education_response(
        self,
        response: str,
        field: str,
        resume_data: Dict
    ) -> ValidationResult:
        """
        Validate education-related responses
        
        Args:
            response: User's response
            field: What field (cgpa, coursework, projects, etc.)
            resume_data: Existing resume data for consistency check
        """
        if field == "cgpa":
            return self._validate_cgpa(response)
        elif field == "coursework":
            return self._validate_coursework(response)
        elif field == "college_projects":
            return self._validate_college_projects(response)
        
        return ValidationResult(
            status=ValidationStatus.VALID,
            is_genuine=True,
            confidence=0.5,
            feedback="Response recorded",
            suggestions=[],
            red_flags=[],
            verification_questions=[]
        )

    def _validate_cgpa(self, response: str) -> ValidationResult:
        """Validate CGPA/Percentage response"""
        red_flags = []
        suggestions = []
        
        # Extract number from response
        match = re.search(r'(\d+\.?\d*)', response)
        if not match:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                is_genuine=False,
                confidence=0.0,
                feedback="❌ CGPA not found. Please provide as: '8.7/10' or '87%'",
                suggestions=["Format: 8.7/10 or 87%", "Example: 'My CGPA is 8.5 out of 10'"],
                red_flags=["No numeric value provided"],
                verification_questions=[]
            )
        
        cgpa = float(match.group(1))
        
        # Check range
        if cgpa > 10 and "/10" not in response and "%" not in response:
            red_flags.append(f"CGPA {cgpa} appears to be out of 100 format, not 10")
            suggestions.append("Clarify: Is this out of 10 or 100? (8.7/10 or 87/100)?")
        
        if cgpa > 100:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                is_genuine=False,
                confidence=0.0,
                feedback=f"❌ CGPA {cgpa} is invalid (max 10 or 100)",
                suggestions=["CGPA should be 0-10 or 0-100"],
                red_flags=["CGPA value exceeds maximum"],
                verification_questions=[]
            )
        
        if cgpa < 2.0:
            red_flags.append(f"Unusual CGPA: {cgpa}. Very low but possible.")
            feedback = f"⚠️ CGPA {cgpa} is quite low. You can skip CGPA if <7."
            status = ValidationStatus.VALID_WITH_WARNING
        else:
            feedback = f"✓ CGPA {cgpa} recorded"
            status = ValidationStatus.VALID
        
        return ValidationResult(
            status=status,
            is_genuine=True,
            confidence=0.95 if not red_flags else 0.7,
            feedback=feedback,
            suggestions=suggestions,
            red_flags=red_flags,
            verification_questions=[]
        )

    def _validate_coursework(self, response: str) -> ValidationResult:
        """Validate coursework/courses response"""
        red_flags = []
        suggestions = []
        
        # Split by comma
        courses = [c.strip() for c in response.split(",")]
        
        if len(courses) < 3:
            return ValidationResult(
                status=ValidationStatus.NEEDS_CLARIFICATION,
                is_genuine=True,
                confidence=0.6,
                feedback=f"⚠️ You listed {len(courses)} courses. Consider 3-5 relevant ones.",
                suggestions=["Add 1-2 more relevant courses"],
                red_flags=[],
                verification_questions=[]
            )
        
        # Check for generic courses
        generic_keywords = ["general", "intro", "basics", "seminar", "lab"]
        for course in courses:
            if any(keyword in course.lower() for keyword in generic_keywords):
                suggestions.append(f"'{course}' sounds generic. Consider removing if better alternatives exist")
        
        # Check for real CS courses
        cs_keywords = ["data structures", "algorithms", "database", "web", "machine learning", 
                      "system design", "os", "networks", "compiler", "graphics"]
        cs_course_count = sum(1 for c in courses 
                             if any(kw in c.lower() for kw in cs_keywords))
        
        if cs_course_count < len(courses) * 0.5:
            red_flags.append("Most courses don't sound CS-related")
            suggestions.append("Include more CS core courses (Algorithms, Databases, etc.)")
        
        return ValidationResult(
            status=ValidationStatus.VALID if not red_flags else ValidationStatus.VALID_WITH_WARNING,
            is_genuine=True,
            confidence=0.85,
            feedback=f"✓ Recorded {len(courses)} courses",
            suggestions=suggestions,
            red_flags=red_flags,
            verification_questions=["Do all these courses relate to your target role?"]
        )

    def _validate_college_projects(self, response: str) -> ValidationResult:
        """Validate college project response"""
        if response.lower() in ["no", "none", "nope", "didn't do any"]:
            return ValidationResult(
                status=ValidationStatus.VALID,
                is_genuine=True,
                confidence=1.0,
                feedback="✓ No college projects. That's fine, your own projects count.",
                suggestions=[],
                red_flags=[],
                verification_questions=[]
            )
        
        # Project exists
        return ValidationResult(
            status=ValidationStatus.VALID,
            is_genuine=True,
            confidence=0.9,
            feedback="✓ College project recorded. Make sure to highlight it!",
            suggestions=["Add this to your Projects section with GitHub link if available"],
            red_flags=[],
            verification_questions=[
                "Do you have code available (GitHub)?",
                "How many people were in your team?"
            ]
        )

    def validate_experience_response(
        self,
        response: str,
        field: str,
        company: str,
        role: str
    ) -> ValidationResult:
        """Validate experience-related responses"""
        if field == "impact_metrics":
            return self._validate_metrics(response)
        elif field == "challenges":
            return self._validate_challenge(response)
        elif field == "reason_for_leaving":
            return self._validate_reason(response)
        
        return ValidationResult(
            status=ValidationStatus.VALID,
            is_genuine=True,
            confidence=0.7,
            feedback="Response recorded",
            suggestions=[],
            red_flags=[],
            verification_questions=[]
        )

    def _validate_metrics(self, response: str) -> ValidationResult:
        """Validate metric claims"""
        red_flags = []
        suggestions = []
        
        # Check for numbers
        numbers = re.findall(r'\d+', response)
        if not numbers:
            return ValidationResult(
                status=ValidationStatus.NEEDS_CLARIFICATION,
                is_genuine=True,
                confidence=0.5,
                feedback="⚠️ No specific metric found. Can you add a number?",
                suggestions=[
                    "Example: 'Improved response time from 2s to 500ms'",
                    "Example: 'Handled 10,000 concurrent users'",
                    "Example: '60% reduction in database queries'"
                ],
                red_flags=[],
                verification_questions=[]
            )
        
        # Check for inflated claims
        for num_str in numbers:
            num = int(num_str)
            
            # Red flags for unrealistic claims
            if num > 1000000:
                red_flags.append(f"Claim of {num} seems very large. Verify this is real.")
            elif "%" in response:
                percentage = num
                if percentage > 100:
                    red_flags.append(f"{percentage}% improvement is impossible")
                    suggestions.append("Percentages can't exceed 100%")
                elif percentage > 80:
                    red_flags.append(f"{percentage}% improvement is very high. Verify calculation.")
                    suggestions.append("Double-check your math. Ask: How would your manager verify this?")
        
        # Check for specific vs. vague
        vague_words = ["improved", "optimized", "enhanced", "better"]
        specific_words = ["reduced", "increased", "achieved", "handled"]
        
        is_specific = any(word in response.lower() for word in specific_words)
        is_vague = any(word in response.lower() for word in vague_words)
        
        if is_vague and not is_specific:
            suggestions.append("Be more specific: 'reduced latency from 2s to 500ms' not just 'improved performance'")
        
        status = ValidationStatus.VALID if not red_flags else ValidationStatus.VALID_WITH_WARNING
        
        return ValidationResult(
            status=status,
            is_genuine=True,
            confidence=0.75 if not red_flags else 0.5,
            feedback="✓ Metric recorded" if not red_flags else "⚠️ Metric looks suspicious",
            suggestions=suggestions,
            red_flags=red_flags,
            verification_questions=[
                "How would your manager verify this metric?",
                "Did you measure this or estimate?"
            ]
        )

    def _validate_challenge(self, response: str) -> ValidationResult:
        """Validate technical challenge response"""
        suggestions = []
        
        # Check length (should have detail)
        if len(response) < 50:
            return ValidationResult(
                status=ValidationStatus.NEEDS_CLARIFICATION,
                is_genuine=True,
                confidence=0.4,
                feedback="⚠️ Challenge description is too brief. More details needed.",
                suggestions=[
                    "Describe: What was the problem? What was difficult? How did you solve it?",
                    "Length: Aim for 2-3 sentences"
                ],
                red_flags=[],
                verification_questions=[]
            )
        
        # Check for technical depth
        if "bug" in response.lower() or "error" in response.lower():
            suggestions.append("Could you describe what type of bug/error? (Memory leak, logic error, etc.)")
        
        # Generic challenges
        generic = ["worked with team", "learned new tool", "met deadline"]
        if any(g in response.lower() for g in generic):
            suggestions.append("This sounds more like responsibility than technical challenge")
        
        return ValidationResult(
            status=ValidationStatus.VALID if not suggestions else ValidationStatus.VALID_WITH_WARNING,
            is_genuine=True,
            confidence=0.8,
            feedback="✓ Challenge recorded",
            suggestions=suggestions,
            red_flags=[],
            verification_questions=[
                "What specific technology caused the challenge?",
                "How long did it take to solve?"
            ]
        )

    def _validate_reason(self, response: str) -> ValidationResult:
        """Validate reason for leaving"""
        red_flags = []
        suggestions = []
        
        # Check for negative language
        negative_words = ["bad", "terrible", "awful", "incompetent", "rude", "stolen"]
        if any(word in response.lower() for word in negative_words):
            red_flags.append("Response sounds negative or blaming")
            suggestions.append("Reframe positively: Focus on what you wanted to do, not what was wrong")
        
        # Check for good signals
        positive_words = ["learn", "growth", "opportunity", "next step", "full-time"]
        has_positive = any(word in response.lower() for word in positive_words)
        
        if not has_positive:
            suggestions.append("Try to emphasize learning/growth: 'Wanted to explore full-stack development'")
        
        status = ValidationStatus.VALID if not red_flags else ValidationStatus.INVALID
        
        return ValidationResult(
            status=status,
            is_genuine=True,
            confidence=0.7 if not red_flags else 0.0,
            feedback="✓ Response recorded" if not red_flags else "❌ Response sounds negative. Rephrase.",
            suggestions=suggestions,
            red_flags=red_flags,
            verification_questions=[]
        )

    def validate_project_response(
        self,
        response: str,
        field: str,
        project_name: str
    ) -> ValidationResult:
        """Validate project-related responses"""
        if field == "github_link":
            return self._validate_github(response)
        elif field == "live_link":
            return self._validate_url(response, "live link")
        elif field == "duration":
            return self._validate_duration(response)
        elif field == "metrics":
            return self._validate_project_metrics(response)
        elif field == "learning":
            return self._validate_learning(response)
        
        return ValidationResult(
            status=ValidationStatus.VALID,
            is_genuine=True,
            confidence=0.7,
            feedback="Response recorded",
            suggestions=[],
            red_flags=[],
            verification_questions=[]
        )

    def _validate_github(self, response: str) -> ValidationResult:
        """Validate GitHub link"""
        red_flags = []
        suggestions = []
        
        # Check for common "no repo" answers
        if response.lower() in ["no", "none", "don't have", "nope"]:
            return ValidationResult(
                status=ValidationStatus.VALID,
                is_genuine=True,
                confidence=1.0,
                feedback="⚠️ GitHub link missing. This will hurt your application.",
                suggestions=[
                    "Create a GitHub repo from your project",
                    "Add this project's code to GitHub",
                    "It doesn't need to be perfect, showing code is key"
                ],
                red_flags=["No GitHub link provided"],
                verification_questions=[]
            )
        
        # Check for GitHub URL format
        if "github.com/" in response:
            if not re.match(r'https?://github\.com/[\w-]+/[\w-]+', response):
                red_flags.append("GitHub URL format looks incorrect")
                suggestions.append("Correct format: https://github.com/username/repo-name")
        else:
            red_flags.append("Doesn't look like a GitHub URL")
            suggestions.append("Expected: https://github.com/username/repo-name")
        
        status = ValidationStatus.VALID if not red_flags else ValidationStatus.VALID_WITH_WARNING
        
        return ValidationResult(
            status=status,
            is_genuine=True,
            confidence=0.9 if not red_flags else 0.6,
            feedback="✓ GitHub link recorded" if not red_flags else "⚠️ GitHub link looks odd",
            suggestions=suggestions,
            red_flags=red_flags,
            verification_questions=[
                "Is this a public repo?",
                "Is the code well-organized and commented?"
            ]
        )

    def _validate_url(self, response: str, link_type: str) -> ValidationResult:
        """Validate any URL"""
        if response.lower() in ["no", "none", "nope"]:
            return ValidationResult(
                status=ValidationStatus.VALID,
                is_genuine=True,
                confidence=0.9,
                feedback=f"✓ No {link_type} provided",
                suggestions=[f"Optional: Adding a {link_type} makes it impressive"],
                red_flags=[],
                verification_questions=[]
            )
        
        # Check URL format
        if not re.match(r'https?://', response):
            red_flags = ["URL doesn't start with http:// or https://"]
        else:
            red_flags = []
        
        return ValidationResult(
            status=ValidationStatus.VALID if not red_flags else ValidationStatus.VALID_WITH_WARNING,
            is_genuine=True,
            confidence=0.8,
            feedback=f"✓ {link_type} recorded" if not red_flags else f"⚠️ {link_type} format odd",
            suggestions=["URL should start with https://"],
            red_flags=red_flags,
            verification_questions=[]
        )

    def _validate_duration(self, response: str) -> ValidationResult:
        """Validate project duration"""
        red_flags = []
        suggestions = []
        
        # Extract number
        match = re.search(r'(\d+)', response)
        if not match:
            return ValidationResult(
                status=ValidationStatus.NEEDS_CLARIFICATION,
                is_genuine=True,
                confidence=0.3,
                feedback="❌ Duration not clear. Provide as: '3 months', '8 weeks', '6 months'",
                suggestions=[],
                red_flags=[],
                verification_questions=[]
            )
        
        duration = int(match.group(1))
        
        # Check if reasonable
        if duration > 60:  # More than 5 years
            red_flags.append(f"Duration {duration}+ looks very long")
            suggestions.append("Projects typically take weeks/months, not years")
        elif duration < 1:
            red_flags.append("Duration < 1 doesn't seem right")
            suggestions.append("Even small projects take at least 1 week")
        
        status = ValidationStatus.VALID if not red_flags else ValidationStatus.VALID_WITH_WARNING
        
        return ValidationResult(
            status=status,
            is_genuine=True,
            confidence=0.9 if not red_flags else 0.6,
            feedback=f"✓ Duration {duration} recorded" if not red_flags else f"⚠️ Duration {duration} seems odd",
            suggestions=suggestions,
            red_flags=red_flags,
            verification_questions=[]
        )

    def _validate_project_metrics(self, response: str) -> ValidationResult:
        """Validate project metrics (users, stars, etc.)"""
        red_flags = []
        suggestions = []
        
        if response.lower() in ["no", "none", "0"]:
            return ValidationResult(
                status=ValidationStatus.VALID,
                is_genuine=True,
                confidence=0.9,
                feedback="✓ No metrics yet. That's okay for new projects.",
                suggestions=["If others use it, even 1 user is impressive to mention"],
                red_flags=[],
                verification_questions=[]
            )
        
        # Extract numbers
        numbers = re.findall(r'\d+', response)
        if numbers:
            num = int(numbers[0])
            
            if num > 100000:
                red_flags.append(f"Metrics {num} seems inflated. Verify against GitHub stars/forks")
                suggestions.append("Be honest: If it's 50 stars, say 50 not 5000")
        else:
            suggestions.append("Add a specific number if possible (users, stars, downloads, etc.)")
        
        status = ValidationStatus.VALID if not red_flags else ValidationStatus.VALID_WITH_WARNING
        
        return ValidationResult(
            status=status,
            is_genuine=True,
            confidence=0.8,
            feedback="✓ Metrics recorded",
            suggestions=suggestions,
            red_flags=red_flags,
            verification_questions=[
                "How are you measuring this? (GitHub stars? Direct usage?)",
                "Can you verify this against public metrics?"
            ]
        )

    def _validate_learning(self, response: str) -> ValidationResult:
        """Validate learning statement"""
        red_flags = []
        suggestions = []
        
        if len(response) < 30:
            suggestions.append("Learning statement is too brief. Explain what you learned and why it matters.")
        
        # Check if generic
        generic = ["learned a lot", "gained experience", "improved myself"]
        if any(g in response.lower() for g in generic):
            suggestions.append("Be specific: 'Learned scalable database design vs learned a lot'")
        
        # Check for technical depth
        if "-" in response and ":" in response:  # Good structure
            pass
        else:
            suggestions.append("Structure: 'What: Database design, Why: Needed for millions of users'")
        
        return ValidationResult(
            status=ValidationStatus.VALID if not suggestions else ValidationStatus.VALID_WITH_WARNING,
            is_genuine=True,
            confidence=0.85,
            feedback="✓ Learning recorded",
            suggestions=suggestions,
            red_flags=red_flags,
            verification_questions=[
                "Is this learning directly applicable to your target role?",
                "Can you explain this concept in an interview?"
            ]
        )

    def validate_soft_skills(self, response: str) -> ValidationResult:
        """Validate soft skills response"""
        red_flags = []
        suggestions = []
        
        skills = [s.strip() for s in response.split(",")]
        
        # Check count
        if len(skills) < 3:
            suggestions.append("List 3-5 soft skills for better coverage")
        elif len(skills) > 7:
            red_flags.append("Too many soft skills listed")
            suggestions.append("Keep top 5 that matter for your role")
        
        # Check for technical skills by mistake
        technical = ["python", "react", "sql", "docker", "aws", "javascript", "java"]
        for skill in skills:
            if any(tech in skill.lower() for tech in technical):
                red_flags.append(f"'{skill}' sounds technical, not soft skill")
                suggestions.append("Soft skills: Communication, Leadership, Problem-solving")
        
        status = ValidationStatus.VALID if not red_flags else ValidationStatus.VALID_WITH_WARNING
        
        return ValidationResult(
            status=status,
            is_genuine=True,
            confidence=0.9,
            feedback="✓ Soft skills recorded",
            suggestions=suggestions,
            red_flags=red_flags,
            verification_questions=[
                "Can you give examples for each soft skill?"
            ]
        )


# Example usage
if __name__ == "__main__":
    validator = ResponseValidator()
    
    # Test CGPA validation
    result = validator._validate_cgpa("My CGPA is 8.7 out of 10")
    print(f"CGPA Validation: {result.status.value}")
    print(f"Confident: {result.confidence}")
    print(f"Feedback: {result.feedback}\n")
    
    # Test metric validation
    result = validator._validate_metrics("Improved performance by 40%")
    print(f"Metrics Validation: {result.status.value}")
    print(f"Confidence: {result.confidence}")
    print(f"Suggestions: {result.suggestions}\n")
    
    # Test GitHub validation
    result = validator._validate_github("https://github.com/johndoe/chat-app")
    print(f"GitHub Validation: {result.status.value}")
    print(f"Feedback: {result.feedback}\n")
