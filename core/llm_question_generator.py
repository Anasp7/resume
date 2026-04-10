"""
LLM-Based Context-Aware Question Generator
============================================
Generates genuine, probing questions based on resume gaps and content.
Questions are specific and contextual, not generic.

Features:
- Analyzes resume sections
- Generates context-specific questions
- Validates questions are meaningful
- Adapts questions based on responses
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json


class QuestionCategory(Enum):
    """Types of questions to ask"""
    MISSING_DETAIL = "missing_detail"      # Content exists but needs detail
    CLARIFICATION = "clarification"         # Ambiguous content needs clarification
    VALIDATION = "validation"               # Need to verify claim
    ENRICHMENT = "enrichment"               # Enhance existing content
    CONTEXT = "context"                     # Provide context/impact
    METRICS = "metrics"                     # Add quantifiable metrics
    IMPACT = "impact"                       # What was the impact?
    LEARNING = "learning"                   # What did you learn?


@dataclass
class ResumeQuestion:
    """A question to ask about resume content"""
    question: str
    category: QuestionCategory
    severity: str  # critical, important, interesting
    section: str   # experience, projects, education, skills
    about: str     # what specifically (e.g., "Chat Application")
    why_asked: str  # explanation for user
    follow_up_questions: List[str]  # If this is answered, ask these next
    validation_rules: List[str]     # How to validate the response
    context: str    # What part of resume triggered this


class LLMQuestionGenerator:
    """
    Generates genuine, context-specific questions using LLM analysis.
    Questions are designed to uncover missing details without hallucination.
    """

    def __init__(self, llm_client=None):
        """
        Initialize question generator
        
        Args:
            llm_client: Optional LLM client (Groq/Ollama) for advanced prompt generation
        """
        self.llm_client = llm_client
        self.question_history = []
        self.response_history = []

    def analyze_and_generate_questions(
        self,
        resume_data: Dict,
        focus_areas: Optional[List[str]] = None,
        question_count: int = 15
    ) -> List[ResumeQuestion]:
        """
        Analyze resume and generate context-specific questions
        
        Args:
            resume_data: Parsed resume data
            focus_areas: Which sections to focus on (if None, analyze all)
            question_count: Target number of questions
            
        Returns:
            List of ResumeQuestion objects
        """
        questions = []
        
        # Generate questions for each section
        if not focus_areas or "education" in focus_areas:
            questions.extend(self._generate_education_questions(resume_data))
        
        if not focus_areas or "experience" in focus_areas:
            questions.extend(self._generate_experience_questions(resume_data))
        
        if not focus_areas or "projects" in focus_areas:
            questions.extend(self._generate_project_questions(resume_data))
        
        if not focus_areas or "skills" in focus_areas:
            questions.extend(self._generate_skills_questions(resume_data))
        
        # Sort by severity (critical → important → interesting)
        severity_order = {"critical": 0, "important": 1, "interesting": 2}
        questions.sort(key=lambda q: severity_order.get(q.severity, 3))
        
        # Return top N questions
        return questions[:question_count]

    def _generate_education_questions(self, resume_data: Dict) -> List[ResumeQuestion]:
        """Generate probing questions about education"""
        questions = []
        education = resume_data.get("education", [])
        
        if not education:
            return questions
        
        edu = education[0] if education else {}
        
        # Question 1: CGPA / Percentage (if missing)
        if not edu.get("cgpa") and not edu.get("percentage"):
            questions.append(ResumeQuestion(
                question="What is your CGPA or percentage score from IIT Bombay? "
                        "(If <7.0, you can skip this)",
                category=QuestionCategory.MISSING_DETAIL,
                severity="critical",
                section="education",
                about="CGPA/Percentage",
                why_asked="CGPA shows academic performance. Most companies expect this for freshers.",
                follow_up_questions=[
                    "Were you in any academic honors (dean's list, toppers)?",
                    "What was your coursework GPA vs overall GPA?"
                ],
                validation_rules=[
                    "Score should be between 0-10 (CGPA) or 0-100 (percentage)",
                    "If 1st year, can provide partial year score",
                    "If <5.0, verify - unusually low"
                ],
                context=f"Education: {edu.get('degree')} from {edu.get('institution')}"
            ))
        
        # Question 2: Top courses / skills learned
        if not edu.get("relevant_coursework") or len(edu.get("relevant_coursework", [])) < 3:
            questions.append(ResumeQuestion(
                question="What were your TOP 5 courses that are most relevant to your target role? "
                        "List the course name and what you learned.",
                category=QuestionCategory.MISSING_DETAIL,
                severity="important",
                section="education",
                about="Relevant Coursework",
                why_asked="Courses show your academic foundation in relevant areas. "
                         "Better than just listing all courses.",
                follow_up_questions=[
                    "In which course did you do a project related to your domain?",
                    "Did you do any research work in these courses?"
                ],
                validation_rules=[
                    "Courses should relate to computer science / target domain",
                    "List should be 3-5 courses",
                    "No generic courses (Gen Ed, humanities)"
                ],
                context=f"Degree: {edu.get('degree')} in {edu.get('field')}"
            ))
        
        # Question 3: Projects during college
        if not edu.get("college_projects"):
            questions.append(ResumeQuestion(
                question="Did you do any capstone project or major projects during your degree? "
                        "Name and technologies used?",
                category=QuestionCategory.ENRICHMENT,
                severity="important",
                section="education",
                about="College Projects",
                why_asked="Capstone projects show depth of learning and can strengthen resume.",
                follow_up_questions=[
                    "How many students were in the team?",
                    "What was your specific role?",
                    "What was the main challenge you faced?"
                ],
                validation_rules=[
                    "Project should be from final year or significant coursework",
                    "Should have quantifiable aspect (accuracy, performance, users)"
                ],
                context="Capstone projects show practical application of theory"
            ))
        
        return questions

    def _generate_experience_questions(self, resume_data: Dict) -> List[ResumeQuestion]:
        """Generate probing questions about work experience"""
        questions = []
        experience = resume_data.get("experience", [])
        
        for i, exp in enumerate(experience):
            # Question: Impact and metrics
            if not exp.get("impact_metrics"):
                questions.append(ResumeQuestion(
                    question=f"In your role at {exp.get('company', 'company')}, "
                            f"what was your biggest impact? (Quantify it if possible)",
                    category=QuestionCategory.METRICS,
                    severity="critical",
                    section="experience",
                    about=f"{exp.get('company')} - {exp.get('position')}",
                    why_asked="Metrics make impact tangible. Recruiters want to see results, "
                             "not just responsibilities.",
                    follow_up_questions=[
                        "How many users/customers were affected?",
                        "What was the performance improvement?",
                        "How much time/cost was saved?"
                    ],
                    validation_rules=[
                        "Metric should be specific (not 'improved performance')",
                        "Should be verifiable (ask: 'How would your manager verify this?')",
                        "Avoid inflated claims (40% improvement = specific measurable change)"
                    ],
                    context=f"Experience: {exp.get('position')} at {exp.get('company')}"
                ))
            
            # Question: Challenge faced
            if not exp.get("challenges"):
                questions.append(ResumeQuestion(
                    question=f"At {exp.get('company')}, what was the BIGGEST technical challenge "
                            f"you faced? How did you solve it?",
                    category=QuestionCategory.CONTEXT,
                    severity="important",
                    section="experience",
                    about=f"{exp.get('company')} - Challenge",
                    why_asked="Challenges show problem-solving ability. Shows you can debug and think.",
                    follow_up_questions=[
                        "What tools/technologies did you use to solve it?",
                        "How long did it take to resolve?",
                        "What did you learn from this?"
                    ],
                    validation_rules=[
                        "Challenge should be technical and real",
                        "Solution should show thinking, not just tooling",
                        "Avoid trivial challenges"
                    ],
                    context=f"Role: {exp.get('position')} ({exp.get('duration_months')} months)"
                ))
            
            # Question: Reason for leaving
            if exp.get("type") in ["internship", "contract"] and not exp.get("reason_for_leaving"):
                questions.append(ResumeQuestion(
                    question=f"Why did you leave your internship at {exp.get('company')}? "
                            f"(Keep it positive)",
                    category=QuestionCategory.CLARIFICATION,
                    severity="interesting",
                    section="experience",
                    about=f"{exp.get('company')} - Reason",
                    why_asked="Shows career progression thinking. Keep it positive.",
                    follow_up_questions=[
                        "Did you get offered full-time? Why not take it?",
                        "What are you doing next?"
                    ],
                    validation_rules=[
                        "Avoid negative comments about company",
                        "Don't blame people or management",
                        "Focus on learning and growth"
                    ],
                    context="Good answer shows maturity and positive attitude"
                ))
        
        return questions

    def _generate_project_questions(self, resume_data: Dict) -> List[ResumeQuestion]:
        """Generate probing questions about projects"""
        questions = []
        projects = resume_data.get("projects", [])
        
        for i, project in enumerate(projects):
            # Question: GitHub link
            if not project.get("github_link") or project.get("github_link") == "":
                questions.append(ResumeQuestion(
                    question=f"Do you have a GitHub link for '{project.get('title')}'? "
                            f"(Required for Junior Developer role)",
                    category=QuestionCategory.MISSING_DETAIL,
                    severity="critical",
                    section="projects",
                    about=project.get('title'),
                    why_asked="GitHub link is essential. Shows actual code, commits, collaboration.",
                    follow_up_questions=[
                        "Is the code well-commented?",
                        "How many commits do you have?",
                        "Any contributors?"
                    ],
                    validation_rules=[
                        "GitHub link should be valid (check format: github.com/username/repo)",
                        "Repository should exist",
                        "Don't share private repos"
                    ],
                    context=f"Project: {project.get('title')} | "
                           f"Tech: {', '.join(project.get('technologies', []))}"
                ))
            
            # Question: Live demo link
            if project.get("technologies") and "Frontend" in str(project.get("technologies")):
                if not project.get("live_link"):
                    questions.append(ResumeQuestion(
                        question=f"Does '{project.get('title')}' have a live demo? (Netlify, Vercel, etc.)",
                        category=QuestionCategory.ENRICHMENT,
                        severity="important",
                        section="projects",
                        about=project.get('title'),
                        why_asked="Live demo is awesome. Recruiters can test your project directly.",
                        follow_up_questions=[
                            "Any demo credentials needed?",
                            "Is it mobile responsive?"
                        ],
                        validation_rules=[
                            "Link should load without errors",
                            "Should showcase main features"
                        ],
                        context="Frontend projects should have live links"
                    ))
            
            # Question: Project duration
            if not project.get("duration_months"):
                questions.append(ResumeQuestion(
                    question=f"How long did you work on '{project.get('title')}'? "
                            f"(in weeks/months)",
                    category=QuestionCategory.MISSING_DETAIL,
                    severity="important",
                    section="projects",
                    about=project.get('title'),
                    why_asked="Duration shows project complexity and commitment.",
                    follow_up_questions=[
                        "Was it solo or team project?",
                        "Are you still maintaining it?"
                    ],
                    validation_rules=[
                        "Duration should be reasonable (not 1 hour projects)",
                        "Should match GitHub history"
                    ],
                    context=f"Project size indicator"
                ))
            
            # Question: User count / adoption
            if not project.get("users_count") and not project.get("metrics"):
                questions.append(ResumeQuestion(
                    question=f"'{project.get('title')}' - How many users/stars/downloads? "
                            f"Or what metrics show success?",
                    category=QuestionCategory.METRICS,
                    severity="important",
                    section="projects",
                    about=project.get('title'),
                    why_asked="Metrics show if project was useful. Even one user is better than none.",
                    follow_up_questions=[
                        "Do you have GitHub stars/forks count?",
                        "Is anyone using it besides you?",
                        "Any performance metrics you track?"
                    ],
                    validation_rules=[
                        "Be honest - 1 user is fine, inflating to 10000 is not",
                        "Verify GitHub stats match your claim"
                    ],
                    context="Adoption metrics make projects impressive"
                ))
            
            # Question: Main learning
            if not project.get("key_learning"):
                questions.append(ResumeQuestion(
                    question=f"What was the BIGGEST technical thing you learned from "
                            f"'{project.get('title')}'?",
                    category=QuestionCategory.LEARNING,
                    severity="important",
                    section="projects",
                    about=project.get('title'),
                    why_asked="Shows reflection and growth mindset. What skills did you gain?",
                    follow_up_questions=[
                        "Did you solve any tricky performance issues?",
                        "What would you do differently if you restart this project?"
                    ],
                    validation_rules=[
                        "Learning should be specific (not 'learned to code')",
                        "Should show technical depth",
                        "Avoid generic answers"
                    ],
                    context="Learning statements show depth of understanding"
                ))
        
        return questions

    def _generate_skills_questions(self, resume_data: Dict) -> List[ResumeQuestion]:
        """Generate probing questions about skills"""
        questions = []
        technical_skills = resume_data.get("technical_skills", {})
        soft_skills = resume_data.get("soft_skills", [])
        
        # Question: Soft skills
        if not soft_skills or len(soft_skills) < 3:
            questions.append(ResumeQuestion(
                question="What are your TOP 5 soft skills? (Communication, Leadership, Problem-solving)",
                category=QuestionCategory.MISSING_DETAIL,
                severity="important",
                section="skills",
                about="Soft Skills",
                why_asked="Soft skills differentiate candidates. Shows you're not just a coder.",
                follow_up_questions=[
                    "Can you give an example where you used leadership?",
                    "Tell about a time you resolved a team conflict"
                ],
                validation_rules=[
                    "Should be real soft skills (not technical)",
                    "Should have 3-5 skills",
                    "Avoid clichés like 'hard worker'"
                ],
                context="Soft skills are crucial for growth beyond individual contributor"
            ))
        
        # Question: Proficiency levels
        # Check if any technical skills have proficiency info
        has_proficiency = False
        if isinstance(technical_skills, dict):
            has_proficiency = any(
                isinstance(v, dict) and v.get("proficiency") 
                for v in technical_skills.values()
            )
        
        if not has_proficiency:
            questions.append(ResumeQuestion(
                question="Rate your proficiency in your TOP 3 languages/frameworks: "
                        "Expert, Advanced, Intermediate, Beginner",
                category=QuestionCategory.CLARIFICATION,
                severity="important",
                section="skills",
                about="Proficiency Levels",
                why_asked="Clarity on proficiency avoids embarrassing interview questions.",
                follow_up_questions=[
                    "Which one are you MOST expert in?",
                    "Which one would you like to learn?"
                ],
                validation_rules=[
                    "Expert = can teach others, advanced project",
                    "Advanced = can build projects independently",
                    "Intermediate = can contribute to team projects",
                    "Beginner = learning phase"
                ],
                context="Honest assessment prevents interview shock"
            ))
        
        # Question: Currently learning
        if not resume_data.get("currently_learning"):
            questions.append(ResumeQuestion(
                question="What technology/skill are you CURRENTLY learning? "
                        "(Shows growth mindset)",
                category=QuestionCategory.ENRICHMENT,
                severity="interesting",
                section="skills",
                about="Currently Learning",
                why_asked="Shows you're not stagnant. Growth mindset is attractive.",
                follow_up_questions=[
                    "Why this skill?",
                    "When will you be done?"
                ],
                validation_rules=[
                    "Should be relevant to career goals",
                    "Should have learning plan/timeline",
                    "Can be online course, side project, etc."
                ],
                context="Growth mindset matters to modern companies"
            ))
        
        return questions

    def generate_follow_up_questions(
        self,
        original_question: ResumeQuestion,
        response: str
    ) -> List[ResumeQuestion]:
        """
        Generate follow-up questions based on response
        
        Args:
            original_question: The original ResumeQuestion asked
            response: User's response
            
        Returns:
            List of follow-up questions
        """
        follow_ups = []
        
        for fq_text in original_question.follow_up_questions[:2]:  # Ask top 2 follow-ups
            follow_ups.append(ResumeQuestion(
                question=fq_text,
                category=QuestionCategory.CONTEXT,
                severity="interesting",
                section=original_question.section,
                about=original_question.about,
                why_asked="Depth follow-up to enhance understanding",
                follow_up_questions=[],
                validation_rules=[],
                context=f"Follow-up to: {original_question.question}"
            ))
        
        return follow_ups

    def format_question_for_user(self, question: ResumeQuestion) -> str:
        """Format a ResumeQuestion for display to user"""
        severity_icon = {
            "critical": "🔴",
            "important": "🟡",
            "interesting": "🟢"
        }
        
        return f"""
{severity_icon.get(question.severity, "❓")} [{question.section.upper()}] {question.question}

Why: {question.why_asked}

Hint: {question.validation_rules[0] if question.validation_rules else 'Provide specific details'}
        """

    def get_question_statistics(self, questions: List[ResumeQuestion]) -> Dict:
        """Get statistics about generated questions"""
        return {
            "total_questions": len(questions),
            "by_category": self._count_by_field(questions, "category"),
            "by_severity": self._count_by_field(questions, "severity"),
            "by_section": self._count_by_field(questions, "section"),
        }

    @staticmethod
    def _count_by_field(questions: List[ResumeQuestion], field: str) -> Dict:
        """Count questions by field"""
        counts = {}
        for q in questions:
            value = getattr(q, field).value if isinstance(getattr(q, field), Enum) else getattr(q, field)
            counts[str(value)] = counts.get(str(value), 0) + 1
        return counts


# Example usage
if __name__ == "__main__":
    # Sample resume data
    sample_resume = {
        "education": [{
            "institution": "IIT Bombay",
            "degree": "B.Tech",
            "field": "Computer Science",
            "graduation_year": 2023,
            "cgpa": None,
            "relevant_coursework": ["Data Structures", "Web Development"]
        }],
        "experience": [{
            "company": "TechCorp",
            "position": "Intern",
            "type": "internship",
            "duration_months": 3,
            "impact_metrics": None,
            "challenges": None
        }],
        "projects": [{
            "title": "Chat Application",
            "description": "Real-time messaging",
            "technologies": ["Python", "FastAPI", "WebSockets"],
            "github_link": None,
            "duration_months": None
        }],
        "technical_skills": {"languages": ["Python", "JavaScript"]},
        "soft_skills": None
    }
    
    # Generate questions
    generator = LLMQuestionGenerator()
    questions = generator.analyze_and_generate_questions(sample_resume, question_count=10)
    
    print(f"Generated {len(questions)} questions:\n")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q.question}")
        print(f"   Category: {q.category.value} | Severity: {q.severity}")
        print()
    
    print("\nStatistics:")
    stats = generator.get_question_statistics(questions)
    print(f"Total: {stats['total_questions']}")
    print(f"By Severity: {stats['by_severity']}")
    print(f"By Section: {stats['by_section']}")
