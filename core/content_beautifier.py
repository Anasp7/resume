"""
Content Beautifier - Improves Resume Content Without Hallucination
==================================================================
Only beautifies/rewrites content that is verified as true and valid.
No making up information - strictly based on validated user input.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum


@dataclass
class BeautificationResult:
    """Result of beautifying a piece of content"""
    original: str
    beautified: str
    improvements: List[str]  # What was improved
    confidence: float        # How confident in beautification (0-1)
    advice: List[str]        # Additional suggestions


class ContentBeautifier:
    """
    Beautifies resume content while maintaining authenticity.
    Only improves/rewrites verified content.
    Never adds information not provided by user.
    """

    def beautify_metrics(
        self,
        original_metric: str,
        context: str = "",
        is_verified: bool = True
    ) -> BeautificationResult:
        """
        Beautify metric claims
        "improved performance" → "Reduced API response time from 2s to 500ms (75% improvement)"
        
        But ONLY if verified by user.
        """
        if not is_verified:
            return BeautificationResult(
                original=original_metric,
                beautified=original_metric,
                improvements=[],
                confidence=0.0,
                advice=["Have user verify metric first"]
            )
        
        improvements = []
        beautified = original_metric
        
        # Pattern: Just a percentage
        if "%" in original_metric and not any(x in original_metric.lower() for x in ["from", "to", "reduction"]):
            # Ask for specifics - but we can't hallucinate
            # Only beautify if we have details
            improvements.append("✓ Percentage metric - clear improvement shown")
            advice = ["Great! Specify the metric: 'response time', 'CPU usage', 'queries', etc."]
        else:
            advice = []
        
        # Pattern: Specific metric
        if "from" in original_metric.lower() and "to" in original_metric.lower():
            beautified = original_metric  # Already well-formatted
            improvements.append("✓ Metric is specific and well-formatted")
            advice = ["This is excellent! Specific before/after metrics are most credible"]
        
        # Pattern: Large number
        if any(word in original_metric.lower() for word in ["users", "downloads", "stars", "commits"]):
            improvements.append("✓ Quantifiable metric")
            advice = ["Excellent: Always quantify user adoption, performance, scale"]
        
        return BeautificationResult(
            original=original_metric,
            beautified=beautified,
            improvements=improvements,
            confidence=0.95 if improvements else 0.5,
            advice=advice
        )

    def beautify_experience_bullet(
        self,
        original: str,
        metrics: Optional[Dict] = None,
        is_verified: bool = True
    ) -> BeautificationResult:
        """
        Beautify individual experience bullet point
        
        Examples:
        "Developed REST APIs" →
        "Developed REST APIs handling 10k+ daily requests with 99.9% uptime"
        (if metrics are provided and verified)
        """
        if not is_verified:
            return BeautificationResult(
                original=original,
                beautified=original,
                improvements=[],
                confidence=0.0,
                advice=["Verify content with user before beautifying"]
            )
        
        improvements = []
        beautified = original
        
        # Remove weak verbs, check for action verbs
        weak_verbs = ["did", "was", "had", "worked on"]
        strong_verbs = {
            "did": "delivered",
            "was": "served as",
            "had": "owned",
            "worked on": "engineered"
        }
        
        for weak, strong in strong_verbs.items():
            if original.lower().startswith(weak):
                beautified = strong + original[len(weak):]
                improvements.append(f"✓ Strengthened verb: '{weak}' → '{strong}'")
        
        # Add metrics if provided
        if metrics:
            # Check if metric already in original
            if not any(str(v) in original for v in metrics.values() if v):
                # Can only add if we have verified metrics
                beautified = beautified.rstrip()
                metric_str = self._format_metrics(metrics)
                if metric_str:
                    beautified += f" {metric_str}"
                    improvements.append(f"✓ Added verified metric: {metric_str}")
        
        # Check for passive voice
        if "was" in beautified.lower() and original.lower().startswith("was"):
            improvements.append("ℹ️ Consider active voice for more impact")
        
        # Check structure: Action - Specifics - Impact
        has_action = any(v in beautified.lower() for v in ["developed", "built", "created", "led"])
        has_specifics = any(c in beautified for c in ["using", "with", "on"])
        has_impact = any(i in beautified.lower() for i in ["improved", "reduced", "increased", "%"])
        
        structure_score = sum([has_action, has_specifics, has_impact]) / 3
        if structure_score > 0.66:
            improvements.append("✓ Good structure: Action - Specifics - Impact")
        else:
            missing = []
            if not has_action:
                missing.append("specific action")
            if not has_specifics:
                missing.append("technologies/context")
            if not has_impact:
                missing.append("quantified impact")
            if missing:
                improvements.append(f"ℹ️ Could add: {', '.join(missing)}")
        
        advice = [
            "Format: Start with action verb + what you did + technologies + impact",
            "Example: 'Developed REST API using Python+FastAPI, serving 1000+ users'"
        ]
        
        return BeautificationResult(
            original=original,
            beautified=beautified,
            improvements=improvements,
            confidence=0.85 if improvements else 0.6,
            advice=advice
        )

    def beautify_project_description(
        self,
        project_title: str,
        original_description: str,
        technologies: List[str],
        metrics: Optional[Dict] = None,
        is_verified: bool = True
    ) -> BeautificationResult:
        """
        Beautify project description
        
        Example:
        "Chat Application - Real-time messaging platform using Python, FastAPI, WebSockets"
        →
        "Chat Application - Real-time messaging platform with 100+ users
         Technologies: Python, FastAPI, WebSockets
         GitHub: [link] | Live Demo: [link]
         Key Achievement: 50ms message delivery latency"
        """
        if not is_verified:
            return BeautificationResult(
                original=original_description,
                beautified=original_description,
                improvements=[],
                confidence=0.0,
                advice=["Verify all details with user before beautifying"]
            )
        
        improvements = []
        beautified_parts = []
        
        # Part 1: Title + Description
        beautified_parts.append(f"**{project_title}**")
        improvements.append("✓ Project title formatted for emphasis")
        
        # Make description more compelling if generic
        if original_description and len(original_description) > 5:
            desc = original_description.strip()
            if desc.endswith("."):
                desc = desc[:-1]
            
            beautified_parts.append(f"*{desc}*")
            
            # Check if description is specific vs generic
            generic_words = ["application", "system", "platform", "tool"]
            if any(w in desc.lower() for w in generic_words):
                improvements.append("✓ Project description is clear")
            else:
                improvements.append("⚠️ Description could be more specific about use case")
        
        # Part 2: Technologies
        if technologies:
            tech_str = " | ".join(technologies)
            beautified_parts.append(f"**Tech:** {tech_str}")
            improvements.append(f"✓ Technologies formatted clearly")
        
        # Part 3: Metrics
        if metrics:
            metric_str = self._format_metrics(metrics)
            if metric_str:
                beautified_parts.append(f"**Key Metrics:** {metric_str}")
                improvements.append("✓ Metrics added for impact")
        
        beautified = "\n".join(beautified_parts)
        
        advice = [
            "Good project descriptions include: What it does + Why it matters + How users benefit",
            "Always include: Title, Description, Tech stack, GitHub link, Live demo (if available)"
        ]
        
        return BeautificationResult(
            original=original_description,
            beautified=beautified,
            improvements=improvements,
            confidence=0.9 if metrics else 0.7,
            advice=advice
        )

    def beautify_soft_skills(
        self,
        original_skills: List[str],
        examples: Optional[Dict[str, str]] = None,
        is_verified: bool = True
    ) -> BeautificationResult:
        """
        Beautify soft skills section
        
        Input: ["Communication", "Problem Solving"]
        Output: "Communication (mentored 2 juniors), Problem-Solving (resolved critical production bug)"
        (if examples provided and verified)
        """
        if not is_verified:
            return BeautificationResult(
                original=", ".join(original_skills),
                beautified=", ".join(original_skills),
                improvements=[],
                confidence=0.0,
                advice=["Get verification before beautifying"]
            )
        
        improvements = []
        beautified_skills = []
        
        for skill in original_skills:
            # Normalize skill name
            skill_normalized = skill.title()  # "communication" → "Communication"
            
            # Add example if available and verified
            if examples and skill_normalized in examples:
                example = examples[skill_normalized]
                beautified_skills.append(f"{skill_normalized} ({example})")
                improvements.append(f"✓ Added verified example for '{skill_normalized}'")
            else:
                beautified_skills.append(skill_normalized)
        
        beautified = ", ".join(beautified_skills)
        
        # Check for redundancy
        skill_keywords = [s.lower() for s in original_skills]
        if "problem solving" in skill_keywords and "analytics" in skill_keywords:
            improvements.append("⚠️ Some skill overlap - consider consolidating")
        
        advice = [
            "Soft skills are strongest when backed by examples",
            "Pair each skill with evidence: 'Communication (led 5-person project)'",
            "Order by relevance to your target role"
        ]
        
        return BeautificationResult(
            original=", ".join(original_skills),
            beautified=beautified,
            improvements=improvements,
            confidence=0.85,
            advice=advice
        )

    def beautify_education_section(
        self,
        institution: str,
        degree: str,
        field: str,
        cgpa: Optional[str] = None,
        coursework: Optional[List[str]] = None,
        projects: Optional[List[str]] = None,
        is_verified: bool = True
    ) -> BeautificationResult:
        """
        Beautify entire education section
        
        Transforms basic info to compelling section with coursework, CGPA, projects
        """
        if not is_verified:
            return BeautificationResult(
                original=f"{degree} in {field}",
                beautified=f"{degree} in {field}",
                improvements=[],
                confidence=0.0,
                advice=["Verify education details before beautifying"]
            )
        
        improvements = []
        beautified_parts = []
        
        # Main degree line
        main_line = f"{degree} in {field}"
        if cgpa:
            main_line += f" | CGPA: {cgpa}"
            improvements.append("✓ CGPA highlighted")
        
        beautified_parts.append(f"**{main_line}**")
        beautified_parts.append(f"{institution}")
        
        # Coursework
        if coursework and len(coursework) > 0:
            course_str = ", ".join(coursework[:5])  # Top 5 courses
            beautified_parts.append(f"Relevant Coursework: {course_str}")
            improvements.append(f"✓ Top courses highlighted ({len(coursework)} courses)")
        
        # Projects
        if projects and len(projects) > 0:
            beautified_parts.append(f"Key Projects: {', '.join(projects[:3])}")
            improvements.append(f"✓ Capstone/major projects noted")
        
        beautified = "\n".join(beautified_parts)
        
        advice = [
            "Education section improves with: Coursework (3-5 relevant) + CGPA (if good) + Projects",
            "CGPA rule: Include if ≥7.5, skip if <7.0, your choice if 7.0-7.5"
        ]
        
        return BeautificationResult(
            original=f"{degree} in {field}",
            beautified=beautified,
            improvements=improvements,
            confidence=0.9 if cgpa and coursework else 0.7,
            advice=advice
        )

    def beautify_summary(
        self,
        experience_years: int,
        key_skills: List[str],
        role_target: str,
        unique_traits: Optional[List[str]] = None,
        is_verified: bool = True
    ) -> BeautificationResult:
        """
        Create compelling professional summary
        
        Input: 0 years, ["Python","React","Node.js"], "Junior Developer"
        Output: "Results-driven Junior Developer with 4 projects...Skills: Python, React, Node.js
                 Value proposition: [unique trait if provided]"
        """
        if not is_verified:
            return BeautificationResult(
                original="",
                beautified="",
                improvements=[],
                confidence=0.0,
                advice=["Get verification before generating summary"]
            )
        
        improvements = []
        
        # Build summary
        summary_parts = []
        
        # Opening line
        if experience_years == 0:
            opening = f"Enthusiastic fresher passionate about {role_target}"
            improvements.append("✓ Fresher positioning statement")
        elif experience_years < 2:
            opening = f"Junior {role_target} with {experience_years} year(s) of experience"
            improvements.append("✓ Experience level clearly stated")
        else:
            opening = f"Experienced {role_target} with {experience_years} years of proven expertise"
            improvements.append("✓ Seniority level established")
        
        summary_parts.append(opening)
        
        # Skills highlight
        top_skills = ", ".join(key_skills[:3])
        summary_parts.append(f"Strong foundation in {top_skills}")
        improvements.append(f"✓ Top skills highlighted")
        
        # Achievement/trait
        if unique_traits:
            trait = unique_traits[0]
            summary_parts.append(f"Known for: {trait}")
            improvements.append(f"✓ Unique value proposition added")
        
        # Call to action
        summary_parts.append(f"Seeking opportunities to contribute and grow as a {role_target}")
        improvements.append("✓ Growth mindset shown")
        
        beautified = ". ".join(summary_parts) + "."
        
        advice = [
            "Professional summary is 3-4 sentences, not a paragraph",
            "Include: Who you are + Key skills + What you want + Why you're valuable",
            "Example: 'Fresher developer with 4 projects using Python/React. Strong problem-solver with passion for scalable systems. Seeking role to contribute and grow.'"
        ]
        
        return BeautificationResult(
            original="",
            beautified=beautified,
            improvements=improvements,
            confidence=0.95,
            advice=advice
        )

    def _format_metrics(self, metrics: Dict) -> str:
        """Format metrics dictionary into readable string"""
        if not metrics:
            return ""
        
        formatted = []
        for key, value in metrics.items():
            if value:
                formatted.append(f"{key}: {value}")
        
        return " | ".join(formatted) if formatted else ""

    def check_resume_polish(self, resume_section: str) -> BeautificationResult:
        """
        Check overall polish of a resume section
        """
        improvements = []
        issues = []
        
        # Check for spelling/grammar (basic)
        if "  " in resume_section:  # Double spaces
            issues.append("Double spaces detected")
        
        if resume_section.count("\n") > 10:
            issues.append("Section is very long - consider condensing")
        
        # Check for consistency
        if resume_section.count("-") > resume_section.count("•"):
            improvements.append("✓ Consistent bullet point style")
        
        # Check for action verbs
        action_verbs = ["developed", "designed", "led", "managed", "built", "created", "optimized"]
        verb_count = sum(1 for verb in action_verbs if verb in resume_section.lower())
        
        if verb_count > 3:
            improvements.append(f"✓ Strong action verbs used ({verb_count} found)")
        else:
            issues.append("Consider more action verbs (developed, designed, led, etc.)")
        
        # Check for metrics
        if any(char in resume_section for char in ["%", "+"]) or any(str(i) in resume_section for i in range(10)):
            improvements.append("✓ Quantified metrics present")
        else:
            issues.append("⚠️ No metrics/numbers found - resume may seem vague")
        
        status = "good" if len(improvements) > len(issues) else "needs_work"
        
        beautified = "\n".join([
            f"✓ {imp}" for imp in improvements
        ] + [
            f"⚠️ {issue}" for issue in issues
        ])
        
        return BeautificationResult(
            original=resume_section,
            beautified=beautified,
            improvements=improvements,
            confidence=0.8,
            advice=[
                "Resume polish matters: Action verbs, metrics, consistent formatting",
                "Target: 3 bullets per role/project with: action + what + metric"
            ]
        )


# Example usage
if __name__ == "__main__":
    beautifier = ContentBeautifier()
    
    # Test 1: Beautify metric
    result = beautifier.beautify_metrics(
        "Reduced database queries by 40%",
        is_verified=True
    )
    print("METRIC BEAUTIFICATION:")
    print(f"Original: {result.original}")
    print(f"Beautified: {result.beautified}")
    print(f"Improvements: {result.improvements}\n")
    
    # Test 2: Beautify experience bullet
    result = beautifier.beautify_experience_bullet(
        "Developed REST APIs",
        metrics={"Response Time": "500ms", "Daily Requests": "10k+"},
        is_verified=True
    )
    print("EXPERIENCE BULLET:")
    print(f"Original: {result.original}")
    print(f"Beautified: {result.beautified}")
    print(f"Improvements: {', '.join(result.improvements)}\n")
    
    # Test 3: Create summary
    result = beautifier.beautify_summary(
        experience_years=0,
        key_skills=["Python", "React", "Node.js"],
        role_target="Junior Developer",
        unique_traits=["Problem solver"],
        is_verified=True
    )
    print("PROFESSIONAL SUMMARY:")
    print(f"Beautified:\n{result.beautified}\n")
    print(f"Advice: {result.advice[0]}")
