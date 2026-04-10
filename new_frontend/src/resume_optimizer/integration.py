"""
Integration layer for Resume Optimization Platform with existing Career Recommendation System
"""
from typing import Dict, Any, List, Optional
import json

from src.resume_optimizer.api import ResumeOptimizationAPI
from .models import JobRoleRequirements
from src.llm_client import generate_career_guidance, generate_clarification_questions


class ResumeOptimizationIntegration:
    """Integration layer that connects resume optimization with career recommendations"""
    
    def __init__(self):
        self.optimization_api = ResumeOptimizationAPI()
    
    def complete_resume_analysis(self, resume_text: str) -> Dict[str, Any]:
        """
        Complete resume analysis including career recommendations
        
        Args:
            resume_text: Raw resume text
            
        Returns:
            Complete analysis with career recommendations
        """
        # Analyze resume
        analysis = self.optimization_api.analyze_resume(resume_text)
        
        # Extract skills and projects for career guidance
        structured_data = analysis['structured_data']
        skills = structured_data['skills'].get('technical', [])
        projects = [proj['name'] for proj in structured_data.get('projects', [])]
        other_keywords = []
        
        # Add other skills to other_keywords
        for category, skill_list in structured_data['skills'].items():
            if category != 'technical':
                other_keywords.extend(skill_list)
        
        # Generate career guidance
        resume_data = {
            'skills': skills,
            'projects': projects,
            'other': other_keywords,
            'education': [edu['degree'] for edu in structured_data.get('education', [])],
            'experience_level': 'Fresher' if len(structured_data.get('work_experience', [])) == 0 else 'Experienced',
            'work_experience': structured_data.get('work_experience', []),
            'full_education': structured_data.get('education', []),
            'raw_structured': structured_data
        }
        
        career_guidance = generate_career_guidance(resume_data)
        
        # Generate smart follow-up questions
        questions = generate_clarification_questions(resume_data)
        
        return {
            'resume_analysis': analysis,
            'career_guidance': career_guidance,
            'extracted_data': resume_data,
            'clarification_questions': questions
        }
    
    def optimize_for_recommended_roles(self, resume_text: str, 
                                     max_roles: int = 3,
                                     user_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize resume for top recommended career roles
        
        Args:
            resume_text: Raw resume text
            max_roles: Maximum number of roles to optimize for
            
        Returns:
            Optimized resumes for top recommended roles
        """
        # Get career guidance first
        complete_analysis = self.complete_resume_analysis(resume_text)
        career_guidance = complete_analysis['career_guidance']
        
        # Extract recommended roles from career guidance
        recommended_roles = self._extract_roles_from_guidance(career_guidance)
        
        # Limit to max_roles
        recommended_roles = recommended_roles[:max_roles]
        
        # Optimize for each role
        optimizations = {}
        for role in recommended_roles:
            # Create job requirements based on role
            job_requirements = self._create_requirements_for_role(role)
            
            # Optimize resume
            optimized = self.optimization_api.optimize_resume(
                resume_text, role, job_requirements, user_data=user_data
            )
            
            optimizations[role] = optimized
        
        return {
            'original_analysis': complete_analysis['resume_analysis'],
            'career_guidance': career_guidance,
            'recommended_roles': recommended_roles,
            'optimized_resumes': optimizations
        }
    
    def batch_optimize_multiple_resumes(self, resume_texts: List[str], 
                                      target_roles: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Optimize multiple resumes for multiple roles
        
        Args:
            resume_texts: List of resume texts
            target_roles: Optional list of target roles
            
        Returns:
            Batch optimization results
        """
        results = {}
        
        for i, resume_text in enumerate(resume_texts):
            resume_id = f"resume_{i+1}"
            
            if target_roles:
                # Optimize for specified roles
                optimizations = {}
                for role in target_roles:
                    job_requirements = self._create_requirements_for_role(role)
                    optimized = self.optimization_api.optimize_resume(
                        resume_text, role, job_requirements
                    )
                    optimizations[role] = optimized
                
                results[resume_id] = {
                    'optimizations': optimizations,
                    'status': 'completed'
                }
            else:
                # Get recommended roles and optimize
                optimization_result = self.optimize_for_recommended_roles(resume_text)
                results[resume_id] = optimization_result
        
        return results
    
    def compare_resume_versions(self, original_text: str, 
                              optimized_text: str) -> Dict[str, Any]:
        """
        Compare original and optimized resume versions
        
        Args:
            original_text: Original resume text
            optimized_text: Optimized resume text
            
        Returns:
            Comparison analysis
        """
        # Analyze both versions
        original_analysis = self.optimization_api.analyze_resume(original_text)
        optimized_analysis = self.optimization_api.analyze_resume(optimized_text)
        
        # Calculate improvements
        score_improvement = optimized_analysis['scores']['overall_score'] - original_analysis['scores']['overall_score']
        fault_reduction = len(original_analysis['faults']) - len(optimized_analysis['faults'])
        
        # Identify specific improvements
        improvements = []
        
        if optimized_analysis['scores']['ats_score'] > original_analysis['scores']['ats_score']:
            improvements.append("ATS optimization improved")
        
        if optimized_analysis['scores']['keyword_score'] > original_analysis['scores']['keyword_score']:
            improvements.append("Keyword optimization improved")
        
        if optimized_analysis['scores']['formatting_score'] > original_analysis['scores']['formatting_score']:
            improvements.append("Formatting improved")
        
        return {
            'original_score': original_analysis['scores']['overall_score'],
            'optimized_score': optimized_analysis['scores']['overall_score'],
            'score_improvement': score_improvement,
            'fault_reduction': fault_reduction,
            'improvements': improvements,
            'original_faults': len(original_analysis['faults']),
            'optimized_faults': len(optimized_analysis['faults'])
        }
    
    def generate_resume_report(self, resume_text: str, 
                              include_optimizations: bool = True) -> str:
        """
        Generate comprehensive resume report
        
        Args:
            resume_text: Raw resume text
            include_optimizations: Whether to include optimization suggestions
            
        Returns:
            Formatted report string
        """
        # Get complete analysis
        complete_analysis = self.complete_resume_analysis(resume_text)
        analysis = complete_analysis['resume_analysis']
        
        # Generate report
        report = []
        report.append("=" * 60)
        report.append("COMPREHENSIVE RESUME ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Scores section
        scores = analysis['scores']
        report.append("PERFORMANCE SCORES:")
        report.append(f"  Overall Score: {scores['overall_score']:.1f}/100")
        report.append(f"  ATS Score: {scores['ats_score']:.1f}/100")
        report.append(f"  Keyword Score: {scores['keyword_score']:.1f}/100")
        report.append(f"  Formatting Score: {scores['formatting_score']:.1f}/100")
        report.append("")
        
        # Faults section
        faults = analysis['faults']
        report.append("IDENTIFIED ISSUES:")
        if faults:
            for fault in faults:
                severity_icon = "🔴" if fault['severity'] == 'critical' else "🟡" if fault['severity'] == 'high' else "🟠"
                report.append(f"  {severity_icon} {fault['description']}")
                if fault['suggestion']:
                    report.append(f"     💡 {fault['suggestion']}")
        else:
            report.append("  ✅ No critical issues found!")
        report.append("")
        
        # Career guidance section
        career_guidance = complete_analysis['career_guidance']
        report.append("CAREER RECOMMENDATIONS:")
        report.append(f"  {career_guidance}")
        report.append("")
        
        # Optimization suggestions
        if include_optimizations:
            report.append("OPTIMIZATION RECOMMENDATIONS:")
            report.append("  1. Add quantifiable achievements")
            report.append("  2. Include more technical keywords")
            report.append("  3. Improve formatting for ATS systems")
            report.append("  4. Add professional summary")
            report.append("  5. Highlight relevant experience")
            report.append("")
        
        report.append("=" * 60)
        report.append("END OF REPORT")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def _extract_roles_from_guidance(self, career_guidance: str) -> List[str]:
        """Extract recommended roles from career guidance text"""
        # Simple extraction - can be enhanced with NLP
        roles = []
        
        # Look for common role patterns
        role_patterns = [
            r'(\w+\s+Engineer)',
            r'(\w+\s+Developer)',
            r'(\w+\s+Analyst)',
            r'(\w+\s+Manager)',
            r'(\w+\s+Architect)',
            r'(\w+\s+Specialist)'
        ]
        
        import re
        for pattern in role_patterns:
            matches = re.findall(pattern, career_guidance, re.IGNORECASE)
            roles.extend(matches)
        
        # Remove duplicates and return
        return list(set(roles))[:3]  # Return top 3
    
    def _create_requirements_for_role(self, role: str) -> JobRoleRequirements:
        """Create job requirements for a given role"""
        # Role-specific requirements database
        role_requirements = {
            'Software Engineer': JobRoleRequirements(
                role='Software Engineer',
                required_skills=['python', 'java', 'javascript', 'sql'],
                preferred_skills=['react', 'nodejs', 'aws', 'docker'],
                experience_level='entry',
                key_responsibilities=['develop software', 'debug code', 'write tests'],
                industry_keywords=['software development', 'programming', 'coding']
            ),
            'Data Analyst': JobRoleRequirements(
                role='Data Analyst',
                required_skills=['sql', 'python', 'excel', 'data analysis'],
                preferred_skills=['tableau', 'power bi', 'statistics', 'machine learning'],
                experience_level='entry',
                key_responsibilities=['analyze data', 'create reports', 'visualize data'],
                industry_keywords=['data analysis', 'analytics', 'reporting']
            ),
            'Robotics Engineer': JobRoleRequirements(
                role='Robotics Engineer',
                required_skills=['python', 'c++', 'ros', 'ros2'],
                preferred_skills=['computer vision', 'machine learning', 'embedded systems'],
                experience_level='mid',
                key_responsibilities=['design robots', 'program controllers', 'test systems'],
                industry_keywords=['robotics', 'automation', 'control systems']
            )
        }
        
        return role_requirements.get(role, JobRoleRequirements(
            role=role,
            required_skills=['python', 'communication'],
            preferred_skills=['teamwork', 'problem solving'],
            experience_level='entry',
            key_responsibilities=['perform job duties'],
            industry_keywords=[role.lower()]
        ))
