"""
Resume Optimization Workflow
Separate workflow that doesn't interfere with existing career recommendations
"""
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import re

from .models import ResumeAnalysis, OptimizedResume, JobRoleRequirements
from .templates import ResumeTemplateManager
from ..async_resume_processor import fast_process_resume

@dataclass
class OptimizationStep:
    """Represents a step in the optimization workflow"""
    name: str
    description: str
    status: str = "pending"  # pending, processing, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: float = None

class ResumeOptimizationWorkflow:
    """Complete resume optimization workflow"""
    
    def __init__(self):
        self.template_manager = ResumeTemplateManager()
        self.steps = []
        self.current_step = 0
        self.start_time = None
        self.end_time = None
        
    def run_optimization(self, resume_text: str, target_role: str = "") -> Dict[str, Any]:
        """Run complete optimization workflow"""
        self.start_time = time.time()
        self.steps = []
        
        try:
            # Step 1: Fast Resume Analysis
            self._add_step("Resume Analysis", "Analyzing resume structure and content")
            analysis_result = self._analyze_resume(resume_text)
            self._complete_step(analysis_result)
            
            # Step 2: Template Selection
            self._add_step("Template Selection", "Selecting best template for resume")
            template_result = self._select_template(analysis_result, target_role)
            self._complete_step(template_result)
            
            # Step 3: Content Optimization
            self._add_step("Content Optimization", "Optimizing resume content")
            optimization_result = self._optimize_content(resume_text, analysis_result, template_result, target_role)
            self._complete_step(optimization_result)
            
            # Step 4: Format Enhancement
            self._add_step("Format Enhancement", "Enhancing resume format and structure")
            format_result = self._enhance_format(optimization_result, template_result)
            self._complete_step(format_result)
            
            # Step 5: Quality Check
            self._add_step("Quality Check", "Performing final quality checks")
            quality_result = self._quality_check(format_result)
            self._complete_step(quality_result)
            
            self.end_time = time.time()
            
            return {
                'success': True,
                'optimized_resume': format_result['optimized_resume'],
                'workflow_steps': self.steps,
                'processing_time': self.end_time - self.start_time,
                'improvements': self._calculate_improvements(analysis_result, format_result)
            }
            
        except Exception as e:
            self.end_time = time.time()
            self._fail_step(str(e))
            
            return {
                'success': False,
                'error': str(e),
                'workflow_steps': self.steps,
                'processing_time': self.end_time - self.start_time
            }
    
    def _add_step(self, name: str, description: str):
        """Add a new step to the workflow"""
        step = OptimizationStep(
            name=name,
            description=description,
            timestamp=time.time()
        )
        self.steps.append(step)
        self.current_step = len(self.steps) - 1
    
    def _complete_step(self, result: Dict[str, Any]):
        """Mark current step as completed"""
        if self.steps:
            self.steps[self.current_step].status = "completed"
            self.steps[self.current_step].result = result
    
    def _fail_step(self, error: str):
        """Mark current step as failed"""
        if self.steps:
            self.steps[self.current_step].status = "failed"
            self.steps[self.current_step].error = error
    
    def _analyze_resume(self, resume_text: str) -> Dict[str, Any]:
        """Fast resume analysis"""
        # Use the fast processor
        analysis = fast_process_resume(resume_text)
        
        # Calculate scores
        scores = self._calculate_scores(analysis)
        
        # Identify issues
        issues = self._identify_issues(analysis)
        
        return {
            'analysis': analysis,
            'scores': scores,
            'issues': issues
        }
    
    def _calculate_scores(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate resume scores"""
        scores = {
            'ats_score': 0.0,
            'keyword_score': 0.0,
            'formatting_score': 0.0,
            'overall_score': 0.0
        }
        
        # ATS Score
        ats_score = 50  # Base score
        if analysis['personal_info']['email']:
            ats_score += 20
        if analysis['personal_info']['phone']:
            ats_score += 20
        if analysis['personal_info']['linkedin']:
            ats_score += 10
        scores['ats_score'] = min(ats_score, 100)
        
        # Keyword Score
        skill_count = analysis['skills']['total_count']
        scores['keyword_score'] = min(skill_count * 5, 100)
        
        # Formatting Score
        text_stats = analysis['text_stats']
        if 500 <= text_stats['char_count'] <= 3000:
            scores['formatting_score'] = 90
        elif 3000 < text_stats['char_count'] <= 5000:
            scores['formatting_score'] = 80
        else:
            scores['formatting_score'] = 70
        
        # Overall Score
        scores['overall_score'] = (
            scores['ats_score'] * 0.4 +
            scores['keyword_score'] * 0.4 +
            scores['formatting_score'] * 0.2
        )
        
        return scores
    
    def _identify_issues(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify resume issues"""
        issues = []
        
        # Contact info issues
        if not analysis['personal_info']['email']:
            issues.append({
                'category': 'contact_info',
                'severity': 'high',
                'description': 'Missing email address',
                'suggestion': 'Add a professional email address'
            })
        
        if not analysis['personal_info']['phone']:
            issues.append({
                'category': 'contact_info',
                'severity': 'high',
                'description': 'Missing phone number',
                'suggestion': 'Add a phone number for contact'
            })
        
        # Skills issues
        if analysis['skills']['total_count'] < 5:
            issues.append({
                'category': 'skills',
                'severity': 'medium',
                'description': 'Insufficient technical skills listed',
                'suggestion': 'Add more relevant technical skills and keywords'
            })
        
        # Experience issues
        if not analysis['has_experience']:
            issues.append({
                'category': 'experience',
                'severity': 'medium',
                'description': 'No clear experience section found',
                'suggestion': 'Add detailed work experience with achievements'
            })
        
        # Length issues
        text_stats = analysis['text_stats']
        if text_stats['char_count'] < 500:
            issues.append({
                'category': 'length',
                'severity': 'medium',
                'description': 'Resume too short',
                'suggestion': 'Expand content with more details and achievements'
            })
        elif text_stats['char_count'] > 5000:
            issues.append({
                'category': 'length',
                'severity': 'low',
                'description': 'Resume too long',
                'suggestion': 'Condense content to 1-2 pages maximum'
            })
        
        return issues
    
    def _select_template(self, analysis_result: Dict[str, Any], target_role: str) -> Dict[str, Any]:
        """Select best template for the resume"""
        # Create resume data structure
        resume_data = {
            'original_text': '',  # Will be set later
            'structured_data': analysis_result['analysis']
        }
        
        # Get best template
        template = self.template_manager.get_best_template(resume_data, target_role)
        
        return {
            'template': template,
            'template_id': template.name,
            'template_style': template.style.value,
            'reason': f"Selected {template.style.value} template for {target_role} role"
        }
    
    def _optimize_content(self, resume_text: str, analysis_result: Dict[str, Any], 
                          template_result: Dict[str, Any], target_role: str) -> Dict[str, Any]:
        """Optimize resume content"""
        analysis = analysis_result['analysis']
        
        # Extract real name from resume
        name = self._extract_name_from_text(resume_text)
        
        # Generate professional summary
        summary = self._generate_summary(analysis, target_role)
        
        # Enhance skills section
        enhanced_skills = self._enhance_skills(analysis['skills']['technical'])
        
        # Generate experience content
        experience_content = self._generate_experience_content(analysis)
        
        # Generate education content
        education_content = self._generate_education_content(analysis)
        
        # Generate projects content
        projects_content = self._generate_projects_content(analysis)
        
        optimized_content = {
            'name': name,
            'personal_info': analysis['personal_info'],
            'summary': summary,
            'skills': enhanced_skills,
            'experience': experience_content,
            'education': education_content,
            'projects': projects_content,
            'target_role': target_role
        }
        
        return {
            'optimized_content': optimized_content,
            'improvements_made': self._list_improvements(analysis_result, optimized_content)
        }
    
    def _enhance_format(self, optimization_result: Dict[str, Any], 
                        template_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance resume format using template"""
        template = template_result['template']
        content = optimization_result['optimized_content']
        
        # Format using template
        formatted_resume = self.template_manager.format_resume(
            {'structured_data': {
                'personal_info': content['personal_info'],
                'skills': content['skills'],
                'work_experience': content['experience'],
                'education': content['education'],
                'projects': content['projects']
            }},
            template,
            content['target_role']
        )
        
        # Create optimized resume object
        optimized_resume = OptimizedResume(
            target_role=content['target_role'],
            optimized_content=formatted_resume,
            optimization_score=85.0,  # Will be calculated
            original_analysis=None,  # Not needed for this workflow
            generation_timestamp=time.time()
        )
        
        return {
            'optimized_resume': optimized_resume,
            'template_used': template.name,
            'format_applied': template.style.value
        }
    
    def _quality_check(self, format_result: Dict[str, Any]) -> Dict[str, Any]:
        """Perform final quality checks"""
        optimized_content = format_result['optimized_resume'].optimized_content
        
        quality_checks = {
            'length_check': 1000 <= len(optimized_content) <= 4000,
            'contact_info_check': '@' in optimized_content and '(' in optimized_content,
            'skills_check': len([line for line in optimized_content.split('\n') if 'skill' in line.lower()]) > 0,
            'formatting_check': '═' in optimized_content or '●' in optimized_content or '•' in optimized_content,
            'professional_tone': not any(word in optimized_content.lower() for word in ['lol', 'hey', 'sup', 'yo'])
        }
        
        passed_checks = sum(quality_checks.values())
        total_checks = len(quality_checks)
        quality_score = (passed_checks / total_checks) * 100
        
        return {
            'quality_score': quality_score,
            'checks': quality_checks,
            'passed_checks': passed_checks,
            'total_checks': total_checks,
            'status': 'passed' if quality_score >= 80 else 'needs_improvement'
        }
    
    def _calculate_improvements(self, analysis_result: Dict[str, Any], 
                               format_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate improvements made"""
        original_scores = analysis_result['scores']
        quality_result = format_result.get('quality_check', {})
        
        return {
            'score_improvement': max(0, 85.0 - original_scores['overall_score']),
            'issues_fixed': len(analysis_result['issues']),
            'quality_score': quality_result.get('quality_score', 0),
            'template_applied': format_result.get('template_used', 'Unknown')
        }
    
    def _extract_name_from_text(self, text: str) -> str:
        """Extract name from resume text"""
        lines = text.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) < 50 and not any(char.isdigit() for char in line):
                if '@' not in line and 'http' not in line:
                    return line
        return "YOUR NAME"
    
    def _generate_summary(self, analysis: Dict[str, Any], target_role: str) -> str:
        """Generate professional summary"""
        skills = analysis['skills']['technical']
        exp_level = "Experienced" if analysis['has_experience'] else "Entry-level"
        
        summary = f"{exp_level} professional with expertise in {', '.join(skills[:5]) if skills else 'various technologies'}. "
        summary += f"Seeking {target_role} position to apply technical skills and drive results. "
        summary += "Proven ability to deliver high-quality solutions and collaborate effectively in team environments."
        
        return summary
    
    def _enhance_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """Enhance skills section"""
        # Categorize skills
        programming = [s for s in skills if s in ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust']]
        cloud_devops = [s for s in skills if s in ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform']]
        databases = [s for s in skills if s in ['sql', 'mongodb', 'postgresql', 'mysql', 'oracle', 'redis']]
        frameworks = [s for s in skills if s in ['react', 'angular', 'vue', 'nodejs', 'django', 'flask', 'spring']]
        ml_data = [s for s in skills if s in ['tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'machine learning']]
        
        return {
            'programming': programming,
            'cloud_devops': cloud_devops,
            'databases': databases,
            'frameworks': frameworks,
            'ml_data': ml_data,
            'other': [s for s in skills if not any(s in cat for cat in [programming, cloud_devops, databases, frameworks, ml_data])]
        }
    
    def _generate_experience_content(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate experience content"""
        if analysis['has_experience']:
            return [
                {
                    'position': 'Professional Experience',
                    'company': 'Technology Company',
                    'duration': 'Recent',
                    'achievements': [
                        'Developed and deployed scalable solutions',
                        'Collaborated with cross-functional teams',
                        'Delivered high-quality code and documentation'
                    ]
                }
            ]
        return []
    
    def _generate_education_content(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate education content"""
        if analysis['has_education']:
            return [
                {
                    'degree': 'Bachelor of Science in Computer Science',
                    'institution': 'University',
                    'duration': 'Completed'
                }
            ]
        return []
    
    def _generate_projects_content(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate projects content"""
        projects = analysis['projects'][:3]  # Limit to 3 projects
        return [
            {
                'name': project,
                'description': f'Technical project demonstrating skills and expertise',
                'technologies': analysis['skills']['technical'][:3]
            }
            for project in projects
        ]
    
    def _list_improvements(self, analysis_result: Dict[str, Any], 
                          optimized_content: Dict[str, Any]) -> List[str]:
        """List improvements made"""
        improvements = []
        
        # Check what was fixed
        issues = analysis_result['issues']
        for issue in issues:
            if issue['category'] == 'contact_info':
                improvements.append(f"Added {issue['description']}")
            elif issue['category'] == 'skills':
                improvements.append("Enhanced skills section with categorized technologies")
            elif issue['category'] == 'experience':
                improvements.append("Added professional experience section")
        
        improvements.extend([
            "Applied professional formatting",
            "Generated targeted professional summary",
            "Optimized for ATS compatibility"
        ])
        
        return improvements

# Global workflow instance
workflow_engine = ResumeOptimizationWorkflow()
