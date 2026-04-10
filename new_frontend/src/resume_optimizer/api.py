"""
API layer for Resume Optimization Platform
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime

from .services import ResumeAnalyzer, ResumeOptimizer, StructuredDataConverter
from .models import ResumeAnalysis, OptimizedResume, JobRoleRequirements


class ResumeOptimizationAPI:
    """API interface for Resume Optimization Platform"""
    
    def __init__(self):
        self.analyzer = ResumeAnalyzer()
        self.optimizer = ResumeOptimizer()
        self.converter = StructuredDataConverter()
    
    def analyze_resume(self, resume_text: str, output_format: str = "dict") -> Dict[str, Any]:
        """
        Analyze resume and return detailed analysis
        
        Args:
            resume_text: Raw resume text
            output_format: "dict", "json", or "structured"
            
        Returns:
            Analysis results in specified format
        """
        analysis = self.analyzer.analyze_resume(resume_text)
        
        if output_format == "json":
            return self._analysis_to_json(analysis)
        elif output_format == "structured":
            return self._analysis_to_structured_dict(analysis)
        else:
            return self._analysis_to_dict(analysis)
    
    def convert_to_structured(self, resume_text: str, output_format: str = "dict") -> Dict[str, Any]:
        """
        Convert unstructured resume to structured format
        
        Args:
            resume_text: Raw resume text
            output_format: "dict" or "json"
            
        Returns:
            Structured resume data
        """
        structured = self.converter.convert_to_structured(resume_text)
        
        if output_format == "json":
            return self.converter.convert_to_json(structured)
        else:
            return self._structured_to_dict(structured)
    def optimize_resume(self, resume_text: str, job_role: str, 
                       job_requirements: Optional[Dict[str, Any]] = None,
                       output_format: str = "dict",
                       user_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize resume for specific job role
        
        Args:
            resume_text: Raw resume text
            job_role: Target job role
            job_requirements: Optional job requirements
            output_format: "dict", "json", or "full"
            
        Returns:
            Optimized resume and analysis
        """
        # Convert requirements if provided
        req_obj = None
        if job_requirements:
            if isinstance(job_requirements, dict):
                req_obj = JobRoleRequirements(
                    role=job_role,
                    required_skills=job_requirements.get('required_skills', []),
                    preferred_skills=job_requirements.get('preferred_skills', []),
                    experience_level=job_requirements.get('experience_level', 'entry'),
                    key_responsibilities=job_requirements.get('key_responsibilities', []),
                    industry_keywords=job_requirements.get('industry_keywords', [])
                )
            else:
                req_obj = job_requirements  # Already a JobRoleRequirements object
        
        optimized = self.optimizer.optimize_for_role(resume_text, job_role, req_obj, user_data=user_data)
        
        if output_format == "json":
            return self._optimized_to_json(optimized)
        elif output_format == "full":
            return self._optimized_to_full_dict(optimized)
        else:
            return self._optimized_to_dict(optimized)
    
    def get_fault_analysis(self, resume_text: str) -> Dict[str, Any]:
        """
        Get detailed fault analysis
        
        Args:
            resume_text: Raw resume text
            
        Returns:
            Fault analysis with recommendations
        """
        analysis = self.analyzer.analyze_resume(resume_text)
        
        faults_by_category = {}
        for fault in analysis.faults:
            category = fault.category.value
            if category not in faults_by_category:
                faults_by_category[category] = []
            faults_by_category[category].append({
                'severity': fault.severity.value,
                'description': fault.description,
                'location': fault.location,
                'suggestion': fault.suggestion,
                'line_number': fault.line_number
            })
        
        return {
            'total_faults': len(analysis.faults),
            'faults_by_category': faults_by_category,
            'critical_faults': len([f for f in analysis.faults if f.severity.value == 'critical']),
            'high_faults': len([f for f in analysis.faults if f.severity.value == 'high']),
            'medium_faults': len([f for f in analysis.faults if f.severity.value == 'medium']),
            'low_faults': len([f for f in analysis.faults if f.severity.value == 'low']),
            'overall_score': analysis.overall_score
        }
    
    def batch_analyze(self, resume_texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple resumes in batch
        
        Args:
            resume_texts: List of resume texts
            
        Returns:
            List of analysis results
        """
        results = []
        for i, text in enumerate(resume_texts):
            try:
                analysis = self.analyze_resume(text)
                analysis['resume_index'] = i
                results.append(analysis)
            except Exception as e:
                results.append({
                    'resume_index': i,
                    'error': str(e),
                    'status': 'failed'
                })
        
        return results
    
    def save_analysis(self, analysis: Dict[str, Any], filename: str, 
                     output_dir: str = "outputs") -> str:
        """
        Save analysis results to file
        
        Args:
            analysis: Analysis results
            filename: Output filename
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        file_path = output_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        return str(file_path)
    
    def _analysis_to_dict(self, analysis: ResumeAnalysis) -> Dict[str, Any]:
        """Convert analysis to dictionary"""
        return {
            'original_text': analysis.original_text,
            'structured_data': self._structured_to_dict(analysis.structured_data),
            'faults': [
                {
                    'category': fault.category.value,
                    'severity': fault.severity.value,
                    'description': fault.description,
                    'location': fault.location,
                    'suggestion': fault.suggestion,
                    'line_number': fault.line_number
                }
                for fault in analysis.faults
            ],
            'scores': {
                'ats_score': analysis.ats_score,
                'keyword_score': analysis.keyword_score,
                'formatting_score': analysis.formatting_score,
                'overall_score': analysis.overall_score
            },
            'analysis_timestamp': analysis.analysis_timestamp.isoformat()
        }
    
    def _analysis_to_json(self, analysis: ResumeAnalysis) -> str:
        """Convert analysis to JSON string"""
        return json.dumps(self._analysis_to_dict(analysis), indent=2, default=str)
    
    def _analysis_to_structured_dict(self, analysis: ResumeAnalysis) -> Dict[str, Any]:
        """Convert analysis to simplified structured format"""
        return {
            'personal_info': self._personal_info_to_dict(analysis.structured_data.personal_info),
            'summary': analysis.structured_data.summary,
            'work_experience': [
                {
                    'company': exp.company,
                    'position': exp.position,
                    'description': exp.description
                }
                for exp in analysis.structured_data.work_experience
            ],
            'education': [
                {
                    'institution': edu.institution,
                    'degree': edu.degree
                }
                for edu in analysis.structured_data.education
            ],
            'skills': analysis.structured_data.skills,
            'projects': [
                {
                    'name': proj.name,
                    'description': proj.description
                }
                for proj in analysis.structured_data.projects
            ]
        }
    
    def _structured_to_dict(self, structured) -> Dict[str, Any]:
        """Convert structured resume to dictionary"""
        return {
            'personal_info': self._personal_info_to_dict(structured.personal_info),
            'summary': structured.summary,
            'work_experience': [
                {
                    'company': exp.company,
                    'position': exp.position,
                    'start_date': exp.start_date,
                    'end_date': exp.end_date,
                    'description': exp.description,
                    'achievements': exp.achievements
                }
                for exp in structured.work_experience
            ],
            'education': [
                {
                    'institution': edu.institution,
                    'degree': edu.degree,
                    'field': edu.field,
                    'start_date': edu.start_date,
                    'end_date': edu.end_date,
                    'gpa': edu.gpa
                }
                for edu in structured.education
            ],
            'projects': [
                {
                    'name': proj.name,
                    'description': proj.description,
                    'technologies': proj.technologies,
                    'duration': proj.duration,
                    'achievements': proj.achievements
                }
                for proj in structured.projects
            ],
            'skills': structured.skills,
            'certifications': structured.certifications,
            'languages': structured.languages,
            'awards': structured.awards
        }
    
    def _personal_info_to_dict(self, personal_info) -> Dict[str, Any]:
        """Convert personal info to dictionary"""
        return {
            'name': personal_info.name,
            'email': personal_info.email,
            'phone': personal_info.phone,
            'linkedin': personal_info.linkedin,
            'github': personal_info.github,
            'location': personal_info.location
        }
    
    def _optimized_to_dict(self, optimized: OptimizedResume) -> Dict[str, Any]:
        """Convert optimized resume to dictionary"""
        return {
            'target_role': optimized.target_role,
            'optimized_content': optimized.optimized_content,
            'improvements_made': optimized.improvements_made,
            'optimization_score': optimized.optimization_score,
            'generation_timestamp': optimized.generation_timestamp.isoformat(),
            'original_analysis': self._analysis_to_dict(optimized.original_analysis)
        }
    
    def _optimized_to_json(self, optimized: OptimizedResume) -> str:
        """Convert optimized resume to JSON string"""
        return json.dumps(self._optimized_to_dict(optimized), indent=2, default=str)
    
    def _optimized_to_full_dict(self, optimized: OptimizedResume) -> Dict[str, Any]:
        """Convert optimized resume to full dictionary with all details"""
        result = self._optimized_to_dict(optimized)
        
        if optimized.job_requirements:
            result['job_requirements'] = {
                'role': optimized.job_requirements.role,
                'required_skills': optimized.job_requirements.required_skills,
                'preferred_skills': optimized.job_requirements.preferred_skills,
                'experience_level': optimized.job_requirements.experience_level,
                'key_responsibilities': optimized.job_requirements.key_responsibilities,
                'industry_keywords': optimized.job_requirements.industry_keywords
            }
        
        return result
