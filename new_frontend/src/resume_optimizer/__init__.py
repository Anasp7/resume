"""
Resume Optimization Platform

A microservice for analyzing resumes, identifying faults,
converting unstructured data to structured format, and
generating job-role-specific optimized resumes.
"""

from .services import ResumeAnalyzer, ResumeOptimizer, StructuredDataConverter
from .models import ResumeAnalysis, OptimizedResume, ResumeFault
from .api import ResumeOptimizationAPI

__all__ = [
    'ResumeAnalyzer',
    'ResumeOptimizer', 
    'StructuredDataConverter',
    'ResumeAnalysis',
    'OptimizedResume',
    'ResumeFault',
    'ResumeOptimizationAPI'
]
