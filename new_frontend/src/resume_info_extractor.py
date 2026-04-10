import re
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class ResumeInfo:
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    skills: List[str] = None
    projects: List[Dict] = None
    education: List[Dict] = None
    experience: List[Dict] = None

class ResumeInfoExtractor:
    def extract_all(self, text: str) -> ResumeInfo:
        pass  # Add implementation later
    
    def extract_name(self, text: str) -> Optional[str]:
        pass  # Add implementation later
    
    def extract_contact_info(self, text: str) -> Dict:
        pass  # Add implementation later
    
    def extract_skills(self, text: str) -> List[str]:
        pass  # Add implementation later
    
    def extract_projects(self, text: str) -> List[Dict]:
        pass  # Add implementation later