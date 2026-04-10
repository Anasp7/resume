import re
import spacy
from typing import List, Dict
import json
import os
from datetime import datetime
from pathlib import Path

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


class KeywordClassifier:
    """
    Enhanced keyword classifier with training capability
    """
    
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Using basic pattern matching.")
            self.nlp = None
        
        # Initialize training data with absolute path
        current_dir = Path(__file__).parent
        self.training_data_file = current_dir / "enhanced_training_data.json"
        self.load_training_data()
    
    # Original keyword sets
    SKILLS = {
        "python", "java", "c++", "machine learning", "deep learning",
        "nlp", "sql", "mysql", "mongodb", "django", "flask",
        "tensorflow", "pytorch", "data analysis", "power bi",
        "excel", "git", "github", "aws", "ros", "ros2",
        "computer vision", "data science"
    }
    
    EDUCATION_KEYWORDS = {
        "btech", "b.tech", "be", "mtech", "m.tech", "msc", "mba",
        "bachelor", "master", "phd", "university", "college",
        "school", "degree", "engineering", "computer science", "cs",
        "information technology", "b.sc", "m.sc", "diploma",
        "computer science and engineering", "software engineering", "electrical engineering"
    }
    
    EXPERIENCE_KEYWORDS = {
        "software engineer", "data scientist", "full stack developer",
        "backend developer", "frontend developer", "devops engineer",
        "project manager", "team lead", "senior developer",
        "intern", "analyst", "consultant"
    }
    
    PROJECT_KEYWORDS = {
        "project", "system", "application", "model",
        "dashboard", "website", "tool", "autonomous vehicle",
        "data analysis project", "machine learning model"
    }
    
    def load_training_data(self):
        """Load training data from JSON file"""
        if os.path.exists(self.training_data_file):
            try:
                with open(self.training_data_file, 'r', encoding='utf-8') as f:
                    self.training_data = json.load(f)
                    print(f"Loaded {len(self.training_data)} training examples")
            except Exception as e:
                print(f"Error loading training data: {e}")
                self.training_data = {
                    "skills": [], "education": [], "experience": [], "projects": []
                }
        else:
            # Create default training data
            self.training_data = {
                "skills": [], "education": [], "experience": [], "projects": []
            }
            with open(self.training_data_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_data, f, indent=2)
                print(f"Created new training file with default structure")
    
    def add_training_example(self, category: str, keywords: List[str]):
        """Add a training example for learning"""
        if category in self.training_data:
            self.training_data[category].extend(keywords)
            print(f"Added {len(keywords)} examples to {category}")
        else:
            print(f"Error: Unknown category '{category}'")
    
    def save_training_data(self):
        """Save training data to JSON file"""
        try:
            with open(self.training_data_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_data, f, indent=2)
                print(f"Training data saved to {self.training_data_file}")
        except Exception as e:
            print(f"Error saving training data: {e}")
    
    def classify_with_training(self, keywords: List[str]) -> Dict[str, List[str]]:
        """Enhanced classification using training data"""
        classified = {
            "NAME": [], "SKILLS": [], "EDUCATION": [], "EXPERIENCE": [], "PROJECTS": [], "OTHER": []
        }
        
        for kw in keywords:
            kw_lower = kw.lower()
            classified_flag = False
            
            # Check against training data first
            for category, trained_keywords in self.training_data.items():
                if any(trained_kw in kw_lower for trained_kw in trained_keywords):
                    classified[category.upper()].append(kw)
                    classified_flag = True
                    break
            
            # If not found in training data, use original classification
            if not classified_flag:
                if self._is_name(kw):
                    classified["NAME"].append(kw)
                elif self._is_skill(kw_lower):
                    classified["SKILLS"].append(kw)
                elif self._is_education(kw_lower):
                    classified["EDUCATION"].append(kw)
                elif self._is_experience(kw_lower):
                    classified["EXPERIENCE"].append(kw)
                elif self._is_project(kw_lower):
                    classified["PROJECTS"].append(kw)
                else:
                    classified["OTHER"].append(kw)
        
        return classified
    
    def _is_name(self, text: str) -> bool:
        doc = nlp(text)
        # Check for PERSON entities - must be actual names
        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text.strip()) > 2:
                return True
        # Check for name patterns (2-3 words, capitalized, no numbers, not technical terms)
        words = text.split()
        technical_terms = self.SKILLS.union(self.EXPERIENCE_KEYWORDS).union(self.EDUCATION_KEYWORDS)
        
        if (2 <= len(words) <= 3 and 
            all(word.isalpha() for word in words) and 
            all(word.istitle() for word in words) and
            not any(word.lower() in technical_terms for word in words)):
            return True
        return False
    
    def _is_skill(self, text: str) -> bool:
        return text in self.SKILLS
    
    def _is_education(self, text: str) -> bool:
        # Check for compound education terms first
        education_compounds = {
            "computer science", "software engineering", 
            "electrical engineering", "mechanical engineering",
            "civil engineering", "information technology"
        }
        if any(compound in text for compound in education_compounds):
            print(f"DEBUG: Found compound education term in: {text}")
            return True
        # Only check specific education keywords, not generic "engineering"
        specific_education = {
            "btech", "b.tech", "be", "mtech", "m.tech", "msc", "mba",
            "bachelor", "master", "phd", "university", "college",
            "school", "degree", "cs", "information technology", 
            "b.sc", "m.sc", "diploma"
        }
        # Exclude common non-education terms
        exclude_terms = {"member", "team", "lead", "manager"}
        words = text.split()
        
        if any(term in words for term in exclude_terms):
            return False
        
        education_found = any(keyword in text for keyword in specific_education)
        print(f"DEBUG: Education check for '{text}': {education_found}")
        return education_found
    
    def _is_experience(self, text: str) -> bool:
        # Check for compound experience terms
        experience_compounds = {
            "ros software", "software engineer", "data scientist",
            "project manager", "team lead", "senior developer"
        }
        if any(compound in text for compound in experience_compounds):
            return True
        return any(keyword in text for keyword in self.EXPERIENCE_KEYWORDS)
    
    def _is_project(self, text: str) -> bool:
        return any(keyword in text for keyword in self.PROJECT_KEYWORDS)
