import os
import re
from typing import Optional
import PyPDF2
import docx2txt

class ResumeParser:
    """
    A class to parse resume files in different formats (PDF, DOCX, TXT)
    and extract raw text content.
    """
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        try:
            with open(file_path, 'rb') as file:
                # Create a PDF reader object
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from each page
                text = ''
                for page in pdf_reader.pages:
                    text += page.extract_text() + '\n'
                return text.strip()
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """
        Extract text from a DOCX file.
        
        Args:
            file_path (str): Path to the DOCX file
            
        Returns:
            str: Extracted text from the DOCX
        """
        try:
            return docx2txt.process(file_path).strip()
        except Exception as e:
            print(f"Error extracting text from DOCX: {e}")
            return ""

    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """
        Extract text from a TXT file.
        
        Args:
            file_path (str): Path to the TXT file
            
        Returns:
            str: Extracted text from the TXT file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            print(f"Error reading text file: {e}")
            return ""

    @classmethod
    def parse_resume(cls, file_path: str) -> Optional[str]:
        """
        Parse a resume file and extract text based on file extension.
        
        Args:
            file_path (str): Path to the resume file
            
        Returns:
            Optional[str]: Extracted text if successful, None otherwise
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
            
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return cls.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            return cls.extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            return cls.extract_text_from_txt(file_path)
        else:
            print(f"Unsupported file format: {file_ext}")
            return None

    @classmethod
    def clean_resume_text(cls, text: str) -> str:
        """
        Basic cleaning of resume text by removing extra whitespace and newlines.
        
        Args:
            text (str): Raw resume text
            
        Returns:
            str: Cleaned resume text
        """
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @classmethod
    def load_and_parse_resume(cls, file_path: str) -> Optional[str]:
        """
        Load and parse a resume file with error handling.
        
        Args:
            file_path (str): Path to the resume file
            
        Returns:
            Optional[str]: Cleaned resume text if successful, None otherwise
        """
        try:
            text = cls.parse_resume(file_path)
            if text:
                return cls.clean_resume_text(text)
            return None
        except Exception as e:
            print(f"Error processing resume {file_path}: {e}")
            return None
