import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Optional

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class TextPreprocessor:
    """
    A class for preprocessing text data with various NLP techniques.
    """
    
    def __init__(self):
        """Initialize the text preprocessor with necessary components."""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation = set(string.punctuation)
        
        # Custom list of words to keep (common in resumes that might be filtered as stopwords)
        self.important_words = {
            'c', 'c++', 'c#', '.net', 'asp.net', 'mvc', 'api', 'apis',
            'javascript', 'python', 'java', 'sql', 'nosql', 'html', 'css',
            'ml', 'ai', 'nlp', 'cv', 'dl', 'iot', 'bigdata', 'aws', 'azure',
            'gcp', 'docker', 'kubernetes', 'ci/cd', 'tcp/ip', 'http', 'https'
        }
        
        # Remove important words from stop words
        self.stop_words = self.stop_words - self.important_words
    
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from the text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with URLs removed
        """
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)
    
    def remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation from the text while preserving important symbols.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with punctuation removed
        """
        # Keep some special characters that are common in tech skills
        keep_chars = {'+', '#', '.', '/', '-'}
        cleaned_text = []
        
        for char in text:
            if char not in self.punctuation or char in keep_chars:
                cleaned_text.append(char)
            else:
                cleaned_text.append(' ')
                
        return ''.join(cleaned_text)
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize the input text into words.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of tokens
        """
        return word_tokenize(text.lower())
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from the list of tokens.
        
        Args:
            tokens (List[str]): List of word tokens
            
        Returns:
            List[str]: Filtered list of tokens
        """
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize the list of tokens.
        
        Args:
            tokens (List[str]): List of word tokens
            
        Returns:
            List[str]: Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def clean_text(self, text: str) -> Optional[str]:
        """
        Clean and preprocess the input text.
        
        Args:
            text (str): Raw input text
            
        Returns:
            Optional[str]: Preprocessed text as a single string, or None if input is invalid
        """
        if not text or not isinstance(text, str):
            return None
            
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = self.remove_urls(text)
            
            # Remove punctuation (with special handling for tech terms)
            text = self.remove_punctuation(text)
            
            # Tokenize
            tokens = self.tokenize_text(text)
            
            # Remove stopwords
            tokens = self.remove_stopwords(tokens)
            
            # Lemmatize
            tokens = self.lemmatize_tokens(tokens)
            
            # Join tokens back into a single string
            return ' '.join(tokens)
            
        except Exception as e:
            print(f"Error in text preprocessing: {e}")
            return None

# Create a global instance for easy importing
preprocessor = TextPreprocessor()

def preprocess_text(text: str) -> Optional[str]:
    """
    Convenience function to preprocess text using the default preprocessor.
    
    Args:
        text (str): Raw input text
        
    Returns:
        Optional[str]: Preprocessed text, or None if input is invalid
    """
    return preprocessor.clean_text(text)
