"""
Async Resume Processor - Non-blocking, fast resume handling
Uses threading and caching for improved performance
"""
import asyncio
import threading
import queue
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import os

@dataclass
class ProcessingTask:
    """Represents a resume processing task"""
    task_id: str
    resume_text: str
    callback: Optional[Callable] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class AsyncResumeProcessor:
    """Non-blocking resume processor with caching and threading"""
    
    def __init__(self, max_workers: int = 4, cache_size: int = 100):
        self.max_workers = max_workers
        self.cache_size = cache_size
        self.processing_queue = queue.Queue()
        self.result_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.is_running = False
        self.worker_thread = None
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_processed = 0
        
    def start(self):
        """Start the async processor"""
        if not self.is_running:
            self.is_running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            print("🚀 Async Resume Processor started")
    
    def stop(self):
        """Stop the async processor"""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        self.executor.shutdown(wait=True)
        print("🛑 Async Resume Processor stopped")
    
    def _generate_cache_key(self, resume_text: str) -> str:
        """Generate cache key from resume text"""
        # Use first 100 chars + length for simple but effective key
        text_sample = resume_text[:100].strip()
        text_length = len(resume_text)
        return hashlib.md5(f"{text_sample}_{text_length}".encode()).hexdigest()
    
    def _worker_loop(self):
        """Main worker loop for processing tasks"""
        while self.is_running:
            try:
                # Get task from queue with timeout
                task = self.processing_queue.get(timeout=1.0)
                
                # Check cache first
                cache_key = self._generate_cache_key(task.resume_text)
                if cache_key in self.result_cache:
                    result = self.result_cache[cache_key]
                    self.cache_hits += 1
                else:
                    # Process the resume
                    result = self._process_resume_sync(task.resume_text)
                    
                    # Add to cache (with size limit)
                    if len(self.result_cache) >= self.cache_size:
                        # Remove oldest entry (simple FIFO)
                        oldest_key = next(iter(self.result_cache))
                        del self.result_cache[oldest_key]
                    
                    self.result_cache[cache_key] = result
                    self.cache_misses += 1
                
                self.total_processed += 1
                
                # Call callback if provided
                if task.callback:
                    task.callback(task.task_id, result)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error processing task: {e}")
                continue
    
    def _process_resume_sync(self, resume_text: str) -> Dict[str, Any]:
        """Synchronous resume processing (fast version)"""
        import re
        
        # Fast text processing
        lines = resume_text.split('\n')
        
        # Extract personal info
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text)
        phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', resume_text)
        linkedin_match = re.search(r'linkedin\.com/in/[\w-]+', resume_text)
        
        # Extract skills using enhanced keyword list
        enhanced_skills = [
            "python", "java", "javascript", "c++", "c#", "ruby", "go", "rust", "swift", "kotlin",
            "sql", "nosql", "mongodb", "postgresql", "mysql", "oracle", "redis", "cassandra",
            "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible", "jenkins",
            "react", "angular", "vue", "nodejs", "express", "django", "flask", "spring", "rails",
            "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "matplotlib",
            "machine learning", "deep learning", "nlp", "computer vision", "data science",
            "git", "github", "gitlab", "svn", "mercurial", "ci/cd", "devops", "agile", "scrum",
            "html", "css", "sass", "less", "bootstrap", "tailwind", "webpack", "babel",
            "linux", "ubuntu", "centos", "windows", "macos", "bash", "powershell", "shell scripting",
            "microservices", "rest api", "graphql", "grpc", "websockets", "mqtt", "kafka",
            "elasticsearch", "solr", "kibana", "logstash", "splunk", "new relic", "datadog"
        ]
        
        found_skills = []
        text_lower = resume_text.lower()
        for skill in enhanced_skills:
            if skill in text_lower:
                found_skills.append(skill)
        
        # Extract experience
        experience_keywords = ['engineer', 'developer', 'manager', 'analyst', 'director', 'lead', 'architect']
        has_experience = any(keyword in text_lower for keyword in experience_keywords)
        
        # Extract education
        edu_keywords = ['university', 'college', 'bachelor', 'master', 'phd', 'btech', 'mtech', 'degree']
        has_education = any(keyword in text_lower for keyword in edu_keywords)
        
        # Extract projects
        project_keywords = ['project', 'developed', 'built', 'created', 'implemented', 'designed']
        projects = []
        for line in lines:
            if any(keyword in line.lower() for keyword in project_keywords):
                if len(line.strip()) > 10 and len(line.strip()) < 100:
                    projects.append(line.strip())
        
        return {
            'personal_info': {
                'email': email_match.group() if email_match else '',
                'phone': phone_match.group() if phone_match else '',
                'linkedin': linkedin_match.group() if linkedin_match else ''
            },
            'skills': {
                'technical': found_skills,
                'total_count': len(found_skills)
            },
            'has_experience': has_experience,
            'has_education': has_education,
            'projects': projects[:5],  # Limit to 5 projects
            'text_stats': {
                'char_count': len(resume_text),
                'word_count': len(resume_text.split()),
                'line_count': len(lines)
            },
            'processing_time': time.time()
        }
    
    def process_resume_async(self, resume_text: str, callback: Optional[Callable] = None) -> str:
        """Process resume asynchronously"""
        task_id = f"task_{int(time.time() * 1000)}"
        task = ProcessingTask(
            task_id=task_id,
            resume_text=resume_text,
            callback=callback
        )
        
        self.processing_queue.put(task)
        return task_id
    
    def get_result_sync(self, task_id: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        """Get result synchronously (wait for completion)"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if result is in cache
            # This is a simplified approach - in production, you'd track task-specific results
            if self.total_processed > 0:
                # Return the most recent result (simplified)
                if self.result_cache:
                    return next(iter(self.result_cache.values()))
            
            time.sleep(0.1)
        
        return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'cache_size': len(self.result_cache),
            'total_processed': self.total_processed,
            'queue_size': self.processing_queue.qsize()
        }

# Global instance
async_processor = AsyncResumeProcessor()

def init_async_processor():
    """Initialize the global async processor"""
    async_processor.start()
    return async_processor

def shutdown_async_processor():
    """Shutdown the global async processor"""
    async_processor.stop()

# Fast resume processing function
def fast_process_resume(resume_text: str) -> Dict[str, Any]:
    """Fast resume processing using async processor"""
    # Check cache first
    cache_key = async_processor._generate_cache_key(resume_text)
    if cache_key in async_processor.result_cache:
        async_processor.cache_hits += 1
        return async_processor.result_cache[cache_key]
    
    # Process synchronously if not in cache
    result = async_processor._process_resume_sync(resume_text)
    
    # Add to cache
    if len(async_processor.result_cache) >= async_processor.cache_size:
        oldest_key = next(iter(async_processor.result_cache))
        del async_processor.result_cache[oldest_key]
    
    async_processor.result_cache[cache_key] = result
    async_processor.cache_misses += 1
    
    return result
