import os
import json
from typing import Dict, Any, List, Optional
import re

def call_llm(prompt: str) -> str:
    """Generate career guidance or optimized resume content using LLM or fallbacks"""
    
    # Priority 1: Groq (Fast & High Reliability)
    groq_key = os.environ.get("GROQ_API_KEY") 
    if groq_key:
        try:
            return _call_groq(prompt, groq_key)
        except Exception as e:
            print(f"Error calling Groq: {e}")

    # Priority 2: Hugging Face 
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        try:
            return _call_huggingface(prompt, hf_token)
        except Exception as e:
            print(f"Error calling Hugging Face: {e}")

    # Priority 3: OpenAI / xAI 
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("XAI_API_KEY")
    if api_key:
        try:
            return _call_openai(prompt, api_key)
        except Exception as e:
            print(f"Error calling LLM (Priority 3): {e}")
            
    # Priority 4: Local Ollama (User Request)
    try:
        ollama_response = _call_ollama(prompt)
        if ollama_response:
            return ollama_response
    except Exception as e:
        print(f"Error calling Ollama (Priority 4): {e}")

    # Final Fallback to high-fidelity mocks if no API keys worked or all failed
    print("⚠️ All AI APIs failed or were missing. Using high-fidelity local fallback...")
    
    if "ResumeForge-X" in prompt or "CONFIRMED EXPERIENCE:" in prompt:
        return _generate_resumeforge_x_response(prompt)
    elif "CLARIFICATION QUESTIONS" in prompt:
        return _generate_clarification_questions_mock(prompt)
    elif "Career Growth Plan" in prompt or "growth plan" in prompt.lower() or "Target Job Goal:" in prompt:
        return _generate_growth_plan_fallback(prompt)
    elif "market insights" in prompt.lower():
        return _generate_market_insights_mock(prompt)
    elif "Optimize this resume" in prompt:
        return _generate_optimized_resume(prompt)
    else:
        # Default career recommendation fallback
        return _generate_career_guidance(prompt)


def _call_groq(prompt: str, api_key: str) -> str:
    """Real Groq API integration"""
    try:
        from openai import OpenAI
        
        # Groq uses the OpenAI-compatible SDK
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        
        # Using Llama 3.3 70B for high quality and speed
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"} if "JSON" in prompt else {"type": "text"}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Groq API Error: {e}")
        raise

def _call_huggingface(prompt: str, token: str) -> str:
    """Hugging Face Inference API integration with robust fallbacks"""
    import requests
    import time
    
    # Mistral is usually more accessible for free-tier users
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    
    # We will try both the new Router and the Dedicated Inference API
    endpoints = [
        ("https://router.huggingface.co/v1/chat/completions", True), # OpenAI format
        (f"https://api-inference.huggingface.co/models/{model_id}", False) # HF format
    ]
    
    last_error = ""
    for url, is_openai_format in endpoints:
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        
        if is_openai_format:
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.7
            }
        else:
            payload = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": 1024, "temperature": 0.7}
            }
            
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if is_openai_format:
                    return result["choices"][0]["message"]["content"]
                else:
                    # Direct HF API returns a list or dict depending on task
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get("generated_text", str(result))
                    return result.get("generated_text", str(result))
            
            # Handle specific permission error
            if response.status_code == 403:
                last_error = "TOKEN_PERMISSION_ERROR: Your Hugging Face token needs 'Make calls to Inference Providers' permission."
                continue
                
            last_error = f"{url}: {response.status_code} - {response.text}"
        except Exception as e:
            last_error = str(e)
            
    # If all fail, provide a clear actionable error
    if "TOKEN_PERMISSION_ERROR" in last_error:
        print("\n" + "="*60)
        print("❌ HUGGING FACE PERMISSION ERROR")
        print("Your token lacks 'Make calls to Inference Providers' permission.")
        print("To fix:")
        print("1. Go to https://huggingface.co/settings/tokens")
        print("2. Edit your token or create a new one.")
        print("3. Check 'Make calls to Inference Providers' under 'Inference' permissions.")
        print("="*60 + "\n")
        
    raise Exception(f"Hugging Face integration failed. {last_error}")

def _call_ollama(prompt: str) -> str:
    """Local Ollama integration as a reliable fallback"""
    import requests
    import os
    
    model = os.environ.get("OLLAMA_MODEL", "phi3")
    url = "http://localhost:11434/api/generate"
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.1,
            "options": {
                "num_predict": 3000,
                "num_ctx": 8192
            }
        }
        if "JSON" in prompt or "```json" in prompt:
            payload["format"] = "json"
            
        r = requests.post(url, json=payload, timeout=60)
        if r.status_code == 200:
            return r.json().get("response", "").strip()
    except Exception as e:
        print(f"Ollama local inference failed: {e}")
    return ""

def _call_openai(prompt: str, api_key: str) -> str:
    """Real OpenAI/xAI API integration"""
    try:
        from openai import OpenAI
        
        # Determine if xAI or OpenAI
        is_xai = api_key.startswith("xai-")
        base_url = "https://api.x.ai/v1" if is_xai else None
        # Default models
        model = "grok-2" if is_xai else "gpt-4o"
        
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"} if "JSON" in prompt else {"type": "text"}
        )
        return response.choices[0].message.content
    except Exception as e:
        # Don't print quota errors to keep output clean unless it's the only option
        raise e

def _generate_resumeforge_x_response(prompt: str) -> str:
    """High-fidelity mock response for ResumeForge-X prompt"""
    
    # Extract data from prompt to make mock relevant
    job_title = "Data Analyst"
    if "Target Job Title:" in prompt:
        job_title = prompt.split("Target Job Title:")[1].split("\n")[0].strip()
        
    company_name = ""
    if "Target Company (if provided):" in prompt:
        company_name = prompt.split("Target Company (if provided):")[1].split("\n")[0].strip()

    # Create high-fidelity JSON mock
    mock_data = {
        "meta": {
            "detected_profile": "mid-level-technical",
            "career_level": "Mid-Level",
            "domain": "Software Engineering",
            "years_experience": 5,
            "target_job": job_title,
            "target_company": company_name,
            "job_match_score": 85,
            "match_rationale": "Strong alignment with role requirements."
        },
        "name": "John Doe",
        "title": job_title,
        "email": "john.doe@email.com",
        "phone": "+1 (555) 000-0000",
        "linkedin": "linkedin.com/in/johndoe",
        "location": "San Francisco, CA",
        "summary": "",
        "experience": [],
        "education": [
            {
                "degree": "Bachelor of Science in Computer Science",
                "school": "University of Technology",
                "year": "2018"
            }
        ],
        "skills": {
            "technical": ["Python", "SQL", "Git"],
            "soft": ["Problem Solving", "Communication"]
        },
        "projects": [],
        "certifications": [],
        "ats_score": 85,
        "improvements": []
    }

    # Use user provided data to override defaults if present (interactive mode)
    if "CONFIRMED EXPERIENCE:" in prompt:
        import re
        exp_match = re.search(r"CONFIRMED EXPERIENCE:\n(.*?)(?:\n\nCONFIRMED|\n\nADDITIONAL|\n\nORIGINAL)", prompt, re.DOTALL)
        if exp_match:
            exp_text = exp_match.group(1)
            current_exp = None
            for line in exp_text.split('\n'):
                line = line.strip()
                if line.startswith("- "):
                    if current_exp: mock_data["experience"].append(current_exp)
                    title_comp = line[2:].split(" at ")
                    current_exp = {
                        "title": title_comp[0].strip(),
                        "company": title_comp[1].strip() if len(title_comp) > 1 else "Various",
                        "bullets": []
                    }
                elif line.startswith("* ") and current_exp:
                    current_exp["bullets"].append(line[2:].strip())
            if current_exp: mock_data["experience"].append(current_exp)

            # Clean up: Remove mistaken objective/summary lines from experience
            valid_exp = []
            for exp in mock_data["experience"]:
                title_lower = exp.get("title", "").lower()
                # If it looks like a paragraph/student objective, push to summary instead
                if len(title_lower) > 50 or "motivated" in title_lower or "aspiring" in title_lower or "student" in title_lower:
                    mock_data["summary"] = exp.get("title", "")
                else:
                    valid_exp.append(exp)
            mock_data["experience"] = valid_exp

    if "CONFIRMED SKILLS:" in prompt:
        import re
        skills_match = re.search(r"CONFIRMED SKILLS: (.*?)\n", prompt)
        if skills_match:
            mock_data["skills"]["technical"] = [s.strip() for s in skills_match.group(1).split(',')]

    if "CONFIRMED PROJECTS:" in prompt:
        import re
        proj_match = re.search(r"CONFIRMED PROJECTS:\n(.*?)(?:\n\nADDITIONAL|\n\nORIGINAL)", prompt, re.DOTALL)
        if proj_match:
            for line in proj_match.group(1).split('\n'):
                line = line.strip()
                if not line: continue
                name = line[2:].strip() if line.startswith("- ") else line
                mock_data["projects"].append({
                    "name": name,
                    "description": f"Lead development of the {name} project, focusing on technical innovation and system performance."
                })

    if "ADDITIONAL PROJECT/EXPERIENCE DETAILS:" in prompt:
        import re
        details_match = re.search(r"ADDITIONAL PROJECT/EXPERIENCE DETAILS:\n(.*?)(?:\n\nORIGINAL)", prompt, re.DOTALL)
        if details_match:
            answers = []
            for line in details_match.group(1).split('\n'):
                if line.startswith("A: "):
                    answers.append(line[3:].strip())
            
            if answers:
                # Inject directly into experience/projects to show it was used
                if mock_data["experience"] and len(mock_data["experience"]) > 0:
                    mock_data["experience"][0]["bullets"].append(f"Successfully implemented key features, utilizing {answers[0] if len(answers) > 0 else 'advanced methodologies'}.")
                    if len(answers) > 1:
                        mock_data["experience"][0]["bullets"].append(f"Delivered measurable impact: {answers[1]}.")
                elif mock_data["projects"] and len(mock_data["projects"]) > 0:
                    mock_data["projects"][0]["description"] += f" Leveraged {answers[0]} and achieved {answers[1] if len(answers) > 1 else 'significant results'}."
                else:
                    if len(answers) >= 2:
                        mock_data["summary"] += f" Specifically leveraged {answers[0]} to drive project success and achieved {answers[1]}."
                    else:
                        mock_data["summary"] += f" Specifically leveraged {answers[0]} to drive project success."

    # Final Summary: Professional and targeted, NOT objective-heavy
    top_skills = mock_data["skills"]["technical"][:3]
    mock_data["summary"] = f"Technical professional with expertise in {', '.join(top_skills)} and hands-on experience in robotic systems. Proven ability to contribute to complex team projects. Committed to technical excellence as a {job_title}."
    
    return json.dumps(mock_data, indent=2)

def _generate_career_guidance(prompt: str) -> str:
    """Generate career guidance using rule-based logic (legacy)"""
    # Safe parsing using regex or keyword search
    skills_section = ""
    projects_section = ""
    other_section = ""
    
    if "Skills:" in prompt:
        parts = prompt.split("Skills:")[1].split("Projects:")
        skills_section = parts[0].strip()
        if len(parts) > 1:
            parts2 = parts[1].split("Experience:") # More likely to exist
            projects_section = parts2[0].strip()
    
    # Fallback to defaults if parsing fails
    if not skills_section: skills_section = "python, sql, ros2"
    if not projects_section: projects_section = "autonomous vehicle"
    if not other_section: other_section = "automation, technology"
    
    skills = [s.strip() for s in skills_section.split(",") if s.strip()]
    projects = [p.strip() for p in projects_section.split(",") if p.strip()]
    other = [o.strip() for o in other_section.split(",") if o.strip()]
    
    # Convert to lowercase for matching
    skills_lower = [s.lower() for s in skills]
    projects_lower = [p.lower() for p in projects]
    other_lower = [o.lower() for o in other]
    
    # Generate career recommendations
    career_roles = []
    if "python" in skills_lower or "sql" in skills_lower:
        career_roles.append("Data Analyst: Strong programming and database skills")
    if "ros2" in skills_lower or "autonomous" in projects_lower:
        career_roles.append("Robotics Engineer: Experience with ROS and autonomous systems")
    if "python" in skills_lower and "software" in other_lower:
        career_roles.append("Software Developer: Programming skills and software background")
    if not career_roles:
        career_roles.append("Entry Level Engineer: Foundational technical skills")
    
    # Generate missing skills
    missing_skills = []
    if "python" in skills_lower and "machine learning" not in other_lower:
        missing_skills.append("Machine Learning: Enhances data analysis capabilities")
    if "ros2" in skills_lower and "c++" not in other_lower:
        missing_skills.append("C++: Essential for robotics development")
    if "sql" in skills_lower and "data visualization" not in other_lower:
        missing_skills.append("Data Visualization: Important for data analysis roles")
    
    # Generate resume improvements
    resume_improvements = []
    resume_improvements.append("Quantify achievements: Add metrics and numbers to projects")
    resume_improvements.append("Technical summary: Add a skills section at the top")
    resume_improvements.append("Project details: Expand on autonomous vehicle project specifics")
    
    # Generate ATS optimization
    ats_tips = []
    ats_tips.append("Keywords: Include more industry-specific terms")
    ats_tips.append("Format: Use simple formatting without tables or images")
    ats_tips.append("File type: Save as .txt or simple .docx for ATS compatibility")
    
    # Generate certifications/projects
    cert_projects = []
    if "python" in skills_lower:
        cert_projects.append("Python Certification: Validates programming expertise")
    if "ros2" in skills_lower:
        cert_projects.append("ROS Certification: Demonstrates robotics proficiency")
    cert_projects.append("Portfolio project: Build a complete automation system demo")
    
    # Format the response
    response = "CAREER ROLES:\n"
    for role in career_roles:
        response += f"- {role}\n"
    
    response += "\nMISSING SKILLS:\n"
    for skill in missing_skills:
        response += f"- {skill}\n"
    
    response += "\nRESUME IMPROVEMENTS:\n"
    for improvement in resume_improvements:
        response += f"- {improvement}\n"
    
    response += "\nATS OPTIMIZATION:\n"
    for tip in ats_tips:
        response += f"- {tip}\n"
    
    response += "\nCERTIFICATIONS/PROJECTS:\n"
    for cert in cert_projects:
        response += f"- {cert}\n"
    
    return response

def generate_career_guidance(resume_data: Dict[str, Any]) -> str:
    """Public wrapper for career guidance generation"""
    # Construct prompt for LLM
    prompt = f"""
    Generate professional career guidance based on this resume data:
    Skills: {', '.join(resume_data.get('skills', []))}
    Projects: {', '.join(resume_data.get('projects', []))}
    Experience: {str(resume_data.get('work_experience', []))}
    Other Keywords: {', '.join(resume_data.get('other', []))}
    
    Provide recommendations for roles, skills to learn, resume improvements, and ATS optimization.
    """
    try:
        return call_llm(prompt)
    except:
        return _generate_career_guidance(prompt)

def _generate_growth_plan_fallback(prompt: str) -> str:
    """Generate growth plan fallback when LLM fails"""
    import json
    
    # Parse Target Job from prompt
    target_job = "None"
    if "Target Job Goal:" in prompt:
        target_job_part = prompt.split("Target Job Goal:")[1].split("(")[0].strip()
        if target_job_part and target_job_part.lower() != "none":
            target_job = target_job_part

    # Parse skills and target job from prompt
    skills = []
    if "Skills:" in prompt:
        try:
            skills_part = prompt.split("Skills:")[1].split("Projects:")[0].strip()
            skills = [s.strip() for s in skills_part.split(",") if s.strip()]
        except:
            pass
            
    # Priority defaults
    short_term_goals = []
    medium_term_goals = []
    long_term_goals = []
    skill_roadmap = {}
    certifications = []
    projects = []
    top_skill = "Technology"
    if skills:
        top_skill = skills[0]
        # Skip generic headers
        if top_skill.lower() in ["technical skills", "skills", "core skills", "professional skills"] and len(skills) > 1:
            top_skill = skills[1]

    # Parse current role from experience section in prompt
    detected_current_role = f"{top_skill} Professional"
    if "Experience:" in prompt:
        try:
            # Extract text between Experience: and the next section header
            exp_part = prompt.split("Experience:")[1].split("Skills:")[0].split("Other:")[0].strip()
            first_line = exp_part.split('\n')[0].strip()
            if first_line and first_line != "Fresher":
                # Clean up "Role at Company (Dates)" format
                clean_title = first_line.split(" at ")[0].split("(")[0].strip()
                if clean_title and len(clean_title) < 50:
                    detected_current_role = clean_title
        except:
            pass
    
    # Generate personalized growth plan based on target role and skills
    is_target_mode = target_job != "None"
    
    # Generic requirements mapping for fallback gap analysis
    role_requirements = {
        "Cloud Solutions Architect": ["AWS/Azure", "Terraform", "System Design", "Kubernetes", "Security"],
        "AI Solutions Architect": ["Python", "PyTorch/Tensorflow", "MLOps", "Model Scaling", "Deep Learning"],
        "Full-Stack System Architect": ["Frontend Architecture", "Backend Engineering", "PostgreSQL", "System Design", "GraphQL"],
        "Distributed Systems Lead": ["Microservices", "Messaging Queues", "CI/CD", "Docker", "Go/Java"],
        "Engineering Manager": ["Agile Leadership", "Strategic Planning", "Mentorship", "Budgeting", "Product Roadmap"],
        "Security Operations Lead": ["Penetration Testing", "Cloud Security", "Incident Response", "Compliance"],
        "Graphic Designer": ["Adobe Creative Suite", "Typography", "UI/UX Principles", "Brand Identity", "Motion Graphics"],
        "UI/UX Designer": ["Figma/Sketch", "User Research", "Prototyping", "Design Systems", "Interaction Design"],
        "Data Scientist": ["Python/R", "Statistical Modeling", "Machine Learning", "Data Visualization", "SQL"],
        "Mobile Developer": ["Swift/Kotlin", "React Native/Flutter", "Mobile UI Patterns", "API Integration", "App Store Deployment"],
        "Backend Developer": ["Server-side Logic", "Database Design", "API Development", "Caching Strategies", "Scalability"],
        "DevOps Engineer": ["CI/CD Pipelines", "Infrastructure as Code", "Monitoring/Logging", "Containerization", "Cloud Platforms"],
        "Product Manager": ["Product Roadmap", "User Analytics", "Agile/Scrum", "Stakeholder Management", "Market Research"]
    }

    # 1. Determine the target role and its requirements
    target_reqs = []
    if is_target_mode:
        next_target = target_job
        found_category = False
        for role_key, reqs in role_requirements.items():
            if role_key.lower() in target_job.lower():
                target_reqs = reqs
                found_category = True
                break
        if not found_category:
            target_reqs = ["High-level System Architecture", "Technical Leadership", "Project Management", "Domain Expertise"]
    else:
        # Fallback to intelligent cluster detection if no target job provided
        skills_str = " ".join(skills).lower()
        if any(s in skills_str for s in ["aws", "cloud", "azure", "devops"]):
            next_target = "Cloud Application Architect"
            target_reqs = role_requirements["Cloud Solutions Architect"]
        elif any(s in skills_str for s in ["machine learning", "ai", "data science"]):
            next_target = "AI Platform Engineer"
            target_reqs = role_requirements["AI Solutions Architect"]
        elif any(s in skills_str for s in ["javascript", "react", "frontend", "html"]):
            next_target = "Senior Full-Stack Technical Lead"
            target_reqs = role_requirements["Full-Stack System Architect"]
        elif any(s in skills_str for s in ["design", "adobe", "figma", "ui", "ux"]):
            next_target = "UI/UX Design Lead"
            target_reqs = role_requirements["UI/UX Designer"]
        elif any(s in skills_str for s in ["swift", "kotlin", "mobile", "android", "ios"]):
            next_target = "Lead Mobile Architect"
            target_reqs = role_requirements["Mobile Developer"]
        else:
            next_target = f"{top_skill} Principal Lead"
            target_reqs = ["Advanced Technical Standards", "System Scalability", "Team Mentorship", "Operational Excellence"]

    # 2. Skill Gap Analysis
    skills_lowered = [s.lower() for s in skills]
    gaps = [r for r in target_reqs if r.lower() not in skills_lowered]
    if not gaps: 
        gaps = ["Enterprise System Design", "Advanced Technical Strategy", "Cross-functional Leadership"]

    # 3. Generate Roadmap & Strategy based on Gaps
    next_level = next_target
    roadmap = [
        {"step": f"Phase 1: {gaps[0]} Mastery", "description": f"Focus on mastering {gaps[0]}, a critical missing competency for reaching the {next_target} level.", "duration": "Month 1-3"},
        {"step": f"Phase 2: {gaps[1] if len(gaps)>1 else 'Technical'} Deep Dive", "description": f"Build advanced proof-of-concept projects that demonstrate your ability to handle {gaps[1] if len(gaps)>1 else gaps[0]} at scale.", "duration": "Month 4-7"},
        {"step": f"Phase 3: {next_target} Transition", "description": f"Position yourself as a {next_target} by leading high-level technical decisions and bridging any remaining {gaps[-1]} gaps.", "duration": "Month 8-12"}
    ]

    detailed_courses = [
        {"name": f"{next_level} Professional Certification", "platform": "Coursera", "relevance": f"Validates your proficiency in {next_level} competencies."},
        {"name": f"Mastering {gaps[0]}", "platform": "Udemy", "relevance": f"Focused training to close your most critical technical skill gap."}
    ]

    skill_roadmap = {}
    for gap in gaps[:3]:
        skill_roadmap[gap] = {
            "current": "Gap / Foundational",
            "target": "Professional / Expert",
            "topics": [f"Deep Dive into {gap}", f"{gap} Security Patterns", f"Scaling {gap} in Enterprise"],
            "learning_time": "3-5 months",
            "suggested_projects": [f"Scalable {gap} Utility", f"Enterprise {gap} Migration Guide"],
            "learning_path": [
                {"step": "1", "title": "Theory Mastery", "activity": f"Master the underlying architecture and theory of {gap}"},
                {"step": "2", "title": "Build & Scale", "activity": f"Implement a comprehensive production-grade project focusing on {gap}"}
            ],
            "resources": [{"name": f"{gap} Official Reference", "url": "#"}],
            "next_skills": ["Systemic Optimization", "Technical Strategy"]
        }

    # 4. Final Growth Plan Object
    growth_plan = {
        "current_professional_role": detected_current_role,
        "next_progressive_level": next_level,
        "salary_range": "$115,000 - $165,000",
        "market_demand": f"High - Critical need for {next_level} roles with {gaps[0]} expertise.",
        "roadmap_steps": roadmap,
        "required_courses": detailed_courses,
        "soft_skills": [
            {"skill": "Stakeholder Negotiation", "importance": f"Required to drive {next_level} initiatives across teams."},
            {"skill": "Architectural Vision", "importance": "Essential for high-level technical leadership."}
        ],
        "interview_strategy": f"Highlight your deliberate closure of the {gaps[0]} gap to prove readiness for the {next_level} role.",
        "short_term": [f"Complete {gaps[0]} training", f"Analyze {next_level} case studies"],
        "medium_term": [f"Project lead involving {gaps[-1]}"],
        "long_term": [f"Secure {next_level} position"],
        "skill_roadmap": skill_roadmap,
        "certification_recommendations": [f"Lead {next_level} Specialist Certification"],
        "project_suggestions": [f"{next_level} Capstone Project"],
        "potential_roles": [next_level, "Technical Product Manager", "Enterprise Architect"]
    }


    
    return json.dumps(growth_plan, indent=2)


def _generate_optimized_resume(prompt: str) -> str:
    """Generate optimized resume content using legacy template system"""
    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from src.resume_optimizer.resume_templates import ResumeTemplateManager
    
    # Extract the original resume from the prompt
    if "CURRENT RESUME:" in prompt:
        original_resume = prompt.split("CURRENT RESUME:")[1].split("ANALYSIS FINDINGS:")[0].strip()
    else:
        original_resume = "Sample resume content"
    
    # Extract target role
    if "TARGET ROLE:" in prompt:
        target_role = prompt.split("TARGET ROLE:")[1].split("JOB REQUIREMENTS:")[0].strip()
    else:
        target_role = "Professional"
    
    # Create resume data structure from original text
    resume_data = {
        'original_text': original_resume,
        'structured_data': _parse_resume_text(original_resume)
    }
    
    # Get best template for this resume
    template_manager = ResumeTemplateManager()
    best_template = template_manager.get_best_template(resume_data, target_role)
    
    # Format resume using the template
    optimized_content = template_manager.format_resume(resume_data, best_template, target_role)
    
    return optimized_content

def optimize_resume(resume_text: str, target_role: str) -> str:
    """Public wrapper for resume optimization"""
    prompt = f"Optimize this resume for the role of {target_role}.\n\nCURRENT RESUME:\n{resume_text}"
    try:
        return call_llm(prompt)
    except:
        return _generate_optimized_resume(prompt)

def _parse_resume_text(text: str) -> Dict[str, Any]:
    """Parse resume text into structured data (legacy)"""
    import re
    
    structured = {
        'personal_info': {},
        'work_experience': [],
        'education': [],
        'skills': {'technical': [], 'soft': [], 'tools': []},
        'projects': []
    }
    
    lines = text.split('\n')
    
    # Extract personal info
    for line in lines[:10]:
        line = line.strip()
        
        # Email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', line)
        if email_match:
            structured['personal_info']['email'] = email_match.group()
        
        # Phone
        phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', line)
        if phone_match:
            structured['personal_info']['phone'] = phone_match.group()
        
        # LinkedIn
        linkedin_match = re.search(r'linkedin\.com/in/[\w-]+', line)
        if linkedin_match:
            structured['personal_info']['linkedin'] = linkedin_match.group()
    
    # Extract work experience
    current_exp = {}
    for line in lines:
        line = line.strip()
        
        # Look for experience patterns, avoiding long sentences (which are usually objectives/summaries)
        if len(line) < 60 and any(keyword in line.lower() for keyword in ['engineer', 'developer', 'manager', 'analyst', 'director', 'intern', 'student']):
            if current_exp:
                structured['work_experience'].append(current_exp)
            current_exp = {
                'position': line,
                'company': line,
                'description': []
            }
        elif current_exp and len(line) > 10:
            current_exp['description'].append(line)
    
    if current_exp:
        structured['work_experience'].append(current_exp)
    
    # Extract skills
    tech_skills = ['python', 'java', 'javascript', 'sql', 'aws', 'docker', 'react', 'nodejs', 'git', 'c++', 'ros', 'ros2']
    for skill in tech_skills:
        if skill.lower() in text.lower():
            structured['skills']['technical'].append(skill)
    
    # Extract projects
    project_keywords = ['project', 'developed', 'built', 'created', 'implemented']
    for line in lines:
        if any(keyword in line.lower() for keyword in project_keywords):
            if len(line) > 10 and len(line) < 100:
                structured['projects'].append({
                    'name': line,
                    'description': line
                })
    
    # Extract education
    edu_keywords = ['university', 'college', 'bachelor', 'master', 'phd', 'btech', 'mtech']
    for line in lines:
        if any(keyword in line.lower() for keyword in edu_keywords):
            structured['education'].append({
                'institution': line,
                'degree': 'Degree'
            })
    
    return structured

def _generate_clarification_questions_mock(prompt: str) -> str:
    """Mock for generating follow-up questions tailored to specific projects/experiences"""
    questions = []
    
    # Simple extraction logic
    projects = []
    try:
        if "Projects:" in prompt and "Experience:" in prompt:
            proj_str = prompt.split("Projects:")[1].split("Experience:")[0].strip()
            for item in proj_str.split(","):
                clean_item = item.replace("[", "").replace("]", "").replace("'", "").replace('"', "").strip()
                if clean_item and len(clean_item) > 2 and clean_item.lower() != 'none':
                    projects.append(clean_item)
    except Exception:
        pass
                
    if projects:
        # Prevent duplicate project questions by using a set
        seen_projects = set()
        for idx, p in enumerate(projects):
            if p in seen_projects: continue
            seen_projects.add(p)
            questions.append({
                "id": f"q_proj_tech_{idx}", 
                "question": f"What specific technologies or programming languages did you use for '{p}'?"
            })
            if len(seen_projects) >= 2:
                break
                
        if len(seen_projects) == 1:
            questions.append({
                "id": "q_proj_impact_0",
                "question": f"What was your most significant contribution to the '{projects[0]}' project and its final outcome?"
            })
    
    # Generic questions to fill the remaining slots
    if len(questions) < 4:
        questions.append({"id": "q_team", "question": "What was the size of your team for these projects, and what was your specific leadership or technical role?"})
    if len(questions) < 4:
        questions.append({"id": "q_metrics", "question": "Can you provide numerical metrics to quantify the impact of your work? (e.g. 20% speed increase or 50% fewer errors)"})
    if len(questions) < 4:
        questions.append({"id": "q_tools", "question": "Are there any additional specialized tools, cloud platforms, or databases (e.g. Gazebo, ROS, Flask) involved in your recent experience that you haven't detailed?"})
            
    return json.dumps(questions[:4])

def generate_clarification_questions(resume_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """AI analyzes the resume and asks for missing details to improve optimization"""
    prompt = f"""
    Analyze this resume data and generate exactly 4 highly specific follow-up questions to fill in missing gaps or clarify vague sentences in the candidate's history.
    
    RULES:
    1. DO NOT ask generic questions like "What was the team size?" or "What programming language was used?".
    2. Read the specific projects and experience provided. Find sentences that lack metrics, business impact, or technical depth.
    3. Ask a direct question about that specific sentence or project. Example: "In your 'E-commerce Platform' project, you mentioned 'improved speed'. What specific database or caching technology did you use, and what was the percentage decrease in load time?"
    4. FOR THE 4TH QUESTION: Specifically ask if there are any additional technical skills, specialized tools, or frameworks (e.g. Gazebo, ROS, Flask, etc.) that the candidate possesses but are not currently clearly detailed in the resume. This question should encourage the candidate to list tools they used in their projects.
    5. If the resume is completely empty, ask 4 questions about their education and personal technical projects.

    Projects: {resume_data.get('projects', [])}
    Experience: {resume_data.get('work_experience', [])}
    Summary: {resume_data.get('summary', '')}
    
    JSON OUTPUT ONLY: [{{"id": "q1", "question": "..."}}, {{"id": "q2", "question": "..."}}, {{"id": "q3", "question": "..."}}, {{"id": "q4", "question": "..."}}]
    CLARIFICATION QUESTIONS:
    """
    try:
        response = call_llm(prompt)
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            return json.loads(response)
        except:
            return json.loads(_generate_clarification_questions_mock(prompt))
    except:
        return json.loads(_generate_clarification_questions_mock(prompt))
def _generate_market_insights_mock(prompt: str) -> str:
    """High-fidelity mock for market insights when AI is unavailable"""
    import re
    
    # Try to detect role
    role = "Software Engineer"
    role_match = re.search(r"job role: (.*?)\n", prompt, re.IGNORECASE)
    if role_match:
        role = role_match.group(1).strip()
        
    # Standard realistic data for common roles in India
    data = {
        "Full Stack Developer": {
            "demand": "High", "competition": "Medium", "trend": "Rising", "barrier": "Medium",
            "salaries": {"entry": "5-9 LPA", "mid": "15-22 LPA", "top": "35+ LPA"},
            "industries": ["eCommerce", "FinTech", "SaaS", "Startups"],
            "companies": ["Flipkart", "Razorpay", "Zomato", "Freshworks"],
            "insight": "Massive demand for MERN/Java Full Stack in Indian startups. Scaling and cloud experience doubles hiring chances."
        },
        "Frontend Developer": {
            "demand": "High", "competition": "Medium", "trend": "Rising", "barrier": "Low",
            "salaries": {"entry": "4-8 LPA", "mid": "12-18 LPA", "top": "30+ LPA"},
            "industries": ["Product Companies", "EdTech", "MarTech"],
            "companies": ["Swiggy", "Unacademy", "Paytm", "Nykaa"],
            "insight": "React and Next.js are the gold standard. Performance optimization and UI/UX sensitivity are key differentiators."
        },
        "Backend Developer": {
            "demand": "High", "competition": "High", "trend": "Stable", "barrier": "Medium",
            "salaries": {"entry": "5-10 LPA", "mid": "18-28 LPA", "top": "45+ LPA"},
            "industries": ["Banking", "Cloud Platforms", "Logistics"],
            "companies": ["PhonePe", "Amazon India", "Microsoft India", "HDFC Bank"],
            "insight": "Strong focus on microservices and system design. Go and Java (Spring Boot) skills are commanding premium salaries."
        }
    }
    
    # Match detected role or provide generic
    result = data.get(role, {
        "demand": "High", "competition": "Medium", "trend": "Rising", "barrier": "Medium",
        "salaries": {"entry": "4-7 LPA", "mid": "10-16 LPA", "top": "25+ LPA"},
        "industries": ["IT Services", "Tech Consulting"],
        "companies": ["TCS", "Infosys", "Wipro", "Accenture"],
        "insight": f"Growing demand for {role} in India's digital transformation. Upskilling in cloud and AI will provide a sustainable edge."
    })
    
    return json.dumps(result, indent=2)
