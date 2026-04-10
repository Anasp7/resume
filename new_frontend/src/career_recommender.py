import json
import re
import os
import copy
from src.llm_client import call_llm

# Load Rule-Based Skills Database
SKILLS_DB_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'skills_database.json')
try:
    with open(SKILLS_DB_FILE, 'r', encoding='utf-8') as f:
        SKILLS_DATABASE = json.load(f)
except Exception:
    SKILLS_DATABASE = {}

# Load Curated Roadmaps (Legacy for specific roles)
ROADMAPS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'roadmaps.json')
try:
    with open(ROADMAPS_FILE, 'r', encoding='utf-8') as f:
        ROADMAP_DATA = json.load(f)
except Exception:
    ROADMAP_DATA = {}

IGNORE_TOOLS = [
    "VS Code", "Postman", "GitHub", "Chrome", "Slack", "Figma", "Canva", 
    "Trello", "Jira", "Discord", "Zoom", "Microsoft Teams"
]

# UNIVERSAL Multi-disciplinary skill database
VALID_SKILLS = [
    # --- SOFTWARE & IT ---
    "Python", "Java", "C", "C++", "C#", "JavaScript", "TypeScript", "Go", "Rust", "PHP", "Ruby", "Swift", "Kotlin", "Scala", "Dart", "Solidity", "SQL", "MATLAB", "R", "Perl", "Shell", "PowerShell", "Assembly",
    "React", "Node.js", "Express", "Angular", "Vue", "Next.js", "Django", "Flask", "Spring Boot", "Laravel", "FastAPI", "Flutter", "React Native", "Tailwind", "Bootstrap", "Web3", "GraphQL",
    "MongoDB", "PostgreSQL", "MySQL", "Redis", "Firebase", "Oracle", "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Linux", "Git", "Terraform", "Ansible", "Jenkins", "CI/CD", "Nginx", "Apache",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "TensorFlow", "PyTorch", "OpenAI", "LangChain", "LLMs", "Data Analytics", "Big Data", "Spark", "Hadoop", "Kafka",
    
    # --- CORE ENGINEERING (Mechanical, Civil, Aero, Pharma, Chemical) ---
    "AutoCAD", "SolidWorks", "CATIA", "ANSYS", "Creo", "Fusion 360", "Revit", "Staad Pro", "Thermodynamics", "Manufacturing", "Robotics", "HVAC", "Structural Design", "Surveying", "Construction Management", "Mechatronics", "Lotus Shark", "CNC Programming", "Materials Science",
    "Process Simulation", "Aspen Plus", "Chemical Engineering", "Biotechnology", "Bioinformatics", "Pharmaceutical Science", "Clinical Research", "Quality Control",
    
    # --- ELECTRICAL & ELECTRONICS ---
    "LabVIEW", "PCB Design", "VLSI", "VHDL", "Verilog", "Embedded Systems", "Microcontrollers", "Arduino", "Raspberry Pi", "Power Systems", "Circuit Design", "FPGA", "Analog Electronics", "Digital Signal Processing", "IoT",
    
    # --- BUSINESS, MANAGEMENT & FINANCE ---
    "Project Management", "Agile", "Scrum", "CRM", "ERP", "Financial Analysis", "Supply Chain", "Operations", "Product Management", "Business Analysis", "Lean Six Sigma", "PMP", "MBA", "Strategy",
    "Salesforce", "SAP", "Excel", "Tally", "QuickBooks", "Power BI", "Tableau", "Corporate Finance", "Investment Banking", "Accounting", "Auditing", "Taxation", "Portfolio Management", "Risk Management",
    
    # --- MARKETING & CREATIVE ---
    "Digital Marketing", "SEO", "SEM", "Content Writing", "Copywriting", "Branding", "Social Media Management", "Google Analytics", "Email Marketing", "Influencer Marketing", "Market Research",
    "Photoshop", "Illustrator", "InDesign", "Figma", "Adobe XD", "After Effects", "Premiere Pro", "Maya", "Blender", "3D Modeling", "Video Editing", "Graphic Design", "UI/UX", "User Research", "Prototyping",
    
    # --- HEALTHCARE, LAW & OTHERS ---
    "Human Resources", "Recruitment", "Talent Acquisition", "Employee Engagement", "Payroll", "Labor Laws", "Performance Management", "Training & Development",
    "Legal Research", "Corporate Law", "Intellectual Property", "Litigation", "Compliance", "Contract Drafting",
    "Clinical Research", "Healthcare Administration", "Public Health", "Epidemiology", "Medical Coding", "Pharmacovigilance",
    "Industrial Engineering", "Quality Assurance", "Six Sigma", "Supply Chain Management", "Logistics"
]

ROLE_MAP = {
    # --- Information Technology & Software ---
    "Full Stack Developer": ["React", "Node.js", "JavaScript", "TypeScript", "MongoDB", "Express", "Next.js"],
    "Frontend Developer": ["React", "Angular", "Vue", "JavaScript", "HTML", "CSS", "Tailwind"],
    "Backend Developer": ["Node.js", "Java", "Python", "SQL", "FastAPI", "Spring Boot", "Go", "MySQL"],
    "Python Developer": ["Python", "Flask", "Django", "FastAPI", "Automation", "Scripts"],
    "Data Analyst": ["SQL", "Python", "Pandas", "Tableau", "Power BI", "Excel", "Statistics"],
    "Data Engineer": ["Python", "SQL", "Big Data", "Spark", "ETL", "Kafka", "Data Pipelines"],
    "AI/ML Engineer": ["Python", "TensorFlow", "PyTorch", "Machine Learning", "NLP", "Deep Learning"],
    "Data Scientist": ["Python", "Pandas", "NumPy", "Scikit-Learn", "SQL", "Tableau", "Power BI", "Statistics"],
    "DevOps Engineer": ["Docker", "Kubernetes", "AWS", "Jenkins", "Terraform", "Git", "Linux", "CI/CD"],
    "Cloud Engineer": ["AWS", "Azure", "GCP", "Cloud Computing", "Serverless", "Security"],
    "Software Engineer": ["Data Structures", "Algorithms", "System Design", "Java", "Python", "C++"],
    "Mobile Developer": ["React Native", "Flutter", "Swift", "Kotlin", "Android", "iOS"],
    "Cybersecurity Analyst": ["Network Security", "Ethical Hacking", "Python", "Security Auditing", "SIEM"],
    "Blockchain Developer": ["Solidity", "Ethereum", "Smart Contracts", "Web3", "Cryptography"],
    "Game Developer": ["Unity", "C#", "C++", "Unreal Engine", "Game Design", "3D Graphics"],
    
    # --- Core Engineering ---
    "Mechanical Engineer": ["SolidWorks", "Creo", "AutoCAD", "Thermodynamics", "Manufacturing", "Robotics"],
    "Mechanical Design Engineer": ["SolidWorks", "Creo", "AutoCAD", "Fusion 360", "Drafting", "Product Design"],
    "CAE / FEA Analyst": ["ANSYS", "Lotus Shark", "Finite Element Analysis", "Simulation", "Aero Design"],
    "Electrical Engineer": ["MATLAB", "Circuit Design", "PCB Design", "Power Systems", "LabVIEW", "Electronics"],
    "Electronics & Communication Engineer": ["Embedded Systems", "VLSI", "Microcontrollers", "Arduino", "VHDL", "Signals"],
    "Civil Engineer": ["AutoCAD", "Staad Pro", "Revit", "Structural Analysis", "Surveying", "Construction Planning"],
    "Chemical Engineer": ["Process Simulation", "Aspen Plus", "Chemical Engineering", "Mass Transfer", "Thermodynamics"],
    "Bioinformatics Scientist": ["Python", "R", "Bioinformatics", "Genomics", "Sequence Analysis"],
    
    # --- Business & Management ---
    "Business Analyst": ["Business Analysis", "Excel", "SQL", "Tableau", "Power BI", "Data Visualization", "CRM"],
    "Product Manager": ["Product Management", "Agile", "Scrum", "Market Research", "Product Roadmap", "User Stories"],
    "Project Manager": ["Project Management", "Agile", "Resource Planning", "Risk Management", "Stakeholder Management"],
    "Supply Chain Manager": ["Supply Chain", "Inventory Management", "Logistics", "Procurement", "Operations"],
    "HR Specialist": ["Recruitment", "Talent Acquisition", "Human Resources", "Employee Engagement", "Labor Laws", "Payroll"],
    "Financial Analyst": ["Financial Analysis", "Excel", "Accounting", "Corporate Finance", "Investment Banking"],
    "Market Research Analyst": ["Market Research", "Data Analytics", "Excel", "Consumer Behavior", "Tableau"],
    "Digital Marketing Manager": ["SEO", "SEM", "Content Strategy", "Google Analytics", "Social Media Marketing"],
    "Sales Strategy Manager": ["Sales", "Negotiation", "CRM", "Business Development", "Lead Generation"],
    
    # --- Creative & Arts ---
    "Graphic Designer": ["Photoshop", "Illustrator", "InDesign", "Branding", "Typography", "Visual Arts"],
    "UI/UX Designer": ["Figma", "Adobe XD", "User Research", "Wireframing", "Prototyping", "UI Design"],
    "Video Editor / Motion Artist": ["Premiere Pro", "After Effects", "Final Cut Pro", "Video Production", "Motion Graphics"],
    "Content Strategist": ["Content Writing", "Copywriting", "Copy Editing", "Storytelling", "SEO Writing"],
    "3D Artist / Animator": ["Blender", "Maya", "3D Modeling", "Character Design", "Texturing"],
    
    # --- Legal & Healthcare ---
    "Corporate Lawyer": ["Corporate Law", "Contract Drafting", "Compliance", "Legal Research", "Due Diligence"],
    "IP Specialist": ["Intellectual Property", "Patents", "Trademarks", "Copyright Law", "Legal Analysis"],
    "Clinical Research Associate": ["Clinical Research", "GCP", "Pharmacovigilance", "Clinical Trials", "Data Management"],
    "Healthcare Administrator": ["Healthcare Administration", "Operations", "Public Health", "Regulatory Compliance"],
    
    # --- Specialized Engineering ---
    "Industrial Engineer": ["Operations Research", "Supply Chain", "Six Sigma", "Lean Manufacturing", "Process Improvement"],
    "Quality Engineer": ["Quality Control", "Six Sigma", "Statistical Analysis", "ISO Standards", "Auditing"]
}

def filter_skills(skills):
    """Filter skills to keep only valid tech skills and remove generic tools"""
    if not skills:
        return []
        
    filtered = []
    for skill in skills:
        if any(tool.lower() in skill.lower() for tool in IGNORE_TOOLS):
            continue
            
        if any(valid.lower() == skill.lower() or valid.lower() in skill.lower() for valid in VALID_SKILLS):
            filtered.append(skill)
            
    return sorted(list(set(filtered)))

def detect_all_suitable_roles(resume_data, raw_resume_text=""):
    """
    Enhanced Role Detection: Uses AI as the primary engine to detect suitable roles 
    and career level based on strict domain validation and skill matching rules.
    """
    try:
        # Priority: AI Detection (Dynamic & Rule-based as per USER_REQUEST)
        roles, level, analysis = detect_roles_ai(resume_data, raw_resume_text)
        if roles and len(roles) > 0:
            return roles, level, analysis
    except Exception as e:
        print(f"AI Role Detection failed, falling back to local map: {str(e)}")
    
    # Fallback: Local rule-based matching if AI fails
    roles, level = detect_roles_local_fallback(resume_data, raw_resume_text)
    return roles, level, {}

def detect_roles_ai(resume_data, raw_resume_text=""):
    """Stage 1 AI: Use LLM to detect realistic career paths and level based on strict rules"""
    prompt_template = load_prompt_template("career_role_detection_prompt.txt")
    
    # Pre-filter skills for the prompt context
    raw_skills = resume_data.get("skills", [])
    filtered_skills = filter_skills(raw_skills)
    
    # Format inputs for LLM
    prompt = prompt_template.format(
        skills=", ".join(filtered_skills),
        projects=", ".join(resume_data.get("projects", [])),
        education=", ".join(resume_data.get("education", []) if isinstance(resume_data.get("education"), list) else [resume_data.get("education", "Engineering Student")]),
        experience=resume_data.get("experience", "Fresher")
    )

    print("\n--- AI Career Intelligence Analysis In Progress ---")
    
    response = call_llm(prompt)
    
    try:
        # Handle cases where LLM might include markdown fences
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            # Extract summary data
            summary = data.get("career_summary", {})
            level = summary.get("level", "Entry-Level")
            
            # Map new deep analysis fields
            analysis_data = {
                "career_domain": summary.get("primary_domain", "Engineering/Technology"),
                "readiness_score": summary.get("readiness_score", 50),
                "primary_skill_domains": summary.get("primary_skill_domains", []),
                "top_strengths": summary.get("top_strengths", []),
                "total_skills_detected": summary.get("total_skills", len(filtered_skills)),
                "profile_summary": f"Level: {level} | Core Skills Detected: {summary.get('total_skills', 0)} | Domains: {', '.join(summary.get('primary_skill_domains', []))}"
            }
            
            # Extract roles with detailed market analysis
            roles_raw = data.get("roles", [])
            valid_roles = []
            
            for r in roles_raw:
                supporting = r.get("matched_skills", [])
                missing = r.get("missing_skills", [])
                
                # Minimum 2 matched skills for a valid suggestion
                if len(supporting) >= 2:
                    valid_roles.append({
                        "role": r.get("role"),
                        "match": r.get("match_percentage", 0),
                        "key_skills": supporting,
                        "missing_skills": missing,
                        "recommended_courses": r.get("recommended_courses", []),
                        "market_analysis": r.get("market_analysis", {}),
                        "hiring_sectors": r.get("hiring_sectors", []),
                        "key_companies": r.get("key_companies", []),
                        "market_insight": r.get("market_insight", "High growth potential.")
                    })
            
            # Sort by match descending
            return sorted(valid_roles, key=lambda x: x["match"], reverse=True), level, analysis_data
            
        return [], "Entry-Level", {}
    except Exception as e:
        print(f"Error parsing AI role detection: {str(e)}")
        return [], "Entry-Level", {}

def detect_roles_local_fallback(resume_data, raw_resume_text=""):
    """Original local rule-based matching as a robust backup with basic level detection"""
    raw_skills = resume_data.get("skills", [])
    filtered = filter_skills(raw_skills)
    
    resume_skills_pool = set(s.lower().strip() for s in (filtered + raw_skills))
    resume_text_lower = raw_resume_text.lower()
    
    potential_roles = []
    
    # Basic level detection logic
    level = "Entry-Level"
    exp_text = str(resume_data.get("experience", "Fresher")).lower()
    
    # 5+ years or "Senior" title
    if "senior" in exp_text or "lead" in exp_text or (re.search(r'5\+\s*years|6\s*years|7\s*years|8\s*years', exp_text)):
        level = "Senior"
    # "Industry" or "Professional" keywords or multiple companies
    elif any(kw in exp_text for kw in ["professional", "industry", "experience", "worked"]):
        level = "Intermediate"
    # Freshers with some projects
    elif len(resume_data.get("projects", [])) >= 2:
        level = "Junior"

    for role, role_skills in ROLE_MAP.items():
        detected_list = []
        missing_list = []
        
        for rs in role_skills:
            rs_lower = rs.lower().strip()
            matched_in_pool = any(rs_lower == us or (len(rs_lower) > 2 and rs_lower in us) for us in resume_skills_pool)
            pattern = re.compile(r'\b' + re.escape(rs_lower) + r'\b', re.IGNORECASE)
            matched_in_text = bool(pattern.search(resume_text_lower))
            
            if matched_in_pool or matched_in_text:
                detected_list.append(rs)
            else:
                missing_list.append(rs)
        
        total_required = len(role_skills)
        match_count = len(detected_list)
        
        # STRICT RULE: ONLY suggest if at least 2 relevant supporting skills are present
        if match_count >= 2:
            score = int((match_count / total_required) * 100) if total_required > 0 else 0
            potential_roles.append({
                "role": role,
                "match": score,
                "key_skills": detected_list,
                "missing_skills": missing_list
            })
            
    # Fallback to general Software Engineer only if matching skills exist
    if not potential_roles and len(filtered) >= 2:
        potential_roles.append({
            "role": "Software Engineer",
            "match": 30,
            "key_skills": list(filtered)[:3],
            "missing_skills": ["System Design", "Cloud / DevOps"]
        })
        
    return sorted(potential_roles, key=lambda x: x["match"], reverse=True), level

def generate_detailed_growth_plan(resume_data, selected_roles, raw_resume_text="", career_level="Entry-Level"):
    """Stage 2: Generate deep intelligence for ANY career domain based on selected roles"""
    
    # Normalize selected_roles to Always be a list of dicts
    if isinstance(selected_roles, str):
        selected_roles = [{"role": selected_roles, "match": 80}]
    elif isinstance(selected_roles, list):
        normalized = []
        for r in selected_roles:
            if isinstance(r, str):
                normalized.append({"role": r, "match": 80})
            elif isinstance(r, dict):
                normalized.append(r)
        selected_roles = normalized

    prompt_template = load_prompt_template("career_growth_prompt.txt")
    
    # Pre-filter skills
    raw_skills = resume_data.get("skills", [])
    filtered_skills_list = filter_skills(raw_skills)
    resume_text_lower = raw_resume_text.lower()
    
    # Format inputs for LLM
    prompt = prompt_template.format(
        skills=", ".join(filtered_skills_list),
        projects=", ".join(resume_data.get("projects", [])),
        other=", ".join(resume_data.get("other", [])),
        education=", ".join(resume_data.get("education", []) if isinstance(resume_data.get("education"), list) else [resume_data.get("education", "Engineering Student")]),
        experience=resume_data.get("experience", "Fresher"),
        selected_roles_json=json.dumps(selected_roles, indent=2)
    )

    print(f"\n--- Generating Structured Multi-Domain Plan for {len(selected_roles)} Roles ---")
    
    try:
        response = call_llm(prompt)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response.strip()
            
        growth_plan = json.loads(json_str)
        
        # PRO-ACTIVE TRANSFER & ACCURACY SCRUB
        all_resume_skills_lower = set(s.lower().strip() for s in (filtered_skills_list + raw_skills))
        
        # Scrub roles in the new "roles" list
        if "roles" in growth_plan:
            for role in growth_plan["roles"]:
                role_name = role.get("role", "")
                
                # 1. Ensure courses are carried over or filled
                if not role.get("recommended_courses"):
                    # Level 1 match from Stage 1 selection
                    s_role = next((r for r in selected_roles if r.get("role") == role_name), None)
                    if s_role and s_role.get("recommended_courses"):
                        role["recommended_courses"] = s_role.get("recommended_courses")
                
                # 2. Scrub skill gaps (new structure)
                if "skill_gap" in growth_plan:
                    gap_entry = next((g for g in growth_plan["skill_gap"] if g.get("role") == role_name), None)
                    if gap_entry:
                        detected = gap_entry.get("detected_skills", [])
                        missing = gap_entry.get("missing_skills", [])
                        new_missing, new_detected = [], list(detected)
                        
                        for s in missing:
                            s_lower = s.lower().strip()
                            matched_in_pool = any(s_lower == rs or s_lower in rs or rs in s_lower for rs in all_resume_skills_lower)
                            pattern = re.compile(r'\b' + re.escape(s_lower) + r'\b', re.IGNORECASE)
                            matched_in_text = bool(pattern.search(resume_text_lower))
                            matched_synonym = (s_lower in ["javascript", "js"]) and bool(re.search(r'\bjs\b|\bjavascript\b', resume_text_lower))
                            
                            if matched_in_pool or matched_in_text or matched_synonym:
                                if s not in new_detected: new_detected.append(s)
                            else:
                                new_missing.append(s)
                        
                        gap_entry["detected_skills"] = new_detected
                        gap_entry["missing_skills"] = new_missing

        if "career_overview" not in growth_plan:
            growth_plan["career_overview"] = {}
        
        growth_plan["career_overview"]["experience_level"] = career_level
        growth_plan["career_overview"]["experience_level"] = career_level
        growth_plan["career_overview"]["total_skills_detected"] = len(filtered_skills_list)
        
        total_raw = len(raw_skills)
        valid_tech = len(filtered_skills_list)
        skill_strength = int((valid_tech / 12) * 80) if valid_tech > 0 else 0
        skill_breadth = min(20, total_raw)
        growth_plan["career_overview"]["skill_strength_score"] = min(100, skill_strength + skill_breadth)
        
        core_skills_detected = list(filtered_skills_list)
        supporting_skills_detected = [s for s in raw_skills if s not in core_skills_detected and not any(tool.lower() in s.lower() for tool in IGNORE_TOOLS)]
        growth_plan["career_overview"]["core_skills"] = core_skills_detected
        growth_plan["career_overview"]["supporting_skills"] = supporting_skills_detected
        
        if selected_roles:
            max_readiness = max([r.get("match", 0) for r in selected_roles] + [r.get("match_percentage", 0) for r in selected_roles] + [0])
        else:
            max_readiness = 0
            
        growth_plan["career_overview"]["career_readiness_score"] = min(max_readiness, 100)
        
        # APPLY DETERMINISTIC RULE-BASED LOGIC (for highest accuracy)
        # This overrides AI/Static fallback if role match exists in SKILLS_DATABASE
        user_skills_set = set(s.lower().strip() for s in (filtered_skills_list + raw_skills))
        
        if selected_roles and "career_roadmap" in growth_plan:
            primary_role_raw = selected_roles[0].get("role", "").lower()
            
            # 1. Check for Rule-Based Match in SKILLS_DATABASE (New Deterministic Engine)
            db_match = None
            for key in SKILLS_DATABASE:
                if key in primary_role_raw or primary_role_raw in key:
                    db_match = SKILLS_DATABASE[key]
                    break
            
            if db_match:
                print(f"Applying Deterministic Rule-Based Roadmap for: {primary_role_raw}")
                
                # Filter Missing Skills
                missing_must = [s for s in db_match["must_have"] if s.lower().strip() not in user_skills_set]
                missing_good = [s for s in db_match["good_to_have"] if s.lower().strip() not in user_skills_set]
                missing_advanced = [s for s in db_match["advanced"] if s.lower().strip() not in user_skills_set]
                
                # Construct Roadmap
                growth_plan["career_roadmap"] = {
                    "short_term": {
                        "focus_areas": missing_must or ["Apply Core Skills in Real Use Cases"],
                        "what_to_do": ["A concrete daily action plan", "Practice coding daily (30-60 mins)"],
                        "suggested_work": db_match["projects"]["short_term"],
                        "projects_count": str(len(db_match["projects"]["short_term"])),
                        "daily_practice": "60 mins coding",
                        "goal": "Build Strong Foundations / Internship Ready",
                        "outcomes": ["Milestones achieved in core tech"]
                    },
                    "medium_term": {
                        "focus_areas": missing_good or ["Deeper domain specialization"],
                        "what_to_do": ["Specific actions like mock interviews, networking"],
                        "suggested_work": db_match["projects"]["medium_term"],
                        "projects_count": str(len(db_match["projects"]["medium_term"])),
                        "daily_practice": "Professional level work",
                        "goal": "Job Ready / Advance Domain Proficiency",
                        "outcomes": ["Level-up certification/projects completed"]
                    },
                    "long_term": {
                        "focus_areas": missing_advanced or ["Strategic tech leadership"],
                        "what_to_do": ["Open-source contribution or niche specialization"],
                        "suggested_work": db_match["projects"]["long_term"],
                        "projects_count": str(len(db_match["projects"]["long_term"])),
                        "daily_practice": "Complex tasks (3-4 hours daily)",
                        "goal": "Senior Level Path / Industry Level Expert",
                        "outcomes": ["Ready for full-time high-impact roles"]
                    }
                }
            
            # 2. Fallback to Curated Roadmaps (roadmaps.json) if no DB match
            elif primary_role_raw in ROADMAP_DATA:
                print("Applying Static Curated Roadmap")
                growth_plan["career_roadmap"] = copy.deepcopy(ROADMAP_DATA[primary_role_raw])
                # Filter Curated (Incremental Subtraction)
                for phase in ['short_term', 'medium_term', 'long_term']:
                    if phase in growth_plan["career_roadmap"]:
                        p_data = growth_plan["career_roadmap"][phase]
                        for skill_key in ['focus_areas', 'skills', 'skills_to_learn', 'technologies_to_focus', 'technologies']:
                            if skill_key in p_data and isinstance(p_data[skill_key], list):
                                p_data[skill_key] = [s for s in p_data[skill_key] if s.lower().strip() not in user_skills_set]

        # 3. Last Fallback: Incremental Subtraction for AI-Generated Roadmaps
        if "career_roadmap" in growth_plan:
             for phase in ['short_term', 'medium_term', 'long_term']:
                 if phase in growth_plan["career_roadmap"]:
                     p_data = growth_plan["career_roadmap"][phase]
                     if not any(f in p_data for f in ['focus_areas', 'skills']): # Only if it's the AI structure
                         for skill_key in ['skills_to_learn', 'technologies_to_focus', 'technologies']:
                             if skill_key in p_data and isinstance(p_data[skill_key], list):
                                 p_data[skill_key] = [s for s in p_data[skill_key] if s.lower().strip() not in user_skills_set]

        return growth_plan
        
    except Exception as e:
        print(f"Error generating universal analysis: {str(e)}")
        # Flexible fallback
        if selected_roles:
            max_readiness = max([r.get("match", 0) for r in selected_roles] + [r.get("match_percentage", 0) for r in selected_roles] + [0])
        else:
            max_readiness = 0
            
        return {
            "career_domain": "Engineering/Technology",
            "career_overview": {
                "career_readiness_score": min(max_readiness, 100),
                "experience_level": career_level,
                "total_skills_detected": len(filtered_skills_list),
                "core_skills": list(filtered_skills_list),
                "supporting_skills": [s for s in resume_data.get("skills", []) if s not in list(filtered_skills_list) and not any(t.lower() in s.lower() for t in IGNORE_TOOLS)],
                "skill_strength_score": min(100, (int((len(filtered_skills_list) / 12) * 80) if len(filtered_skills_list) > 0 else 0) + min(20, len(resume_data.get("skills", [])))),
                "primary_skill_domains": ["Technical Development"],
                "top_strengths": ["Domain Expertise"],
                "improvement_areas": ["Advanced Specialization"]
            },
            "roles": [{
                "role": r["role"], 
                "match_percentage": r["match"], 
                "supporting_skills": r.get("key_skills", []), 
                "recommended_courses": r.get("recommended_courses") or (
                    [
                        {"title": "The Complete 2024 Web Development Bootcamp", "platform": "Udemy", "provider": "Angela Yu", "description": "Master full-stack tools from HTML to MongoDB.", "skills": ["Full Stack", "MERN"], "duration": "65 Hours", "difficulty": "Beginner", "url": "https://www.udemy.com/course/the-complete-web-development-bootcamp/"},
                        {"title": "Full-Stack Web Development with React", "platform": "Coursera", "provider": "HKUST", "description": "Professional certificate in front-end and hybrid mobile development.", "skills": ["React", "Bootstrap", "Node.js"], "duration": "4 Months", "difficulty": "Intermediate", "url": "https://www.coursera.org/specializations/full-stack-mobile-app-development"},
                        {"title": "IBM Full Stack Software Developer Professional Certificate", "platform": "Coursera", "provider": "IBM", "description": "Build and deploy cloud-native full stack applications.", "skills": ["Cloud-native", "Software Engineering"], "duration": "6 Months", "difficulty": "Beginner", "url": "https://www.coursera.org/professional-certificates/ibm-full-stack-cloud-developer"}
                    ] if "full stack" in r["role"].lower() else
                    [
                        {"title": "Meta Front-End Developer Professional Certificate", "platform": "Coursera", "provider": "Meta", "description": "Master React and modern UI/UX.", "skills": ["React", "JS"], "duration": "6 Months", "difficulty": "Beginner", "url": "https://www.coursera.org/professional-certificates/meta-front-end-developer"},
                        {"title": "The Complete JavaScript Course 2024: From Zero to Expert!", "platform": "Udemy", "provider": "Jonas Schmedtmann", "description": "Master JavaScript with projects, challenges and theory.", "skills": ["ES6+", "Async JS"], "duration": "68 Hours", "difficulty": "All Levels", "url": "https://www.udemy.com/course/the-complete-javascript-course/"},
                        {"title": "Modern React with Redux", "platform": "Udemy", "provider": "Stephen Grider", "description": "Master React and Redux with deep-dive hooks.", "skills": ["Redux", "Hooks"], "duration": "30 Hours", "difficulty": "Intermediate", "url": "https://www.udemy.com/course/react-redux/"}
                    ] if "frontend" in r["role"].lower() else
                    [
                        {"title": "Node.js, Express, MongoDB & More: The Complete Bootcamp", "platform": "Udemy", "provider": "Jonas Schmedtmann", "description": "Master backend development from scratch.", "skills": ["Node.js", "MongoDB"], "duration": "42 Hours", "difficulty": "Intermediate", "url": "https://www.udemy.com/course/nodejs-express-mongodb-bootcamp/"},
                        {"title": "Back-End Web Development Specialization", "platform": "Coursera", "provider": "Meta", "description": "Learn the server-side skills needed for professional web apps.", "skills": ["Python", "Django", "APIs"], "duration": "7 Months", "difficulty": "Beginner", "url": "https://www.coursera.org/professional-certificates/meta-back-end-developer"},
                        {"title": "Java Programming and Software Engineering Fundamentals", "platform": "Coursera", "provider": "Duke University", "description": "Learn core programming concepts for enterprise backend.", "skills": ["Java", "Data Structures"], "duration": "5 Months", "difficulty": "Beginner", "url": "https://www.coursera.org/specializations/java-programming"}
                    ] if "backend" in r["role"].lower() else
                    [
                        {"title": "Google Data Analytics Professional Certificate", "platform": "Coursera", "provider": "Google", "description": "Master data analysis foundations.", "skills": ["SQL", "Data Vis"], "duration": "6 Months", "difficulty": "Beginner", "url": "https://www.coursera.org/professional-certificates/google-data-analytics"},
                        {"title": "IBM Data Science Professional Certificate", "platform": "Coursera", "provider": "IBM", "description": "Master data science tools and methodologies.", "skills": ["Python", "Pandas"], "duration": "10 Months", "difficulty": "Beginner", "url": "https://www.coursera.org/professional-certificates/ibm-data-science"},
                        {"title": "Data Analyst Nanodegree", "platform": "Udacity", "provider": "Knowledge Partners", "description": "Prepare for a high-paying career in data analysis.", "skills": ["Wrangling", "ML"], "duration": "4 Months", "difficulty": "Intermediate", "url": "https://www.udacity.com/course/data-analyst-nanodegree--nd002"}
                    ] if any(k in r["role"].lower() for k in ["data", "analyst"]) else
                    [
                        {"title": "AWS Cloud Practitioner Essentials", "platform": "edX", "provider": "AWS", "description": "Cloud fundamentals and architecture.", "skills": ["Cloud", "AWS"], "duration": "4 Weeks", "difficulty": "Beginner", "url": "https://www.edx.org/course/aws-cloud-practitioner-essentials"},
                        {"title": "DevOps on AWS Specialization", "platform": "Coursera", "provider": "AWS", "description": "Master DevOps processes on the AWS cloud.", "skills": ["CI/CD", "Docker"], "duration": "3 Months", "difficulty": "Intermediate", "url": "https://www.coursera.org/specializations/aws-devops"},
                        {"title": "Docker and Kubernetes: The Complete Guide", "platform": "Udemy", "provider": "Stephen Grider", "description": "Master orchestration with Docker and K8s.", "skills": ["Kubernetes", "Docker"], "duration": "22 Hours", "difficulty": "All Levels", "url": "https://www.udemy.com/course/docker-and-kubernetes-the-complete-guide/"}
                    ] if any(k in r["role"].lower() for k in ["devops", "cloud", "aws", "azure", "docker"]) else
                    [
                        {"title": "Machine Learning by Stanford University", "platform": "Coursera", "provider": "Stanford", "description": "Master fundamental AI concepts.", "skills": ["ML", "AI"], "duration": "2 Months", "difficulty": "Intermediate", "url": "https://www.coursera.org/learn/machine-learning"},
                        {"title": "Deep Learning Specialization", "platform": "Coursera", "provider": "DeepLearning.AI", "description": "Become a machine learning expert.", "skills": ["Neural Networks", "NLP"], "duration": "4 Months", "difficulty": "Advanced", "url": "https://www.coursera.org/specializations/deep-learning"},
                        {"title": "AI For Everyone", "platform": "Coursera", "provider": "Andrew Ng", "description": "Non-technical introduction to AI strategy.", "skills": ["AI Strategy", "Concepts"], "duration": "4 Weeks", "difficulty": "Beginner", "url": "https://www.coursera.org/learn/ai-for-everyone"}
                    ] if any(k in r["role"].lower() for k in ["ml", "machine", "intelligence", "ai"]) else
                    [
                        {"title": "Meta Front-End Developer Professional Certificate", "platform": "Coursera", "provider": "Meta", "description": "Verified high-impact certification for modern developers.", "skills": ["React", "UI/UX"], "duration": "6 Months", "difficulty": "Beginner", "url": "https://www.coursera.org/professional-certificates/meta-front-end-developer"},
                        {"title": "Google Project Management Professional Certificate", "platform": "Coursera", "provider": "Google", "description": "Foundations of modern project management.", "skills": ["Agile", "Scrum"], "duration": "6 Months", "difficulty": "Beginner", "url": "https://www.coursera.org/professional-certificates/google-project-management"},
                        {"title": "Communication Skills for Career Success", "platform": "edX", "provider": "Fullbridge", "description": "Master professional soft skills.", "skills": ["Communication", "Soft Skills"], "duration": "4 Weeks", "difficulty": "Beginner", "url": "https://www.edx.org/course/communication-skills-career-fullbridge-fullbridge2-1x"}
                    ]
                )
            } for r in selected_roles],
            "market_analysis": [{"role": r["role"], "demand_level": "High", "competition_level": "Medium", "success_probability": r["match"], "hiring_trend": "🔼 Rising", "entry_barrier": "Medium", "market_insight": "AI processing market depth...", "supporting_skills": r.get("key_skills", [])} for r in selected_roles],
            "job_details": [{"role": r["role"], "entry_salary_inr": "₹4-8 LPA", "mid_salary_inr": "₹12-20 LPA", "top_salary_inr": "₹30+ LPA"} for r in selected_roles],
            "skill_gap": [{"role": r["role"], "detected_skills": r.get("key_skills", []), "missing_skills": r.get("missing_skills", [])} for r in selected_roles],
            "career_roadmap": {
                "short_term": {
                    "focus_areas": ["JavaScript, Git, APIs"], 
                    "what_to_do": ["Daily Practice: 1 hour coding"], 
                    "suggested_work": ["Build 2 projects", "Deploy Portfolio + GitHub Projects"],
                    "projects_count": "2-3",
                    "goal": "Internship Ready",
                    "outcomes": ["Clear understanding of basics"]
                },
                "medium_term": {
                    "focus_areas": ["Advanced React", "Real-world development"], 
                    "what_to_do": ["Solve intermediate coding problems", "Start mock interviews"], 
                    "suggested_work": ["Full-stack application", "Dashboard UI"],
                    "projects_count": "2 major projects",
                    "goal": "Job Ready",
                    "outcomes": ["Strong project portfolio"]
                },
                "long_term": {
                    "focus_areas": ["Cloud basics", "System design"], 
                    "what_to_do": ["Contribute to open source", "Participate in contests"], 
                    "suggested_work": ["Scalable apps", "Complex SaaS UI"],
                    "projects_count": "3+ projects",
                    "goal": "Industry Expert",
                    "outcomes": ["Ready for full-time roles"]
                }
            }
        }

def load_prompt_template(filename: str):
    with open(f"prompts/{filename}", "r", encoding="utf-8") as f:
        return f.read()

def generate_career_guidance(resume_data: dict) -> str:
    prompt_template = load_prompt_template("career_resume_prompt.txt")
    raw_skills = resume_data.get("skills", [])
    filtered = filter_skills(raw_skills)

    prompt = prompt_template.format(
        skills=", ".join(filtered),
        projects=", ".join(resume_data.get("projects", [])),
        other=", ".join(resume_data.get("other", [])),
        education=", ".join(resume_data.get("education", []) if isinstance(resume_data.get("education"), list) else [resume_data.get("education", "Engineering Student")]),
        experience=resume_data.get("experience", "Fresher")
    )
    return call_llm(prompt)

def generate_growth_plan(resume_data: dict, target_job: str = "") -> dict:
    """Legacy wrapper for single-step calls if still needed elsewhere"""
    detected, level, analysis = detect_all_suitable_roles(resume_data, "")
    selected = detected[:5] # Default to top 5
    return generate_detailed_growth_plan(resume_data, selected, "", level)






