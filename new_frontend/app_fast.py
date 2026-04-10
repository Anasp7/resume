"""
Fast Frontend Dashboard - Optimized for Speed
Lightweight version with minimal processing for quick responses
"""
from flask import Flask, render_template, request, jsonify, send_file
import sys
from pathlib import Path
import json
from datetime import datetime
import os
import re
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'src'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Fast resume parser (minimal processing)
def fast_parse_resume(text: str) -> Dict[str, Any]:
    """Fast resume parsing with minimal processing"""
    
    # Extract personal info
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    linkedin_match = re.search(r'linkedin\.com/in/[\w-]+', text)
    
    # Extract skills (fast keyword matching)
    tech_skills = ['python', 'java', 'javascript', 'sql', 'aws', 'docker', 'react', 'nodejs', 'git', 'c++', 'ros', 'ros2', 'machine learning', 'data science']
    found_skills = [skill for skill in tech_skills if skill.lower() in text.lower()]
    
    # Extract experience (simple pattern matching)
    experience_keywords = ['engineer', 'developer', 'manager', 'analyst', 'director', 'lead']
    has_experience = any(keyword in text.lower() for keyword in experience_keywords)
    
    # Extract education
    edu_keywords = ['university', 'college', 'bachelor', 'master', 'phd', 'btech', 'mtech']
    has_education = any(keyword in text.lower() for keyword in edu_keywords)
    
    return {
        'personal_info': {
            'email': email_match.group() if email_match else '',
            'phone': phone_match.group() if phone_match else '',
            'linkedin': linkedin_match.group() if linkedin_match else ''
        },
        'skills': {'technical': found_skills},
        'has_experience': has_experience,
        'has_education': has_education,
        'text_length': len(text),
        'word_count': len(text.split())
    }

# Fast career guidance (rule-based)
def fast_career_guidance(resume_data: Dict[str, Any]) -> str:
    """Fast career guidance using simple rules"""
    
    skills = resume_data.get('skills', {}).get('technical', [])
    has_experience = resume_data.get('has_experience', False)
    
    # Generate career recommendations
    career_roles = []
    
    if 'python' in skills or 'sql' in skills:
        career_roles.append("Data Analyst: Strong programming and database skills")
    
    if 'java' in skills or 'javascript' in skills:
        career_roles.append("Software Developer: Programming and web development skills")
    
    if 'aws' in skills or 'docker' in skills:
        career_roles.append("DevOps Engineer: Cloud and containerization skills")
    
    if 'machine learning' in skills or 'data science' in skills:
        career_roles.append("ML Engineer: Machine learning and data science expertise")
    
    if not career_roles:
        career_roles.append("Entry Level Professional: Foundational skills for growth")
    
    # Generate missing skills
    missing_skills = []
    if 'python' in skills and 'machine learning' not in skills:
        missing_skills.append("Machine Learning: Enhances data analysis capabilities")
    
    if 'sql' in skills and 'data visualization' not in skills:
        missing_skills.append("Data Visualization: Important for data analysis roles")
    
    if 'java' in skills and 'spring' not in skills:
        missing_skills.append("Spring Framework: Essential for enterprise Java development")
    
    # Generate improvements
    improvements = [
        "Quantify achievements: Add metrics and numbers to experience",
        "Add technical summary: Include key skills at the top",
        "Project details: Expand on project descriptions and impact"
    ]
    
    # Format response
    response = "CAREER ROLES:\n"
    for role in career_roles:
        response += f"- {role}\n"
    
    response += "\nMISSING SKILLS:\n"
    for skill in missing_skills:
        response += f"- {skill}\n"
    
    response += "\nRESUME IMPROVEMENTS:\n"
    for improvement in improvements:
        response += f"- {improvement}\n"
    
    response += "\nATS OPTIMIZATION:\n"
    response += "- Keywords: Include more industry-specific terms\n"
    response += "- Format: Use clean, simple formatting\n"
    response += "- Length: Keep resume to 1-2 pages\n"
    
    response += "\nCERTIFICATIONS/PROJECTS:\n"
    if 'python' in skills:
        response += "- Python Certification: Validates programming expertise\n"
    if 'aws' in skills:
        response += "- AWS Certification: Cloud computing expertise\n"
    response += "- Portfolio project: Build a complete application\n"
    
    return response

# Fast resume templates
def fast_resume_template(resume_data: Dict[str, Any], target_role: str = "") -> str:
    """Fast resume template formatting"""
    
    personal_info = resume_data.get('personal_info', {})
    skills = resume_data.get('skills', {}).get('technical', [])
    
    name = extract_name_from_text(resume_data.get('original_text', ''))
    email = personal_info.get('email', 'your.email@example.com')
    phone = personal_info.get('phone', '(555) 123-4567')
    
    # Generate professional summary
    exp_level = "Experienced" if resume_data.get('has_experience', False) else "Entry-level"
    summary = f"{exp_level} professional with expertise in {', '.join(skills[:5]) if skills else 'various technologies'}. "
    summary += f"Seeking {target_role or 'professional'} opportunities to apply technical skills and drive results."
    
    resume = f"""
╔══════════════════════════════════════════════════════════════╗
║                    {name.upper():^57} ║
╚══════════════════════════════════════════════════════════════╝

📧 {email} | 📱 {phone} | 💼 LinkedIn: linkedin.com/in/yourprofile

═══════════════════════════════════════════════════════════════

📋 PROFESSIONAL SUMMARY
{summary}

═══════════════════════════════════════════════════════════════

🛠️ TECHNICAL SKILLS
• Programming: {', '.join(skills[:8]) if skills else 'Python, JavaScript, SQL'}
• Tools: Git, Docker, VS Code
• Technologies: Web Development, Cloud Computing

═══════════════════════════════════════════════════════════════

💼 EXPERIENCE
• Professional experience with focus on {', '.join(skills[:3]) if skills else 'software development'}
• Collaborated on cross-functional teams
• Delivered high-quality solutions and met project deadlines

═══════════════════════════════════════════════════════════════

🎓 EDUCATION
• Bachelor's degree in Computer Science or related field
• Relevant coursework and technical training
• Continuous learning and professional development

═══════════════════════════════════════════════════════════════

🚀 PROJECTS
• Technical projects utilizing {', '.join(skills[:3]) if skills else 'modern technologies'}
• Demonstrated problem-solving and coding abilities
• Portfolio of completed applications and solutions
"""
    
    return resume.strip()

def extract_name_from_text(text: str) -> str:
    """Extract name from text (simple heuristic)"""
    lines = text.split('\n')
    for line in lines[:5]:
        line = line.strip()
        if line and len(line) < 50 and not any(char.isdigit() for char in line):
            if '@' not in line and 'http' not in line and not line.lower().startswith(('summary', 'objective', 'experience')):
                return line
    return "YOUR NAME"

@app.route('/')
def dashboard():
    """Main dashboard with two paths"""
    return render_template('dashboard.html')

@app.route('/resume-optimization')
def resume_optimization():
    """Resume optimization page"""
    return render_template('resume_optimization.html')

@app.route('/career-recommendation')
def career_recommendation():
    """Career recommendation page"""
    return render_template('career_recommendation.html')

@app.route('/career-growth-plan')
def career_growth_plan():
    """Career growth plan page"""
    return render_template('career_growth_plan.html')

@app.route('/test-upload')
def test_upload():
    """Test upload page"""
    return send_file('test_upload.html')

@app.route('/debug-api')
def debug_api():
    """Debug API page"""
    return send_file('debug_api.html')

@app.route('/api/upload-resume', methods=['POST'])
def upload_resume():
    """Handle resume file upload - SAVE ONLY, NO ANALYSIS"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file
    filename = f"resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Get file size for validation
        file_size = os.path.getsize(filepath)
        
        # Return success immediately - NO ANALYSIS
        return jsonify({
            'success': True,
            'filename': filename,
            'file_size': file_size,
            'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'message': 'Resume uploaded successfully. Please select a role to analyze.'
        })
    except Exception as e:
        return jsonify({'error': f'Error saving file: {str(e)}'}), 500

@app.route('/api/analyze-resume', methods=['POST'])
def analyze_resume():
    """Fast resume analysis - REQUIRES filename AND role"""
    data = request.get_json()
    filename = data.get('filename', '')
    target_role = data.get('target_role', '')
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    if not target_role:
        return jsonify({'error': 'Please select a target role for analysis'}), 400
    
    try:
        # Read the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Resume file not found'}), 404
        
        # Extract text from file
        if filename.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                resume_text = f.read()
        else:
            # For PDF/DOCX, read as binary and decode what we can
            with open(filepath, 'rb') as f:
                content = f.read()
                resume_text = content.decode('utf-8', errors='ignore')[:5000]  # Limit to 5000 chars
        
        if not resume_text.strip():
            return jsonify({'error': 'Resume file is empty or could not be read'}), 400
        
        # Fast parsing
        parsed_data = fast_parse_resume(resume_text)
        
        # Calculate scores (simple heuristic)
        ats_score = 50  # Base score
        if parsed_data['personal_info']['email']:
            ats_score += 20
        if parsed_data['personal_info']['phone']:
            ats_score += 20
        if parsed_data['personal_info']['linkedin']:
            ats_score += 10
        
        keyword_score = min(len(parsed_data['skills']['technical']) * 10, 100)
        formatting_score = 90 if parsed_data['text_length'] < 5000 else 70
        overall_score = (ats_score + keyword_score + formatting_score) / 3
        
        # Generate simple faults
        faults = []
        if not parsed_data['personal_info']['email']:
            faults.append({
                'category': 'content',
                'severity': 'medium',
                'description': 'Missing email address',
                'suggestion': 'Add a professional email address'
            })
        if not parsed_data['personal_info']['phone']:
            faults.append({
                'category': 'content',
                'severity': 'medium',
                'description': 'Missing phone number',
                'suggestion': 'Add a phone number for contact'
            })
        if len(parsed_data['skills']['technical']) < 3:
            faults.append({
                'category': 'keyword_optimization',
                'severity': 'high',
                'description': 'Insufficient technical keywords',
                'suggestion': 'Add more relevant technical skills'
            })
        
        return jsonify({
            'success': True,
            'analysis': {
                'structured_data': parsed_data,
                'scores': {
                    'ats_score': ats_score,
                    'keyword_score': keyword_score,
                    'formatting_score': formatting_score,
                    'overall_score': overall_score
                },
                'faults': faults,
                'target_role': target_role
            }
        })
    except Exception as e:
        return jsonify({'error': f'Error analyzing resume: {str(e)}'}), 500

@app.route('/api/optimize-resume', methods=['POST'])
def optimize_resume():
    """Fast resume optimization"""
    data = request.get_json()
    resume_text = data.get('resume_text', '')
    target_role = data.get('target_role', 'Professional')
    
    if not resume_text:
        return jsonify({'error': 'No resume text provided'}), 400
    
    try:
        # Fast parsing
        parsed_data = fast_parse_resume(resume_text)
        parsed_data['original_text'] = resume_text
        
        # Fast template formatting
        optimized_content = fast_resume_template(parsed_data, target_role)
        
        # Calculate improvement
        original_score = 60  # Assumed original score
        optimized_score = 85  # Assumed optimized score
        
        optimized_resume = {
            'target_role': target_role,
            'optimized_content': optimized_content,
            'optimization_score': optimized_score,
            'improvement': optimized_score - original_score,
            'generation_timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'optimization': {
                'optimized_resumes': {
                    'fast_optimized': optimized_resume
                }
            }
        })
    except Exception as e:
        return jsonify({'error': f'Error optimizing resume: {str(e)}'}), 500

@app.route('/api/career-guidance', methods=['POST'])
def get_career_guidance():
    """Fast career guidance"""
    data = request.get_json()
    resume_text = data.get('resume_text', '')
    
    if not resume_text:
        return jsonify({'error': 'No resume text provided'}), 400
    
    try:
        # Fast parsing
        parsed_data = fast_parse_resume(resume_text)
        
        # Fast career guidance
        career_guidance = fast_career_guidance(parsed_data)
        
        # Simple growth plan
        growth_plan = {
            'short_term': [
                f"Master {', '.join(parsed_data['skills']['technical'][:2]) if parsed_data['skills']['technical'] else 'technical skills'}",
                "Build 2-3 portfolio projects",
                "Update professional profiles"
            ],
            'medium_term': [
                "Apply for 50+ targeted positions",
                "Complete relevant certifications",
                "Network with industry professionals"
            ],
            'long_term': [
                "Achieve target role placement",
                "Mentor junior professionals",
                "Consider advanced specialization"
            ],
            'skill_roadmap': {
                skill: {
                    'current': 'Intermediate',
                    'target': 'Advanced',
                    'resources': ['Online courses', 'Practice projects', 'Documentation']
                }
                for skill in parsed_data['skills']['technical'][:3]
            },
            'certification_recommendations': [
                "AWS Certified Solutions Architect",
                "Google Cloud Professional",
                "Microsoft Azure Fundamentals"
            ],
            'project_suggestions': [
                "Build a web application",
                "Create a data analysis project",
                "Develop an automation tool"
            ]
        }
        
        return jsonify({
            'success': True,
            'career_guidance': career_guidance,
            'growth_plan': growth_plan,
            'analysis': {
                'structured_data': parsed_data,
                'scores': {
                    'overall_score': 75,
                    'ats_score': 80,
                    'keyword_score': 70,
                    'formatting_score': 80
                }
            }
        })
    except Exception as e:
        return jsonify({'error': f'Error generating career guidance: {str(e)}'}), 500

@app.route('/api/export-resume', methods=['POST'])
def export_resume():
    """Export optimized resume"""
    data = request.get_json()
    optimized_content = data.get('optimized_content', '')
    format_type = data.get('format', 'text')
    
    if not optimized_content:
        return jsonify({'error': 'No content to export'}), 400
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"optimized_resume_{timestamp}.{format_type}"
        filepath = os.path.join('outputs', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(optimized_content)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'file_path': filepath
        })
    except Exception as e:
        return jsonify({'error': f'Error exporting resume: {str(e)}'}), 500

@app.route('/api/download/<filename>')
def download_file(filename):
    """Download exported files"""
    try:
        file_path = os.path.join('outputs', filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error downloading file: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting FAST Resume Optimization Platform...")
    print("⚡ Optimized for speed with minimal processing")
    print("🌐 Server running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
