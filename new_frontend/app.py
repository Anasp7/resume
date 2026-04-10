"""
Frontend Dashboard for Resume Optimization Platform
Flask web application with two main paths:
1. Resume Optimization
2. Career Recommendation & Growth Plan
"""
from flask import Flask, render_template, request, jsonify, send_file
import sys
from pathlib import Path
import json
from datetime import datetime
import os
import copy
from typing import Dict, Any, List

# Load Roadmaps Data
ROADMAPS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'roadmaps.json')
try:
    with open(ROADMAPS_FILE, 'r', encoding='utf-8') as f:
        ROADMAP_DATA = json.load(f)
except Exception as e:
    print(f"Error loading roadmaps.json: {e}")
    ROADMAP_DATA = {}
from src.market_data import JOB_MARKET_DATA

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.resume_optimizer.integration import ResumeOptimizationIntegration
from src.resume_optimizer.export import ResumeExporter
from src.resume_optimizer.resume_templates import ResumeTemplateManager
from src.resume_parser import ResumeParser
from core.growth.recommender import generate_career_guidance, detect_all_suitable_roles, generate_detailed_growth_plan
from src.resume_section_parser import parse_resume_sections

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize services
integration = ResumeOptimizationIntegration()
exporter = ResumeExporter()
template_manager = ResumeTemplateManager()
parser = ResumeParser()

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('outputs', exist_ok=True)


# ---------------------------------------------------------------------------
# Shared helper: build authoritative resume dict for final template rendering
# ---------------------------------------------------------------------------

def _build_resume_dict(resume_text: str, target_role: str,
                       user_data: Dict[str, Any],
                       llm_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assemble the single authoritative dict that drives every HTML template.
    Refactored to handle skill extraction from answers, education filtering,
    and project tech stack deduplication.
    """
    import re as _re
    from src.resume_section_parser import _TECH_SKILLS_VOCAB, _DISPLAY_NAMES

    # helper to extract skills from any text using the parser's vocab
    def _extract_vocab_skills(text: str) -> list[str]:
        if not text: return []
        found = []
        lower_text = text.lower()
        # match tokens
        tokens = _re.split(r'[^\w.+#/]+', lower_text)
        for tok in tokens:
            if tok in _TECH_SKILLS_VOCAB:
                display = _DISPLAY_NAMES.get(tok, tok.title())
                if display not in found:
                    found.append(display)
        return found

    parsed  = parse_resume_sections(resume_text)
    contact = parsed.get('contact', {})

    # ── Name: first short non-contact line ───────────────────────────────────
    candidate_name = ''
    for line in resume_text.split('\n')[:10]:
        line = line.strip()
        if (line and len(line) < 60
                and not any(c.isdigit() for c in line)
                and '@' not in line and 'http' not in line
                and '/' not in line and ':' not in line):
            candidate_name = line
            break
    if not candidate_name:
        candidate_name = llm_data.get('name', '')

    # ── LinkedIn ─────────────────────────────────────────────────────────────
    lk_m = _re.search(r'linkedin\.com/in/[\w\-]+', resume_text, _re.IGNORECASE)
    linkedin = lk_m.group() if lk_m else llm_data.get('linkedin', '')

    # ── Summary ──────────────────────────────────────────────────────────────
    final_summary = (
        llm_data.get('summary') or
        llm_data.get('professional_summary') or
        parsed.get('summary', '')
    )

    # ── Skills ───────────────────────────────────────────────────────────────
    ud_skills     = user_data.get('skills', [])
    parsed_skills = parsed.get('skills', [])
    if isinstance(parsed_skills, str):
        parsed_skills = [s.strip() for s in parsed_skills.split(',') if s.strip()]

    raw_llm_skills = llm_data.get('skills', {})
    if isinstance(raw_llm_skills, dict):
        llm_skills = [s for v in raw_llm_skills.values() if isinstance(v, list) for s in v]
    elif isinstance(raw_llm_skills, list):
        llm_skills = raw_llm_skills
    else:
        llm_skills = []

    # Extract skills from clarification answers if provided
    answer_skills = []
    project_extra_skills = {} # {project_name_lower: [skills]}
    
    # Handle both potential formats of clarifications
    clarifications = user_data.get('clarifications', [])
    if not clarifications:
        # Fallback to the old dict format if present
        old_clarifications = user_data.get('clarification_answers', {})
        if isinstance(old_clarifications, dict):
            clarifications = [{"id": k, "answer": v, "question": ""} for k, v in old_clarifications.items()]

    if isinstance(clarifications, list):
        for clar in clarifications:
            ans = clar.get('answer', '')
            if not ans: continue
            
            # 1. Extract specified skills (from vocab) -> these CAN go to main skills
            vocab_discovered = _extract_vocab_skills(ans)
            answer_skills.extend(vocab_discovered)
            
            # 2. Extract non-specified skills (regex-based) -> these go to PROJECT tech stack
            potential_new = _re.findall(r'\b[A-Z][a-zA-Z0-9+#.]+\b', ans)
            new_discovered = []
            for s in potential_new:
                if len(s) > 1 and s.lower() not in ['the', 'and', 'this', 'that', 'with', 'for', 'project', 'using']:
                    if s not in vocab_discovered:
                        new_discovered.append(s)
            
            # --- FIX: Add all discovered skills (vocab + new) to global skills ---
            for s in (vocab_discovered + new_discovered):
                if s.lower() not in [sk.lower() for sk in answer_skills]:
                    answer_skills.append(s)
            
            # 3. Associate with projects if mentioned
            q_text = clar.get('question', '').lower()
            a_text = ans.lower()
            
            # Try to find which project this belongs to
            target_proj_name = None
            all_projs = user_data.get('projects', []) or parsed.get('projects', [])
            
            # Context-Aware Matching: 
            # 1. Exact or partial name match
            # 2. If the question was for a specific project ID/number
            # 3. If the answer contains specific keywords like "simulation" or "vehicle" that match a description
            for p in all_projs:
                p_name = (p.get('name') if isinstance(p, dict) else str(p)).lower()
                if not p_name: continue
                
                # Direct Match
                if p_name in q_text or p_name in a_text:
                    target_proj_name = p_name
                    break
                
                # Descriptive Match for "ABAJA" (frequent case)
                if ("baja" in p_name or "vehicle" in p_name) and ("baja" in q_text or "vehicle" in a_text):
                    target_proj_name = p_name
                    break
                    
            if target_proj_name:
                project_extra_skills.setdefault(target_proj_name, []).extend(vocab_discovered + new_discovered)
            else:
                # If it's a technical stack answer but No match found, it might be a general enhancement
                for s in new_discovered:
                    if s.lower() not in [sk.lower() for sk in answer_skills]:
                        answer_skills.append(s)

    # ── Final Skill Deduplication ───────────────────────────────────────────
    all_raw_skills = ud_skills or (parsed_skills + llm_skills + answer_skills)
    seen_skills = set()
    final_skills = []
    
    for s in all_raw_skills:
        s_clean = s.strip()
        s_lower = s_clean.lower()
        if not s_clean or s_lower in seen_skills:
            continue
        if len(s_clean.split()) > 4:
            continue
        # Deduplication e.g. "Sql" vs "SQL", "Python" vs "python"
        final_skills.append(s_clean)
        seen_skills.add(s_lower)

    # ── Experience: (KEEP WORKSHOP FIX) ──────────────────────────────────────
    ud_experience = user_data.get('experience', [])
    raw_experience = []

    if ud_experience:
        for ud_exp in ud_experience:
            bullets = ud_exp.get('bullets', ud_exp.get('achievements', []))
            if isinstance(bullets, str):
                bullets = [b.strip() for b in bullets.split('\n') if b.strip()]
            raw_experience.append({
                'title':   ud_exp.get('title', ''),
                'company': ud_exp.get('company', ''),
                'dates':   ud_exp.get('duration', ud_exp.get('dates', '')),
                'bullets': [b for b in bullets if b],
            })
    else:
        combined_raw_exp = parsed.get('experience', []) + llm_data.get('experience', [])
        for exp in combined_raw_exp:
            bullets = (exp.get('bullets') or exp.get('achievements') or 
                       exp.get('points') or [])
            if isinstance(bullets, str):
                bullets = [b.strip() for b in bullets.split('\n') if b.strip()]
            raw_experience.append({
                'title':   exp.get('title', exp.get('position', '')),
                'company': exp.get('company', exp.get('organization', '')),
                'dates':   exp.get('dates', exp.get('duration', '')),
                'bullets': [b for b in bullets if b],
            })

    # [ENHANCEMENT: WORKSHOP DEDUPLICATION ENGINE]
    final_experience = []
    def _normalize_date(d):
        parts = _re.findall(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|current|present|20\d{2})', str(d).lower())
        return " ".join(parts) if parts else str(d).strip().lower()

    def _are_bullets_similar(b1, b2):
        w1 = set(_re.findall(r'\b\w+\b', str(b1).lower()))
        w2 = set(_re.findall(r'\b\w+\b', str(b2).lower()))
        if not w1 or not w2: return False
        intersect = len(w1.intersection(w2))
        return intersect / max(len(w1), len(w2)) > 0.7

    for exp in raw_experience:
        comp = str(exp.get('company', '')).strip().lower()
        if not comp: continue
        comp_key = comp[:10] 
        date_key = _normalize_date(exp.get('dates', ''))
        title_l = str(exp.get('title', '')).lower()
        
        is_dup = False
        for existing in final_experience:
            ex_comp = str(existing['company']).lower()
            ex_date = _normalize_date(existing['dates'])
            ex_title = str(existing['title']).lower()
            
            # --- FIX: Much more aggressive matching for workshops and identical companies ---
            comp_match = (comp_key in ex_comp or ex_comp[:10] in comp) or (comp == ex_comp and comp != "")
            date_match = (date_key == ex_date or not date_key or not ex_date)
            workshop_match = ('workshop' in title_l and 'workshop' in ex_title)
            title_match = (title_l == ex_title)
            
            if (comp_match and (date_match or workshop_match)) or (title_match and comp_match):
                is_dup = True
                for b in exp['bullets']:
                    if not any(_are_bullets_similar(b, ex_b) for ex_b in existing['bullets']):
                        existing['bullets'].append(b)
                if 'workshop' in title_l and len(title_l) > len(ex_title):
                    existing['title'] = exp['title']
                break
        
        if not is_dup:
            final_experience.append(exp)

    # ── Projects: (REVERTED TO 1:45 PM STABILITY) ───────────────────────────
    def _norm_proj(p):
        if isinstance(p, str):
            name, desc = p[:80], p
        else:
            name = (p.get('name') or p.get('title') or '')
            desc = p.get('description', p.get('details', ''))
        return {'name': name, 'description': desc}

    raw_projects = (
        [_norm_proj(p) for p in user_data.get('projects', [])]     if user_data.get('projects') else
        [_norm_proj(p) for p in parsed.get('projects', [])] + [_norm_proj(p) for p in llm_data.get('projects', [])]
    )

    final_projects = []
    seen_proj_keys = set()
    for proj in raw_projects:
        name = proj['name'].strip()
        desc = proj['description'].strip()
        if not name or not desc: continue
        
        # Hallucination Filter: If Name is just a list of technologies, it's definitely accidental
        is_tech_list = (name.count(',') >= 1 or name.count(' ') >= 5) and any(kw in name.lower() for kw in ['python', 'ros2', 'gazebo', 'flask', 'sql'])
        if is_tech_list:
            # Check if this content is already represented in another project
            already_covered = False
            for fp in final_projects:
                # If titles overlap or descriptions are very similar
                if any(word in fp['name'].lower() for word in name.lower().replace(',',' ').split() if len(word) > 4):
                    already_covered = True
                    break
                if any(word in fp['description'].lower() for word in desc.lower().split() if len(word) > 5):
                    already_covered = True
                    break
            if already_covered:
                continue

        key = (name.lower()[:30], desc.lower()[:50])
        if key in seen_proj_keys:
            continue
        
        final_projects.append({'name': name, 'description': desc})
        seen_proj_keys.add(key)

    # ── Assign Tech Stack from Clarifications to Projects ────────────────────
    for proj in final_projects:
        p_name_l = proj['name'].lower()
        # Find if we have extra skills for this project
        extra = project_extra_skills.get(p_name_l, [])
        if not extra:
            # Try partial match if no exact match
            for stored_name, skills in project_extra_skills.items():
                if stored_name in p_name_l or p_name_l in stored_name:
                    extra = skills
                    break
        
        if extra:
            # Deduplicate and sort
            existing_tech = proj.get('tech_stack', [])
            combined_tech = list(dict.fromkeys(existing_tech + extra))
            proj['tech_stack'] = combined_tech

    # ── Education: (REVERTED TO 1:45 PM STABILITY + Conditionals) ───────────
    edu_src = parsed.get('education', []) or llm_data.get('education', [])
    final_education = []
    
    # Check clarifications for "nil" responses to school questions
    clarifications = user_data.get('clarifications', [])
    if not clarifications:
        old_ans = user_data.get('clarification_answers', {})
        if isinstance(old_ans, dict):
            clarifications = [{"question": "", "answer": v, "id": k} for k, v in old_ans.items()]
            
    skip_12th = False
    skip_10th = False
    _NIL_SET = {"nil","none","no","skip","n/a","na","not applicable"}
    
    for c in clarifications:
        q = (c.get('question', '') or '').lower()
        a = (c.get('answer', '') or '').lower().strip()
        if not a or a in _NIL_SET:
            if any(k in q for k in {"plus two","12th","hsc","higher secondary","class 12"}):
                skip_12th = True
            if any(k in q for k in {"10th","sslc","class 10","secondary","matric"}):
                skip_10th = True

    for edu in edu_src:
        if isinstance(edu, str):
            if "(Not provided)" in edu: continue
            if skip_12th and any(k in edu.lower() for k in {"12th", "xii", "higher secondary"}): continue
            if skip_10th and any(k in edu.lower() for k in {"10th", "class x", "secondary"}): continue
            final_education.append({'degree': edu, 'school': '', 'year': ''})
        else:
            degree = str(edu.get('degree', edu.get('field', ''))).lower()
            school = str(edu.get('school', edu.get('institution', ''))).lower()
            if "(not provided)" in degree or "(not provided)" in school: continue
            
            # Check for Class X/XII markers in degree/school line
            is_12th = any(k in degree or k in school for k in {"12th", "xii", "higher secondary", "senior secondary"})
            is_10th = any(k in degree or k in school for k in {"10th", "class x", "secondary school"})
            
            if is_12th and skip_12th: continue
            if is_10th and skip_10th: continue
            
            final_education.append({
                'degree': str(edu.get('degree', edu.get('field', ''))),
                'school': str(edu.get('school', edu.get('institution', ''))),
                'year':   edu.get('year', edu.get('end_date', '')),
            })

    return {
        'name':           candidate_name,
        'email':          contact.get('email') or llm_data.get('email', ''),
        'phone':          contact.get('phone') or llm_data.get('phone', ''),
        'location':       contact.get('location') or llm_data.get('location', ''),
        'linkedin':       linkedin,
        'title':          target_role.upper() if target_role else '',
        'summary':        final_summary,
        'experience':     final_experience,
        'education':      final_education,
        'skills':         final_skills,
        'projects':       final_projects,
        'certifications': parsed.get('certifications', []),
    }
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

@app.route('/api/upload-resume', methods=['POST'])
def upload_resume():
    """Handle resume file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save uploaded file
        filename = f"resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get file size
        file_size = os.path.getsize(filepath)
        
        # Parse resume
        resume_text = parser.parse_resume(filepath)
        
        return jsonify({
            'success': True,
            'message': 'Resume uploaded successfully!',
            'filename': filename,
            'resume_text': resume_text,
            'file_size': file_size,
            'char_count': len(resume_text)
        })
    except Exception as e:
        return jsonify({'error': f'Error parsing resume: {str(e)}'}), 500

@app.route('/api/analyze-resume', methods=['POST'])
def analyze_resume():
    """Analyze resume text or file"""
    data = request.get_json()
    resume_text = data.get('resume_text', '')
    filename = data.get('filename', '')

    # If filename is provided but no text, load the file
    if filename and not resume_text:
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            resume_text = parser.parse_resume(filepath)
        except Exception as e:
            return jsonify({'error': f'Error loading resume file: {str(e)}'}), 400

    if not resume_text:
        return jsonify({'error': 'No resume content provided'}), 400

    try:
        # --- Structured section extraction (runs BEFORE rendering Step 3) ---
        # parse_resume_sections() classifies content into clean, non-overlapping
        # sections and strips contact / summary text from experience.
        clean_sections = parse_resume_sections(resume_text)

        # --- Legacy scoring / fault analysis (unchanged) --------------------
        full_analysis = integration.complete_resume_analysis(resume_text)

        return jsonify({
            'success': True,
            'analysis': full_analysis.get('resume_analysis', {}),
            'structured_data': full_analysis.get('extracted_data', {}),
            # clean_sections is the authoritative source for Step 3 form mapping
            'clean_sections': clean_sections,
            'career_guidance': full_analysis.get('career_guidance', ''),
            'clarification_questions': full_analysis.get('clarification_questions', [])
        })
    except Exception as e:
        return jsonify({'error': f'Error analyzing resume: {str(e)}'}), 500

@app.route('/api/optimize-resume', methods=['POST'])
def optimize_resume():
    """Optimize resume for specific role"""
    data = request.get_json()
    resume_text = data.get('resume_text', '')
    filename = data.get('filename', '')
    target_role = data.get('target_role', '')
    user_data = data.get('user_data', {})
    
    # Load content if filename provided
    if filename and not resume_text:
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            resume_text = parser.parse_resume(filepath)
        except Exception as e:
            return jsonify({'error': f'Error loading resume file: {str(e)}'}), 400
            
    if not resume_text:
        return jsonify({'error': 'No resume content provided'}), 400
    
    try:
        # If target role is provided, optimize specifically for that role
        if target_role:
            # Create requirements for the role
            requirements = integration._create_requirements_for_role(target_role)
            optimization_result = integration.optimization_api.optimize_resume(
                resume_text, target_role, requirements, user_data=user_data
            )
            
            # Format the output just like template selection does
            optimized_content_raw = optimization_result.get('optimized_content', '')
            try:
                optimized_data = json.loads(optimized_content_raw)
            except Exception:
                optimized_data = {"text": optimized_content_raw}

            # ----------------------------------------------------------------
            # Build authoritative resume_dict (REFACTORED)
            # ----------------------------------------------------------------
            resume_dict = _build_resume_dict(resume_text, target_role, user_data, optimized_data)


            template = template_manager.get_best_template(resume_dict, target_role)
            formatted_output = template_manager.format_resume(resume_dict, template, target_role)
            
            optimization_result = {
                'target_role': target_role,
                'optimized_content': formatted_output,
                'raw_data': resume_dict,
                'template_used': template.name,
                'template_style': template.style.value,
                'optimization_score': optimization_result.get('optimization_score', 0),
                'generation_timestamp': datetime.now().isoformat()
            }
        else:
            # Otherwise optimize for recommended roles (batch fallback)
            optimization_result = integration.optimize_for_recommended_roles(resume_text, user_data=user_data)
        
        return jsonify({
            'success': True,
            'optimization': optimization_result
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error optimizing resume: {str(e)}'}), 500

@app.route('/api/templates')
def get_templates():
    """Get list of available resume templates"""
    try:
        templates = template_manager.get_template_list()
        return jsonify({
            'success': True,
            'templates': templates
        })
    except Exception as e:
        return jsonify({'error': f'Error getting templates: {str(e)}'}), 500

@app.route('/api/optimize-resume-with-template', methods=['POST'])
def optimize_resume_with_template():
    """Optimize resume using specific template"""
    data = request.get_json()
    resume_text = data.get('resume_text', '')
    filename = data.get('filename', '')
    target_role = data.get('target_role', '')
    template_id = data.get('template_id', '')
    user_data = data.get('user_data', {})
    
    # Load content if filename provided
    if filename and not resume_text:
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            resume_text = parser.parse_resume(filepath)
        except Exception as e:
            return jsonify({'error': f'Error loading resume file: {str(e)}'}), 400
            
    if not resume_text:
        return jsonify({'error': 'No resume content provided'}), 400
    
    try:
        requirements = integration._create_requirements_for_role(target_role)
        optimization_result = integration.optimization_api.optimize_resume(
            resume_text, target_role, requirements, user_data=user_data
        )

        optimized_content_raw = optimization_result.get('optimized_content', '')
        try:
            optimized_data = json.loads(optimized_content_raw)
        except Exception:
            optimized_data = {"text": optimized_content_raw}

        # Normalise into authoritative resume_dict (REFACTORED)
        resume_dict = _build_resume_dict(resume_text, target_role, user_data, optimized_data)


        if template_id and template_id in template_manager.templates:
            template = template_manager.templates[template_id]
        else:
            template = template_manager.get_best_template(resume_dict, target_role)

        formatted_output = template_manager.format_resume(resume_dict, template, target_role)

        optimized_resume = {
            'target_role': target_role,
            'optimized_content': formatted_output,
            'raw_data': resume_dict,
            'template_used': template.name,
            'template_style': template.style.value,
            'optimization_score': optimization_result.get('optimization_score', 0),
            'generation_timestamp': datetime.now().isoformat()
        }

        return jsonify({'success': True, 'optimized_resume': optimized_resume})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error optimizing resume with template: {str(e)}'}), 500

@app.route('/api/export-resume', methods=['POST'])
def export_resume():
    """Export optimized resume to various formats"""
    data = request.get_json()
    optimized_resume = data.get('optimized_resume') or data
    format_type = data.get('format', 'text')
    
    if not optimized_resume or (isinstance(optimized_resume, dict) and not optimized_resume.get('optimized_content')):
        return jsonify({'error': 'No optimized resume data provided'}), 400
    
    try:
        if format_type == 'text':
            file_path = exporter.export_to_text(optimized_resume)
        elif format_type == 'markdown':
            file_path = exporter.export_to_markdown(optimized_resume)
        elif format_type == 'html':
            file_path = exporter.export_to_html(optimized_resume)
        elif format_type == 'pdf':
            file_path = exporter.export_to_pdf(optimized_resume)
        else:
            return jsonify({'error': 'Invalid format specified'}), 400
        
        if not file_path:
            return jsonify({'error': f'Failed to generate {format_type} export'}), 500
            
        return jsonify({
            'success': True,
            'file_path': file_path,
            'filename': os.path.basename(file_path)
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

@app.route('/api/test')
def test_endpoint():
    """Simple test endpoint"""
    return jsonify({
        'success': True,
        'message': 'Backend is working',
        'timestamp': str(datetime.now())
    })

@app.route("/detect_career_roles", methods=["POST"])
def detect_career_roles():
    """Step 1: Detect all possible job roles from resume text."""
    try:
        data = request.get_json()
        resume_text = data.get("resume_text", "")
        if not resume_text:
            return jsonify({"status": "error", "message": "No resume content"}), 400
            
        clean_sections = parse_resume_sections(resume_text)
        resume_data = {
            "skills": clean_sections.get("skills", []),
            "projects": clean_sections.get("projects", []),
            "experience": clean_sections.get("experience", "Fresher")
        }
        
        possible_roles, career_level, initial_analysis = detect_all_suitable_roles(resume_data, resume_text)
        
        return jsonify({
            "status": "success",
            "roles": possible_roles,
            "career_level": career_level,
            "initial_analysis": initial_analysis,
            "skills_detected": resume_data["skills"]
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/generate_detailed_plan", methods=["POST"])
def generate_detailed_plan():
    """Step 2: Generate deep intelligence for selected roles."""
    try:
        data = request.get_json()
        resume_text = data.get("resume_text", "")
        selected_roles = data.get("selected_roles", [])
        career_level = data.get("career_level", "Entry-Level")

        if not resume_text or not selected_roles:
            return jsonify({"status": "error", "message": "Missing data"}), 400

        clean_sections = parse_resume_sections(resume_text)

        # Format experience
        exp_list = clean_sections.get('experience', [])
        formatted_exp = ""
        for exp in exp_list:
            formatted_exp += f"{exp.get('title', 'Role')} at {exp.get('company', 'Company')} ({exp.get('duration', 'N/A')})\n"
            for ach in exp.get('achievements', []):
                formatted_exp += f"- {ach}\n"

        resume_data = {
            "skills": clean_sections.get("skills", []),
            "projects": clean_sections.get("projects", []),
            "education": clean_sections.get("education", []),
            "experience": formatted_exp or "Fresher",
            "other": clean_sections.get("certifications", []),
        }

        growth_plan = generate_detailed_growth_plan(resume_data, selected_roles, resume_text, career_level)

        return jsonify({
            "status": "success",
            "growth_plan": growth_plan
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500

@app.route('/api/market-insights', methods=['POST'])
def get_market_insights():
    """API to fetch realistic job market insights: Local lookup first, AI fallback second"""
    try:
        data = request.json
        role = data.get('role', 'Software Engineer')
        skills = data.get('skills', [])
        
        # Priority 1: Smart Lookup from our High-Confidence Local Dataset
        # Improvement: Map specialized role names to core career tracks
        ROLE_MAPPING = {
            "mern": "Full Stack Developer", "fullstack": "Full Stack Developer", 
            "react": "Frontend Developer", "angular": "Frontend Developer",
            "vue": "Frontend Developer", "web developer": "Frontend Developer",
            "node": "Backend Developer", "python developer": "Backend Developer",
            "java developer": "Backend Developer", "spring boot": "Backend Developer",
            "flutter": "Mobile App Developer", "android": "Mobile App Developer",
            "ios": "Mobile App Developer", "react native": "Mobile App Developer",
            "ml": "AI/ML Engineer", "machine learning": "AI/ML Engineer",
            "natural language": "AI/ML Engineer", "computer vision": "AI/ML Engineer",
            "data engineer": "Data Scientist", "data analyst": "Data Scientist",
            "cloud": "DevOps Engineer", "kubernetes": "DevOps Engineer",
            "sre": "DevOps Engineer", "automation": "Quality Assurance (QA)",
            "tester": "Quality Assurance (QA)", "sdet": "Quality Assurance (QA)",
            "ux": "UI/UX Designer", "designer": "UI/UX Designer",
            "soc": "Cybersecurity Analyst", "security": "Cybersecurity Analyst"
        }
        
        # Clean the role for better matching
        clean_role = role.replace('-', ' ').strip().lower()
        
        # Check explicit mapping first
        mapped_role = None
        for key in ROLE_MAPPING:
            if key in clean_role:
                mapped_role = ROLE_MAPPING[key]
                break
        
        # If no mapping, check direct dataset keys
        market_entry = None
        target_key = mapped_role if mapped_role else None
        
        if target_key:
            market_entry = JOB_MARKET_DATA.get(target_key)
        else:
            for key in JOB_MARKET_DATA:
                if key.lower().replace('-', ' ') == clean_role:
                    market_entry = JOB_MARKET_DATA[key]
                    break
        
        if market_entry:
            # Smart Curation: Only return the MOST relevant 'Top' companies 
            # (BigTech and Lead Product firms are prioritized over Services for accuracy)
            curated_companies = (
                market_entry['companies'].get('bigtech', [])[:3] + 
                market_entry['companies'].get('product', [])[:3]
            )
            
            # If we still have space, add one lead service company as a backup
            if len(curated_companies) < 6 and market_entry['companies'].get('service'):
                curated_companies.append(market_entry['companies']['service'][0])

            return jsonify({
                "role": role,
                "demand": market_entry['demand'],
                "competition": market_entry['competition'],
                "trend": market_entry['trend'],
                "salaries": market_entry['salaries'],
                "industries": market_entry['industries'][:4], # Limit to top 4 industries
                "companies": curated_companies,
                "insight": market_entry['insight']
            })

        # Priority 2: AI Generation with Strictest Indian Market Rules
        prompt = f"""
Given the job role: {role}
And skills: {', '.join(skills)}

Use realistic Indian job market data.

STRICT RULES:
- Include IT service companies (TCS, Infosys, Wipro, HCL)
- Include product companies (Zoho, Freshworks, Flipkart)
- Include big tech (Amazon, Google, Microsoft)
- Do NOT give only startup companies
- Keep salary realistic (not extreme) - Use INR LPA format.

Return ONLY valid JSON with this exact structure:
{{
  "demand": "High/Medium/Low",
  "competition": "High/Medium/Low",
  "trend": "Rising/Stable/Declining",
  "salaries": {{ "entry": "X LPA", "mid": "X LPA", "top": "X+ LPA" }},
  "industries": ["...", "..."],
  "companies": ["...", "..."],
  "insight": "2 line insight about market context"
}}
"""
        from src.llm_client import call_llm
        import json
        import re
        raw_response = call_llm(prompt)
        
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            return jsonify(json.loads(json_match.group(0)))
            
        raise Exception("Invalid AI response format")

    except Exception as e:
        # Final safety fallback 
        return jsonify({
            "demand": "High", "competition": "Medium", "trend": "Rising",
            "salaries": { "entry": "4-6 LPA", "mid": "12-18 LPA", "top": "25+ LPA" },
            "industries": ["Software", "IT Services"],
            "companies": ["TCS", "Infosys", "Zoho", "Amazon"],
            "insight": "High demand for this technical role in the current Indian market."
        })

@app.route('/api/roadmap', methods=['POST'])
def get_roadmap():
    """API to fetch static career roadmap for a role from roadmaps.json with skill-gap filtering"""
    try:
        data = request.get_json() or {}
        role = str(data.get('role', '')).lower()
        user_skills = [s.lower().strip() for s in data.get('skills', [])]
        
        base_roadmap = ROADMAP_DATA.get(role)
        if not base_roadmap:
            return jsonify({"error": "No roadmap found for this role"}), 200
            
        # Create a deep copy to avoid modifying the original shared data
        roadmap = copy.deepcopy(base_roadmap)
        
        def filter_skills(skill_list):
            return [s for s in skill_list if s.lower().strip() not in user_skills]
            
        # Dynamically filter skills in each term (short, medium, long)
        if 'short_term' in roadmap:
            roadmap['short_term']['skills'] = filter_skills(roadmap['short_term'].get('skills', []))
        if 'medium_term' in roadmap:
            roadmap['medium_term']['skills'] = filter_skills(roadmap['medium_term'].get('skills', []))
        if 'long_term' in roadmap:
            roadmap['long_term']['skills'] = filter_skills(roadmap['long_term'].get('skills', []))
            
        return jsonify(roadmap)
    except Exception as e:
        return jsonify({"error": str(e)}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
