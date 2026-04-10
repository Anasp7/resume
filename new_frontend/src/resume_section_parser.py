"""
Resume Section Parser
=====================
Structured, vocabulary-driven extractor that classifies raw resume text
into clean, non-overlapping sections BEFORE the UI renders Step 3.

Architecture (ordered passes — each pass removes consumed content):
  1. Contact extraction   → phone / email / location stripped globally
  2. Section-block split  → named headers delimit text into named blocks
  3. Summary extraction   → introductory paragraph captured
  4. Skills extraction    → vocabulary-matched tech skills ONLY
  5. Projects extraction  → project titles + descriptions from project block
  6. Experience extraction→ ONLY real professional accomplishments
  7. Education / Certs    → education and certification lines

Key design principle for Skills:
  Skills extraction uses a comprehensive KNOWN-VOCABULARY set.
  Only tokens / phrases found in that vocabulary are accepted.
  This completely prevents hobby text, location names, or garbage
  comma-lists from leaking into the Technical Skills field.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Contact patterns
# ---------------------------------------------------------------------------

_PHONE_RE  = re.compile(r'\b\d{10}\b')
_EMAIL_RE  = re.compile(r'\S+@\S+\.\S+')

_LOCATION_PATTERNS = [
    re.compile(r'\b[A-Z][a-z]+,\s*Kerala\b',       re.IGNORECASE),
    re.compile(r'\b[A-Z][a-z]+,\s*India\b',         re.IGNORECASE),
    re.compile(r'\b[A-Z][a-z]+,\s*[A-Z]{2}\b'),     # City, ST
    re.compile(
        r'\b(Kerala|Tamil Nadu|Karnataka|Maharashtra|Delhi|Rajasthan|'
        r'Bangalore|Bengaluru|Mumbai|Chennai|Hyderabad|Pune|Kochi|'
        r'Kolkata|Ahmedabad|Surat|Jaipur|Lucknow|Kanpur|Nagpur)\b',
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Section header detection
# ---------------------------------------------------------------------------

_SECTION_HEADERS: Dict[str, List[str]] = {
    'experience': [
        'experience', 'work experience', 'professional experience',
        'employment', 'employment history', 'internship', 'internships',
        'work history', 'career history', 'industrial experience',
    ],
    'education': [
        'education', 'academic', 'academic background',
        'educational qualification', 'qualifications', 'academic history',
    ],
    'skills': [
        'skills', 'technical skills', 'core competencies', 'key skills',
        'expertise', 'technologies', 'tools', 'programming languages',
        'technical expertise', 'core skills', 'competencies',
    ],
    'projects': [
        'projects', 'personal projects', 'key projects',
        'project experience', 'portfolio', 'academic projects',
        'major projects', 'mini projects',
    ],
    'certifications': [
        'certifications', 'certificates', 'courses', 'achievements',
        'professional development', 'training', 'awards',
        'extra-curricular', 'extracurricular', 'activities',
    ],
    'summary': [
        'summary', 'professional summary', 'objective', 'career objective',
        'about me', 'profile', 'overview',
    ],
}


# ---------------------------------------------------------------------------
# Comprehensive technical skills vocabulary
# Only tokens matched against this set will appear in the Skills field.
# ---------------------------------------------------------------------------

_TECH_SKILLS_VOCAB: Set[str] = {
    # ── Programming languages ──────────────────────────────────────────────
    'python', 'java', 'javascript', 'js', 'typescript', 'ts',
    'c', 'c++', 'cpp', 'c#', 'csharp', 'ruby', 'php',
    'swift', 'kotlin', 'go', 'golang', 'rust', 'scala',
    'r', 'matlab', 'perl', 'bash', 'shell', 'powershell', 'vba',
    'assembly', 'dart', 'elixir', 'erlang', 'haskell', 'lua',
    'objective-c', 'prolog', 'groovy', 'cobol', 'fortran',

    # ── Web frontend ───────────────────────────────────────────────────────
    'html', 'html5', 'css', 'css3', 'sass', 'scss', 'less',
    'react', 'reactjs', 'react.js', 'angular', 'angularjs',
    'vue', 'vuejs', 'vue.js', 'svelte', 'nextjs', 'next.js',
    'nuxtjs', 'gatsby', 'jquery', 'bootstrap', 'tailwind',
    'tailwindcss', 'webpack', 'vite', 'babel', 'redux',
    'graphql', 'rest', 'restful api', 'api',

    # ── Web backend ────────────────────────────────────────────────────────
    'nodejs', 'node.js', 'node', 'express', 'expressjs',
    'django', 'flask', 'fastapi', 'springboot', 'spring boot',
    'spring', 'asp.net', '.net', 'dotnet', 'laravel', 'rails',
    'ruby on rails', 'fastify', 'koa', 'nestjs',

    # ── Databases ──────────────────────────────────────────────────────────
    'sql', 'mysql', 'postgresql', 'postgres', 'mongodb', 'mongo',
    'sqlite', 'oracle', 'redis', 'cassandra', 'dynamodb',
    'firebase', 'mariadb', 'nosql', 'elasticsearch', 'neo4j',
    'influxdb', 'cockroachdb', 'supabase', 'prisma', 'sequelize',

    # ── Cloud & DevOps ─────────────────────────────────────────────────────
    'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes',
    'k8s', 'jenkins', 'gitlab', 'github', 'git', 'ci/cd',
    'terraform', 'ansible', 'puppet', 'chef', 'nginx', 'apache',
    'linux', 'ubuntu', 'centos', 'debian', 'unix',
    'heroku', 'vercel', 'netlify', 'cloudflare', 'argocd',

    # ── Machine Learning / Data Science ────────────────────────────────────
    'machine learning', 'ml', 'deep learning', 'dl',
    'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn',
    'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
    'opencv', 'nlp', 'natural language processing',
    'computer vision', 'data science', 'ai', 'artificial intelligence',
    'neural network', 'cnn', 'rnn', 'lstm', 'transformer',
    'bert', 'gpt', 'llm', 'reinforcement learning', 'xgboost',
    'lightgbm', 'random forest', 'svm', 'regression', 'classification',
    'clustering', 'pca', 'data analysis', 'data engineering',
    'data visualization', 'feature engineering', 'mlops',
    'huggingface', 'langchain', 'rag',

    # ── Tools & IDEs ───────────────────────────────────────────────────────
    'excel', 'powerpoint', 'tableau', 'power bi', 'figma',
    'photoshop', 'illustrator', 'postman', 'jira', 'confluence',
    'vs code', 'vscode', 'intellij', 'eclipse', 'pycharm',
    'jupyter', 'colab', 'google colab', 'notion', 'trello',
    'swagger', 'bitbucket', 'sourcetree', 'vim',

    # ── Robotics / Embedded / IoT ──────────────────────────────────────────
    'ros', 'ros2', 'arduino', 'raspberry pi', 'embedded',
    'iot', 'fpga', 'verilog', 'vhdl', 'labview', 'simulink',
    'stm32', 'microcontroller', 'mqtt', 'modbus', 'plc', 'gazebo',

    # ── Networking / Security ──────────────────────────────────────────────
    'tcp/ip', 'http', 'https', 'ssl', 'tls', 'vpn',
    'firewall', 'wireshark', 'nmap', 'metasploit',
    'penetration testing', 'cyber security', 'owasp',

    # ── Mobile ─────────────────────────────────────────────────────────────
    'android', 'ios', 'flutter', 'react native', 'xamarin',
    'jetpack compose', 'swiftui', 'expo',

    # ── Testing ────────────────────────────────────────────────────────────
    'junit', 'pytest', 'selenium', 'cypress', 'jest', 'mocha',
    'chai', 'testng', 'playwright', 'nunit', 'xunit',

    # ── Blockchain ─────────────────────────────────────────────────────────
    'blockchain', 'solidity', 'ethereum', 'web3', 'smart contracts',

    # ── Methodologies ──────────────────────────────────────────────────────
    'agile', 'scrum', 'kanban', 'devops', 'tdd', 'bdd',
    'microservices', 'monolithic', 'event-driven', 'object-oriented',
    'oop', 'functional programming', 'design patterns', 'mvc',
    'mvvm', 'solid principles',
}

# Multi-word skills that must be matched as exact phrases first
_MULTI_WORD_SKILLS: List[str] = sorted(
    [s for s in _TECH_SKILLS_VOCAB if ' ' in s],
    key=len,
    reverse=True,  # longest phrases first to avoid partial matches
)

# Canonical display names for proper capitalisation in the UI
_DISPLAY_NAMES: Dict[str, str] = {
    'python': 'Python', 'java': 'Java', 'javascript': 'JavaScript',
    'js': 'JavaScript', 'typescript': 'TypeScript', 'ts': 'TypeScript',
    'c++': 'C++', 'cpp': 'C++', 'c#': 'C#', 'csharp': 'C#',
    'ruby': 'Ruby', 'php': 'PHP', 'swift': 'Swift', 'kotlin': 'Kotlin',
    'go': 'Go', 'golang': 'Go', 'rust': 'Rust', 'scala': 'Scala',
    'r': 'R', 'matlab': 'MATLAB', 'perl': 'Perl', 'bash': 'Bash',
    'html': 'HTML', 'html5': 'HTML5', 'css': 'CSS', 'css3': 'CSS3',
    'sass': 'Sass', 'scss': 'SCSS', 'react': 'React', 'reactjs': 'React',
    'angular': 'Angular', 'vue': 'Vue.js', 'vuejs': 'Vue.js',
    'svelte': 'Svelte', 'nextjs': 'Next.js', 'gatsby': 'Gatsby',
    'jquery': 'jQuery', 'bootstrap': 'Bootstrap', 'tailwind': 'Tailwind CSS',
    'tailwindcss': 'Tailwind CSS', 'webpack': 'Webpack', 'redux': 'Redux',
    'graphql': 'GraphQL', 'nodejs': 'Node.js', 'node': 'Node.js',
    'express': 'Express.js', 'expressjs': 'Express.js',
    'django': 'Django', 'flask': 'Flask', 'fastapi': 'FastAPI',
    'spring': 'Spring', 'springboot': 'Spring Boot',
    'sql': 'SQL', 'mysql': 'MySQL', 'postgresql': 'PostgreSQL',
    'postgres': 'PostgreSQL', 'mongodb': 'MongoDB', 'mongo': 'MongoDB',
    'sqlite': 'SQLite', 'oracle': 'Oracle', 'redis': 'Redis',
    'cassandra': 'Cassandra', 'dynamodb': 'DynamoDB', 'firebase': 'Firebase',
    'elasticsearch': 'Elasticsearch', 'nosql': 'NoSQL',
    'aws': 'AWS', 'azure': 'Azure', 'gcp': 'GCP',
    'docker': 'Docker', 'kubernetes': 'Kubernetes', 'k8s': 'Kubernetes',
    'jenkins': 'Jenkins', 'gitlab': 'GitLab', 'github': 'GitHub',
    'git': 'Git', 'terraform': 'Terraform', 'ansible': 'Ansible',
    'nginx': 'Nginx', 'linux': 'Linux', 'ubuntu': 'Ubuntu',
    'tensorflow': 'TensorFlow', 'pytorch': 'PyTorch', 'keras': 'Keras',
    'pandas': 'Pandas', 'numpy': 'NumPy', 'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn', 'opencv': 'OpenCV', 'nlp': 'NLP',
    'ai': 'AI', 'ml': 'Machine Learning', 'dl': 'Deep Learning',
    'tableau': 'Tableau', 'figma': 'Figma', 'postman': 'Postman',
    'jira': 'Jira', 'ros': 'ROS', 'ros2': 'ROS2', 'arduino': 'Arduino',
    'iot': 'IoT', 'fpga': 'FPGA', 'android': 'Android', 'ios': 'iOS',
    'flutter': 'Flutter', 'agile': 'Agile', 'scrum': 'Scrum',
    'devops': 'DevOps', 'api': 'REST API', 'rest': 'REST',
    'oop': 'OOP', 'mvc': 'MVC', 'mvvm': 'MVVM',
    'ci/cd': 'CI/CD', 'tdd': 'TDD', 'bdd': 'BDD',
    'excel': 'MS Excel', 'powerpoint': 'PowerPoint',
    'vscode': 'VS Code', 'vs code': 'VS Code',
    'machine learning': 'Machine Learning',
    'deep learning': 'Deep Learning',
    'data science': 'Data Science',
    'computer vision': 'Computer Vision',
    'natural language processing': 'Natural Language Processing',
    'react native': 'React Native',
    'spring boot': 'Spring Boot',
    'power bi': 'Power BI',
    'google cloud': 'Google Cloud',
    'smart contracts': 'Smart Contracts',
    'raspberry pi': 'Raspberry Pi',
    'gazebo': 'Gazebo',
}


# ---------------------------------------------------------------------------
# Experience / accomplishment detection
# ---------------------------------------------------------------------------

_EXPERIENCE_POSITIVE_KW = [
    'intern', 'internship', 'company', 'worked', 'role',
    'responsibilities', 'employed', 'employment', 'position',
    'designation', 'organization', 'organisation', 'firm',
    'corporate', 'ltd', 'inc', 'pvt',
]

_EXPERIENCE_NEGATIVE_KW = [
    'university', 'college', 'school', 'cgpa', 'gpa',
    'b.tech', 'btech', 'bachelor', 'higher secondary',
    'gmail', 'yahoo', 'hotmail', 'outlook', '@',
]

# Action verbs that signal an accomplishment bullet
_ACTION_VERB_RE = re.compile(
    r'^\b(developed|designed|built|created|implemented|deployed|'
    r'managed|led|coordinated|achieved|improved|reduced|increased|'
    r'optimized|automated|integrated|maintained|collaborated|'
    r'researched|analysed|analyzed|tested|debugged|refactored|'
    r'delivered|launched|established|structured|streamlined|'
    r'assisted|supported|conducted|performed|executed|resolved|'
    r'trained|mentored|supervised|monitored|configured|installed|'
    r'operated|handled|processed|prepared|presented|published)\b',
    re.IGNORECASE,
)

# Summary trigger words
_SUMMARY_TRIGGER_WORDS = [
    'motivated', 'aspiring', 'seeking', 'career', 'interested',
    'enthusiastic', 'passionate', 'dedicated', 'objective',
    'self-motivated', 'goal-oriented', 'eager', 'driven',
]
_SUMMARY_SENTENCE_RE = re.compile(
    r'\b(' + '|'.join(_SUMMARY_TRIGGER_WORDS) + r')\b',
    re.IGNORECASE,
)

_EDUCATION_KEYWORDS = [
    'b.tech', 'btech', 'b tech', 'bachelor', "bachelor's",
    'higher secondary', 'high school', 'hse', 'sslc',
    'cgpa', 'gpa', 'university', 'college', 'institute',
    'master', 'm.tech', 'mtech', 'diploma', 'matriculation',
    'class xii', 'class x', '10th', '12th', 'plus two',
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_resume_sections(raw_text: str) -> Dict[str, Any]:
    """
    Parse raw resume text and return a clean structured dictionary:

    {
      "contact":        {"phone": str, "email": str, "location": str},
      "summary":        str,
      "education":      [str, ...],
      "skills":         [str, ...],   # ONLY vocabulary-matched tech skills
      "projects":       [str, ...],   # each entry is one project description
      "experience":     [{"title": str, "company": str, "duration": str,
                          "achievements": [str, ...]}, ...],
      "certifications": [str, ...],
    }
    """
    text = raw_text.replace('\r\n', '\n').replace('\r', '\n')

    # Pass 1 – section blocks
    section_blocks = _split_into_section_blocks(text)

    # Pass 2 – contact (global scan; strips from text)
    contact, text_clean = _extract_contact(text)

    # Pass 3 – summary
    summary, text_clean = _extract_summary(text_clean, section_blocks)

    # Pass 4 – vocabulary-matched skills
    skills = _extract_skills_vocab(section_blocks, text_clean)

    # Pass 5 – projects
    projects = _extract_projects(section_blocks, text_clean)

    # Pass 6 – experience accomplishments
    experience = _extract_experience(section_blocks, contact, summary)

    # Pass 7 – education + certifications
    education      = _extract_education(section_blocks, text_clean)
    certifications = _extract_certifications(section_blocks, text_clean)

    return {
        'contact':        contact,
        'summary':        summary,
        'education':      education,
        'skills':         skills,
        'projects':       projects,
        'experience':     experience,
        'certifications': certifications,
    }


# ---------------------------------------------------------------------------
# Pass 1 – Section header splitter
# ---------------------------------------------------------------------------

def _split_into_section_blocks(text: str) -> Dict[str, str]:
    """
    Walk every line; when a line matches a known section header, all
    subsequent lines go into that section's bucket until the next header.
    """
    lines = text.split('\n')
    blocks: Dict[str, List[str]] = {'header': []}
    current = 'header'

    for line in lines:
        stripped = line.strip()
        matched = _identify_section_header(stripped)
        if matched:
            current = matched
            blocks.setdefault(current, [])
        else:
            blocks.setdefault(current, []).append(line)

    return {k: '\n'.join(v) for k, v in blocks.items()}


def _identify_section_header(line: str) -> Optional[str]:
    """
    Return canonical section name if line looks like a section header.
    Accepts headers with or without trailing colons / underscores.
    """
    if not line or len(line) > 70:
        return None
    normalized = re.sub(r'[_\-:]+$', '', line).strip().lower()
    for section, keywords in _SECTION_HEADERS.items():
        if any(normalized == kw or normalized.startswith(kw)
               for kw in keywords):
            return section
    return None


# ---------------------------------------------------------------------------
# Pass 2 – Contact extraction
# ---------------------------------------------------------------------------

def _extract_contact(text: str) -> Tuple[Dict[str, str], str]:
    contact: Dict[str, str] = {'phone': '', 'email': '', 'location': ''}
    lines = text.split('\n')
    clean: List[str] = []

    for line in lines:
        mod = line

        if not contact['phone']:
            m = _PHONE_RE.search(mod)
            if m:
                contact['phone'] = m.group()
                mod = _PHONE_RE.sub('', mod, count=1).strip()

        if not contact['email']:
            m = _EMAIL_RE.search(mod)
            if m:
                contact['email'] = m.group()
                mod = _EMAIL_RE.sub('', mod, count=1).strip()

        if not contact['location']:
            for loc_re in _LOCATION_PATTERNS:
                m = loc_re.search(mod)
                if m:
                    contact['location'] = m.group().strip()
                    mod = loc_re.sub('', mod, count=1).strip()
                    break

        clean.append(mod)

    return contact, '\n'.join(clean)


# ---------------------------------------------------------------------------
# Pass 3 – Summary extraction
# ---------------------------------------------------------------------------

def _extract_summary(
    text: str,
    section_blocks: Dict[str, str],
) -> Tuple[str, str]:
    # Priority 1: named summary block
    if 'summary' in section_blocks:
        block = section_blocks['summary'].strip()
        if block:
            return block, text.replace(block, '', 1)

    # Priority 2: first paragraph with trigger words
    for para in re.split(r'\n{2,}', text)[:5]:
        stripped = para.strip()
        if len(stripped) > 40 and _SUMMARY_SENTENCE_RE.search(stripped):
            return stripped, text.replace(para, '', 1)

    return '', text


# ---------------------------------------------------------------------------
# Pass 4 – Vocabulary-matched skills extraction
# ---------------------------------------------------------------------------

def _extract_skills_vocab(
    section_blocks: Dict[str, str],
    fallback_text: str,
) -> List[str]:
    """
    Extract ONLY tokens / phrases present in _TECH_SKILLS_VOCAB.
    The search text is the named 'skills' block when present,
    otherwise the entire cleaned text.  This prevents hobby words,
    dates, locations, and free-text sentences from appearing.
    """
    search_text = (
        section_blocks.get('skills', '') or
        section_blocks.get('header', '') + '\n' + fallback_text
    )

    found: List[str] = []
    seen: Set[str] = set()
    lower_text = search_text.lower()

    # Step 1: match multi-word phrases first (longest first)
    for phrase in _MULTI_WORD_SKILLS:
        if phrase in lower_text and phrase not in seen:
            display = _DISPLAY_NAMES.get(phrase, phrase.title())
            found.append(display)
            seen.add(phrase)

    # Step 2: tokenise and match single-word skills
    # Split on anything that isn't alphanumeric, dot, +, or #
    tokens = re.split(r'[^\w.+#/]+', search_text)
    for tok in tokens:
        tok_lower = tok.strip().lower()
        if tok_lower in _TECH_SKILLS_VOCAB and tok_lower not in seen:
            display = _DISPLAY_NAMES.get(tok_lower, tok.strip())
            found.append(display)
            seen.add(tok_lower)

    # Remove single-word matches that are substrings of already-matched phrases
    cleaned: List[str] = []
    for skill in found:
        sk_lower = skill.lower()
        already_covered = any(
            sk_lower != other.lower() and sk_lower in other.lower()
            for other in found
        )
        if not already_covered:
            cleaned.append(skill)

    return cleaned


# ---------------------------------------------------------------------------
# Pass 5 – Project extraction
# ---------------------------------------------------------------------------

def _extract_projects(
    section_blocks: Dict[str, str],
    fallback_text: str,
) -> List[str]:
    """
    Extract project entries from the 'projects' block.
    Each contiguous group of non-empty lines is treated as one project entry.
    """
    projects: List[str] = []
    block = section_blocks.get('projects', '').strip()

    if block:
        current: List[str] = []
        for raw_line in block.split('\n'):
            line = raw_line.strip().lstrip('•·-* ')
            if not line:
                if current:
                    projects.append(' '.join(current))
                    current = []
            else:
                current.append(line)
        if current:
            projects.append(' '.join(current))
    else:
        # Fallback: sentences containing "project" or strong build verbs
        # combined with a tech skill word
        proj_kw_re = re.compile(
            r'\b(project|developed|built|created|implemented)\b',
            re.IGNORECASE,
        )
        for line in fallback_text.split('\n'):
            stripped = line.strip()
            if len(stripped) > 20 and proj_kw_re.search(stripped):
                # Only accept if line also contains at least one known skill
                lower = stripped.lower()
                has_skill = any(sk in lower for sk in _TECH_SKILLS_VOCAB)
                if has_skill:
                    projects.append(stripped)

    return [p for p in projects if p]


# ---------------------------------------------------------------------------
# Pass 6 – Experience extraction
# ---------------------------------------------------------------------------

def _is_valid_exp_header(line: str) -> bool:
    if not line or len(line) > 120:
        return False
    lower = line.lower()
    if _PHONE_RE.search(line) or _EMAIL_RE.search(line):
        return False
    if any(kw in lower for kw in _EXPERIENCE_NEGATIVE_KW):
        return False
    if _SUMMARY_SENTENCE_RE.search(line):
        return False
    return any(kw in lower for kw in _EXPERIENCE_POSITIVE_KW)


def _parse_exp_block(block: str) -> List[Dict[str, Any]]:
    """
    Parse the experience section block line-by-line.
    Short validated lines → entry headers.
    Other non-empty lines → achievement bullets.
    """
    entries: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    duration_re = re.compile(
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
        r'\s*\d{4}\s*[-–—to]+\s*(?:Present|\d{4}|'
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4})\b',
        re.IGNORECASE,
    )

    for raw in block.split('\n'):
        line = raw.strip()
        if not line:
            continue

        dur_m = duration_re.search(line)
        # A short line containing exp keywords → new entry header
        if len(line) <= 80 and _is_valid_exp_header(line):
            if current:
                entries.append(current)
            clean = line.replace(dur_m.group(), '').strip() if dur_m else line
            title, company = _parse_title_company(clean)
            current = {
                'title': title,
                'company': company,
                'duration': dur_m.group() if dur_m else '',
                'achievements': [],
            }
        elif current:
            # Treat as accomplishment bullet
            bullet = line.lstrip('•·-* ').strip()
            if bullet and len(bullet) > 5:
                current['achievements'].append(bullet)
        elif _ACTION_VERB_RE.match(line) and len(line) > 15:
            # Loose accomplishment outside a header — create orphan entry
            if not current:
                current = {'title': '', 'company': '', 'duration': '',
                           'achievements': []}
            current['achievements'].append(line.lstrip('•·-* ').strip())

    if current:
        entries.append(current)

    return entries


def _parse_title_company(text: str) -> Tuple[str, str]:
    at_m = re.match(r'^(.+?)\s+at\s+(.+)$', text, re.IGNORECASE)
    if at_m:
        return at_m.group(1).strip(), at_m.group(2).strip()

    sep_m = re.match(r'^(.+?)\s*[|–\-]\s*(.+)$', text)
    if sep_m:
        left, right = sep_m.group(1).strip(), sep_m.group(2).strip()
        role_kw = ['intern', 'engineer', 'developer', 'analyst',
                   'manager', 'lead', 'consultant']
        if any(kw in right.lower() for kw in role_kw):
            return right, left
        return left, right

    return text, ''


def _extract_experience(
    section_blocks: Dict[str, str],
    contact: Dict[str, str],
    summary: str,
) -> List[Dict[str, Any]]:
    block = section_blocks.get('experience', '').strip()
    if not block:
        return []

    raw_entries = _parse_exp_block(block)
    validated: List[Dict[str, Any]] = []

    for exp in raw_entries:
        title   = exp.get('title', '')
        company = exp.get('company', '')
        combined = f"{title} {company}".lower()

        if contact.get('phone') and contact['phone'] in combined:
            continue
        if contact.get('email') and contact['email'].lower() in combined:
            continue
        if _SUMMARY_SENTENCE_RE.search(title):
            continue
        if any(kw in title.lower() for kw in _EDUCATION_KEYWORDS):
            continue
        if not title.strip() and not company.strip() and not exp['achievements']:
            continue

        validated.append(exp)

    return validated


# ---------------------------------------------------------------------------
# Pass 7 – Education extraction
# ---------------------------------------------------------------------------

def _extract_education(
    section_blocks: Dict[str, str],
    fallback_text: str,
) -> List[str]:
    text = section_blocks.get('education', '') or fallback_text
    results: List[str] = []
    for line in text.split('\n'):
        s = line.strip()
        if not s:
            continue
        if any(kw in s.lower() for kw in _EDUCATION_KEYWORDS):
            if s not in results:
                results.append(s)
    return results


# ---------------------------------------------------------------------------
# Certifications extraction
# ---------------------------------------------------------------------------

def _extract_certifications(
    section_blocks: Dict[str, str],
    fallback_text: str,
) -> List[str]:
    block = section_blocks.get('certifications', '').strip()
    certs: List[str] = []
    cert_kw_re = re.compile(
        r'\b(certified|certification|certificate|course|training|'
        r'completion|award|achievement|hackathon|competition)\b',
        re.IGNORECASE,
    )

    if block:
        for line in block.split('\n'):
            s = line.strip().lstrip('•·-* ')
            if s and len(s) > 3:
                certs.append(s)
    else:
        for line in fallback_text.split('\n'):
            s = line.strip()
            if s and cert_kw_re.search(s) and len(s) > 10:
                certs.append(s)

    return certs
