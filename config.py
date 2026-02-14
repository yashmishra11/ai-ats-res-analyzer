"""
Configuration constants for ATS Resume Analyzer
"""

# Application constants
APP_TITLE = "ATS Resume Analyzer"
APP_SUBTITLE = "Optimize your resume for Applicant Tracking Systems"
APP_ICON = "𖤓"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "AI-powered resume analysis tool that helps you optimize your resume for ATS systems"

# UI Color scheme
PRIMARY_COLOR = "#00D4AA"
SECONDARY_COLOR = "#1E1E1E"
ACCENT_COLOR = "#FF6B6B"
SUCCESS_COLOR = "#00D4AA"
WARNING_COLOR = "#FFA500"
ERROR_COLOR = "#FF6B6B"

# Score thresholds for status badges
SCORE_EXCELLENT_THRESHOLD = 80
SCORE_GOOD_THRESHOLD = 60
SCORE_FAIR_THRESHOLD = 40

# Important keywords that should appear in resume
IMPORTANT_KEYWORDS = [
    'agile', 'scrum', 'team', 'project', 'collaboration', 'communication',
    'leadership', 'problem-solving', 'analytical', 'performance', 'optimization',
    'deployment', 'testing', 'debugging', 'documentation', 'workflow',
    'api', 'database', 'frontend', 'backend', 'fullstack', 'development',
    'design', 'architecture', 'scalable', 'efficient', 'responsive',
    'ci/cd', 'devops', 'cloud', 'security', 'authentication', 'authorization'
]

# Education keywords
EDUCATION_KEYWORDS = [
    'bachelor', 'master', 'phd', 'degree', 'diploma', 'certification',
    'computer science', 'engineering', 'information technology', 'software',
    'b.tech', 'm.tech', 'b.e', 'm.e', 'bsc', 'msc', 'bca', 'mca'
]

# Job type keywords for role detection
JOB_TYPE_KEYWORDS = {
    'frontend': [
        'frontend', 'front-end', 'ui', 'ux', 'react', 'angular', 'vue',
        'html', 'css', 'javascript', 'typescript', 'responsive', 'web design'
    ],
    'backend': [
        'backend', 'back-end', 'server', 'api', 'database', 'sql', 'nosql',
        'node', 'django', 'flask', 'spring', 'microservices', 'rest', 'graphql'
    ],
    'fullstack': [
        'fullstack', 'full-stack', 'full stack', 'mern', 'mean', 'lamp'
    ],
    'devops': [
        'devops', 'sre', 'infrastructure', 'ci/cd', 'docker', 'kubernetes',
        'aws', 'azure', 'gcp', 'jenkins', 'terraform', 'ansible'
    ],
    'data': [
        'data scientist', 'data engineer', 'data analyst', 'machine learning',
        'ml', 'ai', 'deep learning', 'nlp', 'computer vision', 'analytics'
    ]
}

# Score weights for similarity calculation
SCORE_WEIGHTS = {
    'tfidf': 0.30,       # Text similarity
    'skills': 0.40,      # Skills match
    'keywords': 0.20,    # Important keywords
    'sections': 0.10     # Section completeness
}

# Improvement potential settings
IMPROVEMENT_POTENTIAL = {
    'section_missing': 8.0,           # Points gained by adding missing section
    'section_weak': 4.0,              # Points gained by improving weak section
    'max_section_improvement': 25.0,  # Maximum total improvement from sections
    'max_overall_score': 98.0         # Maximum achievable score
}

# Status score mappings for visualization
STATUS_SCORES = {
    'missing': 45,
    'weak': 65,
    'good': 90,
    'present': 90  # Alias for 'good'
}

# Expected scores after improvements
EXPECTED_SCORES_AFTER_FIX = {
    'missing': 75,
    'weak': 85,
    'good': 95
}

# Section display order
SECTION_ORDER = [
    'Skills & Technologies',
    'Education',
    'Experience Level',
    'Location',
    'Important Keywords',
    'Projects'
]

# Section label mapping for display
SECTION_LABEL_MAP = {
    'Skills & Technologies': 'Skills &\nTechnologies',
    'Education': 'Education',
    'Experience Level': 'Experience\nLevel',
    'Location': 'Location',
    'Important Keywords': 'Important\nKeywords',
    'Projects': 'Projects'
}