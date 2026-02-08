import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download NLTK resources
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Page Setup
st.set_page_config(
    page_title="ATS Resume Analyzer",
    page_icon="𖤓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'STXihei', 'Microsoft YaHei', 'Inter', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        border: 1px solid #404040;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0;
    }
    
    .logo {
        width: 45px;
        height: 45px;
        background: linear-gradient(135deg, #c0c0c0 0%, #808080 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        box-shadow: 0 4px 16px rgba(192, 192, 192, 0.3);
        flex-shrink: 0;
    }
    
    .title-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .main-title {
        font-size: 1.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #e0e0e0 0%, #a0a0a0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
        font-family: 'STXihei', 'Microsoft YaHei', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        line-height: 1.2;
    }
    
    .subtitle {
        color: #a0a0a0;
        font-size: 0.85rem;
        font-weight: 400;
        margin-top: 0.25rem;
        line-height: 1.3;
    }
    
    /* Card styles */
    .custom-card {
        background: rgba(40, 40, 40, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid #404040;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
    }
    
    /* Section Analysis Cards */
    .section-card {
        background: rgba(40, 40, 40, 0.8);
        border: 1px solid #404040;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .section-card:hover {
        border-color: #606060;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #404040;
    }
    
    .section-icon {
        font-size: 1.5rem;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        background: rgba(128, 128, 128, 0.2);
    }
    
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e0e0e0;
        margin: 0;
    }
    
    .section-status {
        margin-left: auto;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .status-missing {
        background: rgba(96, 96, 96, 0.3);
        color: #ff8080;
        border: 1px solid rgba(255, 128, 128, 0.3);
    }
    
    .status-weak {
        background: rgba(144, 144, 144, 0.2);
        color: #ffd080;
        border: 1px solid rgba(255, 208, 128, 0.3);
    }
    
    .status-good {
        background: rgba(192, 192, 192, 0.2);
        color: #80ff80;
        border: 1px solid rgba(128, 255, 128, 0.3);
    }
    
    .section-content {
        color: #b0b0b0;
        line-height: 1.7;
        font-size: 1rem;
    }
    
    .missing-items {
        background: rgba(60, 40, 40, 0.5);
        border-left: 3px solid #ff6b6b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 6px;
    }
    
    .missing-items-title {
        color: #ff8080;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 1.05rem;
    }
    
    .keyword-tag {
        display: inline-block;
        background: rgba(96, 96, 96, 0.4);
        color: #c0c0c0;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        margin: 0.35rem;
        font-size: 0.95rem;
        border: 1px solid rgba(128, 128, 128, 0.3);
    }
    
    .recommendation-box {
        background: rgba(60, 60, 40, 0.4);
        border-left: 3px solid #c0c0c0;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 6px;
    }
    
    .recommendation-title {
        color: #c0c0c0;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 1.05rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a1a 0%, #0a0a0a 100%);
        border-right: 1px solid #404040;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e0e0e0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #808080 0%, #505050 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(128, 128, 128, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(128, 128, 128, 0.4);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(40, 40, 40, 0.4);
        border: 2px dashed #404040;
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
        min-height: 235px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #808080;
        background: rgba(128, 128, 128, 0.1);
    }
    
    /* Text area */
    .stTextArea textarea {
        background: rgba(40, 40, 40, 0.6);
        border: 1px solid #404040;
        border-radius: 12px;
        color: #e0e0e0;
        font-size: 0.95rem;
        padding: 1rem;
        min-height: 235px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #808080;
        box-shadow: 0 0 0 1px #808080;
    }
    
    /* Ensure both columns have same height */
    [data-testid="column"] {
        display: flex;
        flex-direction: column;
    }
    
    [data-testid="column"] > div {
        flex: 1;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #e0e0e0 0%, #a0a0a0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Success/Warning/Info boxes */
    .stSuccess, .stWarning, .stInfo {
        background: rgba(40, 40, 40, 0.6);
        border-radius: 12px;
        border-left: 4px solid;
        padding: 1rem 1.5rem;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #404040;
    }
    
    /* Subheader */
    .stMarkdown h3 {
        color: #e0e0e0;
        font-weight: 600;
        letter-spacing: -0.3px;
    }
    
    /* Info boxes in sidebar */
    .element-container .stMarkdown .stAlert {
        background: rgba(40, 40, 40, 0.8);
        border-radius: 12px;
        border: 1px solid #404040;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #808080 !important;
    }
    
    /* Feature cards */
    .feature-card {
        background: rgba(128, 128, 128, 0.1);
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    
    .feature-card h4 {
        color: #c0c0c0;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-card p {
        color: #a0a0a0;
        font-size: 0.9rem;
        margin: 0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(40, 40, 40, 0.6);
        border-radius: 8px;
        color: #e0e0e0 !important;
        font-weight: 600;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(60, 60, 60, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# Header with logo
st.markdown("""
<div class="main-header">
    <div class="logo-container">
        <div class="logo">𖤓</div>
        <div class="title-container">
            <h1 class="main-title">ATS Resume Analyzer</h1>
            <p class="subtitle">AI-Powered Resume Optimization Platform</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⓘ About This Tool")
    st.markdown("""
    <div class="feature-card">
        <h4>Smart Analysis</h4>
        <p>Advanced TF-IDF algorithm measures resume-job alignment</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4>Section Detection</h4>
        <p>Identifies missing or weak sections in your resume</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4>Keyword Extraction</h4>
        <p>Identifies critical missing terms from job descriptions</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4>Instant Feedback</h4>
        <p>Get actionable insights to improve your resume</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### ⚙ How It Works")
    st.markdown("""
    1. **Upload** your resume in PDF format
    2. **Paste** the target job description
    3. **Analyze** with our AI algorithm
    4. **Review** section-by-section feedback
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #808080; font-size: 2.85rem;'>
        <p>Powered by Machine Learning</p>
        <p style='color: #404040; font-size: 0.75rem;'>Version 2.0</p>
    </div>
    """, unsafe_allow_html=True)

# Helper functions
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text = text + page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = set()
    words = word_tokenize(text)
    return " ".join([word for word in words if word not in stop_words])

def calculate_similarity(resume_text, job_description):
    resume_processed = remove_stopwords(clean_text(resume_text))
    job_processed = remove_stopwords(clean_text(job_description))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_processed, job_processed])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    return round(score, 2), resume_processed, job_processed

def extract_section(text, section_patterns):
    """Extract a section from text based on patterns"""
    text_lower = text.lower()
    for pattern in section_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL)
        if match:
            # Try to get content until next section or reasonable length
            start = match.end()
            # Common section headers to stop at
            next_section = re.search(r'\n\s*(experience|education|skills|projects|certifications?|awards?|publications?|summary|objective)\s*[:\n]', 
                                    text_lower[start:], re.IGNORECASE)
            if next_section:
                end = start + next_section.start()
            else:
                end = start + 500  # Default length
            return text[start:end].strip()
    return ""

def extract_skills(text):
    """Extract skills from text"""
    # Common skill keywords
    skill_patterns = [
        r'skills?\s*[:\n](.+?)(?=\n\s*[a-z]+\s*:|$)',
        r'technical skills?\s*[:\n](.+?)(?=\n\s*[a-z]+\s*:|$)',
        r'core competencies\s*[:\n](.+?)(?=\n\s*[a-z]+\s*:|$)',
    ]
    skills_text = extract_section(text, skill_patterns)
    if skills_text:
        # Extract individual skills (comma or newline separated)
        skills = re.split(r'[,\n•\-\|]', skills_text)
        return [s.strip() for s in skills if len(s.strip()) > 2]
    return []

def extract_technologies(text):
    """Extract programming languages and technologies"""
    tech_keywords = [
        'python', 'java', 'javascript', 'typescript', 'c\\+\\+', 'c#', 'ruby', 'go', 'rust', 'swift',
        'react', 'angular', 'vue', 'node', 'express', 'django', 'flask', 'spring', 'bootstrap',
        'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'ci/cd',
        'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn',
        'html', 'css', 'sass', 'less', 'webpack', 'babel',
        'rest', 'graphql', 'api', 'microservices', 'agile', 'scrum'
    ]
    
    text_lower = text.lower()
    found_tech = []
    for tech in tech_keywords:
        if re.search(r'\b' + tech + r'\b', text_lower):
            found_tech.append(tech)
    return found_tech

def extract_education(text):
    """Extract education information"""
    education_patterns = [
        r'education\s*[:\n](.+?)(?=\n\s*[a-z]+\s*:|$)',
        r'academic background\s*[:\n](.+?)(?=\n\s*[a-z]+\s*:|$)',
    ]
    return extract_section(text, education_patterns)

def extract_experience_years(text):
    """Try to determine years of experience"""
    # Look for year patterns
    year_patterns = re.findall(r'\b(19|20)\d{2}\b', text)
    if len(year_patterns) >= 2:
        years = [int(y) for y in year_patterns]
        return max(years) - min(years)
    
    # Look for explicit mentions
    exp_match = re.search(r'(\d+)\+?\s*years?\s*(of)?\s*experience', text.lower())
    if exp_match:
        return int(exp_match.group(1))
    
    return None

def extract_location(text):
    """Extract location information"""
    # Common location patterns
    location_patterns = [
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2})\b',  # City, ST
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+area\b',  # City area
        r'location[:\s]+([^\n]+)',
        r'based in[:\s]+([^\n]+)',
    ]
    
    for pattern in location_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None

def analyze_sections(resume_text, job_description):
    """Perform detailed section-by-section analysis"""
    
    sections_analysis = []
    
    # 1. Skills Analysis
    resume_skills = set([s.lower() for s in extract_skills(resume_text)])
    job_skills = set([s.lower() for s in extract_skills(job_description)])
    resume_tech = set(extract_technologies(resume_text))
    job_tech = set(extract_technologies(job_description))
    
    missing_skills = job_skills - resume_skills
    missing_tech = job_tech - resume_tech
    
    if missing_skills or missing_tech:
        status = "missing" if len(missing_skills) + len(missing_tech) > 5 else "weak"
    else:
        status = "good"
    
    # Generate skills section rewrite
    skills_rewrite = generate_skills_section_rewrite(list(missing_skills) + list(missing_tech), 
                                                      list(resume_tech))
    
    sections_analysis.append({
        'icon': '⚙️',
        'title': 'Skills & Technologies',
        'status': status,
        'missing': list(missing_skills) + list(missing_tech),
        'recommendation': 'Add these missing skills to your resume if you have experience with them. Place them prominently in a Skills section.' if missing_skills or missing_tech else 'Your skills section aligns well with the job requirements.',
        'skills_rewrite': skills_rewrite
    })
    
    # 2. Education Analysis
    resume_education = extract_education(resume_text)
    job_education = extract_education(job_description)
    
    if job_education:
        edu_keywords = ['bachelor', 'master', 'phd', 'degree', 'bs', 'ms', 'ba', 'ma', 'mba']
        job_edu_reqs = [k for k in edu_keywords if k in job_education.lower()]
        resume_has_edu = any(k in resume_education.lower() for k in edu_keywords) if resume_education else False
        
        if job_edu_reqs and not resume_has_edu:
            status = "missing"
            recommendation = "The job requires specific educational qualifications. Make sure your Education section is clearly visible and matches the requirements."
        elif job_edu_reqs and resume_has_edu:
            status = "good"
            recommendation = "Your education qualifications are present and appear to meet the requirements."
        else:
            status = "good"
            recommendation = "Education section looks adequate."
    else:
        status = "good"
        recommendation = "No specific education requirements detected in job description."
    
    sections_analysis.append({
        'icon': '🎓',
        'title': 'Education',
        'status': status,
        'missing': job_edu_reqs if job_education and not resume_has_edu else [],
        'recommendation': recommendation
    })
    
    # 3. Experience Level Analysis
    resume_exp_years = extract_experience_years(resume_text)
    job_exp_years = extract_experience_years(job_description)
    
    if job_exp_years and resume_exp_years:
        if resume_exp_years < job_exp_years:
            status = "weak"
            recommendation = f"The job requires {job_exp_years}+ years of experience. Emphasize your {resume_exp_years} years and highlight relevant accomplishments to bridge the gap."
        else:
            status = "good"
            recommendation = f"Your {resume_exp_years} years of experience meets or exceeds the {job_exp_years} years requirement."
    elif job_exp_years and not resume_exp_years:
        status = "missing"
        recommendation = f"The job requires {job_exp_years}+ years of experience. Make sure this is clearly stated in your resume summary or experience section."
    else:
        status = "good"
        recommendation = "Experience level appears adequate."
    
    sections_analysis.append({
        'icon': '💼',
        'title': 'Experience Level',
        'status': status,
        'missing': [f"{job_exp_years}+ years experience"] if job_exp_years and (not resume_exp_years or resume_exp_years < job_exp_years) else [],
        'recommendation': recommendation
    })
    
    # 4. Location Analysis
    resume_location = extract_location(resume_text)
    job_location = extract_location(job_description)
    
    if job_location:
        if resume_location:
            # Basic check if locations are compatible
            if any(word in resume_location.lower() for word in job_location.lower().split()):
                status = "good"
                recommendation = "Your location aligns with the job location."
            else:
                status = "weak"
                recommendation = f"Job location: {job_location}. Your resume shows: {resume_location}. Clarify if you're willing to relocate or work remotely."
        else:
            status = "missing"
            recommendation = f"The job specifies location: {job_location}. Consider adding your location or relocation preferences to your resume."
    else:
        status = "good"
        recommendation = "No specific location requirements detected."
    
    sections_analysis.append({
        'icon': '📍',
        'title': 'Location',
        'status': status,
        'missing': [job_location] if job_location and not resume_location else [],
        'recommendation': recommendation
    })
    
    # 5. Keywords Analysis with rewrite suggestions
    job_words = set(job_description.lower().split())
    resume_words = set(resume_text.lower().split())
    
    # Common important keywords
    important_keywords = ['leadership', 'management', 'team', 'project', 'strategic', 'analysis', 
                         'development', 'communication', 'collaboration', 'innovation', 'agile',
                         'scrum', 'data', 'research', 'optimization', 'performance', 'quality']
    
    missing_important = [kw for kw in important_keywords if kw in job_words and kw not in resume_words]
    
    if len(missing_important) > 3:
        status = "weak"
    elif len(missing_important) > 0:
        status = "weak"
    else:
        status = "good"
    
    # Generate rewrite examples for missing keywords
    rewrite_examples = generate_keyword_rewrites(missing_important[:5])
    
    sections_analysis.append({
        'icon': '🔑',
        'title': 'Important Keywords',
        'status': status,
        'missing': missing_important[:10],  # Top 10
        'recommendation': 'Incorporate these keywords naturally throughout your resume, especially in your experience descriptions and summary.' if missing_important else 'Good keyword coverage detected.',
        'rewrite_examples': rewrite_examples
    })
    
    return sections_analysis

def generate_keyword_rewrites(missing_keywords):
    """Generate example bullet points incorporating missing keywords"""
    
    keyword_templates = {
        'leadership': [
            "Demonstrated **leadership** by mentoring a team of 5 junior developers, improving code quality by 40%",
            "Provided technical **leadership** in architecting scalable solutions for enterprise clients",
            "Exercised **leadership** in driving cross-functional initiatives that reduced deployment time by 60%"
        ],
        'management': [
            "Project **management** of end-to-end development lifecycle for 3+ concurrent applications",
            "Resource **management** and task allocation across distributed teams to meet tight deadlines",
            "**Managed** stakeholder relationships and gathered requirements for critical business systems"
        ],
        'team': [
            "Collaborated with cross-functional **teams** including designers, PMs, and QA engineers",
            "Led **team** code reviews and established best practices for version control",
            "Worked within an agile **team** environment to deliver features in 2-week sprints"
        ],
        'project': [
            "Delivered 10+ **projects** on time and under budget, serving 100K+ users",
            "Spearheaded **project** to migrate legacy systems to modern cloud infrastructure",
            "**Project** contributions resulted in 35% improvement in application performance"
        ],
        'strategic': [
            "Contributed to **strategic** planning for technology roadmap and architecture decisions",
            "Made **strategic** technical choices that reduced infrastructure costs by 25%",
            "Developed **strategic** partnerships with vendors to enhance product capabilities"
        ],
        'analysis': [
            "Conducted technical **analysis** of system bottlenecks and implemented optimization strategies",
            "Performed data **analysis** to identify user behavior patterns and improve UX",
            "**Analyzed** business requirements and translated them into technical specifications"
        ],
        'development': [
            "Full-stack **development** using React, Node.js, and PostgreSQL for e-commerce platform",
            "Led **development** efforts for mobile-responsive web applications",
            "Drove **development** of RESTful APIs serving 1M+ requests daily"
        ],
        'communication': [
            "Strong **communication** with stakeholders to align technical solutions with business goals",
            "Effective **communication** of complex technical concepts to non-technical audiences",
            "Regular **communication** through documentation, presentations, and team meetings"
        ],
        'collaboration': [
            "**Collaboration** with product teams to define features and prioritize backlog items",
            "Fostered **collaboration** between frontend and backend teams for seamless integration",
            "**Collaborated** with DevOps to implement CI/CD pipelines and automated testing"
        ],
        'innovation': [
            "Drove **innovation** by introducing modern frameworks that improved development velocity",
            "**Innovative** problem-solving approach reduced critical bug resolution time by 50%",
            "Championed **innovation** through hackathons and proof-of-concept projects"
        ],
        'agile': [
            "**Agile** development methodology with daily standups, sprint planning, and retrospectives",
            "Participated in **agile** ceremonies and contributed to continuous improvement initiatives",
            "**Agile** approach to iterative development and rapid feature deployment"
        ],
        'scrum': [
            "Active participant in **Scrum** framework including sprint planning and backlog refinement",
            "**Scrum** team member contributing to sprint goals and velocity improvements",
            "Followed **Scrum** best practices for transparent and efficient project delivery"
        ],
        'data': [
            "**Data**-driven decision making through analytics, monitoring, and A/B testing",
            "Worked with large **data** sets using SQL, ETL processes, and data visualization tools",
            "**Data** pipeline development for real-time analytics and reporting"
        ],
        'research': [
            "**Research** and evaluation of emerging technologies to enhance product offerings",
            "Conducted **research** on industry best practices and competitive analysis",
            "**Research** initiatives led to adoption of new tools improving team productivity"
        ],
        'optimization': [
            "Performance **optimization** resulting in 3x faster page load times",
            "Database **optimization** and query tuning for improved application responsiveness",
            "**Optimized** algorithms reducing computational complexity from O(n²) to O(n log n)"
        ],
        'performance': [
            "Enhanced application **performance** through code refactoring and caching strategies",
            "Monitored **performance** metrics and implemented improvements based on analytics",
            "**Performance** tuning of backend services handling high-volume traffic"
        ],
        'quality': [
            "Ensured code **quality** through comprehensive unit testing and code reviews",
            "Improved **quality** assurance processes with automated testing frameworks",
            "**Quality**-focused development with test coverage above 85%"
        ]
    }
    
    examples = []
    for keyword in missing_keywords:
        if keyword in keyword_templates:
            import random
            examples.append({
                'keyword': keyword,
                'example': random.choice(keyword_templates[keyword])
            })
    
    return examples

def generate_skills_section_rewrite(missing_items, existing_tech):
    """Generate a complete skills section incorporating missing items"""
    if not missing_items:
        return None
    
    # Categorize skills
    categories = {
        'Languages': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'rust', 'swift', 'php', 'kotlin'],
        'Frontend': ['react', 'angular', 'vue', 'html', 'css', 'sass', 'less', 'webpack', 'babel', 'next.js', 'svelte'],
        'Backend': ['node', 'express', 'django', 'flask', 'spring', 'fastapi', 'rails', 'asp.net'],
        'Databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'dynamodb', 'cassandra'],
        'Cloud & DevOps': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'ci/cd', 'terraform', 'ansible'],
        'Tools & Frameworks': ['git', 'jira', 'confluence', 'rest', 'graphql', 'api', 'microservices'],
        'Methodologies': ['agile', 'scrum', 'kanban', 'tdd', 'bdd']
    }
    
    # Organize missing and existing items by category
    organized = {}
    all_items = set(missing_items + existing_tech)
    
    for category, keywords in categories.items():
        items_in_category = [item for item in all_items if item.lower() in [k.lower() for k in keywords]]
        if items_in_category:
            # Mark which are missing
            organized[category] = {
                'items': items_in_category,
                'missing': [item for item in items_in_category if item in missing_items]
            }
    
    return organized

# Main app
def main():
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📄 Upload Resume")
        uploaded_file = st.file_uploader(
            "Drag and drop your resume here",
            type=['pdf'],
            help="Upload your resume in PDF format for analysis",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### 💼 Job Description")
        job_description = st.text_area(
            "Paste the complete job description",
            height=235,
            placeholder="Copy and paste the job posting here, including requirements, responsibilities, and qualifications...",
            label_visibility="collapsed"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Centered analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Resume Match", use_container_width=True)
    
    if analyze_button:
        if not uploaded_file:
            st.warning("⚠️ Please upload your resume to continue")
            return
        if not job_description:
            st.warning("⚠️ Please paste the job description to continue")
            return
        
        with st.spinner("👀 Analyzing your resume with AI..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            if not resume_text:
                st.error("❌ Could not extract text from PDF. Please try another file.")
                return 
            
            # Calculate similarity
            similarity_score, resume_processed, job_processed = calculate_similarity(resume_text, job_description)
            
            st.markdown("---")
            st.markdown("## 📈 Analysis Results")
            
            # Score display
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.metric("Match Score", f"{similarity_score:.1f}%", delta=None)
            
            # Visual gauge
            fig, ax = plt.subplots(figsize=(10, 1.5))
            fig.patch.set_facecolor('#1a1a1a')
            ax.set_facecolor('#1a1a1a')
            
            # Color scheme based on score
            if similarity_score < 40:
                color = '#606060'
                label = 'Needs Improvement'
            elif similarity_score < 70:
                color = '#909090'
                label = 'Good Match'
            else:
                color = '#c0c0c0'
                label = 'Excellent Match'
            
            ax.barh([0], [similarity_score], color=color, height=0.6, alpha=0.8)
            ax.barh([0], [100-similarity_score], left=[similarity_score], 
                   color='#2a2a2a', height=0.6, alpha=0.3)
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlabel('Match Percentage', color='#a0a0a0', fontsize=11)
            ax.tick_params(colors='#a0a0a0')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#404040')
            ax.set_yticks([])
            ax.grid(axis='x', alpha=0.2, color='#404040')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Feedback based on score
            st.markdown("<br>", unsafe_allow_html=True)
            
            if similarity_score < 40:
                st.error(f"""
                **{label}** - Your resume needs significant optimization to match this role.
                
                **Recommendations:**
                - Incorporate more keywords from the job description
                - Highlight relevant skills and experiences
                - Tailor your resume specifically for this position
                """)
            elif similarity_score < 70:
                st.info(f"""
                **{label}** - Your resume aligns fairly well with the job requirements.
                
                **Recommendations:**
                - Add more specific keywords from the job posting
                - Emphasize achievements related to the role
                - Fine-tune your experience descriptions
                """)
            else:
                st.success(f"""
                **{label}** - Your resume strongly aligns with this job posting!
                
                **You're on the right track:**
                - Strong keyword alignment detected
                - Relevant experience highlighted
                - Consider minor refinements for perfection
                """)
            
            # Section-by-Section Analysis
            st.markdown("---")
            st.markdown("## 🔍 Section-by-Section Analysis")
            st.markdown("*Detailed breakdown of what needs attention in your resume*")
            
            sections = analyze_sections(resume_text, job_description)
            
            for section in sections:
                status_class = f"status-{section['status']}"
                status_text = {
                    'missing': '❌ Missing',
                    'weak': '⚠️ Needs Work',
                    'good': '✓ Good'
                }[section['status']]
                
                st.markdown(f"""
                <div class="section-card">
                    <div class="section-header">
                        <div class="section-icon">{section['icon']}</div>
                        <h3 class="section-title">{section['title']}</h3>
                        <span class="section-status {status_class}">{status_text}</span>
                    </div>
                    <div class="section-content">
                        <div class="recommendation-box">
                            <div class="recommendation-title">💡 Recommendation</div>
                            <p>{section['recommendation']}</p>
                        </div>
                """, unsafe_allow_html=True)
                
                # Special handling for Skills & Technologies section
                if section['title'] == 'Skills & Technologies' and section.get('skills_rewrite'):
                    st.markdown("""
                        <div class="missing-items">
                            <div class="missing-items-title">✏️ Enhanced Skills Section:</div>
                            <p style='color: #909090; font-size: 1.45rem; margin-top: 0.5rem; margin-bottom: 1rem;'>
                                Here's how to organize your skills section. Items in <strong style='color: #c0c0c0;'>bold</strong> are missing from your resume.
                            </p>
                    """, unsafe_allow_html=True)
                    
                    for category, data in section['skills_rewrite'].items():
                        items_html = []
                        for item in data['items']:
                            if item in data['missing']:
                                items_html.append(f"<strong style='color: #c0c0c0;'>{item}</strong>")
                            else:
                                items_html.append(f"<span style='color: #909090;'>{item}</span>")
                        
                        st.markdown(f"""
                            <div style='background: rgba(50, 50, 50, 0.5); padding: 1rem 1.25rem; margin: 0.75rem 0; border-radius: 8px;'>
                                <div style='color: #a0a0a0; font-weight: 600; font-size: 1rem; margin-bottom: 0.75rem;'>
                                    {category}
                                </div>
                                <div style='color: #b0b0b0; line-height: 2; font-size: 1rem;'>
                                    {' • '.join(items_html)}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Special handling for Important Keywords section - show rewrite examples
                elif section['title'] == 'Important Keywords' and section.get('rewrite_examples'):
                    st.markdown("""
                        <div class="missing-items">
                            <div class="missing-items-title">✏️ Example Bullet Points to Add:</div>
                            <p style='color: #909090; font-size: 0.95rem; margin-top: 0.5rem; margin-bottom: 1rem;'>
                                Copy and adapt these examples to your actual experience. Replace generic details with your specific achievements.
                            </p>
                    """, unsafe_allow_html=True)
                    
                    for example in section['rewrite_examples']:
                        # Highlight the keyword in the example
                        example_text = example['example'].replace(f"**{example['keyword']}**", 
                                                                  f"<strong style='color: #c0c0c0;'>{example['keyword']}</strong>")
                        st.markdown(f"""
                            <div style='background: rgba(50, 50, 50, 0.5); padding: 1rem 1.25rem; margin: 0.75rem 0; border-radius: 8px; border-left: 3px solid #808080;'>
                                <span style='color: #707070; font-size: 0.8rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;'>
                                    {example['keyword']}
                                </span>
                                <p style='color: #b0b0b0; margin: 0.5rem 0 0 0; line-height: 1.7; font-size: 1rem;'>
                                    • {example_text}
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # For other sections, show missing items as tags
                elif section['missing'] and section['title'] not in ['Important Keywords', 'Skills & Technologies']:
                    st.markdown("""
                        <div class="missing-items">
                            <div class="missing-items-title">Missing or Weak Elements:</div>
                    """, unsafe_allow_html=True)
                    
                    # Display as tags
                    tags_html = "".join([f'<span class="keyword-tag">{item}</span>' for item in section['missing'][:15]])
                    st.markdown(tags_html, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Additional insights
            st.markdown("---")
            st.markdown("### 💡 Pro Tips")
            
            tip_col1, tip_col2 = st.columns(2)
            
            with tip_col1:
                st.markdown("""
                <div class="custom-card">
                    <h4 style='color: #c0c0c0;'>✓ ATS Best Practices</h4>
                    <ul style='color: #a0a0a0;'>
                        <li>Use standard fonts (Arial, Calibri)</li>
                        <li>Avoid tables and text boxes</li>
                        <li>Include keywords naturally</li>
                        <li>Use clear section headings</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with tip_col2:
                st.markdown("""
                <div class="custom-card">
                    <h4 style='color: #c0c0c0;'>⚡ Quick Wins</h4>
                    <ul style='color: #a0a0a0;'>
                        <li>Match your skills to requirements</li>
                        <li>Quantify your achievements</li>
                        <li>Use action verbs</li>
                        <li>Keep format simple and clean</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()

>>>>>>> dca1c5cd8c47cf0752237bd5939f2b2da36defdd
