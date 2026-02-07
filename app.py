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
    page_icon="🎯",
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
    
    /* Score badge */
    .score-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1rem 0;
    }
    
    .score-low {
        background: linear-gradient(135deg, #606060 0%, #404040 100%);
        color: white;
    }
    
    .score-medium {
        background: linear-gradient(135deg, #909090 0%, #707070 100%);
        color: white;
    }
    
    .score-high {
        background: linear-gradient(135deg, #c0c0c0 0%, #a0a0a0 100%);
        color: #1a1a1a;
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
</style>
""", unsafe_allow_html=True)

# Header with logo
st.markdown("""
<div class="main-header">
    <div class="logo-container">
        <div class="logo">🎯</div>
        <div class="title-container">
            <h1 class="main-title">ATS Resume Analyzer</h1>
            <p class="subtitle">AI-Powered Resume Optimization Platform</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 About This Tool")
    st.markdown("""
    <div class="feature-card">
        <h4>Smart Analysis</h4>
        <p>Advanced TF-IDF algorithm measures resume-job alignment</p>
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
    
    st.markdown("### 🚀 How It Works")
    st.markdown("""
    1. **Upload** your resume in PDF format
    2. **Paste** the target job description
    3. **Analyze** with our AI algorithm
    4. **Review** your match score & recommendations
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #808080; font-size: 0.85rem;'>
        <p>Powered by Machine Learning</p>
        <p style='color: #404040; font-size: 0.75rem;'>Version 1.0</p>
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
        
        with st.spinner("🤖 Analyzing your resume with AI..."):
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
    main()
