"""
ATS Resume Analyzer - Main Application
Streamlit-based tool for analyzing resume-job match scores
"""

import streamlit as st
import matplotlib.pyplot as plt

# Import custom modules
from nltk_setup import setup_nltk
from ui_components import (
    apply_custom_css, render_header, render_sidebar,
    render_section_card, render_pro_tips
)
from text_extractors import extract_text_from_pdf
from similarity_calculator import calculate_similarity, calculate_expected_score
from section_analyzer import analyze_sections
from visualization import create_section_impact_chart

# Setup NLTK
setup_nltk()

# Page configuration
st.set_page_config(
    page_title="ATS Resume Analyzer",
    page_icon="𖤓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
apply_custom_css()

# Render header
render_header()

# Render sidebar
render_sidebar()


def main():
    """Main application logic"""
    
    # Input section
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
    
    # Relocation preference section
    st.markdown("### 📍 Relocation Preference (Optional)")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        willing_to_relocate = st.checkbox(
            "I am willing to relocate for this position",
            help="Check this if you're open to relocating to the job location"
        )
    
    with col2:
        not_willing_to_relocate = st.checkbox(
            "I prefer not to relocate",
            help="Check this if you prefer to stay in your current location"
        )
    
    # Determine relocation status
    if willing_to_relocate and not not_willing_to_relocate:
        relocation_preference = True
    elif not_willing_to_relocate and not willing_to_relocate:
        relocation_preference = False
    else:
        relocation_preference = None  # User hasn't specified or contradictory input
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Centered analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Resume Match", use_container_width=True)
    
    # Analysis logic
    if analyze_button:
        if not uploaded_file:
            st.warning("⚠️ Please upload your resume to continue")
            return
        if not job_description:
            st.warning("⚠️ Please paste the job description to continue")
            return
        
        with st.spinner("👀 Analyzing your resume..."):
            # Extract text from PDF
            try:
                resume_text = extract_text_from_pdf(uploaded_file)
            except Exception as e:
                st.error(f"❌ Error reading PDF: {str(e)}")
                return
            
            if not resume_text:
                st.error("❌ Could not extract text from PDF. Please try another file.")
                return 
            
            # Analyze sections (with relocation preference)
            sections = analyze_sections(resume_text, job_description, relocation_preference)
            
            # Calculate similarity (pass sections for section score calculation)
            similarity_score, resume_processed, job_processed = calculate_similarity(resume_text, job_description, sections)
            
            # Calculate expected score
            expected_score, potential_gain = calculate_expected_score(similarity_score, sections)
            
            # Display info message
            st.info(
                "ℹ️ Visual PDFs may affect section extraction. "
                "The analyzer uses semantic fallbacks where possible."
            )
            
            # Results display
            st.markdown("---")
            st.markdown("## 📈 Analysis Results")
            
            # Score display - Current vs Expected
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="score-label">Current Match Score</div>', unsafe_allow_html=True)
                st.metric("", f"{similarity_score:.1f}%", delta=None)
            
            with col2:
                st.markdown('<div class="score-label">Expected After Improvements</div>', unsafe_allow_html=True)
                delta_text = f"+{potential_gain:.1f}%"
                st.metric("", f"{expected_score:.1f}%", delta=delta_text, delta_color="normal")
            
            # Visual line chart showing section-by-section impact
            st.markdown("#### Section-by-Section Impact Analysis")
            
            fig = create_section_impact_chart(sections)
            st.pyplot(fig)
            plt.close()
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Section-by-section analysis
            st.markdown("---")
            st.markdown("## 🔍 Section-by-Section Analysis")
            st.markdown("*Detailed breakdown of what needs attention in your resume*")
            
            # Render section cards
            for section in sections:
                render_section_card(section)
            
            # Pro tips
            st.markdown("---")
            render_pro_tips()


if __name__ == "__main__":
    main()