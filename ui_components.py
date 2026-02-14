"""
UI Components for ATS Resume Analyzer
Streamlit UI components and styling
"""

import streamlit as st
from config import APP_TITLE, APP_SUBTITLE, APP_ICON, APP_VERSION


def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app"""
    st.markdown("""
    <style>
        /* Main container */
        .main {
            background-color: #0E1117;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #FAFAFA;
        }
        
        /* Score cards */
        .score-label {
            font-size: 14px;
            color: #8B8B8B;
            margin-bottom: 5px;
        }
        
        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 2.5rem;
            font-weight: bold;
        }
        
        /* Status badges */
        .status-good {
            background-color: #00D4AA;
            color: #000;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .status-weak {
            background-color: #FFA500;
            color: #000;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .status-missing {
            background-color: #FF6B6B;
            color: #FFF;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        
        /* Section cards */
        .section-card {
            background-color: #1E1E1E;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            border-left: 4px solid #00D4AA;
        }
        
        /* Bold keywords in recommendations */
        .section-card strong {
            color: #00D4AA;
            font-weight: 700;
        }
        
        /* Buttons */
        .stButton>button {
            background-color: #00D4AA;
            color: #000;
            font-weight: 600;
            border-radius: 8px;
            padding: 12px 24px;
            border: none;
        }
        
        .stButton>button:hover {
            background-color: #00B894;
        }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the application header"""
    st.markdown(f"""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 3rem; margin-bottom: 10px;">{APP_ICON} {APP_TITLE}</h1>
        <p style="color: #8B8B8B; font-size: 1.1rem;">{APP_SUBTITLE}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")


def render_sidebar():
    """Render the sidebar with app information"""
    with st.sidebar:
        st.markdown(f"### {APP_ICON} About")
        st.info(
            f"**{APP_TITLE}** v{APP_VERSION}\n\n"
            "An AI-powered tool to analyze and optimize your resume for "
            "Applicant Tracking Systems (ATS)."
        )
        
        st.markdown("### 📊 How It Works")
        st.markdown("""
        1. **Upload** your resume in PDF format
        2. **Paste** the job description
        3. **Analyze** to get your match score
        4. **Improve** based on recommendations
        """)
        
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Use a clean, ATS-friendly resume format
        - Include relevant keywords from the job description
        - Quantify your achievements
        - Keep formatting simple
        """)


def render_section_card(section):
    """
    Render a section analysis card.
    
    Args:
        section (dict): Section analysis data containing:
            - icon: Emoji icon
            - title: Section name
            - status: 'good', 'weak', or 'missing'
            - recommendation: Recommendation text
            - missing: List of missing elements
    """
    status = section.get('status', 'missing')
    icon = section.get('icon', '📌')
    title = section.get('title', 'Section')
    recommendation = section.get('recommendation', 'No recommendation available.')
    missing = section.get('missing', [])
    
    # Convert markdown bold (**text**) to HTML bold (<strong>text</strong>)
    import re
    recommendation_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', recommendation)
    
    # Determine status badge
    if status == 'good':
        status_html = '<span class="status-good">✓ Good</span>'
        border_color = "#00D4AA"
    elif status == 'weak':
        status_html = '<span class="status-weak">⚠ Needs Work</span>'
        border_color = "#FFA500"
    else:  # missing
        status_html = '<span class="status-missing">✗ Missing</span>'
        border_color = "#FF6B6B"
    
    st.markdown(f"""
    <div class="section-card" style="background-color: #1E1E1E; border-radius: 10px; padding: 20px; margin: 15px 0; border-left: 4px solid {border_color};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <h3 style="margin: 0; font-size: 1.3rem;">{icon} {title}</h3>
            {status_html}
        </div>
        <div style="background-color: #2A2A2A; border-radius: 8px; padding: 15px; margin: 10px 0;">
            <div style="color: #FFA500; font-weight: 600; margin-bottom: 8px;">💡 Recommendation</div>
            <p style="color: #CCCCCC; margin: 0;">
                <style>
                    .section-card strong {{
                        color: #00D4AA;
                        font-weight: 700;
                    }}
                </style>
                {recommendation_html}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show missing elements if any
    if missing and len(missing) > 0:
        st.markdown("**Missing or Weak Elements:**")
        for item in missing[:5]:  # Show max 5 items
            if item:  # Only show non-empty items
                st.markdown(f"• {item}")


def render_pro_tips():
    """Render professional tips section"""
    st.markdown("## 💡 Pro Tips")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ Do's
        - Use action verbs (Built, Developed, Managed)
        - Quantify achievements with numbers
        - Match keywords from job description
        - Keep format clean and ATS-friendly
        - Use standard section headings
        """)
    
    with col2:
        st.markdown("""
        ### ❌ Don'ts
        - Don't use images or graphics
        - Avoid headers/footers
        - Don't use tables for layout
        - Avoid uncommon fonts
        - Don't use text boxes
        """)
    
    st.info(
        "💡 **Remember:** Most companies use ATS to filter resumes before human review. "
        "Make sure your resume is optimized for both systems!"
    )


def render_score_card(score, label, delta=None):
    """
    Render a score card with optional delta.
    
    Args:
        score (float): The score to display
        label (str): Label for the score
        delta (float, optional): Delta value to show
    """
    st.markdown(f'<div class="score-label">{label}</div>', unsafe_allow_html=True)
    if delta is not None:
        st.metric("", f"{score:.1f}%", delta=f"+{delta:.1f}%")
    else:
        st.metric("", f"{score:.1f}%")