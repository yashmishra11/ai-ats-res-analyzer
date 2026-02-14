"""
Similarity calculation and scoring utilities
"""

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import IMPORTANT_KEYWORDS, SCORE_WEIGHTS, JOB_TYPE_KEYWORDS
from feature_extractors import extract_skills, extract_technologies, normalize_skill
from text_extractors import clean_text

def calculate_expected_score(current_score, sections_analysis):
    """
    Calculate the expected score after improving missing/weak sections.
    Args:
        current_score (float): Current similarity score.
        sections_analysis (list): List of section analysis dictionaries.
    Returns:
        tuple: (expected_score, total_gain)
    """
    from config import IMPROVEMENT_POTENTIAL

    total_gain = 0
    for section in sections_analysis:
        if section['status'] == 'missing':
            total_gain += IMPROVEMENT_POTENTIAL.get('section_missing', 0)
        elif section['status'] == 'weak':
            total_gain += IMPROVEMENT_POTENTIAL.get('section_weak', 0)

    total_gain = min(total_gain, IMPROVEMENT_POTENTIAL['max_section_improvement'])
    expected = min(
        current_score + total_gain,
        IMPROVEMENT_POTENTIAL['max_overall_score']
    )

    return round(expected, 2), total_gain


# --------------------------------------------------
# Stopword Removal
# --------------------------------------------------
def remove_stopwords(text):
    """
    Remove stopwords from text with fallback tokenization.
    """
    # Get stopwords with fallback
    try:
        stop_words = set(stopwords.words('english'))
    except:
        # Fallback: basic English stopwords if NLTK fails
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
    
    # Tokenize with fallback
    try:
        words = word_tokenize(text)
    except:
        # Fallback: simple word splitting if NLTK fails
        words = text.split()
    
    return " ".join([word for word in words if word.lower() not in stop_words])

# --------------------------------------------------
# Main Similarity Calculation
# --------------------------------------------------
def calculate_similarity(resume_text, job_description, sections=None):
    resume_processed = remove_stopwords(clean_text(resume_text))
    job_processed = remove_stopwords(clean_text(job_description))

    # ---------- TF-IDF ----------
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
    tfidf_matrix = vectorizer.fit_transform([resume_processed, job_processed])
    raw_tfidf = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    if len(job_description.split()) < 150:
        tfidf_score = min(1.0, raw_tfidf * 1.2)
    else:
        tfidf_score = raw_tfidf

    # ---------- Skills ----------
    resume_skills = set([s.lower() for s in extract_skills(resume_text)])
    job_skills = set([s.lower() for s in extract_skills(job_description)])
    resume_tech = set(normalize_skill(t) for t in extract_technologies(resume_text))
    job_tech = set(normalize_skill(t) for t in extract_technologies(job_description))
    all_resume = resume_skills | resume_tech
    all_job = job_skills | job_tech

    if all_job:
        skills_score = len(all_resume & all_job) / len(all_job)
    else:
        skills_score = 0.5

    # Required skills boost
    required_match = re.search(
        r'required skills.*?(?=\n\n|\Z)',
        job_description.lower(),
        re.DOTALL
    )

    if required_match:
        required_text = required_match.group(0)
        required_tech = set(extract_technologies(required_text))

        if required_tech:
            matched_required = required_tech & all_resume
            ratio = len(matched_required) / len(required_tech)
            skills_score *= (1 + ratio * 0.3)

    skills_score = min(skills_score, 1.0)

    # ---------- Important Keywords ----------
    job_words = set(job_description.lower().split())
    resume_words = set(resume_text.lower().split())

    job_imp = set([kw for kw in IMPORTANT_KEYWORDS if kw in job_words])
    resume_imp = set([kw for kw in IMPORTANT_KEYWORDS if kw in resume_words])

    if job_imp:
        keywords_score = len(resume_imp & job_imp) / len(job_imp)
    else:
        keywords_score = 0.5

    # ---------- Section Score ----------
    # Use sections if provided, otherwise create basic score
    if sections:
        section_scores = []
        for s in sections:
            if s['status'] == "missing":
                section_scores.append(0.4)
            elif s['status'] == "weak":
                section_scores.append(0.7)
            else:
                section_scores.append(0.95)

        sections_score = sum(section_scores) / len(section_scores) if section_scores else 0.8
    else:
        sections_score = 0.8

    # ---------- Dynamic Role Weight ----------
    job_type = None
    job_lower = job_description.lower()

    for role, keywords in JOB_TYPE_KEYWORDS.items():
        if any(kw in job_lower for kw in keywords):
            job_type = role
            break

    if job_type == "frontend":
        weights = {'tfidf': 0.25, 'skills': 0.45, 'keywords': 0.20, 'sections': 0.10}
    elif job_type == "backend":
        weights = {'tfidf': 0.35, 'skills': 0.35, 'keywords': 0.20, 'sections': 0.10}
    else:
        weights = SCORE_WEIGHTS

    final_score = (
        tfidf_score * weights['tfidf'] +
        skills_score * weights['skills'] +
        keywords_score * weights['keywords'] +
        sections_score * weights['sections']
    ) * 100

    return round(final_score, 2), resume_processed, job_processed