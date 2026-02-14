"""
Section-by-section resume analysis
Analyzes different sections of resume against job requirements
"""

import re
from config import (
    EDUCATION_KEYWORDS, JOB_TYPE_KEYWORDS, IMPORTANT_KEYWORDS
)
from feature_extractors import (
    extract_skills, extract_technologies, extract_education,
    extract_experience_years, extract_location, extract_projects,
    count_projects, normalize_skill
)


def analyze_sections(resume_text, job_description, willing_to_relocate=None):
    """
    Main function to analyze all resume sections against job requirements.
    
    Args:
        resume_text (str): Resume text
        job_description (str): Job description text
        willing_to_relocate (bool or None): User's relocation preference
        
    Returns:
        list: List of section analysis dictionaries
    """
    sections = []
    
    # Analyze Skills & Technologies
    try:
        sections.append(analyze_skills_section(resume_text, job_description))
    except Exception as e:
        print(f"Error analyzing skills: {e}")
        sections.append({
            'icon': '⚙️',
            'title': 'Skills & Technologies',
            'status': 'missing',
            'missing': [],
            'recommendation': 'Unable to analyze skills section.'
        })
    
    # Analyze Projects
    try:
        sections.append(analyze_projects_section(resume_text, job_description))
    except Exception as e:
        print(f"Error analyzing projects: {e}")
        sections.append({
            'icon': '🚀',
            'title': 'Projects',
            'status': 'missing',
            'missing': [],
            'recommendation': 'Unable to analyze projects section.'
        })
    
    # Analyze Experience
    try:
        sections.append(analyze_experience_section(resume_text, job_description))
    except Exception as e:
        print(f"Error analyzing experience: {e}")
        sections.append({
            'icon': '💼',
            'title': 'Experience Level',
            'status': 'missing',
            'missing': [],
            'recommendation': 'Unable to analyze experience section.'
        })
    
    # Analyze Education
    try:
        sections.append(analyze_education_section(resume_text, job_description))
    except Exception as e:
        print(f"Error analyzing education: {e}")
        sections.append({
            'icon': '🎓',
            'title': 'Education',
            'status': 'good',
            'missing': [],
            'recommendation': 'Unable to analyze education section.'
        })
    
    # Analyze Location
    try:
        sections.append(analyze_location_section(resume_text, job_description, willing_to_relocate))
    except Exception as e:
        print(f"Error analyzing location: {e}")
        sections.append({
            'icon': '📍',
            'title': 'Location',
            'status': 'good',
            'missing': [],
            'recommendation': 'Unable to analyze location section.'
        })
    
    # Analyze Keywords
    try:
        sections.append(analyze_keywords_section(resume_text, job_description))
    except Exception as e:
        print(f"Error analyzing keywords: {e}")
        sections.append({
            'icon': '🔑',
            'title': 'Important Keywords',
            'status': 'good',
            'missing': [],
            'recommendation': 'Unable to analyze keywords.'
        })
    
    return sections


def analyze_skills_section(resume_text, job_description):
    """
    Analyze skills and technologies section
    
    Args:
        resume_text (str): Resume text
        job_description (str): Job description text
        
    Returns:
        dict: Analysis results for skills section
    """
    resume_skills = set([
        normalize_skill(s) for s in extract_skills(resume_text)
    ])

    job_skills = set([
        normalize_skill(s) for s in extract_skills(job_description)
    ])
    resume_tech = set(extract_technologies(resume_text))
    job_tech = set(extract_technologies(job_description))

    # Normalize before comparing
    resume_tech_normalized = set([normalize_skill(t) for t in resume_tech])
    job_tech_normalized = set([normalize_skill(t) for t in job_tech])

    missing_tech = job_tech_normalized - resume_tech_normalized
    missing_skills = job_skills - resume_skills
    
    # Combine and deduplicate missing items
    all_missing = list(set(list(missing_skills) + list(missing_tech)))
    missing_count = len(all_missing)
    
    if missing_count == 0:
        status = "good"
        recommendation = "Excellent! Your skills and technologies align well with the job requirements."
    elif missing_count <= 3:
        status = "weak"
        # Make each skill bold
        bold_skills = ', '.join([f"**{skill}**" for skill in all_missing])
        recommendation = (
            f"Add these {missing_count} missing skill{'s' if missing_count > 1 else ''}: "
            f"{bold_skills}. "
            "Include them in your Skills section or demonstrate them through project descriptions."
        )
    else:
        status = "missing"
        top_missing = all_missing[:5]
        # Make each skill bold
        bold_skills = ', '.join([f"**{skill}**" for skill in top_missing])
        recommendation = (
            f"Your resume is missing {missing_count} key technical skills. Priority skills to add: "
            f"{bold_skills}. "
            "Add these to your Skills section and showcase them in your projects."
        )
    
    return {
        'icon': '⚙️',
        'title': 'Skills & Technologies',
        'status': status,
        'missing': [],  # Empty - all info is in recommendation
        'recommendation': recommendation,
        'match_ratio': (
            len(resume_tech_normalized & job_tech_normalized) /
            len(job_tech_normalized)
            if job_tech_normalized else 1
        )
    }


def analyze_projects_section(resume_text, job_description):
    """
    Analyze projects section
    
    Args:
        resume_text (str): Resume text
        job_description (str): Job description text
        
    Returns:
        dict: Analysis results for projects section
    """
    project_section = extract_projects(resume_text)
    job_tech = set(extract_technologies(job_description))
    job_skills = set([s.lower() for s in extract_skills(job_description)])
    
    # Extract technologies mentioned in projects
    project_tech = set()
    if project_section:
        project_tech = set(extract_technologies(project_section))
    
    # Check if projects use relevant technologies
    relevant_project_tech = project_tech & (job_tech | job_skills)
    
    # Count number of projects - improved detection
    project_count = count_projects(project_section)
    
    # If project section is empty but we see project-like content in resume
    if project_count == 0:
        # Check for project indicators in the full resume
        project_indicators = [
            r'(?:developed|built|created|engineered|designed)\s+(?:a|an)\s+\w+\s+(?:website|application|app|platform|system)',
            r'(?:project|portfolio)\s*:',
            r'\d{2}/\d{4}\s*-\s*\d{2}/\d{4}',  # Date ranges
        ]
        
        for pattern in project_indicators:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            if matches:
                project_count = max(project_count, len(matches))
    
    # Determine project type alignment
    job_type = None
    for role_type, keywords in JOB_TYPE_KEYWORDS.items():
        if any(kw in job_description.lower() for kw in keywords):
            job_type = role_type
            break
    
    resume_has_relevant_projects = False
    if project_section and job_type:
        project_lower = project_section.lower()
        if job_type == 'frontend':
            frontend_indicators = ['react', 'ui', 'ux', 'frontend', 'front-end', 'responsive', 'component', 'css', 'html', 'interface']
            resume_has_relevant_projects = any(kw in project_lower for kw in frontend_indicators)
        elif job_type == 'backend':
            backend_indicators = ['api', 'backend', 'back-end', 'server', 'database', 'rest', 'restful', 'node', 'express']
            resume_has_relevant_projects = any(kw in project_lower for kw in backend_indicators)
        elif job_type == 'fullstack':
            fullstack_indicators = ['fullstack', 'full-stack', 'mern', 'mean', 'frontend', 'backend', 'full stack']
            resume_has_relevant_projects = any(kw in project_lower for kw in fullstack_indicators)
    
    # Determine status based on project count
    if project_count == 0:
        projects_status = "missing"
        projects_recommendation = (
            "Add a Projects section showcasing 2-4 relevant projects that demonstrate "
            "your skills with technologies mentioned in the job description."
        )
        projects_missing = ["Projects section"]

    elif project_count >= 3 and len(relevant_project_tech) >= 2:
        # Has 3+ projects with relevant tech
        projects_status = "good"
        projects_recommendation = (
            f"Excellent! You have {project_count} projects that align well with the job requirements."
        )
        projects_missing = []

    elif project_count >= 2 and len(relevant_project_tech) >= 2:
        # Has 2+ projects with relevant tech
        projects_status = "good"
        projects_recommendation = (
            f"Good! You have {project_count} projects aligned with required technologies."
        )
        projects_missing = []

    elif project_count >= 2:
        # Has 2+ projects but not enough relevant tech
        projects_status = "weak"
        missing_techs = list(job_tech - project_tech)[:5]
        if missing_techs:
            # Make each tech bold
            bold_techs = ', '.join([f"**{tech}**" for tech in missing_techs])
            projects_recommendation = (
                f"You have {project_count} projects. Enhance them by incorporating these job-relevant technologies: "
                f"{bold_techs}. "
                "Update existing projects or start a new one using the required tech stack."
            )
        else:
            projects_recommendation = (
                f"You have {project_count} projects. Add more measurable outcomes and technical details to strengthen them."
            )
        projects_missing = []

    else:
        # Has only 1 project
        projects_status = "weak"
        top_tech = list(job_tech)[:5]
        if top_tech:
            # Make each tech bold
            bold_techs = ', '.join([f"**{tech}**" for tech in top_tech])
            projects_recommendation = (
                f"You have {project_count} project. Add 1-2 more projects using technologies like: "
                f"{bold_techs}. "
                "This will demonstrate your ability to work with the required tech stack."
            )
        else:
            projects_recommendation = (
                f"You have {project_count} project. Add 1-2 more relevant projects to strengthen your technical credibility."
            )
        projects_missing = []

    
    return {
        'icon': '🚀',
        'title': 'Projects',
        'status': projects_status,
        'missing': projects_missing,
        'recommendation': projects_recommendation,
        'project_count': project_count,
        'relevant_project_ratio': (
            len(relevant_project_tech) / len(job_tech)
            if job_tech else 1
        )
    }


def analyze_education_section(resume_text, job_description):
    """
    Analyze education section
    
    Args:
        resume_text (str): Resume text
        job_description (str): Job description text
        
    Returns:
        dict: Analysis results for education section
    """
    resume_education = extract_education(resume_text)
    job_education = extract_education(job_description)
    
    if job_education:
        job_edu_reqs = [k for k in EDUCATION_KEYWORDS if k in job_education.lower()]
        resume_has_edu = (
            any(k in resume_education.lower() for k in EDUCATION_KEYWORDS) 
            if resume_education else False
        )
        
        if job_edu_reqs and not resume_has_edu:
            status = "missing"
            # Make each requirement bold
            bold_reqs = ', '.join([f"**{req}**" for req in job_edu_reqs])
            recommendation = (
                f"The job requires these educational qualifications: {bold_reqs}. "
                "Make sure your Education section is clearly visible and matches the requirements."
            )
        elif job_edu_reqs and resume_has_edu:
            status = "good"
            recommendation = "Your education qualifications are present and appear to meet the requirements."
        else:
            status = "good"
            recommendation = "Education section looks adequate."
    else:
        status = "good"
        recommendation = "No specific education requirements detected in the job description."
    
    return {
        'icon': '🎓',
        'title': 'Education',
        'status': status,
        'missing': [],  # Empty - all info is in recommendation
        'recommendation': recommendation
    }


def analyze_experience_section(resume_text, job_description):
    """
    Analyze experience section
    
    Args:
        resume_text (str): Resume text
        job_description (str): Job description text
        
    Returns:
        dict: Analysis results for experience section
    """
    resume_exp_years = extract_experience_years(resume_text)
    job_exp_years = extract_experience_years(job_description)
    
    if resume_exp_years and job_exp_years:
        resume_min, resume_max = resume_exp_years if isinstance(resume_exp_years, tuple) else (resume_exp_years, resume_exp_years)
        job_min, job_max = job_exp_years if isinstance(job_exp_years, tuple) else (job_exp_years, job_exp_years)
        
        # Handle "X+ years" pattern
        if job_max is None or job_max > 50:  # "5+ years" case
            if resume_max >= job_min:
                status = "good"
                recommendation = f"Your experience meets the {job_min}+ years requirement."
            else:
                status = "weak"
                recommendation = (
                    f"The job requires {job_min}+ years of experience. "
                    f"Emphasize your {resume_max} years and highlight relevant accomplishments to bridge the gap."
                )
        else:  # Range like "0-1 years" or exact like "3 years"
            # Special handling for 0-X year ranges (entry-level/fresher roles)
            if job_min == 0:
                # This is an entry-level role
                if resume_max <= job_max or not resume_exp_years:
                    status = "good"
                    recommendation = f"Perfect! This role accepts entry-level candidates ({job_min}-{job_max} year experience)."
                else:
                    status = "good"
                    recommendation = f"Your experience exceeds the {job_min}-{job_max} year range, which strengthens your application."
            elif resume_min <= job_max and resume_max >= job_min:
                # Experience overlaps with requirement
                status = "good"
                recommendation = f"Your experience aligns with the {job_min}-{job_max} year requirement."
            elif resume_max < job_min:
                # Underqualified
                status = "weak"
                recommendation = (
                    f"The job typically requires {job_min}-{job_max} years of experience. "
                    f"Emphasize your {resume_max} years and highlight relevant accomplishments."
                )
            else:  # overqualified
                status = "good"
                recommendation = (
                    f"You have more experience than the typical {job_min}-{job_max} year range, "
                    "which strengthens your application."
                )
        
        missing_items = []
        
    elif job_exp_years and not resume_exp_years:
        job_min, job_max = job_exp_years if isinstance(job_exp_years, tuple) else (job_exp_years, job_exp_years)
        
        # Check if this is entry-level/fresher role
        if job_min == 0:
            status = "good"
            recommendation = (
                f"This role accepts entry-level candidates ({job_min}-{job_max} year experience). "
                "As a fresh graduate or entry-level candidate, you're a good fit!"
            )
            missing_items = []
        else:
            status = "missing"
            if job_max and job_max < 50:
                recommendation = (
                    f"The job requires {job_min}-{job_max} year{'s' if job_max > 1 else ''} of experience. "
                    "Make sure this is clearly stated in your resume summary or experience section."
                )
                missing_items = [f"{job_min}-{job_max} year{'s' if job_max > 1 else ''} experience"]
            else:
                recommendation = (
                    f"The job requires {job_min}+ years of experience. "
                    "Make sure this is clearly stated in your resume summary or experience section."
                )
                missing_items = [f"{job_min}+ years experience"]
    else:
        status = "good"
        recommendation = "Experience level appears adequate."
        missing_items = []
    
    return {
        'icon': '💼',
        'title': 'Experience Level',
        'status': status,
        'missing': [],  # Empty - all info is in recommendation
        'recommendation': recommendation
    }


def analyze_location_section(resume_text, job_description, willing_to_relocate=None):
    """
    Analyze location section with relocation preference
    
    Args:
        resume_text (str): Resume text
        job_description (str): Job description text
        willing_to_relocate (bool or None): User's relocation preference
        
    Returns:
        dict: Analysis results for location section
    """
    resume_location = extract_location(resume_text)
    job_location = extract_location(job_description)
    
    if job_location:
        if resume_location:
            # Check if locations match
            locations_match = any(word in resume_location.lower() for word in job_location.lower().split())
            
            if locations_match:
                status = "good"
                recommendation = "Your location aligns with the job location."
            else:
                # Location doesn't match - check relocation preference
                if willing_to_relocate is True:
                    status = "good"
                    recommendation = (
                        f"Job location: {job_location}. Your resume shows: {resume_location}. "
                        f"Since you're willing to relocate, consider adding this to your resume or cover letter."
                    )
                elif willing_to_relocate is False:
                    status = "weak"
                    recommendation = (
                        f"Job location: {job_location}. Your resume shows: {resume_location}. "
                        f"Location mismatch may affect your application. Consider applying to local positions."
                    )
                else:
                    # User hasn't specified preference
                    status = "weak"
                    recommendation = (
                        f"Job location: {job_location}. Your resume shows: {resume_location}. "
                        f"Clarify if you're willing to relocate or work remotely."
                    )
        else:
            # No location on resume
            if willing_to_relocate is True:
                status = "good"
                recommendation = (
                    f"The job specifies location: {job_location}. "
                    f"Since you're willing to relocate, add this preference to your resume."
                )
            elif willing_to_relocate is False:
                status = "weak"
                recommendation = (
                    f"The job specifies location: {job_location}. "
                    f"Add your location to your resume if it matches, or note remote work preference."
                )
            else:
                status = "missing"
                recommendation = (
                    f"The job specifies location: {job_location}. "
                    f"Consider adding your location or relocation preferences to your resume."
                )
    else:
        status = "good"
        recommendation = "No specific location requirements detected."
    
    return {
        'icon': '📍',
        'title': 'Location',
        'status': status,
        'missing': [],  # Empty - all info is in recommendation
        'recommendation': recommendation,
        'job_location': job_location,
        'resume_location': resume_location,
        'willing_to_relocate': willing_to_relocate
    }


def analyze_keywords_section(resume_text, job_description):
    """
    Analyze important keywords section
    
    Args:
        resume_text (str): Resume text
        job_description (str): Job description text
        
    Returns:
        dict: Analysis results for keywords section
    """
    job_words = set(job_description.lower().split())
    resume_words = set(resume_text.lower().split())
    
    missing_important = [kw for kw in IMPORTANT_KEYWORDS if kw in job_words and kw not in resume_words]
    
    if len(missing_important) > 5:
        status = "weak"
        top_keywords = missing_important[:7]
        # Make each keyword bold
        bold_keywords = ', '.join([f"**{kw}**" for kw in top_keywords])
        recommendation = (
            f"Add these {len(missing_important)} important keywords throughout your resume: "
            f"{bold_keywords}. "
            "Incorporate them naturally in your experience descriptions, skills section, and professional summary."
        )
    elif len(missing_important) > 0:
        status = "weak"
        # Make each keyword bold
        bold_keywords = ', '.join([f"**{kw}**" for kw in missing_important])
        recommendation = (
            f"Incorporate these missing keywords: {bold_keywords}. "
            "Weave them naturally into your experience bullet points and skills section where relevant."
        )
    else:
        status = "good"
        recommendation = "Excellent keyword coverage! Your resume aligns well with the job requirements."
    
    return {
        'icon': '🔑',
        'title': 'Important Keywords',
        'status': status,
        'missing': [],  # Empty - all info is in recommendation
        'recommendation': recommendation
    }