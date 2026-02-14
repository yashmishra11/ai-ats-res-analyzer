def analyze_sections(resume_text, job_description=None):
    """
    Analyze resume sections (Skills, Projects, Experience) and return their status.
    Args:
        resume_text (str): Raw text of the resume.
        job_description (str, optional): Raw text of the job description. Defaults to None.
    Returns:
        list: List of dictionaries with section titles and statuses.
    """
    sections = [
        {
            "title": "Skills & Technologies",
            "keywords": ["skills", "technologies", "tech stack"],
            "status": "missing"
        },
        {
            "title": "Projects",
            "keywords": ["projects", "project work", "portfolio"],
            "status": "missing"
        },
        {
            "title": "Experience",
            "keywords": ["experience", "internship", "work"],
            "status": "missing"
        },
    ]

    resume_lower = resume_text.lower()
    for section in sections:
        for keyword in section["keywords"]:
            if keyword.lower() in resume_lower:
                section["status"] = "present"
                break
        else:
            section["status"] = "missing"

    return sections
