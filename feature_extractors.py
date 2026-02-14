"""
Feature extraction utilities
Extracts skills, technologies, education, experience, location, and projects from text
"""

import re
from typing import Set, List, Tuple, Optional


# Common technology and skill sets
TECHNOLOGIES = {
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin',
    'go', 'rust', 'scala', 'r', 'matlab', 'html', 'css', 'sql', 'nosql',
    'react', 'angular', 'vue', 'svelte', 'next.js', 'nuxt', 'gatsby',
    'node.js', 'express', 'django', 'flask', 'fastapi', 'spring', 'laravel', 'rails',
    'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch', 'cassandra',
    'docker', 'kubernetes', 'jenkins', 'git', 'github', 'gitlab', 'bitbucket',
    'aws', 'azure', 'gcp', 'heroku', 'vercel', 'netlify',
    'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
    'rest', 'graphql', 'grpc', 'websockets', 'api',
    'jest', 'mocha', 'pytest', 'junit', 'selenium', 'cypress',
    'webpack', 'babel', 'vite', 'rollup', 'prisma', 'sequelize', 'mongoose'
}

SOFT_SKILLS = {
    'leadership', 'communication', 'teamwork', 'problem-solving', 'analytical',
    'creative', 'organized', 'detail-oriented', 'time-management', 'adaptable',
    'collaborative', 'initiative', 'critical-thinking', 'decision-making'
}


def normalize_skill(skill: str) -> str:
    """Normalize skill name for comparison"""
    skill = skill.lower().strip()
    
    # Handle common variations
    variations = {
        'reactjs': 'react',
        'react.js': 'react',
        'nodejs': 'node.js',
        'node': 'node.js',
        'nextjs': 'next.js',
        'next': 'next.js',
        'vuejs': 'vue',
        'vue.js': 'vue',
        'postgres': 'postgresql',
        'mongo': 'mongodb',
        'k8s': 'kubernetes',
        'js': 'javascript',
        'ts': 'typescript',
        'py': 'python',
    }
    
    return variations.get(skill, skill)


def extract_skills(text: str) -> Set[str]:
    """Extract skills from text"""
    text_lower = text.lower()
    found_skills = set()
    
    # Extract technologies
    for tech in TECHNOLOGIES:
        if tech in text_lower:
            found_skills.add(tech)
    
    # Extract soft skills
    for skill in SOFT_SKILLS:
        if skill in text_lower:
            found_skills.add(skill)
    
    return found_skills


def extract_technologies(text: str) -> Set[str]:
    """Extract only technical skills/technologies from text"""
    text_lower = text.lower()
    found_tech = set()
    
    for tech in TECHNOLOGIES:
        if tech in text_lower:
            found_tech.add(tech)
    
    return found_tech


def extract_education(text: str) -> str:
    """Extract education information from text"""
    # Look for education section
    education_pattern = r'(?:education|qualification|academic|degree)(.*?)(?:experience|skills|projects|$)'
    match = re.search(education_pattern, text.lower(), re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    return ""


def extract_experience_years(text: str) -> Optional[Tuple[int, int]]:
    """
    Extract years of experience from text.
    Returns tuple (min_years, max_years) or None if not found.
    For "X+ years", returns (X, 999)
    For "0-1 years" or fresher roles, returns (0, 1)
    """
    text_lower = text.lower()
    
    # Check for fresher/entry-level indicators first
    fresher_keywords = ['fresher', 'entry-level', 'entry level', 'graduate', 'new grad']
    if any(keyword in text_lower for keyword in fresher_keywords):
        # Still check if there's a specific range mentioned
        pass
    
    # Pattern for "X-Y years" (including 0-1)
    range_pattern = r'(\d+)\s*-\s*(\d+)\s*(?:years?|yrs?)'
    range_match = re.search(range_pattern, text_lower)
    if range_match:
        min_years = int(range_match.group(1))
        max_years = int(range_match.group(2))
        return (min_years, max_years)
    
    # Pattern for "X+ years"
    plus_pattern = r'(\d+)\s*\+\s*(?:years?|yrs?)'
    plus_match = re.search(plus_pattern, text_lower)
    if plus_match:
        years = int(plus_match.group(1))
        return (years, 999)
    
    # Pattern for just "X year(s)" without + or range
    single_pattern = r'\b(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)\b'
    single_match = re.search(single_pattern, text_lower)
    if single_match:
        years = int(single_match.group(1))
        return (years, years)
    
    return None


def extract_location(text: str) -> Optional[str]:
    """
    Extract location from text with improved accuracy.
    Focuses on actual city/country names, not company descriptions.
    """
    text_lower = text.lower()
    
    # First, try explicit location markers
    location_patterns = [
        r'location\s*:\s*([a-z][a-z\s,]+?)(?:\n|$|\||•|based)',
        r'based\s+(?:in|at)\s+([a-z][a-z\s,]+?)(?:\n|$|\.|\|)',
        r'office(?:s)?\s+(?:in|at)\s+([a-z][a-z\s,]+?)(?:\n|$|\.|\|)',
        r'work\s+(?:from|in)\s+([a-z][a-z\s,]+?)(?:\n|$|\.|\|)',
    ]
    
    for pattern in location_patterns:
        match = re.search(pattern, text_lower)
        if match:
            location = match.group(1).strip()
            # Clean up
            location = re.sub(r'\s+', ' ', location)
            # Remove common non-location words
            location = re.sub(r'\b(the|a|an|in|at|of|for|to|and|or|is|are|with)\b', '', location).strip()
            
            # Validate it's actually a location (not too long, no technical terms)
            if 2 < len(location) < 40 and not any(tech in location for tech in ['developer', 'engineer', 'software', 'quality', 'devops']):
                return location.title()
    
    # Look for common city patterns: "City, State/Country" or "City, XX"
    city_state_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2,}|[A-Z][a-z]+)\b'
    for match in re.finditer(city_state_pattern, text):
        city = match.group(1)
        state_country = match.group(2)
        
        # Exclude if it looks like a company name or technical term
        combined = f"{city} {state_country}".lower()
        if not any(tech in combined for tech in ['ltd', 'inc', 'llc', 'pvt', 'private', 'limited', 'corporation', 'technologies', 'solutions', 'systems']):
            return f"{city}, {state_country}"
    
    # Look for specific well-known cities and countries
    # Indian cities
    indian_cities = [
        'bangalore', 'bengaluru', 'mumbai', 'delhi', 'hyderabad', 'pune', 'chennai',
        'kolkata', 'ahmedabad', 'jaipur', 'surat', 'lucknow', 'kanpur', 'nagpur',
        'indore', 'thane', 'bhopal', 'visakhapatnam', 'vadodara', 'kochi', 'trivandrum',
        'thiruvananthapuram', 'noida', 'gurgaon', 'gurugram'
    ]
    
    # Global cities
    global_cities = [
        'london', 'new york', 'san francisco', 'seattle', 'austin', 'boston',
        'toronto', 'vancouver', 'sydney', 'melbourne', 'singapore', 'dubai',
        'paris', 'berlin', 'amsterdam', 'tokyo', 'beijing', 'shanghai'
    ]
    
    # Countries
    countries = [
        'india', 'usa', 'uk', 'united kingdom', 'united states', 'canada',
        'australia', 'germany', 'france', 'singapore', 'uae', 'netherlands',
        'japan', 'china', 'remote'
    ]
    
    all_locations = indian_cities + global_cities + countries
    
    for location in all_locations:
        # Use word boundary to match whole words
        pattern = r'\b' + re.escape(location) + r'\b'
        if re.search(pattern, text_lower):
            return location.title()
    
    return None


def extract_projects(text: str) -> str:
    """
    Extract the projects section from text.
    Returns the full projects section text.
    """
    # Look for projects section with various header formats
    project_patterns = [
        r'(?:^|\n)\s*projects?\s*(?:\n|:)(.*?)(?=\n\s*(?:experience|education|skills|certifications?|achievements?|$))',
        r'(?:^|\n)\s*(?:key\s+)?projects?\s*(?:\n|:)(.*?)(?=\n\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*(?:\n|$))',
    ]
    
    for pattern in project_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return ""


def count_projects(project_section: str) -> int:
    """
    Count the number of projects in a project section.
    Uses multiple heuristics to identify individual projects.
    """
    if not project_section:
        return 0
    
    count = 0
    
    # Method 1: Count project titles (lines that look like headers)
    # These are usually short lines (< 50 chars) followed by descriptions
    lines = project_section.split('\n')
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        # Skip empty lines
        if not line_stripped:
            continue
        # Check if this looks like a project title
        if (len(line_stripped) < 50 and 
            len(line_stripped) > 5 and
            not line_stripped.startswith('-') and
            not line_stripped.startswith('•') and
            i + 1 < len(lines) and  # Has following content
            len(lines[i + 1].strip()) > 30):  # Following line is descriptive
            count += 1
    
    # Method 2: Count date ranges (MM/YYYY - MM/YYYY pattern)
    date_pattern = r'\d{2}/\d{4}\s*-\s*\d{2}/\d{4}'
    date_matches = re.findall(date_pattern, project_section)
    if len(date_matches) > count:
        count = len(date_matches)
    
    # Method 3: Count bullet point groups
    # Projects often have multiple bullets under each title
    bullet_groups = 0
    in_bullet_group = False
    
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith(('-', '•', '▪', '●', '*')):
            if not in_bullet_group:
                bullet_groups += 1
                in_bullet_group = True
        elif len(line_stripped) < 50 and len(line_stripped) > 5:
            in_bullet_group = False
    
    if bullet_groups > count:
        count = bullet_groups
    
    # Method 4: Count explicit "Project:" markers
    explicit_projects = len(re.findall(r'(?:^|\n)\s*(?:project\s*:|\d+\.)', project_section, re.IGNORECASE))
    if explicit_projects > count:
        count = explicit_projects
    
    # If we still found nothing but there's substantial content, assume at least 1
    if count == 0 and len(project_section.strip()) > 100:
        count = 1
    
    return count