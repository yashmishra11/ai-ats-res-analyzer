"""
Recommendation generation utilities
Generates improvement suggestions and rewrite examples
"""

import random
from config import SKILL_CATEGORIES


def generate_keyword_rewrites(missing_keywords):
    """
    Generate example bullet points incorporating missing keywords
    
    Args:
        missing_keywords (list): List of missing keywords
        
    Returns:
        list: List of dictionaries with keyword and example
    """
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
            examples.append({
                'keyword': keyword,
                'example': random.choice(keyword_templates[keyword])
            })
    
    return examples


def generate_skills_section_rewrite(missing_items, existing_tech):
    """
    Generate a complete skills section incorporating missing items
    
    Args:
        missing_items (list): List of missing skills/technologies
        existing_tech (list): List of existing technologies in resume
        
    Returns:
        dict or None: Organized skills by category or None if no missing items
    """
    if not missing_items:
        return None
    
    # Organize missing and existing items by category
    organized = {}
    all_items = set(missing_items + existing_tech)
    
    for category, keywords in SKILL_CATEGORIES.items():
        items_in_category = [item for item in all_items if item.lower() in [k.lower() for k in keywords]]
        if items_in_category:
            # Mark which are missing
            organized[category] = {
                'items': items_in_category,
                'missing': [item for item in items_in_category if item in missing_items]
            }
    
    return organized
