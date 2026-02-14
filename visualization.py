"""
Visualization utilities for ATS Resume Analyzer
Creates charts and graphs for analysis results
"""

import matplotlib.pyplot as plt
from config import STATUS_SCORES, EXPECTED_SCORES_AFTER_FIX, SECTION_ORDER, SECTION_LABEL_MAP


def calculate_section_scores(sections_analysis):
    """
    Calculate current and expected scores for each section
    """

    section_scores = {
        'current': [],
        'expected': [],
        'labels': []
    }

    for section_name in SECTION_ORDER:
        section = next((s for s in sections_analysis if s['title'] == section_name), None)

        if not section:
            continue

        label = SECTION_LABEL_MAP.get(section_name, section_name)

        # 🔹 Dynamic scoring
        if section_name == "Projects":
            project_count = section.get("project_count", 0)
            tech_ratio = section.get("relevant_project_ratio", 0)

            score = round(min(95, 50 + (project_count * 10) + (tech_ratio * 30)))

        elif section_name == "Skills & Technologies":
            ratio = section.get("match_ratio", 0)
            score = round(min(95, 40 + (ratio * 55)))

        else:
            status = section['status']
            if status == "missing":
                score = 45
            elif status == "weak":
                score = 65
            else:
                score = 90

        # Expected improvement boost
        expected_score = round(min(95, score + 15))

        section_scores['labels'].append(label)
        section_scores['current'].append(score)
        section_scores['expected'].append(expected_score)

    return section_scores



def create_section_impact_chart(sections_analysis):
    """
    Create section-by-section impact analysis chart
    
    Args:
        sections_analysis (list): List of section analysis dictionaries
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    section_scores = calculate_section_scores(sections_analysis)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    
    # Plot lines
    x_positions = range(len(section_scores['labels']))
    
    # Current score line (whitish/gray)
    ax.plot(x_positions, section_scores['current'], 
           marker='o', markersize=8, linewidth=2.5, 
           color='#b0b0b0', label='Current', alpha=0.8)
    
    # Expected score line (greenish)
    ax.plot(x_positions, section_scores['expected'], 
           marker='o', markersize=8, linewidth=2.5, 
           color='#80d080', label='After Improvements', alpha=0.9)
    
    # Fill area between lines to show improvement potential
    ax.fill_between(x_positions, section_scores['current'], section_scores['expected'], 
                   alpha=0.2, color='#80d080')
    
    # Customize chart
    ax.set_xticks(x_positions)
    ax.set_xticklabels(section_scores['labels'], rotation=0, ha='center', fontsize=9, color='#a0a0a0')
    ax.set_ylim(40, 100)
    ax.set_ylabel('Score (%)', color='#a0a0a0', fontsize=11)
    ax.set_xlabel('Resume Sections', color='#a0a0a0', fontsize=11)
    ax.tick_params(colors='#a0a0a0')
    ax.grid(axis='y', alpha=0.2, color='#404040', linestyle='--')
    ax.grid(axis='x', alpha=0)
    
    # Style spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#404040')
    ax.spines['bottom'].set_color('#404040')
    
    # Legend
    ax.legend(loc='lower right', framealpha=0.3, facecolor='#2a2a2a', 
             edgecolor='#404040', fontsize=10, labelcolor='#a0a0a0')
    
    # Add value labels on points
    for i, (curr, exp) in enumerate(zip(section_scores['current'], section_scores['expected'])):
        ax.text(i, curr - 3, f'{int(curr)}', ha='center', va='top', 
               fontsize=8, color='#b0b0b0', weight='bold')
        if exp != curr:
            ax.text(i, exp + 2, f'{int(exp)}', ha='center', va='bottom', 
                   fontsize=8, color='#80d080', weight='bold')
    
    plt.tight_layout()
    return fig