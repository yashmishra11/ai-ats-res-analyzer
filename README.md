# ATS Resume Analyzer

AI-Powered Resume Optimization Platform that analyzes resume-job match scores using machine learning algorithms.

## 📁 Project Structure

```
ats-resume-analyzer/
│
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration and constants
├── nltk_setup.py                   # NLTK initialization
│
├── text_extractors.py              # PDF and text extraction utilities
├── feature_extractors.py           # Resume feature extraction (skills, education, etc.)
├── similarity_calculator.py        # Similarity scoring algorithms
├── section_analyzer.py             # Section-by-section analysis
├── recommendation_generator.py     # Improvement recommendations
├── visualization.py                # Charts and visualizations
├── ui_components.py                # Streamlit UI components and styling
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🔧 Module Descriptions

### Core Modules

- **app.py**: Main application entry point. Orchestrates all components and handles user interactions.

- **config.py**: Central configuration file containing all constants, skill categories, regex patterns, and score weights.

### Utility Modules

- **nltk_setup.py**: Handles NLTK initialization and downloads required packages.

- **text_extractors.py**: Functions for extracting and normalizing text from PDF files.

- **feature_extractors.py**: Extracts specific resume features like skills, technologies, education, experience, location, and projects.

### Analysis Modules

- **similarity_calculator.py**: Calculates resume-job match scores using:
  - TF-IDF similarity (40% weight)
  - Skills matching (30% weight)
  - Keyword matching (20% weight)
  - Section completeness (10% weight)

- **section_analyzer.py**: Performs detailed analysis of each resume section:
  - Skills & Technologies
  - Projects
  - Education
  - Experience Level
  - Location
  - Important Keywords

- **recommendation_generator.py**: Generates actionable recommendations and rewrite examples for improvement.

### UI Modules

- **ui_components.py**: Contains all Streamlit UI components, CSS styling, and rendering functions.

- **visualization.py**: Creates charts and graphs for visualizing analysis results.

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ats-resume-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## 📖 Usage

1. **Upload Resume**: Upload your resume in PDF format
2. **Paste Job Description**: Copy and paste the target job description
3. **Analyze**: Click the "Analyze Resume Match" button
4. **Review Results**: 
   - View your current match score
   - See expected score after improvements
   - Review section-by-section analysis
   - Get actionable recommendations

## 🎯 Features

- **Multi-factor Analysis**: Combines TF-IDF, skills matching, keywords, and section completeness
- **Section Detection**: Identifies missing or weak sections in your resume
- **Keyword Extraction**: Identifies critical missing terms from job descriptions
- **Visual Analytics**: Section-by-section impact analysis chart
- **Actionable Recommendations**: Specific suggestions for each section with examples
- **Skills Rewrite**: Organized missing skills by category
- **Keyword Examples**: Sample bullet points demonstrating keyword usage

## 🛠️ Customization

### Adding New Skills/Technologies

Edit `config.py` and add to `SKILL_CATEGORIES` or `TECH_KEYWORDS`:

```python
SKILL_CATEGORIES = {
    'New Category': ['skill1', 'skill2', 'skill3'],
    # ...
}
```

### Adjusting Score Weights

Modify weights in `config.py`:

```python
SCORE_WEIGHTS = {
    'tfidf': 0.40,
    'skills': 0.30,
    'keywords': 0.20,
    'sections': 0.10
}
```

### Adding New Recommendation Templates

Add templates to `recommendation_generator.py` in `keyword_templates` dictionary.

## 𖤓 Analysis Components

### Similarity Score Calculation
- **TF-IDF Similarity** (40%): Measures overall text similarity
- **Skills Matching** (30%): Compares technical skills and technologies
- **Keywords Matching** (20%): Checks for important keywords
- **Section Completeness** (10%): Verifies all necessary sections are present

### Section Analysis
Each section is rated as:
- ✓ **Good**: Meets requirements
- ⚠️ **Weak**: Needs improvement
- ❌ **Missing**: Critical gap

## 🎨 UI Customization

All styling is contained in `ui_components.py`. The app uses a dark theme with:
- Gradient backgrounds
- Card-based layout
- Interactive charts
- Responsive design

## ⚠️ Known Limitations

- PDF text extraction may vary based on PDF structure
- Visual/image-based PDFs may have reduced accuracy
- Semantic fallbacks are used for edge cases

## 📝 Version

Current Version: 2.1

## 🤝 Contributing

Contributions are welcome! The modular structure makes it easy to:
- Add new analysis sections
- Improve extraction algorithms
- Enhance UI components
- Add new visualizations

## 📄 License

MIT License (or your preferred license)