"""
Text extraction utilities for ATS Resume Analyzer
Handles PDF text extraction and text cleaning
"""

import re
from io import BytesIO
try:
    from PyPDF2 import PdfReader
except ImportError:
    try:
        from pypdf import PdfReader
    except ImportError:
        PdfReader = None


def extract_text_from_pdf(uploaded_file):
    """
    Extract text from uploaded PDF file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        str: Extracted text from PDF
        
    Raises:
        Exception: If PDF reading fails
    """
    if PdfReader is None:
        raise ImportError("PyPDF2 or pypdf is required. Install with: pip install PyPDF2")
    
    try:
        # Read the uploaded file
        pdf_file = BytesIO(uploaded_file.read())
        
        # Create PDF reader
        pdf_reader = PdfReader(pdf_file)
        
        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
        
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {str(e)}")


def clean_text(text):
    """
    Clean and normalize text for processing.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep important ones
    # Keep: letters, numbers, spaces, @, +, -, ., #, /
    text = re.sub(r'[^\w\s@+\-\.#/]', ' ', text)
    
    # Remove extra spaces again
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def extract_email(text):
    """
    Extract email address from text.
    
    Args:
        text (str): Text to search
        
    Returns:
        str or None: Email address if found
    """
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    match = re.search(email_pattern, text)
    return match.group(0) if match else None


def extract_phone(text):
    """
    Extract phone number from text.
    
    Args:
        text (str): Text to search
        
    Returns:
        str or None: Phone number if found
    """
    # Pattern for various phone formats
    phone_patterns = [
        r'\b\d{10}\b',  # 10 digits
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # XXX-XXX-XXXX
        r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b',  # (XXX) XXX-XXXX
        r'\+\d{1,3}[-.\s]?\d{10}\b',  # +XX XXXXXXXXXX
    ]
    
    for pattern in phone_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    
    return None


def extract_urls(text):
    """
    Extract URLs from text.
    
    Args:
        text (str): Text to search
        
    Returns:
        list: List of URLs found
    """
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    return urls


def extract_github_username(text):
    """
    Extract GitHub username from text.
    
    Args:
        text (str): Text to search
        
    Returns:
        str or None: GitHub username if found
    """
    # Pattern for github.com/username
    github_pattern = r'github\.com/([a-zA-Z0-9-]+)'
    match = re.search(github_pattern, text.lower())
    return match.group(1) if match else None


def extract_linkedin_username(text):
    """
    Extract LinkedIn username from text.
    
    Args:
        text (str): Text to search
        
    Returns:
        str or None: LinkedIn username if found
    """
    # Pattern for linkedin.com/in/username
    linkedin_pattern = r'linkedin\.com/in/([a-zA-Z0-9-]+)'
    match = re.search(linkedin_pattern, text.lower())
    return match.group(1) if match else None