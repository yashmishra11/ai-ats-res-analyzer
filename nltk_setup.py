"""
NLTK Setup and Initialization
Downloads required NLTK data packages
"""

import nltk
import os
import ssl

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def setup_nltk():
    """
    Download required NLTK data packages if not already present.
    This function ensures all necessary NLTK resources are available.
    """
    # Set NLTK data path
    nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
    
    # Create directory if it doesn't exist
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    
    # List of required NLTK packages
    required_packages = [
        'punkt',           # Tokenizer (old version)
        'punkt_tab',       # Tokenizer (new version for NLTK 3.9+)
        'stopwords',       # Stopwords
        'averaged_perceptron_tagger',  # POS tagger
        'maxent_ne_chunker',  # Named entity chunker
        'words'            # Word list
    ]
    
    # Download each package if not present
    for package in required_packages:
        try:
            # Try to find the package first
            try:
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                try:
                    nltk.data.find(f'corpora/{package}')
                except LookupError:
                    # Package not found, download it
                    print(f"Downloading NLTK package: {package}")
                    nltk.download(package, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download NLTK package '{package}': {e}")
            # Continue anyway - some packages might not be critical


# Run setup when module is imported
setup_nltk()


if __name__ == "__main__":
    setup_nltk()