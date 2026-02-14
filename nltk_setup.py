"""
NLTK Setup and Initialization
Downloads required NLTK data packages
"""

import nltk
import os


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
        'punkt',        # Tokenizer
        'stopwords',    # Stopwords
        'averaged_perceptron_tagger',  # POS tagger
        'maxent_ne_chunker',  # Named entity chunker
        'words'         # Word list
    ]
    
    # Download each package if not present
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{package}')
            except LookupError:
                try:
                    nltk.download(package, quiet=True)
                except Exception as e:
                    print(f"Warning: Could not download NLTK package '{package}': {e}")


if __name__ == "__main__":
    setup_nltk()