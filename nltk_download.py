# download_nltk.py

import nltk
import os

# Define the download directory relative to this script
download_dir = "nltk_data"

# Create the directory if it doesn't exist
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Download the necessary NLTK resources to the specified directory
nltk.download('punkt', download_dir=download_dir)
nltk.download('stopwords', download_dir=download_dir)

print(f"NLTK data successfully downloaded to the '{download_dir}' directory.")