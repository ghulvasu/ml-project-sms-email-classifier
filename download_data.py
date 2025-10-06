import nltk
# This downloads the complete 'punkt' tokenizer package
nltk.download('punkt', download_dir='./nltk_data')
# Also re-download stopwords to be safe
nltk.download('stopwords', download_dir='./nltk_data')
print("NLTK data downloaded successfully.")