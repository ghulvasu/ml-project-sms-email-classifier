import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# --- Download NLTK data for deployment ---
# This checks if the data is present and downloads it if not.
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)
# -----------------------------------------

ps = PorterStemmer()

def transform_text(text):
    """
    Performs text preprocessing:
    1. Lowercases the text.
    2. Tokenizes the text into words.
    3. Removes non-alphanumeric characters.
    4. Removes stop words and punctuation.
    5. Applies stemming to each word.
    Returns the processed text as a single string.
    """
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

