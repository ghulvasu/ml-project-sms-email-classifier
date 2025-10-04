import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    """
    Performs text preprocessing:
    1. Converts to lowercase
    2. Tokenizes
    3. Removes non-alphanumeric characters
    4. Removes stop words and punctuation
    5. Stems the words
    """
    # Convert to lowercase
    text = text.lower()

    # Tokenize the text
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    alphanumeric_chars = []
    for i in text:
        if i.isalnum():
            alphanumeric_chars.append(i)

    # Remove stop words and punctuation
    text = alphanumeric_chars[:]
    processed_text = []
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            processed_text.append(i)

    # Stem the words
    text = processed_text[:]
    stemmed_text = []
    for i in text:
        stemmed_text.append(ps.stem(i))

    return " ".join(stemmed_text)