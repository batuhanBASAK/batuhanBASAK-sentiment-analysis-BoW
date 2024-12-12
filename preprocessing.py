import re
from tqdm import tqdm
import nltk
# Download required NLTK data files
nltk.download('punkt')  # For tokenization
nltk.download('punkt_tab')
nltk.download('stopwords')  # For stop words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import spacy
# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")



def handle_links_and_tags(text):
    # Regular expression to match URLs, mentions, and hashtags
    pattern = r'http[s]?://\S+|@\w+|#\w+'
    
    # Remove matches
    cleaned_tweet = re.sub(pattern, '', text)
    
    # Remove extra spaces
    cleaned_tweet = ' '.join(cleaned_tweet.split())

    return cleaned_tweet



def handle_stop_words(text):    
    # Tokenize the text
    words = word_tokenize(text)

    # Get the stop words for English
    stop_words = set(stopwords.words('english'))
    
    # Remove stop words
    filtered_words = [word for word in words if word.lower() not in stop_words]

    return filtered_words


def handle_lemmatization(text):
    # Process the text
    doc = nlp(text)
    
    # Extract lemmatized words
    lemmatized_words = [token.lemma_ for token in doc]

    return lemmatized_words



def preprocess_text(text):
    # Remove any tag and link
    text = handle_links_and_tags(text)
    
    # Remove stop words
    text = handle_stop_words(text)
    text = ' '.join(text)

    # Do lemmatization
    text = handle_lemmatization(text)

    return ' '.join(text)