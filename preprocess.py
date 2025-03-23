import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Initialize tools
lemmatizer = WordNetLemmatizer()

# 1. Base stopwords from NLTK
base_stopwords = set(stopwords.words("english"))

# 2. Add domain-specific and weak-value words
custom_stopwords = {
    "thank", "committee", "central", "bank", "year", "would", "could", "also", "shall", "may",
    "per", "upon", "like", "one", "two", "three", "must", "even", "among", "whether", "however",
    "further", "another", "many", "much", "still", "let", "us", "today"
}

# 3. Merge and lowercase all stopwords
stop_words = set(word.lower() for word in base_stopwords.union(custom_stopwords))

def preprocess_text(text):
    tokens = word_tokenize(text, preserve_line=True)

    # Normalize: lowercase + lemmatize
    lemmatized = [
        lemmatizer.lemmatize(word.lower())
        for word in tokens
        if word.isalpha()
    ]

    # Additional garbage/junk words to exclude
    junk_words = {"ha", "wa", "erm", "hmm", "uh", "ah", "yeah", "gotta", "gonna"}

    return [
        word for word in lemmatized
        if word not in stop_words
        and word not in junk_words
        and len(word) > 3
    ]

def extract_year_from_filename(filename):
    match = re.search(r"\d{4}", filename)
    return match.group() if match else "Unknown"

def load_and_preprocess_documents(folder_path):
    documents = []
    filenames = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                tokens = preprocess_text(text)
                documents.append(tokens)
                filenames.append(filename)

    return documents, filenames
