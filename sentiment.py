import numpy as np
from gensim.models import Word2Vec

# Load Loughran-McDonald lexicon (once at the top level)
with open("lexicons/loughran_positive.txt") as f:
    POSITIVE_WORDS = [line.strip().lower() for line in f if line.strip()]

with open("lexicons/loughran_negative.txt") as f:
    NEGATIVE_WORDS = [line.strip().lower() for line in f if line.strip()]

# Train Skip-gram model
def train_skipgram_model(docs):
    return Word2Vec(sentences=docs, vector_size=100, window=5, min_count=2, sg=1)

# Calculate sentiment score for a topic
def get_topic_sentiment(topic_words, model):
    pos_similarities = []
    neg_similarities = []

    for word in topic_words:
        if word in model.wv:
            pos_similarities += [model.wv.similarity(word, pos) for pos in POSITIVE_WORDS if pos in model.wv]
            neg_similarities += [model.wv.similarity(word, neg) for neg in NEGATIVE_WORDS if neg in model.wv]

    if not pos_similarities and not neg_similarities:
        return "neutral", 0.0

    avg_pos = np.mean(pos_similarities) if pos_similarities else 0
    avg_neg = np.mean(neg_similarities) if neg_similarities else 0
    score = avg_pos - avg_neg

    if score > 0.05:
        return "positive", score
    elif score < -0.05:
        return "negative", score
    else:
        return "neutral", score
