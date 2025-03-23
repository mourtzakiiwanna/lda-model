import os
from gensim import corpora, models
import pyLDAvis.gensim_models
import pyLDAvis
import webbrowser

def train_lda_model(documents, num_topics, passes):
    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
    return lda_model, corpus, dictionary

def save_model_and_dictionary(model, dictionary, path):
    os.makedirs(path, exist_ok=True)
    model.save(os.path.join(path, "lda_model"))
    dictionary.save(os.path.join(path, "dictionary.dict"))

def visualize_topics(lda_model, corpus, dictionary, path):
    os.makedirs(path, exist_ok=True)
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    html_path = os.path.join(path, "lda_visualization.html")
    pyLDAvis.save_html(vis, html_path)
    webbrowser.open(f"file://{os.path.abspath(html_path)}")
