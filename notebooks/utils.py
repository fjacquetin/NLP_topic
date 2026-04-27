import os
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from langdetect import detect
import re
import spacy

def load_texts_from_dir(text_dir):
    texts = []
    meta = []

    for filename in os.listdir(text_dir):
        if not filename.endswith(".txt"):
            continue
        
        path = os.path.join(text_dir, filename)
        doc_id = filename.replace(".txt", "")
        
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()  # enlever les espaces en début/fin
            if len(text) == 0:
                continue  # ignorer les fichiers vides
        
        texts.append(text)
        meta.append({
            "id": doc_id,
            "text": text
        })

    df = pd.DataFrame(meta)
    print(f"Nombre de documents chargés : {len(df)}")
    return df

def associer_partis_nlp_strict(texte, partis_ref):
    if not isinstance(texte, str):
        return None

    texte_clean = re.sub(r'[-/;]', ' ', texte.lower())
    texte_clean = re.sub(r'[^\w\s]', '', texte_clean)
    
    nlp = spacy.load("fr_core_news_sm")

    doc = nlp(texte_clean)
    tokens = [token.text for token in doc]

    partis_detectes = set()

    for parti, aliases in partis_ref.items():
        for alias in aliases:
            alias_tokens = alias.lower().split()

            if len(alias_tokens) > 1 and all(tok in tokens for tok in alias_tokens):
                partis_detectes.add(parti)
                break

            if len(alias_tokens) == 1 and alias_tokens[0] in tokens:
                partis_detectes.add(parti)
                break

    if partis_detectes:
        return list(partis_detectes)[0]
    return None

def get_parti_associe(texte,partis_ref):
    if not isinstance(texte, str):
        return None
    fragments = [frag.strip() for frag in texte.split(';')]
    for frag in fragments:
        # Si le fragment correspond exactement à un parti
        if frag in partis_ref.keys():
            return frag
        # Sinon on applique la fonction stricte
        parti_detecte = associer_partis_nlp_strict(frag, partis_ref)
        if parti_detecte:
            return parti_detecte
    return None


def detect_lang(text):
    try:
        return detect(text)
    except:
        return "unknown"

def print_top_words(model, vectorizer, n_top_words):
    
    feature_names = vectorizer.get_feature_names_out()
    
    for topic_idx, topic in enumerate(model.components_):
        
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        
        print(f"\nTopic {topic_idx+1}")
        print(", ".join(top_features))

def plot_top_words(model, vectorizer, topic_titles, n_top_words, title):
    
    feature_names = vectorizer.get_feature_names_out()
    n_topics = len(model.components_)

    n_cols = 5
    n_lines = int(np.ceil(n_topics / n_cols))

    fig, axes = plt.subplots(n_lines, n_cols, figsize=(30, 30), sharex=True)
    axes = axes.flatten()

    for topic_idx, topic in enumerate(model.components_):

        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]

        ax.barh(top_features, weights, height=0.7)

        topic_name = topic_titles[topic_idx]
        ax.set_title(f"Topic {topic_idx+1}\n{topic_name}", fontsize=22)

        ax.tick_params(axis="both", which="major", labelsize=18)

        for i in "top right left".split():
            ax.spines[i].set_visible(False)

    fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.4)
    plt.show()