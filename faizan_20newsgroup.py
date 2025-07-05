import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import pandas as pd
import os
from PIL import Image

# --- Constants ---
TOPIC_NAMES = [
    "alt.atheism", "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware", "comp.windows.x", "misc.forsale", "rec.autos", "rec.motorcycles",
    "rec.sport.baseball", "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med",
    "sci.space", "soc.religion.christian", "talk.politics.guns", "talk.politics.mideast",
    "talk.politics.misc", "talk.religion.misc"
]
n_components = 20
ngram_range = (1, 2)

# --- Title ---
st.title("20 Newsgroups Topic Modeling Dashboard")

# --- Load base data ---
@st.cache_data
def load_base_data():
    try:
        nmf_refined = joblib.load('newgroups_pickle/nmf_refined_model.pkl')
        vectorizer_ngrams = joblib.load('newgroups_pickle/tfidf_refined_nmf_vectorizer.pkl')
        df = pd.read_pickle('newgroups_pickle/newsgroups_cleaned.pkl')
        X_ngrams = vectorizer_ngrams.transform(df['cleaned_text'])
        topic_distributions = nmf_refined.transform(X_ngrams)
        dominant_topics = np.argmax(topic_distributions, axis=1)
        return nmf_refined, vectorizer_ngrams, df, X_ngrams, topic_distributions, dominant_topics
    except Exception as e:
        st.error(f"Error loading base data: {e}")
        return None, None, None, None, None, None

nmf_refined, vectorizer_ngrams, df, X_ngrams, topic_distributions, dominant_topics = load_base_data()
if dominant_topics is None:
    st.stop()

# --- Re-run NMF to get dynamic features ---
@st.cache_data
def get_dynamic_nmf(df, n_components, ngram_range):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', min_df=5, max_df=0.95, ngram_range=ngram_range)
    X_dynamic = vectorizer.fit_transform(df['cleaned_text'])
    nmf = NMF(n_components=n_components, random_state=42, max_iter=200, init='nndsvda', solver='mu')
    nmf.fit(X_dynamic)
    return nmf, vectorizer, X_dynamic

try:
    nmf_dynamic, vectorizer_dynamic, X_dynamic = get_dynamic_nmf(df, n_components, ngram_range)
    dynamic_distributions = nmf_dynamic.transform(X_dynamic)
    dynamic_topics = np.argmax(dynamic_distributions, axis=1)
except Exception as e:
    st.error(f"Error generating NMF model: {e}")
    st.stop()

# --- Use actual topic names ---
topic_labels = TOPIC_NAMES  # Fixed list of 20 topics
selected_topic_index = st.sidebar.selectbox("Select a Topic", list(range(n_components)), format_func=lambda x: topic_labels[x])

# --- Topic Explorer ---
st.subheader("Topic Explorer")
feature_names = vectorizer_dynamic.get_feature_names_out()
try:
    top_words = [feature_names[i] for i in nmf_dynamic.components_[selected_topic_index].argsort()[:-11:-1]]
    st.write(f"Top 10 Words for **{topic_labels[selected_topic_index]}**: {top_words}")
except Exception as e:
    st.error(f"Error in topic exploration: {e}")

# --- Top Documents per Topic ---
st.subheader(f"Top 5 Documents for Topic: {topic_labels[selected_topic_index]}")
try:
    topic_docs = df[dynamic_topics == selected_topic_index]['cleaned_text']
    if len(topic_docs) > 0:
        sample_docs = topic_docs.sample(min(5, len(topic_docs)), random_state=42)
        for i, doc in enumerate(sample_docs, 1):
            st.write(f"**Document {i}:** {doc[:200]}...")
    else:
        st.warning("No documents found for this topic.")
except Exception as e:
    st.error(f"Error loading documents: {e}")

# --- t-SNE Visualization ---
st.subheader("t-SNE Visualization (Optional Topic Filter)")
filter_topic = st.selectbox("Filter by Topic", [None] + list(range(n_components)), format_func=lambda x: topic_labels[x] if x is not None else "All")

try:
    if filter_topic is not None:
        mask = dynamic_topics == filter_topic
        if np.sum(mask) < 2:
            st.warning(f"Not enough documents to visualize topic: {topic_labels[filter_topic]}")
        else:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            X_tsne = tsne.fit_transform(dynamic_distributions[mask])
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=dynamic_topics[mask], palette='tab20', ax=ax)
            ax.set_title(f"t-SNE for Topic: {topic_labels[filter_topic]}")
            st.pyplot(fig)
    else:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(dynamic_distributions)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=dynamic_topics, palette='tab20', ax=ax)
        ax.set_title("t-SNE for All Topics")
        st.pyplot(fig)
except Exception as e:
    st.error(f"Error generating t-SNE visualization: {e}")

# --- Topic Distribution ---
st.subheader("Topic Distribution (Bar Chart)")
if len(dynamic_topics) > 0:
    topic_counts = np.bincount(dynamic_topics, minlength=n_components)
    fig, ax = plt.subplots()
    ax.bar(range(len(topic_counts)), topic_counts)
    ax.set_xticks(range(n_components))
    ax.set_xticklabels(topic_labels, rotation=90)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Document Count")
    ax.set_title("Distribution of Documents per Topic")
    st.pyplot(fig)
else:
    st.warning("No topic assignments available.")

# --- Optional: Load saved plot image ---
st.subheader("Top 20 Words (Pre-Saved Plot)")
plot_path = "C:/Users/Faizan/Desktop/csi_proj/newsgroupplots/top_20_words.png"
if os.path.exists(plot_path):
    img = Image.open(plot_path)
    st.image(img, use_column_width=True)
else:
    st.warning("Top 20 Words plot not found at the specified path.")
