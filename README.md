Topic Modeling and Document Clustering on 20 Newsgroups Dataset

Celebal Summer Internship Final Project Report – Data Science

Author: Faizan RasoolDuration: Summer 2025Dataset: 20 Newsgroups (20K documents)


🎯 Objective
Extract meaningful latent topics from a large text corpus using unsupervised topic modeling and evaluate their alignment with document clusters, both analytically and visually.


🛠️ Methodology Overview

Preprocessing: Lowercasing, stopword removal, lemmatization
Vectorization: TF-IDF with bigrams (ngram_range=(1,2), max_features=5000)
Modeling Techniques: LDA (default + tuned), NMF (base + refined)
Evaluation Metrics: Adjusted Rand Score (ARS), Topic Distribution, t-SNE
Clustering: KMeans (20 clusters) on PCA-reduced TF-IDF features
Dashboard: Interactive Streamlit dashboard for exploration

🧾 Topic Interpretations (Refined NMF)



Topic ID
Top Words (Refined NMF)
Interpreted Label



0
people, just, like, think, time
General Discussion


1
program, window, file, image, graphics
Software / Graphics


2
god, jesus, bible, christ, faith
Religion / Christianity


3
article, writes, dod, mark, john
Posting / Replies


4
game, team, hockey, players, season
Sports


5
drive, scsi, disk, hard, floppy
Hardware / Storage


6
university, phone, email, internet
Academia / Communication


7
windows, dos, os, microsoft, run
Operating Systems


8
key, encryption, clipper, pgp, algorithm
Cryptography / Security


9
thanks, advance, info, help, anybody
Help Requests


10
gordon banks, jxp, satire phrases
Satirical / Named Threads


11
moral, values, frank dwyer, objective
Ethics / Philosophy


12
armenian, turkish, genocide, soviet
History / Politics


13
card, video, monitor, mhz, dx
PC Hardware


14
israel, jews, arab, peace, state
Middle East Politics


15
car, bike, engine, dealer, miles
Automobiles


16
don, know, want, say, writes
Opinion / Freeform Thought


17
mailing list, post, send, address
Lists / Communication


18
kent, ksand, private, alink, activities
Named Author Threads


19
write today, company, investors
Marketing / Commercial Posts


📊 Adjusted Rand Score (Cluster-Topic Alignment)



Model Type
Description
ARS Score



LDA (default)
Baseline LDA with unigram TF-IDF
0.0404


LDA (tuned)
Batch LDA + n-gram TF-IDF
0.1231


NMF (base)
NMF + n-gram TF-IDF
0.2875


NMF (refined)
NMF + nndsvda init + mu solver
0.2953 ✅


Insight: Refined NMF achieved the highest cluster-topic alignment.
💻 📊 Streamlit Dashboard

Live URL: 20 Newsgroups Topic Explorer
Features:
Topic Selector: View top 10 words per topic
Top Documents Viewer: Shows 5 sample documents per topic
t-SNE Visualizer: Visualize topics with optional filtering
Topic Distribution Chart: Bar chart of document counts



📈 Final t-SNE Visualization

Each point represents a document
Colors indicate dominant refined NMF topics
Clear separation validates topic distinctiveness

✅ Key Takeaways

Refined NMF with n-grams provided the best topic coherence and cluster alignment (ARS: 0.2953).
t-SNE plots and cluster overlap confirm high model fidelity.
The Streamlit dashboard enables dynamic exploration.

📦 Project Deliverables



File
Description



nmf_refined_model.pkl
Final topic model


tfidf_refined_nmf_vectorizer.pkl
Final vectorizer


nmf_refined_assignments.npy
Topic labels


kmeans_clusters_pca.npy
Cluster labels


faizan_20newsgroup.py
Streamlit app code


t-SNE Plot (.png)
Final 2D visualization


🚀 Future Work

Incorporate topic coherence metrics (e.g., UMass, NPMI)
Test advanced models like BERTopic, Top2Vec
Add search functionality and clustering visualization overlays

🛠️ Setup Instructions

Clone the Repository:
git clone https://github.com/faizan1343/20newsgroup.git
cd 20newsgroup


Install Git LFS (for large files like newgroups_cleaned.pkl):
git lfs install


Install Dependencies:
pip install -r requirements.txt


Run the Dashboard Locally:
streamlit run faizan_20newsgroup.py


Access the Live App: 20 Newsgroups Topic Explorer
https://20newsgroup-w2bhljl9sxmqvvhvsq36gq.streamlit.app/

📝 Notes

Large files (e.g., newgroups_cleaned.pkl) are managed with Git LFS.
Ensure all .pkl files are in the newgroups_pickle folder.
