# embeddings/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

# Embedding modelini yükleyin
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(text):
    sentences = text.split('\n')  # Metni satırlara ayıralım (veya istediğiniz gibi bölebilirsiniz)
    embeddings = model.encode(sentences)
    return embeddings

def retrieve_top_k_similar_embeddings(query, embeddings, k=3):
    query_embedding = model.encode([query])
    similarities = np.dot(query_embedding, embeddings.T)  # Kosinüs benzerliği hesaplayalım
    top_k_idx = similarities.argsort()[0][-k:][::-1]  # En benzer k öğeyi seç
    return top_k_idx
