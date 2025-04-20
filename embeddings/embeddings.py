from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def create_embeddings(text):
    if isinstance(text, list):
        sentences = text
    else:
        sentences = text.split('\n')

    sentences = [s.strip() for s in sentences if s.strip()]

    embeddings = model.encode(sentences)
    return embeddings


def create_txt_embeddings(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()

    text_chunks = text.split('\n')

    text_chunks = [chunk.strip() for chunk in text_chunks if chunk.strip()]

    embeddings = model.encode(text_chunks)

    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(np.array(embeddings).astype(np.float32))

    return faiss_index, text_chunks


def retrieve_top_k_similar_embeddings(query, embeddings, k=3):
    query_embedding = model.encode([query])
    similarities = np.dot(query_embedding, embeddings.T)
    top_k_idx = similarities.argsort()[0][-k:][::-1]
    return top_k_idx
