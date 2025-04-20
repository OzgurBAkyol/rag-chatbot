from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Embedding modelini yükleyin
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def create_embeddings(text):
    """
    Gelen text parametresinden embedding'ler oluşturur.
    Eğer text bir listeyse, her bir öğeyi işleme alır.
    Eğer text bir stringse, '\n' ile böler ve işleme alır.
    """
    # Eğer gelen veri zaten listeyse, split etme
    if isinstance(text, list):
        sentences = text
    else:
        sentences = text.split('\n')

    # Satırları temizle
    sentences = [s.strip() for s in sentences if s.strip()]

    # Embedding'leri oluştur
    embeddings = model.encode(sentences)
    return embeddings


def create_txt_embeddings(txt_path):
    """
    TXT dosyasından metin okur, embedding oluşturur, FAISS index döndürür.
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Satırlara böl (veya ihtiyacına göre paragraflara/kelime gruplarına da bölebilirsin)
    text_chunks = text.split('\n')

    # Boş satırları filtrele
    text_chunks = [chunk.strip() for chunk in text_chunks if chunk.strip()]

    # Embedding oluştur
    embeddings = model.encode(text_chunks)

    # FAISS index oluştur
    dim = embeddings.shape[1]  # embedding boyutunu al
    faiss_index = faiss.IndexFlatL2(dim)  # L2 mesafesi ile FAISS index oluştur
    faiss_index.add(np.array(embeddings).astype(np.float32))  # FAISS index'e embedding'leri ekle

    return faiss_index, text_chunks


def retrieve_top_k_similar_embeddings(query, embeddings, k=3):
    """
    Verilen sorguya en benzer k adet embedding'i döndürür.
    """
    query_embedding = model.encode([query])  # Sorgu için embedding oluştur
    similarities = np.dot(query_embedding, embeddings.T)  # Kosinüs benzerliği hesapla (dot product ile)
    top_k_idx = similarities.argsort()[0][-k:][::-1]  # En benzerleri sırala
    return top_k_idx
