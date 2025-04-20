# chatbot/chatbot.py
import openai
from config import MODEL_NAME
from pdf_processing.pdf_extraction import extract_text_from_pdf
from embeddings.embeddings import create_embeddings, retrieve_top_k_similar_embeddings

# OpenAI API anahtarınızı buraya eklemelisiniz
openai.api_key = 'your_openai_api_key'

# PDF metnini çıkarma ve embedding işlemi
pdf_text = extract_text_from_pdf('data/your_pdf_file.pdf')
pdf_embeddings = create_embeddings(pdf_text)


def get_answer_from_llm(query):
    # LLM kullanarak cevap almak
    response = openai.Completion.create(
        model=MODEL_NAME,
        prompt=query,
        max_tokens=150
    )
    return response.choices[0].text.strip()


def get_answer_from_rag(query):
    # RAG için benzer metinleri getir
    top_k_idx = retrieve_top_k_similar_embeddings(query, pdf_embeddings)
    context = "\n".join([pdf_text.split('\n')[i] for i in top_k_idx])

    # Bağlamla birlikte LLM'e sorgu gönder
    query_with_context = f"Bağlam: {context}\n\nSoru: {query}\nCevap:"

    response = openai.Completion.create(
        model=MODEL_NAME,
        prompt=query_with_context,
        max_tokens=150
    )
    return response.choices[0].text.strip()

