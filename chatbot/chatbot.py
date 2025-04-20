# chatbot/chatbot.py
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from config import MODEL_NAME, HF_TOKEN
from pdf_processing.pdf_extraction import extract_text_from_pdf
from embeddings.embeddings import create_embeddings, retrieve_top_k_similar_embeddings
import faiss
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline


# Hugging Face modelini ve token'ı kullanarak model ve tokenizer'ı yükleyin
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)

# Hugging Face pipeline'ını kurma
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

# PDF metnini çıkarma ve embedding işlemi
pdf_text = extract_text_from_pdf('data/franz-kafka-donusum-tr.pdf')
pdf_embeddings = create_embeddings(pdf_text)


def get_answer_from_llm(query):
    """
    Kullanıcı sorgusuna Hugging Face Llama modelini kullanarak cevap döndürür.
    """
    result = llm(query, max_length=150)
    return result[0]['generated_text'].strip()


def get_answer_from_rag(query):
    """
    RAG için benzer metinleri getir
    """
    # FAISS indexini oluştur
    faiss_index = create_pdf_embeddings('data/your_pdf_file.pdf')

    # Benzer metinleri al
    top_k_idx = retrieve_top_k_similar_embeddings(query, faiss_index)
    context = "\n".join([pdf_text.split('\n')[i] for i in top_k_idx])

    # Bağlamla birlikte LLM'e sorgu gönder
    query_with_context = f"Bağlam: {context}\n\nSoru: {query}\nCevap:"

    # LangChain kullanarak soruyu yönlendirme
    prompt = PromptTemplate(input_variables=["context", "query"], template="Bağlam: {context}\nSoru: {query}\nCevap:")
    llm_chain = LLMChain(llm=HuggingFacePipeline(pipeline=llm), prompt=prompt)

    response = llm_chain.run({"context": context, "query": query_with_context})
    return response.strip()


def create_pdf_embeddings(pdf_path):
    """
    PDF dosyasını alır, metni çıkarır ve metinden vektörler oluşturur.
    FAISS indexi döndürür.
    """
    # PDF'ten metin çıkar
    pdf_text = extract_text_from_pdf(pdf_path)

    # Metinden embedding oluştur
    pdf_embeddings = create_embeddings(pdf_text)

    # FAISS index oluştur
    dim = len(pdf_embeddings[0])  # Embedding boyutu
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(np.array(pdf_embeddings).astype(np.float32))

    return faiss_index
