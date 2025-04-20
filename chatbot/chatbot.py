from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from config import MODEL_NAME, HF_TOKEN, TXT_PATH
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
llm = pipeline("text-generation", model=model, tokenizer=tokenizer, truncation=True)


def load_text_from_txt(txt_path):
    """
    TXT dosyasından metni okur ve satır satır döner.
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.split("\n")  # Satır bazlı ayırıyoruz, istersen başka bölme yöntemi de kullanılabilir


def create_txt_embeddings(txt_path):
    """
    TXT dosyasından metinleri okur ve FAISS için embedding'leri oluşturur.
    """
    text_chunks = load_text_from_txt(txt_path)
    txt_embeddings = create_embeddings(text_chunks)

    dim = len(txt_embeddings[0])
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(np.array(txt_embeddings).astype(np.float32))

    return faiss_index, text_chunks


def get_answer_from_llm(query):
    """
    Kullanıcı sorgusuna Hugging Face Llama modelini kullanarak cevap döndürür.
    """
    result = llm(query, max_length=150)
    return result[0]['generated_text'].strip()


def get_answer_from_rag(query, faiss_index, text_chunks):
    """
    RAG için benzer metinleri getir
    """
    # Metinler için embeddings oluşturulmuşsa, burada create_embeddings kullanmaya gerek yoktur
    txt_embeddings = create_embeddings(text_chunks)

    # Top-k benzer indeksleri al
    top_k_idx = retrieve_top_k_similar_embeddings(query, txt_embeddings, k=5)  # K sayısını ihtiyaca göre değiştirebilirsiniz

    # Bağlam metnini oluştur
    context = "\n".join([text_chunks[i] for i in top_k_idx])

    # Prompt template hazırla
    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template="Bağlam: {context}\nSoru: {query}\nCevap:"
    )

    llm_chain = LLMChain(llm=HuggingFacePipeline(pipeline=llm), prompt=prompt)

    # Sorguyu gönder
    response = llm_chain.run({"context": context, "query": query})
    return response.strip()
