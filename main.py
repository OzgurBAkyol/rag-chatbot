# main.py
from chatbot.chatbot import get_answer_from_llm, get_answer_from_rag, create_pdf_embeddings
from config import PDF_PATH

def main():
    # PDF embedding'lerini oluştur
    print("PDF embedding'leri oluşturuluyor...")
    faiss_index = create_pdf_embeddings(PDF_PATH)

    # Kullanıcıdan sorgu alın
    user_query = input("Gregor Samsa neden işe gitmek istemiyor?")

    # Normal LLM ile cevap al
    print("\nNormal LLM Cevabı:")
    llm_response = get_answer_from_llm(user_query)
    print(f"LLM: {llm_response}")

    # RAG ile cevap al
    print("\nRAG Cevabı:")
    rag_response = get_answer_from_rag(user_query, faiss_index)
    print(f"RAG: {rag_response}")


if __name__ == "__main__":
    main()