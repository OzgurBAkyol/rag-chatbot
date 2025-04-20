from chatbot.chatbot import get_llm_response, get_rag_response, create_pdf_embeddings
from config import PDF_PATH


def main():
    # PDF embedding'lerini oluştur
    faiss_index = create_pdf_embeddings(PDF_PATH)

    # Kullanıcıdan sorgu alın
    user_query = input("Hangi türde bir film izlemek istersiniz? ")

    # Normal LLM ile cevap al
    print("\nNormal LLM Cevabı:")
    llm_response = get_llm_response(user_query)
    print(f"LLM: {llm_response}")

    # RAG ile cevap al
    print("\nRAG Cevabı:")
    rag_response = get_rag_response(user_query, faiss_index)
    print(f"RAG: {rag_response}")


if __name__ == "__main__":
    main()
