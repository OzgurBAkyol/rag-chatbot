from chatbot.chatbot import get_answer_from_llm, get_answer_from_rag
from embeddings.embeddings import create_txt_embeddings
from config import TXT_PATH


def main():
    print("TXT embedding'leri oluşturuluyor...")
    faiss_index, text_chunks = create_txt_embeddings(TXT_PATH)

    user_query = input("Gregor Samsa neden işe gitmek istemiyor?\n> ")

    print("\nNormal LLM Cevabı:")
    llm_response = get_answer_from_llm(user_query)
    print(f"LLM: {llm_response}")

    print("\nRAG Cevabı:")
    rag_response = get_answer_from_rag(user_query, faiss_index, text_chunks)
    print(f"RAG: {rag_response}")


if __name__ == "__main__":
    main()
