from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from config import MODEL_NAME, HF_TOKEN, TXT_PATH
from embeddings.embeddings import create_embeddings, retrieve_top_k_similar_embeddings
import faiss
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline


model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)

llm = pipeline("text-generation", model=model, tokenizer=tokenizer, truncation=True)


def load_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.split("\n")


def create_txt_embeddings(txt_path):
    text_chunks = load_text_from_txt(txt_path)
    txt_embeddings = create_embeddings(text_chunks)

    dim = len(txt_embeddings[0])
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(np.array(txt_embeddings).astype(np.float32))

    return faiss_index, text_chunks


def get_answer_from_llm(query):
    result = llm(query, max_length=150)
    return result[0]['generated_text'].strip()


def get_answer_from_rag(query, faiss_index, text_chunks):
    txt_embeddings = create_embeddings(text_chunks)

    top_k_idx = retrieve_top_k_similar_embeddings(query, txt_embeddings, k=5)  # K sayısını ihtiyaca göre değiştirebilirsiniz

    context = "\n".join([text_chunks[i] for i in top_k_idx])

    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template="Bağlam: {context}\nSoru: {query}\nCevap:"
    )

    llm_chain = LLMChain(llm=HuggingFacePipeline(pipeline=llm), prompt=prompt)

    response = llm_chain.run({"context": context, "query": query})
    return response.strip()
