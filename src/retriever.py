import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from src.config import (
    VECTORSTORE_DIR,
    CATEGORIES,
    EMBEDDING_MODEL_NAME,
    RELEVANCE_SCORE_THRESHOLD,
    TOP_K_RETRIEVAL,
)

embed = HuggingFaceEndpointEmbeddings(
    model=EMBEDDING_MODEL_NAME, task="feature-extraction"
)


def load_vector_stores():
    vectorstore_map = {}
    for cat in CATEGORIES:
        persist_path = os.path.join(VECTORSTORE_DIR, f"{cat.capitalize()}VecStore")
        if os.path.exists(persist_path):
            vectorstore_map[cat] = Chroma(
                persist_directory=persist_path,
                collection_name=cat.capitalize(),
                embedding_function=embed,
            )
    return vectorstore_map


def retrieve_relevant_docs(vector_store, query: str, k: int = TOP_K_RETRIEVAL):
    results = vector_store.similarity_search_with_relevance_scores(query, k=k)
    relevant_docs = [
        doc for doc, score in results if score >= RELEVANCE_SCORE_THRESHOLD
    ]
    return relevant_docs