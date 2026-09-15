import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from src.config import (
    DATA_DIR,
    VECTORSTORE_DIR,
    CATEGORIES,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

load_dotenv()

def run_ingestion():
    embed = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL_NAME, task="feature-extraction"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    for cat in CATEGORIES:
        cat_dir = os.path.join(DATA_DIR, cat)
        if not os.path.exists(cat_dir):
            continue

        pdf_files = [f for f in os.listdir(cat_dir) if f.lower().endswith(".pdf")]
        if not pdf_files:
            continue

        cat_docs = []
        for file_name in pdf_files:
            file_path = os.path.join(cat_dir, file_name)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["category"] = cat
            cat_docs.extend(docs)

        if cat_docs:
            split_docs = splitter.split_documents(cat_docs)
            persist_path = os.path.join(VECTORSTORE_DIR, f"{cat.capitalize()}VecStore")
            Chroma.from_documents(
                documents=split_docs,
                persist_directory=persist_path,
                collection_name=cat.capitalize(),
                embedding=embed,
            )