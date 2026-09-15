from dotenv import load_dotenv

load_dotenv()

from src.ingestion import run_ingestion

if __name__ == "__main__":
    print("Starting document ingestion and vector store creation...")
    run_ingestion()
    print("Ingestion complete. Vector stores saved in vector_stores directory.")