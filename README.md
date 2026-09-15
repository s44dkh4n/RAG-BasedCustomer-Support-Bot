# Multi-Retriever Customer Support RAG Bot

A modular, domain-routed Customer Support AI Assistant built with LangChain, ChromaDB, Semantic Router, and Streamlit.

The application intelligently routes incoming user inquiries across technical, billing, and shipping domains using a hybrid classification approach (Semantic Router with LLM fallback), retrieves context with strict relevance filtering, and answers questions using Llama 3.1.

---

## Key Features

* **Hybrid Domain Routing**: Fast vector-based classification via `semantic-router` with automatic fallback to Llama 3.1 for low-confidence queries.
* **Domain-Isolated Retrieval**: PDF documents are indexed into separate Chroma vector store collections (`technical`, `billing`, `shipping`).
* **Relevance Thresholding**: Drops weakly relevant document chunks using similarity score filtering (`similarity_search_with_relevance_scores`) before sending context to the LLM.
* **Modular Architecture**: Complete separation of concerns between ingestion, configuration, vector retrieval, domain routing, and UI presentation.
* **Streamlit Chat Interface**: Interactive web interface featuring session-persisted message history and routed category indicators.

---

## Project Structure

```text
customer-support-rag/
├── data/
│   ├── billing/        # Place billing PDFs here
│   ├── shipping/       # Place shipping PDFs here
│   └── technical/      # Place technical PDFs here
├── vector_stores/      # Generated ChromaDB collections
├── src/
│   ├── __init__.py     # Package initialization
│   ├── config.py       # Global parameters and path setups
│   ├── ingestion.py    # PDF loading, splitting, and vector indexing
│   ├── rag_chain.py    # LangChain prompt and LLM pipeline initialization
│   ├── retriever.py  # Vector store loading and relevance scoring
│   └── router.py     # Hybrid semantic + LLM fallback router
├── .env                # Environment variables (API tokens)
├── app.py              # Streamlit Web UI application
├── ingest_cli.py       # CLI tool to run document ingestion
└── requirements.txt    # Python dependencies
```

# Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/customer-support-rag.git](https://github.com/your-username/customer-support-rag.git)
cd customer-support-rag
```

2. Set Up Virtual Environment
```
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

3. Configure Environment Variables
Create a .env file in the root directory and put your API keys in it.

4. Run Document Ingestion
Execute the ingestion script to chunk PDFs, compute embeddings, and build local Chroma vector stores inside vector_stores/:
```
python ingest_cli.py
```

5. Launch the Web Application
```
streamlit run app.py
```
### Note: This is a demo project created just to apply the techniques i Learned and the performance of the ChatBot is not exemplary.
