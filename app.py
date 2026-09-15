from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from src.rag_chain import format_docs, rag_chain
from src.retriever import load_vector_stores, retrieve_relevant_docs
from src.router import query_router_hybrid

st.set_page_config(page_title="Customer Support Assistant")
st.title("Customer Support AI Assistant")


@st.cache_resource
def get_vector_stores():
    return load_vector_stores()


vectorstore_map = get_vector_stores()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input(
    "Ask a question about billing, shipping, or technical issues..."
)

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Processing query..."):
            category, vector_store = query_router_hybrid(
                user_query, vectorstore_map
            )

            if vector_store is None:
                response_text = (
                    "Support vector stores are not loaded. Please run ingestion first."
                )
            else:
                retrieved_docs = retrieve_relevant_docs(vector_store, user_query)
                context = format_docs(retrieved_docs)
                response_text = rag_chain.invoke(
                    {"question": user_query, "context": context}
                )
                st.caption(f"Routed Category: {category.capitalize()}")

            st.markdown(response_text)
            st.session_state.messages.append(
                {"role": "assistant", "content": response_text}
            )