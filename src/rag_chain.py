from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import LLM_REPO_ID

llm = HuggingFaceEndpoint(
    repo_id=LLM_REPO_ID, task="text-generation", temperature=0.1
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

rag_prompt = ChatPromptTemplate.from_template(
    """You are an AI customer support assistant.
Your job is to answer the user's question using ONLY the provided context.
If the answer is not available in the context, say:
'I could not find that information in the support documents.'

Rules:
1. Be clear and professional.
2. Do not make up policies, prices, or procedures.
3. If steps are needed, present them in numbered form.
4. If the context includes policy conditions, mention them clearly.
5. Keep the answer focused on the user's question.

Question -> {question}

Context -> {context}

Answer -> """
)

rag_chain = rag_prompt | model | parser


def format_docs(docs):
    if not docs:
        return "No relevant context found."
    return "\n\n".join(doc.page_content for doc in docs)