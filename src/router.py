from semantic_router import Route, SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder
from langchain_core.prompts import PromptTemplate
from src.config import EMBEDDING_MODEL_NAME, SEMANTIC_ROUTER_THRESHOLD
from src.rag_chain import model, parser

encoder = HuggingFaceEncoder(name=EMBEDDING_MODEL_NAME)

technical_route = Route(
    name="technical",
    utterances=[
        "How do I reset my password?",
        "How to troubleshoot API setup issues?",
        "I am getting an error when logging into my account",
        "Where can I find installation instructions?",
        "Application throws a runtime error",
        "API endpoint returns 500 status code",
        "Cannot establish database connection",
        "Where are the system logs stored?",
        "How do I configure my environment variables?",
        "Integration failing with invalid credentials error",
        "Port conflict error on server startup",
        "Software keeps crashing unexpectedly",
    ],
)

billing_route = Route(
    name="billing",
    utterances=[
        "I was double charged on my invoice",
        "What is your refund policy?",
        "How do I upgrade or cancel my subscription plan?",
        "My credit card payment was rejected",
        "Can I get a copy of my billing receipt?",
        "Where can I update my credit card details?",
        "Why was I charged an extra fee this month?",
        "Do you offer annual payment discounts?",
        "How do I request tax exempt status on billing?",
        "Can I switch from monthly to annual billing?",
        "My payment failed but money was deducted",
        "How long does a refund processing take?",
    ],
)

shipping_route = Route(
    name="shipping",
    utterances=[
        "Where is my order delivery package?",
        "How do I return a damaged product?",
        "What are your delivery times and shipping rates?",
        "My tracking number shows delayed status",
        "How do I issue an exchange or product return?",
        "Package says delivered but I have not received it",
        "Do you provide international shipping options?",
        "How can I change my delivery address for an active order?",
        "Carrier lost my shipment in transit",
        "What courier service do you use for express shipping?",
        "How do I print a return shipping label?",
        "What is the return window for purchased items?",
    ],
)

routes = [technical_route, billing_route, shipping_route]

# Initialize router and explicitly index the routes
route_layer = SemanticRouter(encoder=encoder)
route_layer.add(routes=routes)

llm_router_prompt = PromptTemplate.from_template(
    """Classify the following question into exactly one category: 'technical', 'billing', or 'shipping'.
If the question does not match any specifically, output 'billing'.
Return ONLY the single category name in lowercase with no punctuation or extra words.

Question: {question}
Category:"""
)

llm_router_chain = llm_router_prompt | model | parser


def query_router_hybrid(query: str, vectorstore_map: dict):
    route_choice = route_layer(query)
    category = getattr(route_choice, "name", None)
    score = getattr(route_choice, "similarity_score", 0.0) or 0.0

    if category in vectorstore_map and score >= SEMANTIC_ROUTER_THRESHOLD:
        return category, vectorstore_map[category]

    fallback_category = llm_router_chain.invoke({"question": query}).strip().lower()

    if fallback_category in vectorstore_map:
        return fallback_category, vectorstore_map[fallback_category]

    return "billing", vectorstore_map.get("billing")