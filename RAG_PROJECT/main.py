# RAG + LangGraph Customer Support Bot

# Install required packages before running:
# pip install langchain langchain-community langgraph chromadb sentence-transformers pypdf


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langgraph.graph import StateGraph

from typing import TypedDict
import os

# Step 1: Load PDF

PDF_PATH = "data/customer_support.pdf"   

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# Step 2: Chunking

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

# Step 3: Embeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 4: Vector DB (Chroma)

db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# Step 5: Define State (LangGraph)

class GraphState(TypedDict):
    query: str
    context: str
    answer: str
    confidence: float
    escalate: bool

# Step 6: Processing Node

def process_node(state: GraphState) -> GraphState:
    query = state["query"]

    docs = retriever.invoke(query)

    if docs:
        context = "\n".join([doc.page_content for doc in docs])
        confidence = 0.8
    else:
        context = ""
        confidence = 0.3

    if context:
        q = query.lower()

        if "refund" in q:
            answer = "Customers can request a refund within 7 days of purchase."
        elif "shipping" in q or "delivery" in q:
            answer = "Delivery takes 3–5 business days."
        elif "cancel" in q:
            answer = "Orders can be cancelled before they are shipped."
        elif "payment" in q:
            answer = "We accept UPI, Debit Card, Credit Card, and Net Banking."
        elif "support" in q or "contact" in q:
            answer = "You can contact support at support@example.com."
        else:
            answer = "I found relevant information in the document, but couldn't match your query exactly."
    else:
        answer = "Sorry, I couldn't find relevant information."
    return {
        "query": query,
        "context": context,
        "answer": answer,
        "confidence": confidence,
        "escalate": False
    }

# Step 7: Routing Logic

def route(state: GraphState):
    if state["confidence"] < 0.5:
        return "escalate"
    return "answer"

# Step 8: HITL Node

def human_node(state: GraphState) -> GraphState:
    return {
        **state,
        "answer": "⚠️ Escalated to human support. Please wait for assistance.",
        "escalate": True
    }

# Step 9: Build LangGraph

graph = StateGraph(GraphState)

graph.add_node("process", process_node)
graph.add_node("answer", lambda x: x)
graph.add_node("escalate", human_node)

graph.set_entry_point("process")

graph.add_conditional_edges(
    "process",
    route,
    {
        "answer": "answer",
        "escalate": "escalate"
    }
)

app = graph.compile()

# Step 10: CLI Interface

def run_chat():
    print("\n🤖 RAG Customer Support Bot (type 'exit' to quit)\n")

    while True:
        query = input("You: ")

        if query.lower() == "exit":
            print("Goodbye!")
            break

        result = app.invoke({
            "query": query,
            "context": "",
            "answer": "",
            "confidence": 0.0,
            "escalate": False
        })

        print(f"\nBot: {result["answer"]} \n")


if __name__ == "__main__":
    run_chat()