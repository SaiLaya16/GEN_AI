https://medium.com/@sailaya1606/langchain-deep-dive-building-modular-llm-applications-from-scratch-ce345a945b1f

LangChain Deep Dive: Building Modular LLM Applications from Scratch
Sai laya B.
Sai laya B.
4 min read
·
Just now





1. Introduction to LangChain
The rise of Large Language Models (LLMs) like GPT has transformed how we build intelligent applications. However, using raw APIs alone quickly becomes messy when applications grow complex. This is where LangChain comes in.

What is LangChain?
LangChain is a framework designed to build applications powered by LLMs using modular, reusable components. Instead of writing scattered API calls, LangChain helps structure workflows into clean pipelines.

Why is it Important?
Modern AI apps require:

Context handling (memory)
Multi-step reasoning (chains)
External tools (APIs, databases)
Decision-making (agents)
LangChain provides all of this in a unified architecture.

2. Core Components of LangChain
Let’s break down each component with concept + code.

2.1 LLMs and Chat Models
Concept
LLMs are the core engines that generate text. LangChain wraps these models to standardize usage.

Why it Exists
Simplifies API interaction
Supports multiple providers (OpenAI, HuggingFace)
Code Example
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

response = llm.invoke("Explain LangChain in simple terms")
print(response.content)
2.2 Prompt Templates
Concept
Prompts are structured inputs sent to LLMs.

Why it Exists
Avoid hardcoding prompts
Enable dynamic input injection
Code Example
from langchain.prompts import PromptTemplate

template = PromptTemplate(
input_variables=[“topic”],
template=”Explain {topic} in simple terms”
)

prompt = template.format(topic=”LangChain”)
print(prompt)

2.3 Chains
Concept
Chains connect multiple steps into a pipeline.

Why it Exists
Automates multi-step workflows
Improves modularity
Code Example
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

chain = LLMChain(llm=llm, prompt=template)

output = chain.run(“Artificial Intelligence”)
print(output)

2.4 Memory
Concept
Memory stores previous interactions.

Why it Exists
Enables conversational AI
Maintains context
Code Example
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()

conversation = ConversationChain(
llm=llm,
memory=memory
)

print(conversation.run(“Hi, my name is Sai”))
print(conversation.run(“What is my name?”))

2.5 Agents
Concept
Agents decide what action to take dynamically.

Why it Exists
Handles complex reasoning
Chooses tools automatically
Code Example
from langchain.agents import initialize_agent, Tool

def calculator_tool(input):
return eval(input)

tools = [
Tool(
name=”Calculator”,
func=calculator_tool,
description=”Useful for math calculations”
)
]

agent = initialize_agent(
tools,
llm,
agent=”zero-shot-react-description”,
verbose=True
)

agent.run(“What is 25 * 4?”)

2.6 Tools
Concept
Tools are external functions/APIs that agents use.

Why it Exists
Extends LLM capabilities
Enables real-world interaction
Example:

Calculator
Search APIs
Database queries
2.7 Document Loaders
Concept
Loads external data into LangChain.

Why it Exists
Enables knowledge-based apps
Supports PDFs, CSVs, web pages
Code Example

Become a Medium member
from langchain.document_loaders import TextLoader

loader = TextLoader(“data.txt”)
documents = loader.load()

print(documents[0].page_content)

2.8 Vector Stores (Indexes)
Concept
Stores embeddings for semantic search.

Why it Exists
Enables retrieval-based QA
Supports RAG (Retrieval Augmented Generation)
Code Example
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

db = FAISS.from_texts(
[“LangChain is powerful”, “AI is the future”],
embeddings
)

query = “What is LangChain?”
results = db.similarity_search(query)

print(results[0].page_content)

3. Architecture Explanation
The flow of a LangChain application looks like this:

Press enter or click to view image in full size

Flow Breakdown
User Input
Prompt Template
LLM Processing
Chain Execution
Agent Decision
Tool Usage
Final Output
This modular flow makes applications scalable and maintainable.

4. Hands-on Code Implementation

Basic LLM Call

response = llm.invoke(“What is Machine Learning?”)
print(response.content)

Prompt Template + Chain

template = PromptTemplate(
input_variables=[“topic”],
template=”Explain {topic} in 3 bullet points”
)

chain = LLMChain(llm=llm, prompt=template)

print(chain.run(“Deep Learning”))

Memory Example

memory = ConversationBufferMemory()

conversation = ConversationChain(llm=llm, memory=memory)

conversation.run(“My favorite subject is AI”)
print(conversation.run(“What is my favorite subject?”))

Agent with Tool

agent.run(“Calculate 15 * 8 and explain the result”)

5. Real-World Use Cases
1. AI Chatbot
Problem: Customer support automation
Solution: Use conversation chain + memory
Components: LLM + Memory + PromptTemplate

2. Document Q&A System (RAG)
Problem: Answer questions from PDFs
Solution: Use vector store + retriever + LLM
Components: Document Loader + Embeddings + Vector Store

3. Smart Data Assistant
Problem: Query databases using natural language
Solution: Use agent with SQL tool
Components: Agent + Tools + LLM

6. Advantages and Limitations
Advantages
Modular architecture
Easy integration with APIs
Rapid prototyping
Supports RAG and Agents

Limitations
High latency for multi-step chains
Debugging is complex
Cost increases with API calls
Requires careful prompt engineering

When NOT to Use LangChain
Simple one-shot LLM tasks
Low-latency production systems
When custom lightweight pipelines are enough


7. Conclusion
LangChain is a powerful framework that transforms raw LLMs into structured, production-ready applications.

Key Takeaways
Chains simplify workflows
Memory enables context-aware apps
Agents bring intelligence and autonomy
Vector stores enable RAG systems
Future Scope
LangGraph for advanced workflows
Multi-agent systems
Autonomous AI applications
LLM
Llm Agent
Langchain
Chatmodels
