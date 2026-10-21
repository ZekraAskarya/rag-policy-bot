import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API Key not found! Check your .env file.")

# 2. Connect to the Database (Must use the same embedding model as ingest.py!)
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001", google_api_key=api_key, transport="rest")

# Load the database we just created
vectorstore = Chroma(persist_directory="./chroma_db",
                     embedding_function=embeddings)

# Create a retriever that gets the top 3 most relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. Initialize the LLM (Using Gemini 3.8 Flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash", google_api_key=api_key, transport="rest")

# 4. Create the Prompt
template = """
Answer the question based ONLY on the following context:
{context}

Question: {question}

If the answer is not in the context, say "I don't know based on the provided document."
"""
prompt = PromptTemplate.from_template(template)

# 5. Build the RAG Chain


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Chat Loop (Debug Version)
print("🤖 RAG Bot is ready! Type 'quit' to exit.")
while True:
    query = input("\nYou: ")
    if query.lower() == 'quit':
        break

    print("🔍 Searching the database...")
    docs = retriever.invoke(query)

    # --- DEBUG PRINTING ---
    print(f"\n[DEBUG] Found {len(docs)} chunks. Here they are:")
    for i, doc in enumerate(docs):
        print(f"--- Chunk {i+1} ---")
        # Print first 200 chars of each chunk
        print(doc.page_content[:200].replace('\n', ' ') + "...")
    print("---------------------\n")
    # ----------------------

    print("Asking Gemini...")
    try:
        response = rag_chain.invoke(query)
        print(f"\nAI: {response}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
