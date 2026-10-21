import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# 1. Load API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# ADD THIS SAFETY CHECK:
if not api_key:
    raise ValueError("API Key not found! Please check your .env file.")

# 2. Load the PDF (Make sure your file is named exactly policy.pdf)
print("Loading PDF...")
loader = PyPDFLoader("policy.pdf")
docs = loader.load()

# 3. Split into chunks
print("Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

# 4. Convert chunks to embeddings and save to Vector Database (Chroma)
print("Saving to vector database...")

# Workaround for the DefaultCredentialsError
# Explicitly set the environment variable the Google SDK expects
os.environ["GOOGLE_API_KEY"] = api_key

embeddings = GoogleGenerativeAIEmbeddings(
    # Force the use of REST transport
    model="gemini-embedding-001", google_api_key=api_key)

vectorstore = Chroma.from_documents(
    documents=chunks, embedding=embeddings, persist_directory="./chroma_db")

print("✅ Ingestion complete! Database is ready.")
