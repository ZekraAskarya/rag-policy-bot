import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Page Configuration (Sets the browser tab title and icon)
st.set_page_config(page_title="Company Policy Bot",
                   page_icon="📄", layout="centered")
st.title("📄 Company Policy Assistant")
st.markdown("Ask me anything about the company policy document!")

# 2. Load API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 3. Initialize the RAG Chain (We use @st.cache_resource so it only loads once, not on every button click)


@st.cache_resource
def load_rag_chain():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key,
        transport="rest"
    )
    vectorstore = Chroma(persist_directory="./chroma_db",
                         embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        transport="rest"
    )

    template = """
    Answer the question based ONLY on the following context:
    {context}
    
    Question: {question}
    
    If the answer is not in the context, say "I don't know based on the provided document."
    """
    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


rag_chain = load_rag_chain()

# 4. Initialize Chat History in the Web Session
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Handle New User Input
if prompt := st.chat_input("Example: What is the STOP principle?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display AI response
    with st.chat_message("assistant"):
        with st.spinner("Searching policy..."):
            response = rag_chain.invoke(prompt)
            st.markdown(response)

    # Add AI response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response})
