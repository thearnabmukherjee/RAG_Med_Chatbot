import os
import pickle
import faiss
import requests
import wikipediaapi
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Get Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ ERROR: Groq API key is missing! Check your .env file.")
    st.stop()

# Available Groq models
GROQ_MODELS = {
    "Llama3-8B": "llama3-8b-8192",
    "Llama3-70B": "llama3-70b-8192",
    "Mixtral-8x7B": "mixtral-8x7b-32768"
}

# Streamlit Sidebar
st.sidebar.title("⚙️ Settings")
selected_model = st.sidebar.selectbox("Choose AI Model:", list(GROQ_MODELS.keys()))

# **Embeddings Model (BioBERT)**
EMBEDDING_MODEL = "dmis-lab/biobert-v1.1"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# **Wikipedia API Setup**
wiki = wikipediaapi.Wikipedia(user_agent="MyMedicalBot/1.0 (contact: myemail@example.com)", language="en")

# **Path to Preloaded Medical Book**
book_path = r"D:\WEB PROJECT\Langchain\Learnings\nuc_med.pdf"

# **Load and Process the Book**
@st.cache_resource
def load_book(_book_path):
    """Loads and splits the book into smaller chunks for embedding."""
    try:
        loader = PyPDFLoader(_book_path)
        documents = loader.load()

        if not documents:
            st.error("❌ ERROR: No pages loaded. Check the file path.")
            return []

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        return texts

    except Exception as e:
        st.error(f"❌ ERROR: Failed to load PDF - {str(e)}")
        return []

# **Load or Create FAISS Index**
@st.cache_resource
def create_faiss_index(_texts):
    """Loads or creates a FAISS vector store."""
    index_path = "faiss_index.pkl"

    if os.path.exists(index_path):
        with open(index_path, "rb") as f:
            faiss_index = pickle.load(f)
    else:
        if not _texts:
            st.error("❌ ERROR: No text found to create FAISS index.")
            return None

        faiss_index = FAISS.from_documents(_texts, embeddings)
        
        with open(index_path, "wb") as f:
            pickle.dump(faiss_index, f)

    return faiss_index

# **Load Book and FAISS Index**
texts = load_book(book_path)
faiss_index = create_faiss_index(texts)
retriever = faiss_index.as_retriever() if faiss_index else None

# **Wikipedia Retrieval**
def search_wikipedia(query):
    """Search Wikipedia for relevant medical information."""
    try:
        page = wiki.page(query)
        if page.exists():
            return page.summary[:1000]  # Return first 1000 characters
    except Exception as e:
        return f"⚠️ Wikipedia fetch error: {str(e)}"
    return "No Wikipedia data found."

# **Fetch AI Response**
def get_groq_response(messages):
    """Sends messages to the Groq API and gets a response."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
    data = {"model": GROQ_MODELS[selected_model], "messages": messages, "max_tokens": 500}
    
    response = requests.post(url, json=data, headers=headers)
    return response.json().get("choices", [{}])[0].get("message", {}).get("content", "Error fetching response.")

# **Streamlit UI**
st.title("💬 Medical RAG Chatbot")
st.write("Ask anything related to medicine!")

# **Chat History**
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# **Display Chat History**
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# **User Input**
user_input = st.chat_input("Type your medical question...")

if user_input:
    # Store user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve relevant context from FAISS
    context = ""
    if retriever:
        past_messages = retriever.get_relevant_documents(user_input)
        context = "\n".join([msg.page_content for msg in past_messages])

    if not context:
        st.warning("⚠️ No relevant context found in the book. Answer may be generic.")

    # Fetch Wikipedia Data
    wikipedia_data = search_wikipedia(user_input)

    # **Build Conversation with Retrieved Context**
    messages = [{"role": "system", "content": "You are a medical AI assistant."}]
    
    for msg in st.session_state.chat_history[-5:]:  # Keep last 5 messages
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": f"Context: {context}\nWikipedia: {wikipedia_data}\nUser Query: {user_input}"})

    # **Get AI Response**
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_groq_response(messages)
            st.markdown(response)

    # Store AI response in history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# **Sidebar: Reprocess Book**
if st.sidebar.button("🔄 Reprocess Book"):
    texts = load_book(book_path)
    faiss_index = create_faiss_index(texts)
    retriever = faiss_index.as_retriever() if faiss_index else None
    st.sidebar.success("✅ Book reprocessed successfully!")
