# RAG_Med_Chatbot 
An advanced **Retrieval-Augmented Generation (RAG) chatbot** built with **LangChain, FAISS, Groq API, Wikipedia, and BioBERT**, designed to provide **medical insights** by retrieving information from a preloaded medical book and Wikipedia.

---

## **🚀 Features**
### ✅ **Retrieval-Augmented Generation (RAG)**  
- Uses **FAISS (Facebook AI Similarity Search)** to store and retrieve medical knowledge from a preloaded **medical textbook**.  

### ✅ **BioBERT-Based Embeddings**  
- Uses **dmis-lab/biobert-v1.1** for **medical text embeddings**, improving response accuracy.  

### ✅ **Wikipedia Integration**  
- Automatically **fetches medical information from Wikipedia** when no direct answer is found in the textbook.  

### ✅ **Groq API (Llama3 / Mixtral Models)**  
- Provides **AI-generated medical insights** using **Groq’s Llama3 and Mixtral models**.  

### ✅ **Optimized FAISS Caching**  
- Uses **Streamlit's caching system** (`@st.cache_resource`) to prevent reloading FAISS unnecessarily.  

### ✅ **Preloaded Medical Textbook**  
- The medical textbook is **embedded internally** and **not uploaded through the frontend** for efficiency.  

---

## **📂 Project Structure**
```
medical-rag-chatbot/
│── env/                      # Virtual environment (optional)
│── medical_book.pdf          # Preloaded medical textbook
│── faiss_index.pkl           # FAISS vector database
│── .env                      # API keys (Groq API key required)
│── app.py                    # Main Streamlit app
│── requirements.txt          # Required dependencies
│── README.md                 # Project details
```

---

## **🛠️ Installation**
### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/yourusername/medical-rag-chatbot.git
cd medical-rag-chatbot
```

### 2️⃣ **Create a Virtual Environment (Optional)**
```bash
python -m venv env
source env/bin/activate  # On MacOS/Linux
env\Scripts\activate     # On Windows
```

### 3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4️⃣ **Set Up API Keys**
Create a `.env` file in the project root and add:
```
GROQ_API_KEY=your-groq-api-key
```

### 5️⃣ **Run the Application**
```bash
streamlit run app.py
```

---

## **📌 Usage**
1. Open **Streamlit UI** in your browser.  
2. **Ask medical questions** in the chat.  
3. The chatbot will **retrieve relevant context** from the medical textbook.  
4. If needed, it will **fetch extra information from Wikipedia**.  
5. **AI generates a response** using Groq’s Llama3/Mixtral models.  

---

## **🔧 Future Enhancements**
🔹 **ICD-10 Disease Classification**  
🔹 **Voice-Based Chatbot (Speech-to-Text & Text-to-Speech)**  
🔹 **Integration with PubMed, WHO, CDC Databases**  
🔹 **Multimodal AI (Images & Medical Reports Processing)**  

---

## **📝 License**
This project is **open-source**. Feel free to contribute!  

---

### **🚀 Ready to push this to GitHub?**
Use these commands:  
```bash
git init
git add .
git commit -m "Initial commit - Medical RAG Chatbot"
git branch -M main
git remote add origin https://github.com/yourusername/medical-rag-chatbot.git
git push -u origin main
```

---
## **License**  

This project is licensed under the **MIT License**.  

---

## **Contributors**  

👨‍💻 Developed by **Arnab Mukherjee**  
📧 Contact: **arnabjaymukherjee@gmail.com**  
