# AirBot

An AI-powered chatbot built on top of a Large Language Model (LLM) enhanced with Retrieval-Augmented Generation (RAG).
This chatbot answers questions using both pretrained knowledge and a custom knowledge base.

---

##  Features

- Natural language conversations
- Custom knowledge base integration
- Retrieval-Augmented Generation (RAG)
- Context-aware responses
- Lightweight and efficient architecture
- Can run locally (no mandatory cloud dependency)

---

##  Architecture Overview

User Query  
↓  
Retriever (Searches Knowledge Base)  
↓  
Relevant Context Extracted  
↓  
LLM (Generates Answer using Retrieved Context)  
↓  
Final Response  

---

##  Tech Stack

- Python
- Vector Database (FAISS / Chroma)
- Embedding Model
- Open-source LLM using Ollama
- CLI Interface

---

## 📂 Project Structure

project/
│── app.py
│── retriever.py
│── embeddings.py
│── vector_store/
│── knowledge_base/
│── requirements.txt
│── README.md

---

##  Installation

git clone <your-repo>
cd project
pip install -r requirements.txt

---

## ▶ Running the Chatbot

python app.py

---

##  How It Works

1. User asks a question.
2. The retriever searches the vector database.
3. Relevant documents are fetched.
4. The LLM uses those documents as context.
5. A grounded response is generated.

---
