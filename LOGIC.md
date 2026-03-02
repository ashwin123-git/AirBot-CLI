#  Training Process & RAG Implementation

This project enhances an existing Large Language Model (minimax-m2.5:cloud) using Retrieval-Augmented Generation (RAG).
Instead of fine-tuning model weights, it dynamically injects relevant knowledge at runtime.

---

##  What is RAG?

Retrieval-Augmented Generation (RAG) is a technique where:

1. A knowledge base is created.
2. Documents are converted into vector embeddings.
3. A retriever finds relevant documents for a query.
4. The LLM generates answers using retrieved context.

---

##  Training Approach Used

### Step 1 — Data Collection
- Gathered domain-specific documents
- Cleaned and structured the data

### Step 2 — Text Chunking
- Split documents into smaller chunks
- Added overlap for better semantic continuity

### Step 3 — Embedding Generation
- Used an embedding model to convert text into vectors
- Stored vectors in a vector database

### Step 4 — Vector Database
- Stored embeddings in Chroma
- Enabled fast similarity search

### Step 5 — Retrieval
- Convert query into embedding
- Perform similarity search
- Retrieve top-k relevant chunks

### Step 6 — Augmented Prompting
- Inject retrieved context into prompt
- Send augmented prompt to LLM

### Step 7 — Response Generation
- LLM generates grounded answer

---

##  Knowledge Base

- Supports PDFs, Markdown, TXT, JSON
- Converted into embeddings
- Stored in vector database
- Queried during runtime

Updating knowledge only requires:
- Adding new documents
- Re-embedding them
- Updating vector store

No retraining needed.

---

