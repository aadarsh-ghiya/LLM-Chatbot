## Custom Q&A Chatbot

This project implements a complete **Retrieval‑Augmented Generation (RAG)** pipeline using Python, LangChain, FAISS, and OpenAI.  
The system processes PDF files, chunks them, stores them in SQLite + FAISS, and finally enables a conversational chatbot that answers questions based on the PDF content.

---

##  **Project Structure**

```
Custom Q&A Chatbot/
│
├── input_pdf/              # Place your PDFs here
├── output/                 # Auto‑generated: text files, chunks, SQLite DB
│   ├── texts/
│   ├── chunks/
│   └── chunks.db
│
├── vector_store/           # Auto‑generated FAISS vector index
│
├── extract_and_chunk.py    # Part A & B
├── conversation_chain.py   # Part C, D & E
├── requirements.txt
├── .env                    # Place your OPEN_API_KEY
└── README.md
```

---

## **Part A - Extract Text from PDFs**

`extract_and_chunk.py` loads PDFs from `input_pdf/`, extracts text using **pdfplumber**, cleans it, and saves:

- Raw text --> `output/texts/*.txt`
- Cleaned text --> used for chunking

---

## **Part B - Chunking + SQLite Storage**

The cleaned text is split into overlapping chunks using:

- `CharacterTextSplitter`
- Default: **500 characters**, **50 overlap**

Each PDF produces:

- JSON chunk file --> `output/chunks/*.json`
- SQLite entries --> `output/chunks.db`

This database is later used to build embeddings.

---

##  **Part C - Embeddings + FAISS Vector Store**

`conversation_chain.py` loads chunks from SQLite and creates:

- OpenAI embeddings (`text-embedding-3-small`)
- FAISS vector store saved to `vector_store/`

This enables fast similarity search during retrieval.

---

##  **Part D - Conversational Retrieval Chain**

A LangChain pipeline is built:

- FAISS retriever (`k=3`)
- `ConversationBufferMemory` for chat history
- `ConversationalRetrievalChain` for RAG

This allows the chatbot to answer questions grounded in the PDF.

---

##  **Part E - Driver Function (Chatbot Interface)**

Running `conversation_chain.py` launches an interactive chatbot:

```
Ask a question (type 'exit' to quit):
```

The chatbot retrieves relevant chunks from FAISS and generates answers using `gpt-4o-mini`.

---

## **How to Run the Project**

### **1. Install dependencies**
```
pip install -r requirements.txt
```

### **2. Add your OpenAI API key**
Create a `.env` file:

```
OPEN_API_KEY=your_key_here
```

### **3. Place PDFs**
Put your PDF(s) inside:

```
input_pdf/
```

### **4. Run Part A & B (Extraction + Chunking)**

```
python extract_and_chunk.py --input_folder ./input_pdf --output_folder ./output
```

This generates:

- `output/texts/`
- `output/chunks/`
- `output/chunks.db`

### **5. Run Part C, D, E (Vector Store + Chatbot)**

```
python conversation_chain.py
```

You will see:

```
Ask a question (type 'exit' to quit):
```

Now you can ask questions like:

- “What topics does this PDF cover”
- “Explain harmonic balance simulation”
- “Summarize the section on RF system design”

---