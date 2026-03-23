import os
import sqlite3
import warnings
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ignore warnings
warnings.filterwarnings("ignore")

# loading the open-ai-api-key
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")

DEFAULT_DB_PATH = "./output/chunks.db"
DEFAULT_VECTOR_STORE = "./vector_store"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


def ensure_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing OpenAI API key. Set OPENAI_API_KEY or OPEN_API_KEY in your .env file."
        )

    os.environ["OPENAI_API_KEY"] = key
    return key

####### Creating and store embeddigns in the vector database
# Step-1 : Load chunks from sqlite
def load_chunks_from_sqlite(db_path: str) -> Tuple[List[str], List[dict]]:
    # connection string
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    #query
    query = """
    SELECT file_id, chunk_index, text
    from chunks 
    ORDER BY file_id, chunk_index;
    """
    cur.execute(query)

    # fetch all the rows
    rows = cur.fetchall()
    
    #close the connection
    conn.close()

    # adding chunked texts
    chunks = []
    metadatas = []

    for file_id, idx, text in rows:
        chunks.append(text)
        metadatas.append({"file_id": file_id, "chunk_index": idx})

    return chunks, metadatas


def create_embeddings(model: str = EMBEDDING_MODEL) -> OpenAIEmbeddings:
    ensure_openai_api_key()
    return OpenAIEmbeddings(model=model)


def create_chat_llm(model: str = CHAT_MODEL) -> ChatOpenAI:
    ensure_openai_api_key()
    return ChatOpenAI(model=model)

# Step-2 : creating Embeddings + Vector Store - FAISS
def create_vector_store(
    db_path: str,
    out_folder: str,
    embedding_model: str = EMBEDDING_MODEL,
):

    # load the chunks from the database
    chunks, metadatas = load_chunks_from_sqlite(db_path)
    if not chunks:
        raise ValueError(f"No chunks were found in SQLite database: {db_path}")
    print(f"Loaded {len(chunks)} chunks from SQLite.")

    # generate embeddings
    embeddings = create_embeddings(embedding_model)

    # store it in FAISS vector database
    vector_store = FAISS.from_texts(
        texts = chunks,
        embedding = embeddings,
        metadatas = metadatas
    )

    # save FAISS vector store locally
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    vector_store.save_local(out_folder)
    print(f"Vector store saved to {out_folder}")

    return vector_store


def create_vector_store_from_chunks(
    chunks: List[str],
    metadatas: List[dict],
    out_folder: str,
    embedding_model: str = EMBEDDING_MODEL,
):
    if not chunks:
        raise ValueError("No chunks were provided to build the vector store.")

    embeddings = create_embeddings(embedding_model)
    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
    )

    Path(out_folder).mkdir(parents=True, exist_ok=True)
    vector_store.save_local(out_folder)
    print(f"Vector store saved to {out_folder}")
    return vector_store


def load_vector_store(
    out_folder: str,
    embedding_model: str = EMBEDDING_MODEL,
):
    embeddings = create_embeddings(embedding_model)
    return FAISS.load_local(
        out_folder,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )



###### Create Conversational chain
def create_conversation_chain(llm, vector_store):
    
    #Step-1: Create retriever from FAISS vector store
    retriever = vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k":3}

    )

    #Step-2: Create memory buffer
    memory = ConversationBufferMemory(
        memory_key = "chat_history",
        return_messages = True
    )   

    #Step-3: Build conversational chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        memory = memory
    )

    return conversation_chain


def build_conversation_chain(
    db_path: str = DEFAULT_DB_PATH,
    vector_store_folder: str = DEFAULT_VECTOR_STORE,
    rebuild_vector_store: bool = True,
    embedding_model: str = EMBEDDING_MODEL,
    chat_model: str = CHAT_MODEL,
):
    vector_store_path = Path(vector_store_folder)
    if rebuild_vector_store or not vector_store_path.exists():
        vector_store = create_vector_store(
            db_path=db_path,
            out_folder=vector_store_folder,
            embedding_model=embedding_model,
        )
    else:
        vector_store = load_vector_store(
            out_folder=vector_store_folder,
            embedding_model=embedding_model,
        )

    llm = create_chat_llm(chat_model)
    return create_conversation_chain(llm, vector_store)


def build_conversation_chain_from_vector_store(
    vector_store,
    chat_model: str = CHAT_MODEL,
):
    llm = create_chat_llm(chat_model)
    return create_conversation_chain(llm, vector_store)


def ask_question(chain, question: str) -> str:
    response = chain.invoke({"question": question})
    return response["answer"]

def driver():
    chain = build_conversation_chain(
        db_path=DEFAULT_DB_PATH,
        vector_store_folder=DEFAULT_VECTOR_STORE,
        rebuild_vector_store=True,
    )

    # Use the Chat bot
    while True:
        question = input("Ask a question (type 'exit' to quit): ").strip()
        if question.lower() == 'exit':
            break
        if not question:
            continue

        answer = ask_question(chain, question)
        print("\nAnswer: ", answer, "\n")

if __name__ == "__main__":
    driver()
    

