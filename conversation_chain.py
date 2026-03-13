import sqlite3
import os
import warnings

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

# ignore warnings
warnings.filterwarnings("ignore")

# loading the open-ai-api-key
load_dotenv()
api_key = os.getenv("OPEN_API_KEY")

####### Creating and store embeddigns in the vector database
# Step-1 : Load chunks from sqlite
def load_chunks_from_sqlite(db_path:str):
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

# Step-2 : creating Embeddings + Vector Store - FAISS
def create_vector_store(db_path:str, out_folder:str):

    # load the chunks from the database
    chunks, metadatas = load_chunks_from_sqlite(db_path)
    print(f"Loaded {len(chunks)} chunks from SQLite.")

    # generate embeddings
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")

    # store it in FAISS vector database
    vector_store = FAISS.from_texts(
        texts = chunks,
        embedding = embeddings,
        metadatas = metadatas
    )

    # save FAISS vector store locally
    vector_store.save_local(out_folder)
    print(f"Vector store saved to {out_folder}")

    return vector_store



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

def driver():
    # call to create the vector store
    create_vector_store(
        db_path="./output/chunks.db",
        out_folder="./vector_store"
    )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # loading the saved FAISS vector store
    vector_store = FAISS.load_local(
        "./vector_store",
        embeddings = embeddings,
        allow_dangerous_deserialization=True
    )

    # Create the LLM
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Create conversation chain
    chain = create_conversation_chain(llm, vector_store)

    # Use the Chat bot
    while True:
        question = input("Ask a question (type 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        response = chain({"question":question})
        print("\nAnswer: ", response["answer"], "\n")

if __name__ == "__main__":
    driver()
    




