import os
import logging
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS

from src import settings


def create_vector_store(file_path: str, persist_directory: str = "db")-> FAISS:

    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
    
    if os.path.exists(persist_directory):
        logging.info(f"Vector Store already exists. Loading from '{persist_directory}'...")
        db = FAISS.load_local(
            persist_directory, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        logging.info("✅ Vector Store loaded successfully.")
        return db
    
    logging.info(f"Vector Store not found. Creating a new one from '{file_path}'...")

    loader = CSVLoader(file_path=file_path, encoding="utf-8")
    documents = loader.load()

    """ chubkization """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE, 
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)
    logging.info(f"Splitted {len(documents)} documents into {len(texts)} chunks.")

    """ VDB FAISS by CSV """
    logging.info("Creating embeddings and building the vector store. This may take a while...")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(persist_directory)
    logging.info(f"✅ Vector Store created and saved to '{persist_directory}'.")
    
    return db