import os
import logging
import pandas as pd
import ast
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS

from src import settings


def create_vector_store(file_path: str, persist_directory: str = "db"):

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

    # --- PRE-PROCESSING CSV WITH PANDAS ---
    logging.info(f"Loading and preprocessing data from '{file_path}'...")
    
    try:
        df = pd.read_csv(file_path).fillna('')
        df = df.head(200) # testing
    except FileNotFoundError:
        logging.error(f"FATAL: CSV file not found at path: {file_path}")
        raise

    documents = []

    for index, row in df.iterrows():
        ingredienti_str = ""
        try:
            ingredienti_list = ast.literal_eval(row['Ingredienti'])
            ingredienti_str = "\n".join([f"- {item[0]}: {item[1]}" for item in ingredienti_list])
        except (ValueError, SyntaxError):
            logging.warning(f"Could not parse ingredients for recipe '{row['Nome']}' at row {index}. Using raw string.")
            ingredienti_str = row['Ingredienti']

        page_content = (
            f"Titolo: {row['Nome']}\n"
            f"Categoria: {row['Categoria']}\n"
            f"Porzioni: {row['Persone/Pezzi']}\n\n"
            f"Ingredienti:\n{ingredienti_str}\n\n"
            f"Procedimento:\n{row['Steps']}\n\n"
            f"Link: {row['Link']}"
        )
        
        doc = Document(
            page_content=page_content,
            metadata={
                "source": file_path,
                "recipe_name": row['Nome'],
                "category": row['Categoria'],
                "row_index": index
            }
        )
        documents.append(doc)

    logging.info(f"Successfully created {len(documents)} structured documents from CSV rows.")

    # --- CHUNKIZATION ---
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    texts = text_splitter.split_documents(documents)
    logging.info(f"Splitted {len(documents)} documents into {len(texts)} semantic chunks.")

    # --- VDB FAISS by CSV ---
    logging.info("Creating embeddings and building the vector store. This may take a while...")

    db = FAISS.from_documents(texts, embeddings)

    db.save_local(persist_directory)
    logging.info(f"✅ Vector Store created and saved to '{persist_directory}'.")
    
    return db
