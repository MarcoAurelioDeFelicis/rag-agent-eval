import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS


def create_vector_store(file_path: str, persist_directory: str = "db"):
    
    if os.path.exists(persist_directory):
        print("✅ Vector Store already loaded, loading...")
        #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        return FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)

    print("📚 Vdb is being created, loading of CSV file...")
    loader = CSVLoader(file_path=file_path, encoding="utf-8")
    documents = loader.load()

    """ chubkization """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    """ embeddizzatoion """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    """ VDB FAISS by CSV """
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(persist_directory)
    print("✅ Vector Store has been created succesfully.")
    
    return db