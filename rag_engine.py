import os
import warnings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

# Suppress HuggingFace/LangChain deprecation warnings for cleaner output
warnings.filterwarnings("ignore")

class RAGEngine:
    def __init__(self, db_path="./chroma_db"):
        self.db_path = db_path
        # Small, fast local embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def ingest_document(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # Splitter: 500 char chunks with 10% overlap for context continuity
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embeddings, 
            persist_directory=self.db_path
        )
        return vector_store.as_retriever(search_kwargs={"k": 3})

    def get_existing_retriever(self):
        vector_store = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
        return vector_store.as_retriever(search_kwargs={"k": 3})