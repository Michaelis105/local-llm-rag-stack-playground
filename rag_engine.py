import os
import sys
import warnings

# Still highly recommend the mock just in case other sub-packages trigger it
if sys.platform == "win32":
    import types
    sys.modules["pwd"] = types.ModuleType("pwd")

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Suppress HuggingFace/LangChain deprecation warnings for cleaner output
warnings.filterwarnings("ignore")

class RAGEngine:
    def __init__(self, db_path="./chroma_db"):
        self.db_path = db_path
        self.total_chunks = 0  # Track total chunks for smart retrieval

        # Small, fast local embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False},
            cache_folder="./models" # Stores the 90MB model locally in your repo
        )

    def ingest_document(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # Debug: Show how many pages were loaded
        print(f"Loaded {len(docs)} pages from PDF")
        for i, doc in enumerate(docs):
            print(f"  Page {i+1}: {len(doc.page_content)} characters")
        
        # Splitter: 500 char chunks with 10% overlap for context continuity
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        
        print(f"Split into {len(chunks)} chunks")
        self.total_chunks = len(chunks) # Store for dynamic retrieval
        
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embeddings, 
            persist_directory=self.db_path
        )
                
        return vector_store.as_retriever(search_kwargs={"k": self.total_chunks})

    def get_existing_retriever(self):
        vector_store = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
        # For existing retriever, use same logic: 50% of estimated chunks (minimum 4)
        # Note: This is an estimate; for exact count, you'd need to query the vector store
        return vector_store.as_retriever(search_kwargs={"k": self.total_chunks})