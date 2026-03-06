import sys

from llm_manager import LLMManager
from rag_engine import RAGEngine

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

if sys.platform == "win32":
    import types
    sys.modules["pwd"] = types.ModuleType("pwd")

def format_docs(docs):    
    return "\n\n".join(doc.page_content for doc in docs)

def run_pipeline(pdf_path, user_query):
    # Initialize components
    llm_handler = LLMManager()
    rag_handler = RAGEngine()

    # Process document and get retriever
    print(f"Ingesting {pdf_path}...")
    retriever = rag_handler.ingest_document(pdf_path)
    # Modern LCEL Chain: Avoids 'langchain.chains' entirely
    # Flow: Context Retrieval -> Prompt -> LLM -> String Output
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | llm_handler.get_rag_prompt()
        | llm_handler.get_model()
        | StrOutputParser()
    )

    print(f"Querying: {user_query}")
    return rag_chain.invoke(user_query)

if __name__ == "__main__":
    text_file = "test.pdf"
    answer = run_pipeline(text_file, sys.argv[1])
    print(f"\n--- Analysis ---\n{answer}")