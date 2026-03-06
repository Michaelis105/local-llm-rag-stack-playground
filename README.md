# local-llm-rag-stack-playground
Playground experimenting with local LLM-based RAG stack

# Setup

## Install Ollama

```
curl -fsSL https://ollama.com/install.sh | sh
ollama run gemma3:270m
```

## Install Repo

```
pip install langchain langchain-community langchain-core chromadb sentence-transformers pypdf
```

OR

```
pip install -r requirements.txt
```