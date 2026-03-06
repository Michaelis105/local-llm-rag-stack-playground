# local-llm-rag-stack-playground
Playground experimenting with local LLM-based RAG stack

# Setup

Runs successfully on Python 3.11.9.

## Install Ollama

```
curl -fsSL https://ollama.com/install.sh | sh
ollama run gemma3:270m
```

## Install Repo

```
pip install -r requirements.txt
```

# Usage

1. Place a text PDF file in the working directory.
2. Invoke with query:
```
python main.py "How many chickens crossed the road"
```

# Random Caveats

The RAG ingested data persists as chunked "cache" in `chroma_db`. Deleting and rerunning will recreate the chunks.