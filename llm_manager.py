from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

class LLMManager:
    def __init__(self, model_name="gemma3"):
        self.llm = Ollama(model=model_name)

    def get_rag_prompt(self):
        system_prompt = (
            "You are an assistant answering questions based strictly on the provided context. "
            "Use the following pieces of retrieved context to answer the question. "
            "If the answer is not in the context, state that you do not know. Do not hallucinate. "
            "Context: {context}"
        )
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

    def get_model(self):
        return self.llm