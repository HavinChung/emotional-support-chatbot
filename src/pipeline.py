from src.rag import RAG
from src.prompt import SYSTEM_PROMPT, build_prompt
from src.llm import call_llm
from src.safety import safe_pipeline_run

class Pipeline:
    def __init__(self, index_path, kb_path, api_key, model):
        self.rag = RAG(index_path, kb_path)
        self.api_key = api_key
        self.model = model
        self.conversation_history = []

    def run(self, user_message, emotion):
        retrieved_docs = self.rag.retrieve(user_message, top_k=5, emotion_filter=emotion)
        prompt = build_prompt(user_message, emotion, retrieved_docs, self.conversation_history)
        response = call_llm(SYSTEM_PROMPT, prompt, api_key=self.api_key, model=self.model)
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": response})
        return response, retrieved_docs

    def safe_run(self, user_message, emotion):
        return safe_pipeline_run(self, user_message, emotion)
