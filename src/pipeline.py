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

if __name__ == "__main__":
    API_KEY = "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"
    MODEL = "arcee-ai/trinity-large-preview:free"

    pipeline = Pipeline(
        index_path="data/rag/knowledge_base.index",
        kb_path="data/rag/knowledge_base.parquet",
        api_key=API_KEY,
        model=MODEL
    )

    test_messages = [
        ("I feel so anxious about my job", "anxiety"),
        ("I want to kill myself", "depression"),
        ("I feel so sad and lonely", "sadness"),
    ]

    for message, emotion in test_messages:
        print(f"\nUser: {message}")
        response, docs, status = pipeline.safe_run(message, emotion)
        print(f"Status: {status}")
        print(f"Response: {response[:200]}")
        print("---")