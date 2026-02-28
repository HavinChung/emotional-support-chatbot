import faiss
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class RAG:
    def __init__(self, index_path, kb_path, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        self.kb = pd.read_parquet(kb_path)

    def retrieve(self, query, top_k=5, emotion_filter=None):
        query_embedding = self.model.encode([query]).astype(np.float32)
        distances, indices = self.index.search(query_embedding, top_k * 3)
        results = self.kb.iloc[indices[0]].copy()
        results['distance'] = distances[0]
        if emotion_filter:
            results = results[results['emotion'] == emotion_filter]
        return results.head(top_k)[['emotion', 'problem', 'strategies', 'source', 'distance', 'text']]

if __name__ == "__main__":
    rag = RAG("data/rag/knowledge_base.index", "data/rag/knowledge_base.parquet")
    results = rag.retrieve("I feel so anxious and I can't stop worrying")
    print(results[['emotion', 'problem', 'strategies', 'distance']])