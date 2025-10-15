import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class EmbeddingStore:
    def __init__(self, model_name="all-MiniLM-L6-v2", json_path="data/train_data.json"):
        self.model = SentenceTransformer(model_name)
        with open(json_path, "r", encoding="utf-8") as f:
            self.qa_data = json.load(f)
        self.questions = [item["instruction"] for item in self.qa_data]
        self.answers = [item["response"] for item in self.qa_data]

    def build_index(self, index_file="data/faiss.index"):
        embeddings = self.model.encode(self.questions, convert_to_numpy=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, index_file)
        np.save("data/questions.npy", self.questions)
        np.save("data/answers.npy", self.answers)
        print(f"FAISS index built and saved → {index_file}")

if __name__ == "__main__":
    store = EmbeddingStore()
    store.build_index()
