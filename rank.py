
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import importlib, subprocess, sys
from rank_bm25 import BM25Okapi
# class HybridChatBot:
#     def __init__(self, model_name="all-MiniLM-L6-v2", index_file="Chatbot/data/faiss.index"):
#         # Load embeddings index
#         self.model = SentenceTransformer(model_name)
#         self.index = faiss.read_index(index_file)
#         self.questions = np.load("Chatbot/data/questions.npy", allow_pickle=True)
#         self.answers = np.load("Chatbot/data/answers.npy", allow_pickle=True)

#         # Prepare BM25
#         tokenized_corpus = [q.lower().split() for q in self.questions]
#         self.bm25 = BM25Okapi(tokenized_corpus)

#     def search(self, query, top_k, alpha):
#         """
#         Hybrid search:
#         alpha = weight for BM25 vs embeddings (0.5 = equal weight)
#         """
#         # --- Embedding Search ---
#         query_embedding = self.model.encode([query], convert_to_numpy=True)
#         distances, indices = self.index.search(query_embedding, top_k)
#         embedding_scores = {idx: 1/(1+dist) for idx, dist in zip(indices[0], distances[0])}

#         # --- BM25 Search ---
#         bm25_scores = self.bm25.get_scores(query.lower().split())
#         bm25_top = np.argsort(bm25_scores)[::-1][:top_k]
#         bm25_scores = {idx: bm25_scores[idx] for idx in bm25_top}

#         # --- Combine Scores ---
#         combined_scores = {}
#         for idx in set(list(embedding_scores.keys()) + list(bm25_scores.keys())):
#             emb_score = embedding_scores.get(idx, 0)
#             bm_score = bm25_scores.get(idx, 0)
#             combined_scores[idx] = alpha * bm_score + (1 - alpha) * emb_score

#         # --- Sort and Return ---
#         best = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

#         results = []
#         for idx, score in best[:top_k]:
#             results.append({
#                 "matched_question": self.questions[idx],
#                 "answer": self.answers[idx],
#                 "score": float(score)
#             })
#         return results
class HybridChatBot:
    def __init__(self, model_name="all-MiniLM-L6-v2", index_file="data/faiss.index", fallback_threshold=0.05):
        # Load embeddings index
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_file)
        self.questions = np.load("data/questions.npy", allow_pickle=True)
        self.answers = np.load("data/answers.npy", allow_pickle=True)

        # Prepare BM25
        tokenized_corpus = [q.lower().split() for q in self.questions]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Threshold for fallback
        self.fallback_threshold = fallback_threshold

    def search(self, query, top_k=5, alpha=0.5):
        """
        Hybrid search:
        alpha = weight for BM25 vs embeddings (0.5 = equal weight)
        """
        # --- Embedding Search ---
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        embedding_scores = {idx: 1/(1+dist) for idx, dist in zip(indices[0], distances[0])}

        # --- BM25 Search ---
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top = np.argsort(bm25_scores)[::-1][:top_k]
        bm25_scores = {idx: bm25_scores[idx] for idx in bm25_top}

        # --- Combine Scores ---
        combined_scores = {}
        for idx in set(list(embedding_scores.keys()) + list(bm25_scores.keys())):
            emb_score = embedding_scores.get(idx, 0)
            bm_score = bm25_scores.get(idx, 0)
            combined_scores[idx] = alpha * bm_score + (1 - alpha) * emb_score

        # --- Sort and Return ---
        best = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        if not best or best[0][1] < self.fallback_threshold:
            # Low confidence → fallback message
            results.append({
                "matched_question": None,
                "answer": "Sorry, I couldn't find a reliable answer. Please contact our support team.",
                "score": 0.0
            })
        else:
            for idx, score in best[:top_k]:
                results.append({
                    "matched_question": self.questions[idx],
                    "answer": self.answers[idx],
                    "score": float(score)
                })

        return results
