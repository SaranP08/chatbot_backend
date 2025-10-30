# recommend.py (Upgraded Version)

import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings # Import this

class QuestionRecommender:
    def __init__(self, faiss_index_path, questions_path, embedding_model: HuggingFaceEmbeddings, top_k=5):
        """
        MODIFIED: Now requires an embedding_model to handle new queries.
        """
        print("Initializing Question Recommender...")
        self.index = faiss.read_index(faiss_index_path)
        self.questions = np.load(questions_path, allow_pickle=True)
        self.embedding_model = embedding_model # <-- NEW: Store the embedding model
        self.top_k = top_k
        self.start_questions = [
            "What is Sat2Farm?",
            "Can someone without farming background do farming using your advisories?",
            "How to add my farm in the App?",
            "Is the app available for Iphone?",
            "Is the app free?",
        ]
        self.history = []
        self.current_recommendations = []
        print("✅ Question Recommender initialized.")

    def get_initial_questions(self):
        """Gets the initial set of questions and resets the state for a new session."""
        self.history = []
        self.current_recommendations = self.start_questions
        return self.current_recommendations

    def recommend(self, query: str):
        """
        MODIFIED: Now handles any query string, not just a selected question.
        """
        if self.current_recommendations:
            self.history.append(self.current_recommendations)

        embedding = None
        # First, try the fast path: see if the query is a known question
        try:
            q_idx = np.where(self.questions == query)[0][0]
            # If found, reconstruct its embedding directly from the index (very fast)
            embedding = self.index.reconstruct(int(q_idx)).reshape(1, -1)
            print(f"Recommending based on known question: '{query}'")
        except IndexError:
            # This is the new, powerful part!
            # If the query is not in our list, embed it on the fly.
            print(f"Recommending based on new user query: '{query}'")
            embedding = np.array(self.embedding_model.embed_query(query)).reshape(1, -1)

        if embedding is not None:
            # Search for similar question embeddings in our FAISS index
            distances, indices = self.index.search(embedding, self.top_k + 1)
            
            # Create the list of recommended questions from the search results
            recommended = [
                self.questions[i] for i in indices[0]
                if i < len(self.questions) and self.questions[i] != query
            ]
            
            self.current_recommendations = recommended[:self.top_k]
            return self.current_recommendations
        
        # Fallback if something went wrong
        return self.start_questions


    def go_back(self):
        """Returns the previous set of recommended questions from history."""
        if self.history:
            self.current_recommendations = self.history.pop()
            return self.current_recommendations
        else:
            print("💡 No more history. Returning to initial questions.")
            return self.get_initial_questions()