# recommend.py
import faiss
import numpy as np

class QuestionRecommender:
    def __init__(self, faiss_index_path, questions_path, top_k=5):
        self.index = faiss.read_index(faiss_index_path)
        self.questions = np.load(questions_path, allow_pickle=True)
        self.top_k = top_k
        self.start_questions = [
            "What is Sat2Farm?",
            "Can someone without farming background do farming using your advisories?",
            "How to add my farm in the App?",
            "Is the app available for Iphone?",
            "Is the app free?",
        ]
        # --- NEW: State management ---
        self.history = []
        self.current_recommendations = []

    def get_initial_questions(self):
        """
        Gets the initial set of questions and resets the state for a new session.
        """
        # Reset history and set the initial questions as the current ones
        self.history = []
        self.current_recommendations = self.start_questions
        return self.current_recommendations

    def recommend(self, selected_question):
        """
        Recommend questions based on similarity search and saves the previous state.
        """
        # --- NEW: Save the current list of questions to history before changing it ---
        if self.current_recommendations:
            self.history.append(self.current_recommendations)

        # Find the index of the selected question in dataset
        try:
            q_idx = np.where(self.questions == selected_question)[0][0]
        except IndexError:
            # Fallback for questions not in the original list (e.g., user-typed query)
            # This part requires a model to embed the query, which is beyond this scope.
            # For now, we'll just return the initial questions as a safe fallback.
            print("⚠️ User-typed question, cannot find exact embedding. Reverting to initial questions.")
            return self.start_questions

        # Search similar questions
        embedding = self.index.reconstruct(int(q_idx)).reshape(1, -1)
        distances, indices = self.index.search(embedding, self.top_k + 1)

        # Exclude the same question itself
        recommended = [
            self.questions[i] for i in indices[0]
            if i < len(self.questions) and self.questions[i] != selected_question
        ]
        
        # --- NEW: Update the current recommendations ---
        self.current_recommendations = recommended[:self.top_k]
        return self.current_recommendations

    def go_back(self):
        """
        Returns the previous set of recommended questions from history.
        """
        if self.history:
            # Pop the last list from history and make it the current one
            self.current_recommendations = self.history.pop()
            return self.current_recommendations
        else:
            # If there's no history, return the initial set of questions
            print("💡 No more history. Returning to initial questions.")
            return self.get_initial_questions()