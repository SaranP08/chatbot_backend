# recommender.py

import faiss
import numpy as np
import random  # Import the random module

# This list is still needed for the first "More Questions" click
FIRST_MORE_QUESTIONS = [
    "How does Sat2Farm help in monitoring crop health?",
    "Can you explain the Vegetation Index in detail?",
    "What kind of satellite data does Sat2Farm use?",
    "Is there a mobile application for Sat2Farm?",
    "How can I get started with a subscription?",
]

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
        self.history = []
        self.current_recommendations = []
        self.more_questions_level = 0

    def get_initial_questions(self):
        self.history = []
        self.current_recommendations = self.start_questions
        self.more_questions_level = 0
        return self.current_recommendations

    def recommend(self, selected_question):
        if self.current_recommendations:
            self.history.append(self.current_recommendations)
        self.more_questions_level = 0
        try:
            q_idx = np.where(self.questions == selected_question)[0][0]
        except IndexError:
            print("⚠️ User-typed question, cannot find exact embedding. Reverting to initial questions.")
            return self.start_questions
        embedding = self.index.reconstruct(int(q_idx)).reshape(1, -1)
        distances, indices = self.index.search(embedding, self.top_k + 1)
        recommended = [
            self.questions[i] for i in indices[0]
            if i < len(self.questions) and self.questions[i] != selected_question
        ]
        self.current_recommendations = recommended[:self.top_k]
        return self.current_recommendations

    def go_back(self):
        self.more_questions_level = 0
        if self.history:
            self.current_recommendations = self.history.pop()
            return self.current_recommendations
        else:
            print("💡 No more history. Returning to initial questions.")
            return self.get_initial_questions()

    # ---- THIS IS THE MODIFIED METHOD ----
    def get_more_questions(self):
        """
        Provides the next set of recommended questions based on user request.
        """
        if self.more_questions_level == 0:
            # First time user clicks "More Questions", return the hardcoded list
            self.more_questions_level += 1
            return FIRST_MORE_QUESTIONS
        elif self.more_questions_level == 1:
            # Second time user clicks, generate dynamic recommendations
            self.more_questions_level += 1
            
            # We need to exclude questions the user has already seen on the screen
            exclude_list = set(self.current_recommendations + FIRST_MORE_QUESTIONS)
            return self._generate_dynamic_recommendations(exclude_list)
        else:
            # Subsequent clicks can return an empty list
            return ["No more questions to suggest at this time."]

    # ---- THIS IS THE NEW HELPER METHOD ----
    def _generate_dynamic_recommendations(self, exclude_list: set) -> list:
        """
        Generates new recommendations based on the current context, excluding
        questions that have already been shown.
        """
        if not self.current_recommendations:
            # If there's no context, we can't generate anything
            return []

        # Pick a random question from the current context as a seed
        seed_question = random.choice(self.current_recommendations)

        try:
            q_idx = np.where(self.questions == seed_question)[0][0]
        except IndexError:
            # Fallback if seed not found
            return []

        embedding = self.index.reconstruct(int(q_idx)).reshape(1, -1)
        
        # Search for more candidates than we need, as we will be filtering
        num_candidates_to_find = self.top_k * 3 
        distances, indices = self.index.search(embedding, num_candidates_to_find)

        new_recommendations = []
        for i in indices[0]:
            candidate_question = self.questions[i]
            if candidate_question not in exclude_list:
                new_recommendations.append(candidate_question)
            
            # Stop once we have enough new questions
            if len(new_recommendations) >= self.top_k:
                break
        
        return new_recommendations