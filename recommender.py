# recommender.py

import faiss
import numpy as np

# Define the hardcoded lists for "More Questions"
FIRST_MORE_QUESTIONS = [
    "How does Sat2Farm help in monitoring crop health?",
    "Can you explain the Vegetation Index in detail?",
    "What kind of satellite data does Sat2Farm use?",
    "Is there a mobile application for Sat2Farm?",
    "How can I get started with a subscription?",
]

# This second list demonstrates how the feature can be extended.
SECOND_MORE_QUESTIONS = [
    "How accurate is the weather forecast feature?",
    "Can I manage multiple farms with one account?",
    "What support options are available for users?",
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
        # State to track the level of "more questions" requested
        self.more_questions_level = 0

    def get_initial_questions(self):
        """
        Gets the initial set of questions and resets the state for a new session.
        """
        self.history = []
        self.current_recommendations = self.start_questions
        # Reset the level on initial load
        self.more_questions_level = 0
        return self.current_recommendations

    def recommend(self, selected_question):
        """
        Recommend questions based on similarity search and saves the previous state.
        """
        if self.current_recommendations:
            self.history.append(self.current_recommendations)

        # Reset the "more questions" flow whenever a new primary question is asked
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
        """
        Returns the previous set of recommended questions from history.
        """
        # Reset the "more questions" flow when going back
        self.more_questions_level = 0
        
        if self.history:
            self.current_recommendations = self.history.pop()
            return self.current_recommendations
        else:
            print("💡 No more history. Returning to initial questions.")
            return self.get_initial_questions()

    def get_more_questions(self):
        """
        Provides the next set of recommended questions based on user request.
        """
        if self.more_questions_level == 0:
            # First time user clicks "More Questions"
            self.more_questions_level += 1
            return FIRST_MORE_QUESTIONS
        elif self.more_questions_level == 1:
            # Second time user clicks "More Questions"
            self.more_questions_level += 1
            return SECOND_MORE_QUESTIONS
        else:
            # Subsequent clicks can return an empty list or a message
            return ["No more questions to suggest at this time."]