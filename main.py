# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import traceback # <-- IMPORT THE TRACEBACK MODULE
from deep_translator import GoogleTranslator
from rag_chatbot import RAGChatBot
from recommender import QuestionRecommender
from langchain_huggingface import HuggingFaceEmbeddings
import os

app = FastAPI(
    title="Sat2Farm AI Assistant API",
    version="1.0.0",
    description="An API powered by a RAG model for document Q&A and question recommendations."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    user_language: str = "en"
    chat_history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    answer: str
    recommendations: List[str]
    status: str = "success"

class RecommendationRequest(BaseModel):
    action: str  
    user_language: str = "en"

class TranslationRequest(BaseModel):
    text: str
    target_lang: str
    source_lang: str = "auto"

bot: RAGChatBot = None
recommender: QuestionRecommender = None

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"


@app.on_event("startup")
@app.on_event("startup")
async def startup_event():
    """
    This function runs once when the API server starts.
    It's the perfect place to load heavy models into memory.
    """
    global bot, recommender
    try:
        # 1. Create the embedding model - THE SINGLE SOURCE OF TRUTH
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        print("✅ Embedding model loaded.")

        # 2. Initialize the RAG ChatBot (pass embeddings)
        print("Loading RAG ChatBot...")
        bot = RAGChatBot(embeddings)   # ✅ FIXED

        faiss_path = os.path.join("data", "faiss.index") # Corrected name to match your file
        questions_path = os.path.join("data", "questions.npy")

        print(f"Attempting to load Recommender index from: '{faiss_path}'") # Add for debugging

        # Initialize the Question Recommender with the corrected paths
        recommender = QuestionRecommender(
            faiss_index_path=faiss_path,
            questions_path=questions_path,
            embedding_model=embeddings
        )
        print("🎉 All models loaded successfully! API is ready. 🎉")

    except Exception as e:
        print("💥 Critical Error: Failed to load models during startup:")
        traceback.print_exc()
        raise e


# --- Helper Functions ---

def translate_text(text: str, target_lang: str, source_lang: str = "auto") -> str:
    if not text or source_lang == target_lang or (target_lang == "en" and source_lang == "auto"):
        return text
    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception as e:
        print(f"Translation failed for text: '{text}'. Error: {e}")
        return text

def get_bot_response(query_en: str) -> tuple[str, List[str]]:
    global bot, recommender
    
    query_en = str(query_en).strip()
    if not query_en:
        return "Please provide a question.", []

    rag_response = bot.ask(query_en)
    answer_en = rag_response.get('answer', "I'm sorry, I couldn't find an answer to your question.")
    
    new_recommendations = recommender.recommend(query_en)
    
    return answer_en, new_recommendations

# --- API Endpoints ---

@app.get("/")
async def root():
    return {"message": "Sat2Farm AI Assistant API is running!"}

@app.get("/health")
async def health_check():
    models_ready = bot is not None and recommender is not None
    return {"status": "healthy" if models_ready else "degraded", "models_loaded": models_ready}

@app.get("/languages")
async def get_languages():
    supported_languages = {
        "English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te", "Kannada": "kn",
        "Malayalam": "ml", "Marathi": "mr", "Gujarati": "gu", "Punjabi": "pa", "Bengali": "bn",
        "Odia": "or", "Urdu": "ur", "Assamese": "as", "French": "fr", "Spanish": "es",
        "German": "de", "Italian": "it", "Manipuri": "mni", "Swahili": "sw", "Malay": "ms", "Portuguese": "pt",
        "Turkish": "tr", "Arabic": "ar", "Indonesian": "id", "Thai": "th", "Chichewa": "ny"
    }
    return {"languages": supported_languages}

@app.get("/recommendations/initial")
async def get_initial_recommendations():
    if not recommender:
        raise HTTPException(status_code=503, detail="Recommender service is not available.")
    try:
        recommendations = recommender.get_initial_questions()
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main endpoint to process a user's chat message."""
    if not bot or not recommender:
        raise HTTPException(status_code=503, detail="Chat service is not available. Models are not loaded.")
    
    # ========================== THE FIX IS HERE ==========================
    try:
        query_en = translate_text(request.message, target_lang="en", source_lang=request.user_language)
        
        answer_en, new_recommendations_en = get_bot_response(query_en)
        
        answer_translated = translate_text(answer_en, request.user_language)
        
        return ChatResponse(
            answer=answer_translated,
            recommendations=new_recommendations_en
        )
    except Exception as e:
        # This will now print the FULL error traceback to your logs
        print(f"Error during /chat processing. Full traceback below:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    # =====================================================================

@app.post("/recommendations/action")
async def handle_recommendation_action(request: RecommendationRequest):
    if not recommender:
        raise HTTPException(status_code=503, detail="Recommender service is not available.")
    try:
        if request.action == "go_back":
            recommendations = recommender.go_back()
            return {"recommendations": recommendations}
        else:
            chat_request = ChatRequest(message=request.action, user_language=request.user_language)
            return await chat(chat_request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error handling recommendation action: {str(e)}")

@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        translated_text = translate_text(request.text, request.target_lang, request.source_lang)
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)