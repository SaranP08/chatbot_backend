from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from deep_translator import GoogleTranslator

# Import your existing modules
from rank import HybridChatBot
from recommender import QuestionRecommender

app = FastAPI(title="Sat2Farm AI Assistant API", version="1.0.0")

# CORS middleware to allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
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
    action: str  # "go_back" or question text
    user_language: str = "en"

class TranslationRequest(BaseModel):
    text: str
    target_lang: str
    source_lang: str = "auto"

# Global variables for models (similar to st.cache_resource)
bot = None
recommender = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global bot, recommender
    try:
        bot = HybridChatBot()
        recommender = QuestionRecommender(
            faiss_index_path="data/faiss.index",
            questions_path="data/questions.npy"
        )
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise e

def translate_text(text: str, target_lang: str, source_lang: str = "auto") -> str:
    """Translates text, returning original text if translation is not needed or fails."""
    if not text or source_lang == target_lang or (target_lang == "en" and source_lang == "auto"):
        return text
    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception:
        return text

def get_bot_response(query_en: str) -> tuple[str, List[str]]:
    """
    Searches for an answer and new recommendations based on an English query.
    Returns the English answer and a list of new English recommendations.
    """
    global bot, recommender
    
    query_en = str(query_en).strip()
    if not query_en:
        return "Please enter a question.", []

    results = bot.search(query_en, top_k=5, alpha=0.8)
    answer_en = results[0].get('answer') if results else "I'm sorry, I couldn't find an answer."
    new_recommendations = recommender.recommend(query_en)
    return answer_en, new_recommendations

@app.get("/")
async def root():
    return {"message": "Sat2Farm AI Assistant API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": bot is not None and recommender is not None}

@app.get("/languages")
async def get_languages():
    """Get available languages"""
    indian_languages = {
        "English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te", "Kannada": "kn",
        "Malayalam": "ml", "Marathi": "mr", "Gujarati": "gu", "Punjabi": "pa", "Bengali": "bn",
        "Odia": "or", "Urdu": "ur", "Assamese": "as", "French": "fr", "Spanish": "es",
        "German": "de", "Italian": "it"
    }
    return {"languages": indian_languages}

@app.get("/recommendations/initial")
async def get_initial_recommendations():
    """Get initial recommended questions"""
    global recommender
    if not recommender:
        raise HTTPException(status_code=500, detail="Recommender not loaded")
    
    try:
        recommendations = recommender.get_initial_questions()
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Process chat message and return response with recommendations"""
    global bot, recommender
    
    if not bot or not recommender:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        # Translate user message to English for processing
        query_en = translate_text(
            request.message,
            target_lang="en",
            source_lang=request.user_language
        )
        
        # Get bot response and new recommendations
        answer_en, new_recommendations = get_bot_response(query_en)
        
        # Translate answer back to user's language
        answer_translated = translate_text(answer_en, request.user_language)
        
        return ChatResponse(
            answer=answer_translated,
            recommendations=new_recommendations
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.post("/recommendations/action")
async def handle_recommendation_action(request: RecommendationRequest):
    """Handle recommendation actions (go back or select question)"""
    global recommender
    
    if not recommender:
        raise HTTPException(status_code=500, detail="Recommender not loaded")
    
    try:
        if request.action == "go_back":
            recommendations = recommender.go_back()
            return {"recommendations": recommendations}
        else:
            # The action is a question, so we process it like a chat message
            chat_request = ChatRequest(
                message=request.action,
                user_language=request.user_language
            )
            return await chat(chat_request)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error handling recommendation: {str(e)}")

@app.post("/translate")
async def translate(request: TranslationRequest):
    """Translate text between languages"""
    try:
        translated_text = translate_text(
            request.text,
            request.target_lang,
            request.source_lang
        )
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )