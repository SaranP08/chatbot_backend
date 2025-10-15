import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
from deep_translator import GoogleTranslator

# --- Production-Ready Setup ---
# Set up logging to see clear output in Hugging Face logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set a writable cache path for Hugging Face models BEFORE importing your modules
# This is crucial for read-only container environments
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface/transformers"

# Import your custom modules AFTER setting the cache path
try:
    from rank import HybridChatBot
    from recommender import QuestionRecommender
except ImportError as e:
    logger.error(f"Failed to import custom modules: {e}. Ensure rank.py and recommender.py are present.")
    raise e

app = FastAPI(title="Sat2Farm AI Assistant API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
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

class ActionRequest(BaseModel):
    action: str
    user_language: str = "en"

class TranslationRequest(BaseModel):
    text: str
    target_lang: str
    source_lang: str = "auto"

# Global variables for models
bot = None
recommender = None

@app.on_event("startup")
async def startup_event():
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
    if not text or source_lang == target_lang or (target_lang == "en" and source_lang == "auto"):
        return text
    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception:
        return text

def get_bot_response(query_en: str) -> tuple[str, List[str]]:
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
    indian_languages = {
        "English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te", "Kannada": "kn",
        "Malayalam": "ml", "Marathi": "mr", "Gujarati": "gu", "Punjabi": "pa", "Bengali": "bn",
        "Odia": "or", "Urdu": "ur", "Assamese": "as", "French": "fr", "Spanish": "es",
        "German": "de", "Italian": "it", "manipuri": "mni", "swahili": "sw", "malay": "ms", "portuguese": "pt",
        "turkish": "tr", "arabic": "ar", "indonesian": "id", "thai": "th", "chichewa": "ny" }
    return {"languages": indian_languages}

@app.get("/recommendations/initial")
async def get_initial_recommendations():
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
    global bot, recommender
    if not bot or not recommender:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        query_en = translate_text(request.message, "en", request.user_language)
        answer_en, new_recommendations = get_bot_response(query_en)
        answer_translated = translate_text(answer_en, request.user_language)
        
        return ChatResponse(answer=answer_translated, recommendations=new_recommendations)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.post("/recommendations/action")
async def handle_recommendation_action(request: ActionRequest):
    """Handle recommendation actions ('go_back' or 'get_more')"""
    global recommender
    
    if not recommender:
        raise HTTPException(status_code=500, detail="Recommender not loaded")
    
    try:
        if request.action == "go_back":
            recommendations = recommender.go_back()
            return {"recommendations": recommendations}
        
        elif request.action == "get_more":
            recommendations = recommender.get_more_questions()
            return {"recommendations": recommendations}
            
        else:
            # Proper handling for unknown actions
            raise HTTPException(status_code=400, detail=f"Invalid action: '{request.action}'")
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error handling recommendation: {str(e)}")

@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        translated_text = translate_text(request.text, request.target_lang, request.source_lang)
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)