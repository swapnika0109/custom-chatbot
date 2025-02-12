from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn
from ..chatbot import FashionChatbot
from ..utils.rate_limiter import RateLimiter

app = FastAPI(title="Fashion Chatbot API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot
chatbot = FashionChatbot()
rate_limiter = RateLimiter()

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    intent: Optional[str]
    confidence: float

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not rate_limiter.can_process(request.user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    result = chatbot.process_message(request.user_id, request.message)
    return ChatResponse(
        response=result['response'],
        intent=result.get('intent'),
        confidence=result.get('response_confidence', 0.0)
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 