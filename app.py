import os
import json
import asyncio
import logging
from typing import Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
from threading import Thread
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
BASE_MODEL_ID = "openlm-research/open_llama_3b_v2"
LORA_MODEL_PATH = "./swinz-3b-lora"  # Path to your LoRA model
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "swinz-3b-lora"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False
    top_p: float = 0.9

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[Dict[str, Any]]

async def load_model():
    """Load the base model and apply LoRA weights"""
    global model, tokenizer
    
    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        if device.type == "cpu":
            base_model = base_model.to(device)
        
        logger.info("Loading LoRA weights...")
        if os.path.exists(LORA_MODEL_PATH):
            model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
            logger.info("LoRA model loaded successfully!")
        else:
            logger.warning(f"LoRA model path {LORA_MODEL_PATH} not found, using base model")
            model = base_model
        
        model.eval()
        logger.info(f"Model loaded on device: {device}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up...")
    await load_model()
    yield
    # Shutdown
    logger.info("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Swinz LoRA API",
    description="API for Swinz fine-tuned LoRA model with streaming support",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def format_prompt(messages: list[ChatMessage]) -> str:
    """Format messages into a prompt for the model"""
    # Extract the last user message as instruction
    instruction = ""
    for message in reversed(messages):
        if message.role == "user":
            instruction = message.content
            break
    
    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n### Response:"
    )
    return prompt

async def generate_stream(prompt: str, **generation_kwargs) -> AsyncGenerator[str, None]:
    """Generate streaming response"""
    try:
        # Setup streaming
        streamer = TextIteratorStreamer(
            tokenizer, 
            timeout=10.0, 
            skip_special_tokens=True,
            skip_prompt=True
        )
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generation parameters
        generation_config = {
            **inputs,
            "max_new_tokens": generation_kwargs.get("max_tokens", MAX_NEW_TOKENS),
            "temperature": generation_kwargs.get("temperature", TEMPERATURE),
            "top_p": generation_kwargs.get("top_p", TOP_P),
            "do_sample": True,
            "streamer": streamer,
            "pad_token_id": tokenizer.eos_token_id,
        }
        
        # Start generation in separate thread
        generation_thread = Thread(
            target=model.generate,
            kwargs=generation_config
        )
        generation_thread.start()
        
        # Stream tokens
        for token in streamer:
            if token:
                yield token
                
        generation_thread.join()
        
    except Exception as e:
        logger.error(f"Error in generation: {e}")
        yield f"Error: {str(e)}"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "swinz-3b-lora",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "swinz"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Chat completions endpoint (OpenAI compatible)"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Format prompt
        prompt = format_prompt(request.messages)
        
        # Generate unique ID
        completion_id = f"chatcmpl-{int(time.time())}"
        created = int(time.time())
        
        if request.stream:
            # Streaming response
            async def stream_response():
                yield f"data: {json.dumps({})}\n\n"  # Initial chunk
                
                async for token in generate_stream(
                    prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p
                ):
                    chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=request.model,
                        choices=[{
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None
                        }]
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    await asyncio.sleep(0.01)  # Small delay for better streaming
                
                # Final chunk
                final_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                stream_response(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/plain; charset=utf-8"
                }
            )
        
        else:
            # Non-streaming response
            full_response = ""
            async for token in generate_stream(
                prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            ):
                full_response += token
            
            response = ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_response.strip()
                    },
                    "finish_reason": "stop"
                }],
                usage={
                    "prompt_tokens": len(tokenizer.encode(prompt)),
                    "completion_tokens": len(tokenizer.encode(full_response)),
                    "total_tokens": len(tokenizer.encode(prompt)) + len(tokenizer.encode(full_response))
                }
            )
            
            return response
    
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def simple_chat(request: dict):
    """Simplified chat endpoint"""
    try:
        message = request.get("message", "")
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        messages = [ChatMessage(role="user", content=message)]
        chat_request = ChatCompletionRequest(
            messages=messages,
            stream=request.get("stream", False),
            temperature=request.get("temperature", 0.7),
            max_tokens=request.get("max_tokens", 512)
        )
        
        return await chat_completions(chat_request)
        
    except Exception as e:
        logger.error(f"Error in simple chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)