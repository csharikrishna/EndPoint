# main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from llama_cpp import Llama
import os
import requests
import logging
import psutil
import resource
from typing import Optional

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
MODEL_URL = "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q4_k_s.gguf"
MODEL_PATH = "model.gguf"
MODEL_READY = False

# llama.cpp optimization params for 512MB RAM
LLM_CONFIG = {
    "n_ctx": 512,           # Small context window
    "n_threads": 2,         # Render free tier = 2 vCPU max
    "n_gpu_layers": 0,      # Force CPU-only (no GPU on free tier)
    "use_mlock": False,     # Don't lock model in RAM
    "use_mmap": True,       # Memory-map model file (saves RAM)
    "n_batch": 32,          # Small batch to reduce peak memory
    "verbose": False,
}

# Generation defaults
GENERATION_CONFIG = {
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "stop": ["</s>", "###", "\n\n", "User:", "assistant:"],
    "echo": False,
}

# ─────────────────────────────────────────────────────────────
# Logging & Memory Limits
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Limit process memory to ~450MB (leave ~62MB for OS/Python overhead)
# NOTE: We have disabled manual `resource.setrlimit(RLIMIT_AS)` because limiting 
# VIRTUAL memory to 450MB crashes Python's internal imports (like `re` and Pydantic)
# on startup with "SystemError". Render's container will automatically enforce the 
# 512MB physical RAM limit, so we don't need this manual strict limit.
logger.info("OS will handle memory limits (512MB free tier constraint).")

# ─────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="SmolLM2 API",
    description="Lightweight LLM inference on Render free tier",
    version="1.0.0"
)

# ─────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500, description="Input prompt for the model")
    max_tokens: Optional[int] = Field(default=100, ge=1, le=200, description="Max tokens to generate")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=1.0, description="Sampling temperature")

class GenerateResponse(BaseModel):
    response: str
    model: str
    tokens_used: int
    status: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    ram_usage_percent: float
    ram_available_mb: float

# ─────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────
llm: Optional[Llama] = None

def download_model():
    """Download model with streaming to avoid memory spikes"""
    if os.path.exists(MODEL_PATH):
        logger.info(f"Model already exists at {MODEL_PATH}")
        return
    
    logger.info(f"Downloading model from {MODEL_URL}...")
    try:
        with requests.get(MODEL_URL, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logger.info(f"✅ Model downloaded successfully: {MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Model download failed: {e}")
        raise

def load_model():
    """Load the LLM model with optimized settings"""
    global llm, MODEL_READY
    
    try:
        logger.info("Loading SmolLM2-360M model...")
        llm = Llama(model_path=MODEL_PATH, **LLM_CONFIG)
        MODEL_READY = True
        logger.info("✅ Model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        MODEL_READY = False
        raise

def init_model_background():
    """Initialize model: download + load (run in background)"""
    try:
        download_model()
        load_model()
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")

# ─────────────────────────────────────────────────────────────
# Startup Events
# ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Start model loading in background to avoid cold-start timeout"""
    import asyncio
    logger.info("🚀 Starting SmolLM2 API...")
    # Run model init in background thread to not block startup
    asyncio.create_task(asyncio.to_thread(init_model_background))

# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API info"""
    return {
        "name": "SmolLM2-360M API",
        "version": "1.0.0",
        "model": "SmolLM2-360M-Instruct (Q4_K_S)",
        "status": "ready" if MODEL_READY else "loading",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint with memory stats"""
    mem = psutil.virtual_memory()
    return HealthResponse(
        status="healthy" if MODEL_READY else "initializing",
        model_loaded=MODEL_READY,
        ram_usage_percent=mem.percent,
        ram_available_mb=mem.available / 1024 / 1024
    )

@app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
async def generate(request: GenerateRequest):
    """Generate text from prompt"""
    if not MODEL_READY or llm is None:
        raise HTTPException(
            status_code=503,
            detail="Model not ready yet. Please wait ~30-60 seconds after deploy and try again."
        )
    
    try:
        # Check memory before inference
        mem = psutil.virtual_memory()
        if mem.percent > 95:
            logger.warning(f"High memory usage: {mem.percent}%")
        
        # Merge request params with defaults
        gen_config = {
            **GENERATION_CONFIG,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        
        # Run inference
        logger.info(f"Generating for prompt: {request.prompt[:50]}...")
        output = llm(request.prompt, **gen_config)
        
        # Extract and clean response
        response_text = output["choices"][0]["text"].strip()
        tokens_used = len(output["choices"][0]["tokens"]) if "tokens" in output["choices"][0] else 0
        
        logger.info(f"✅ Generated {tokens_used} tokens")
        return GenerateResponse(
            response=response_text,
            model="SmolLM2-360M-Instruct",
            tokens_used=tokens_used,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"❌ Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/warmup", tags=["Utilities"])
async def warmup():
    """Endpoint to trigger a dummy generation (helps with cold starts)"""
    if not MODEL_READY:
        raise HTTPException(503, "Model not ready")
    
    try:
        _ = llm("Hello", max_tokens=10, temperature=0)
        return {"status": "warmed up", "message": "Dummy generation complete"}
    except Exception as e:
        raise HTTPException(500, f"Warmup failed: {str(e)}")

# ─────────────────────────────────────────────────────────────
# Optional: Shutdown cleanup
# ─────────────────────────────────────────────────────────────
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🔌 Shutting down...")
    global llm
    if llm:
        del llm