# main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from llama_cpp import Llama
import os
import requests
import logging
import psutil
import json
import asyncio
from typing import Optional

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
# ✅ FIXED URL: Using bartowski repo which has the correct filename casing
# File: SmolLM2-360M-Instruct-Q4_K_S.gguf (~200MB)
MODEL_URL = "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_S.gguf"
MODEL_PATH = "model.gguf"
MODEL_READY = False

# llama.cpp optimization params for 512MB RAM
LLM_CONFIG = {
    "n_ctx": 512,           # Small context window
    "n_threads": 2,         # Render free tier = 2 vCPU max
    "n_gpu_layers": 0,      # Force CPU-only
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
    "stop": ["</s>", "###", "\n\n", "User:", "assistant:", "<|im_end|>"],
    "echo": False,
}

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("OS will handle memory limits (512MB free tier constraint).")

# ─────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="SmolLM2-360M API",
    description="Lightweight LLM inference on Render free tier",
    version="1.0.0"
)

# ─────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500, description="Input prompt")
    max_tokens: Optional[int] = Field(default=100, ge=1, le=200)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)

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
# Model Loading Logic
# ─────────────────────────────────────────────────────────────
llm: Optional[Llama] = None

def download_model():
    """Download model with streaming"""
    if os.path.exists(MODEL_PATH):
        logger.info(f"✅ Model already exists at {MODEL_PATH}")
        return
    
    logger.info(f"⬇️ Downloading model from {MODEL_URL}...")
    try:
        with requests.get(MODEL_URL, stream=True, timeout=300) as r:
            r.raise_for_status()  # This will raise 404 if URL is wrong
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Simple progress log
                        if total_size > 0 and downloaded % (1024 * 1024 * 10) == 0:
                            logger.info(f"Downloaded {downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB")
        
        logger.info(f"✅ Model downloaded successfully: {MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Model download failed: {e}")
        raise

def load_model():
    """Load the LLM model"""
    global llm, MODEL_READY
    try:
        logger.info("🧠 Loading SmolLM2-360M model into memory...")
        llm = Llama(model_path=MODEL_PATH, **LLM_CONFIG)
        MODEL_READY = True
        logger.info("✅ Model loaded successfully! Ready for inference.")
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        MODEL_READY = False
        raise

def init_model_background():
    """Initialize model: download + load"""
    try:
        download_model()
        load_model()
    except Exception as e:
        logger.error(f"💥 Model initialization failed: {e}")

# ─────────────────────────────────────────────────────────────
# Startup Events
# ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    import asyncio
    logger.info("🚀 Starting SmolLM2 API...")
    # Run in background to prevent Render timeout
    asyncio.create_task(asyncio.to_thread(init_model_background))

# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────
# ── Serve UI ──────────────────────────────────────────────────
@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_ui():
    return FileResponse("static/index.html")

# ── SSE: Real-time metrics stream ─────────────────────────────
@app.get("/metrics/stream", tags=["Monitoring"])
async def metrics_stream(request: Request):
    async def event_generator():
        psutil.cpu_percent()          # prime the CPU baseline (first call is always 0)
        while True:
            if await request.is_disconnected():   # clean up when browser tab closes
                logger.info("SSE client disconnected")
                break
            try:
                mem  = psutil.virtual_memory()
                cpu  = psutil.cpu_percent()
                disk = psutil.disk_usage("/")
                payload = {
                    "cpu_percent":      round(cpu, 1),
                    "cpu_count":        psutil.cpu_count(logical=True),
                    "ram_percent":      round(mem.percent, 1),
                    "ram_used_mb":      round(mem.used      / 1024 / 1024, 1),
                    "ram_available_mb": round(mem.available / 1024 / 1024, 1),
                    "ram_total_mb":     round(mem.total     / 1024 / 1024, 1),
                    "disk_percent":     round(disk.percent, 1),
                    "disk_free_gb":     round(disk.free  / 1024 ** 3, 2),
                    "disk_total_gb":    round(disk.total / 1024 ** 3, 2),
                    "model_loaded":     MODEL_READY,
                }
                yield f"data: {json.dumps(payload)}\n\n"
            except Exception as e:
                logger.error(f"Metrics SSE error: {e}")
                yield "data: {}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",      # ← disables Nginx/Render proxy buffer
            "Connection":        "keep-alive",
        },
    )

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    mem = psutil.virtual_memory()
    return HealthResponse(
        status="healthy" if MODEL_READY else "initializing",
        model_loaded=MODEL_READY,
        ram_usage_percent=mem.percent,
        ram_available_mb=mem.available / 1024 / 1024
    )

@app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
async def generate(request: GenerateRequest):
    if not MODEL_READY or llm is None:
        raise HTTPException(
            status_code=503,
            detail="Model not ready yet. Please wait ~45-60 seconds after deploy."
        )
    
    try:
        mem = psutil.virtual_memory()
        if mem.percent > 95:
            logger.warning(f"⚠️ High memory usage: {mem.percent}%")
        
        gen_config = {
            **GENERATION_CONFIG,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        
        logger.info(f"Generating for: {request.prompt[:30]}...")
        output = llm(request.prompt, **gen_config)
        
        response_text = output["choices"][0]["text"].strip()
        tokens_used = len(response_text.split())  # Approximate
        
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
    if not MODEL_READY:
        raise HTTPException(503, "Model not ready")
    try:
        _ = llm("Hello", max_tokens=5, temperature=0)
        return {"status": "warmed up"}
    except Exception as e:
        raise HTTPException(500, f"Warmup failed: {str(e)}")