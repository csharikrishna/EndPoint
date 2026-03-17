# main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from llama_cpp import Llama
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional
import os, json, requests, logging, psutil, asyncio


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
MODEL_URL = (
    "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF"
    "/resolve/main/SmolLM2-360M-Instruct-Q4_K_S.gguf"
)
MODEL_PATH  = "model.gguf"
MODEL_READY = False

LLM_CONFIG = {
    "n_ctx":        256,    # KV cache sized for 256 tokens (~8–12 MB saved vs 512)
    "n_threads":    2,      # Render free tier = 2 vCPUs
    "n_gpu_layers": 0,      # Force CPU-only
    "use_mlock":    False,  # Don't pin model in RAM
    "use_mmap":     True,   # Memory-map model file (saves RSS)
    "n_batch":      16,     # Low peak RAM during prompt prefill
    "n_ubatch":     8,      # Physical micro-batch (llama-cpp-python ≥ 0.2.56)
    "logits_all":   False,  # Only compute last-token logit → saves CPU cycles
    "embedding":    False,  # Disable unused embedding endpoint overhead
    "verbose":      False,
}

STOP_TOKENS = ["</s>", "<|im_end|>", "###", "User:", "Human:", "Assistant:"]


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────────────────────
llm: Optional[Llama] = None
llm_semaphore = asyncio.Semaphore(1)   # serialize inference — Llama is not re-entrant


# ─────────────────────────────────────────────────────────────
# Prompt Template  (ChatML — required for instruct fine-tune)
# ─────────────────────────────────────────────────────────────
def format_prompt(user_input: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_input}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# ─────────────────────────────────────────────────────────────
# Model Download + Load
# ─────────────────────────────────────────────────────────────
def download_model() -> None:
    if os.path.exists(MODEL_PATH):
        logger.info("✅ Model already exists at %s", MODEL_PATH)
        return

    logger.info("⬇️  Downloading model from %s …", MODEL_URL)
    with requests.get(MODEL_URL, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total and downloaded % (10 * 1024 * 1024) == 0:
                        logger.info(
                            "  … %d MB / %d MB",
                            downloaded // (1024 * 1024),
                            total // (1024 * 1024),
                        )
    logger.info("✅ Model downloaded → %s", MODEL_PATH)


def load_model() -> None:
    global llm, MODEL_READY
    logger.info("🧠 Loading SmolLM2-360M into memory …")
    llm = Llama(model_path=MODEL_PATH, **LLM_CONFIG)
    MODEL_READY = True
    logger.info("✅ Model loaded and ready for inference.")


def init_model_background() -> None:
    try:
        download_model()
        load_model()
    except Exception as exc:
        logger.error("💥 Model initialization failed: %s", exc)


# ─────────────────────────────────────────────────────────────
# Lifespan (replaces deprecated @app.on_event)
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting SmolLM2 API …")
    # Run download + load in a thread so startup doesn't block Render's health check
    asyncio.create_task(asyncio.to_thread(init_model_background))
    yield
    logger.info("👋 Shutting down.")


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="SmolLM2-360M API",
    description="Lightweight LLM inference on Render free tier",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt:      str           = Field(..., min_length=1, max_length=500)
    max_tokens:  Optional[int]   = Field(default=100, ge=1, le=200)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    response:    str
    model:       str
    tokens_used: int
    status:      str


class HealthResponse(BaseModel):
    status:             str
    model_loaded:       bool
    ram_usage_percent:  float
    ram_available_mb:   float


# ─────────────────────────────────────────────────────────────
# LRU Cache  (temperature = 0 deterministic requests only)
# ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=32)
def _cached_infer(prompt: str, max_tokens: int) -> tuple[str, int]:
    """Synchronous cached inference for temperature=0 calls."""
    out   = llm(
        format_prompt(prompt),
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=0.9,
        stop=STOP_TOKENS,
        echo=False,
    )
    text   = out["choices"][0]["text"].strip()
    tokens = out["usage"]["completion_tokens"]      # ← real token count, not word split
    return text, tokens


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

# ── UI (serves static/index.html) ─────────────────────────────
@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_ui():
    return FileResponse("static/index.html")


# ── API status (former root) ───────────────────────────────────
@app.get("/api/status", tags=["Info"])
async def api_status():
    return {
        "name":    "SmolLM2-360M API",
        "version": "1.0.0",
        "model":   "SmolLM2-360M-Instruct (Q4_K_S)",
        "status":  "ready" if MODEL_READY else "loading",
        "docs":    "/docs",
    }


# ── Health ────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    mem = psutil.virtual_memory()
    return HealthResponse(
        status            = "healthy" if MODEL_READY else "initializing",
        model_loaded      = MODEL_READY,
        ram_usage_percent = mem.percent,
        ram_available_mb  = round(mem.available / 1024 / 1024, 1),
    )


# ── SSE: Real-time metrics stream ─────────────────────────────
@app.get("/metrics/stream", tags=["Monitoring"])
async def metrics_stream(request: Request):
    """
    Server-Sent Events endpoint.
    Pushes CPU / RAM / Disk metrics every 2 seconds.
    X-Accel-Buffering: no  →  disables Render/Nginx proxy buffer so
                               events are delivered immediately.
    """
    async def event_generator():
        psutil.cpu_percent()          # prime baseline — first call always returns 0.0
        while True:
            if await request.is_disconnected():
                logger.info("SSE client disconnected — closing stream.")
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
            except Exception as exc:
                logger.error("Metrics SSE error: %s", exc)
                yield "data: {}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",          # critical for Render/Nginx
            "Connection":        "keep-alive",
        },
    )


# ── Generate ──────────────────────────────────────────────────
@app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
async def generate(request: GenerateRequest):
    if not MODEL_READY or llm is None:
        raise HTTPException(
            status_code=503,
            detail="Model not ready yet. Please wait ~60 seconds after deploy.",
        )

    try:
        # Fast path: deterministic requests served from LRU cache
        if request.temperature == 0.0:
            text, tokens = _cached_infer(request.prompt, request.max_tokens)
            return GenerateResponse(
                response=text, model="SmolLM2-360M-Instruct",
                tokens_used=tokens, status="success (cached)",
            )

        # Normal path: serialized async inference
        async with asyncio.timeout(60):                      # hard 60-second deadline
            async with llm_semaphore:                        # one request at a time
                out = await asyncio.to_thread(               # off the event loop
                    llm,
                    format_prompt(request.prompt),
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=0.9,
                    stop=STOP_TOKENS,
                    echo=False,
                )

        text   = out["choices"][0]["text"].strip()
        tokens = out["usage"]["completion_tokens"]           # real token count

        logger.info("Generated %d tokens for: %.40s…", tokens, request.prompt)
        return GenerateResponse(
            response=text, model="SmolLM2-360M-Instruct",
            tokens_used=tokens, status="success",
        )

    except TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timed out after 60s.")
    except Exception as exc:
        logger.error("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")


# ── Warmup ────────────────────────────────────────────────────
@app.get("/warmup", tags=["Utilities"])
async def warmup():
    """Pre-warm the KV cache with a trivial call to cut first-user latency."""
    if not MODEL_READY or llm is None:
        raise HTTPException(status_code=503, detail="Model not ready.")
    try:
        _cached_infer("Hello", 5)        # cached → zero cost on repeat calls
        return {"status": "warmed up"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {exc}")
