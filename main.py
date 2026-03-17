from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
import os
import requests

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str

# Safe default for Phi-3 Mini 4K Instruct (Q4 quantized, fits in ~2.5GB RAM)
MODEL_URL = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
MODEL_PATH = "model.gguf"

if not os.path.exists(MODEL_PATH):
    print("Downloading model... This may take a few minutes.")
    # Streaming the download to limit memory consumption
    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Model downloaded successfully!")

# Load the model with optimizations for constrained environments (like Render's limits)
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,       # Reduced context window to save RAM
    n_threads=2      # Render free tier gives minimal CPU cores
)

@app.get("/")
def home():
    return {"status": "running", "message": "Phi-3 Mini is ready"}

@app.post("/generate")
def generate(request: GenerateRequest):
    output = llm(request.prompt, max_tokens=100)
    return {"response": output["choices"][0]["text"]}
