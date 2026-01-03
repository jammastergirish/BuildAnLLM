"""FastAPI entrypoint for the transformer backend."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.routers import checkpoints, docs, finetune, inference, pretrain, system, tokenizers

app = FastAPI(title="Transformer Training API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(docs.router)
app.include_router(checkpoints.router)
app.include_router(pretrain.router)
app.include_router(finetune.router)
app.include_router(inference.router)
app.include_router(tokenizers.router)

frontend_dir = Path(__file__).resolve().parents[2] / "frontend" / "out"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


@app.get("/")
async def root() -> dict:
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"service": "transformer-backend"}
