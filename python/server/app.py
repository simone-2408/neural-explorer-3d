"""
FastAPI backend for the LLM and Experiments tabs.

    cd python && .venv/bin/python -m uvicorn server.app:app --port 8000
    (or, from the project root: npm run llm)

The Vite dev server proxies /api to http://127.0.0.1:8000, so the browser
talks to the same origin. The model loads in the background at start-up;
/api/llm/status tells the UI when it is ready.
"""
from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .experiments import BadRequest, Experiments
from .llm import LLM, ModelNotReady

llm = LLM()
experiments = Experiments(llm)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=llm.load, daemon=True).start()
    yield


app = FastAPI(title="Neural Explorer 3D — LLM backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str | None = Field(default=None, max_length=4000)
    ids: list[int] | None = Field(default=None, max_length=256)


class GenerateRequest(BaseModel):
    text: str = Field(max_length=4000)
    temperature: float = Field(default=1.0, ge=0, le=5)
    topK: int = Field(default=0, ge=0, le=1000)
    topP: float = Field(default=1.0, gt=0, le=1)
    maxNewTokens: int = Field(default=12, ge=1, le=40)
    seed: int = 0


class ExperimentRequest(TextRequest):
    # Interventions as described in experiments.py (validated there, with readable errors).
    interventions: list[dict] = Field(default_factory=list, max_length=512)
    target: int | None = None
    contrast: int | None = None


class NeuronsRequest(ExperimentRequest):
    layer: int = Field(ge=0, le=47)
    k: int = Field(default=24, ge=1, le=64)
    extra: list[int] = Field(default_factory=list, max_length=8)


class CurveRequest(ExperimentRequest):
    component: dict
    scales: list[float] = Field(min_length=1, max_length=41)


class AttentionRequest(ExperimentRequest):
    layer: int = Field(ge=0, le=47)
    head: int = Field(ge=0, le=63)


class TokenizeRequest(BaseModel):
    text: str = Field(max_length=200)


def run(fn, *args):
    try:
        with llm.compute_lock:
            return fn(*args)
    except ModelNotReady as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except BadRequest as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"{type(error).__name__}: {error}") from error


@app.get("/api/llm/status")
def status() -> dict:
    return llm.status()


@app.post("/api/llm/analyze")
def analyze(request: TextRequest) -> dict:
    return run(llm.analyze, request.text, request.ids)


@app.post("/api/llm/generate")
def generate(request: GenerateRequest) -> dict:
    return run(llm.generate, request.text, request.temperature, request.topK, request.topP, request.maxNewTokens, request.seed)


@app.post("/api/llm/tokenize")
def tokenize(request: TokenizeRequest) -> dict:
    return run(experiments.tokenize, request.text)


@app.post("/api/llm/experiment")
def experiment(request: ExperimentRequest) -> dict:
    return run(experiments.experiment, request.text, request.ids, request.interventions, request.target, request.contrast)


@app.post("/api/llm/sweep")
def sweep(request: ExperimentRequest) -> dict:
    return run(experiments.sweep, request.text, request.ids, request.interventions, request.target, request.contrast)


@app.post("/api/llm/neurons")
def neurons(request: NeuronsRequest) -> dict:
    return run(experiments.neurons, request.text, request.ids, request.interventions, request.layer, request.target, request.contrast, request.k, request.extra)


@app.post("/api/llm/curve")
def curve(request: CurveRequest) -> dict:
    return run(experiments.curve, request.text, request.ids, request.interventions, request.component, request.scales, request.target, request.contrast)


@app.post("/api/llm/attention")
def attention(request: AttentionRequest) -> dict:
    return run(experiments.attention, request.text, request.ids, request.interventions, request.layer, request.head)
