"""FastAPI service for the contract-clause demo: POST /ask (grounded Q&A) and POST /classify
(clause type), the two tasks the adapter was trained and evaluated on.

    uvicorn app.api:app --port 8000

Env vars: LEGAL_FT_ADAPTER (Hub id or local path; default the published adapter),
LEGAL_FT_BASE_MODEL, LEGAL_FT_EMBED_ON_CPU (default "1", needed on 6 GB GPUs).
The model loads once at startup (~1 min); requests are served one at a time.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from legal_ft.inference import DEFAULT_ADAPTER, DEFAULT_BASE, DISCLAIMER, InvalidRequest

MAX_EXCERPT_CHARS = 12_000  # cheap pre-check; the real limit is in tokens (inference.py)


class AskRequest(BaseModel):
    excerpt: str = Field(min_length=1, max_length=MAX_EXCERPT_CHARS,
                         description="Contract excerpt (up to ~1,300 tokens)")
    clause_type: str = Field(description="One of GET /clause-types")


class QuoteOut(BaseModel):
    text: str
    verified: bool


class AskResponse(BaseModel):
    clause_type: str
    present: bool | None
    confidence: float
    confidence_label: str
    p_present: float
    quotes: list[QuoteOut]
    all_quotes_verified: bool
    valid_output: bool
    latency_s: float
    model: str
    disclaimer: str = DISCLAIMER


class ClassifyRequest(BaseModel):
    clause: str = Field(min_length=1, max_length=MAX_EXCERPT_CHARS,
                        description="A single contract clause (up to ~500 tokens)")


class ClassifyResponse(BaseModel):
    label: str | None
    confidence: float
    confidence_label: str
    valid_output: bool
    latency_s: float
    model: str
    disclaimer: str = DISCLAIMER


def default_factory():
    from legal_ft.inference import ClauseQA

    return ClauseQA(
        adapter=os.environ.get("LEGAL_FT_ADAPTER", DEFAULT_ADAPTER) or None,
        base_model=os.environ.get("LEGAL_FT_BASE_MODEL", DEFAULT_BASE),
        embed_on_cpu=os.environ.get("LEGAL_FT_EMBED_ON_CPU", "1") == "1",
    )


def create_app(qa_factory: Callable = default_factory) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.qa = qa_factory()
        yield

    app = FastAPI(title="Contract clause QA (QLoRA demo)", version="0.1.0", lifespan=lifespan,
                  description=DISCLAIMER)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": app.state.qa.model_name}

    @app.get("/clause-types")
    def clause_types():
        return {"clause_types": app.state.qa.clause_types}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest):
        # sync handler: FastAPI runs it in a worker thread; ClauseQA serializes GPU access
        try:
            answer = app.state.qa.ask(req.excerpt, req.clause_type)
        except InvalidRequest as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        return AskResponse(**{k: v for k, v in answer.to_dict().items() if k != "raw_output"},
                           model=app.state.qa.model_name)

    @app.post("/classify", response_model=ClassifyResponse)
    def classify(req: ClassifyRequest):
        try:
            result = app.state.qa.classify(req.clause)
        except InvalidRequest as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        return ClassifyResponse(**{k: v for k, v in result.to_dict().items() if k != "raw_output"},
                                model=app.state.qa.model_name)

    return app


app = create_app()
