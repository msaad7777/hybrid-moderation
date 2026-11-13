"""The hybrid moderator as a service: a fast classifier decides clear cases, a local LLM decides the grey zone.

    uv run uvicorn hybrid_moderation.pipeline:app --port 8000
    curl -s localhost:8000/moderate -H 'content-type: application/json' -d '{"text": "you are an idiot"}'

Thresholds come from results/results.json (chosen on the tune set by `evaluate report`).
MODERATION_PROFILE=f1 (default) uses the best-F1 thresholds; MODERATION_PROFILE=wide sends 20-25% to the LLM.
"""
import json
import os
import time
from pathlib import Path

from .moderators import LLM, Decision

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "results.json"


def thresholds(ml: str = "detoxify", llm: str = "llm_gpt-oss", profile: str | None = None) -> tuple[float, float]:
    wide = (profile or os.environ.get("MODERATION_PROFILE", "f1")) == "wide"
    for h in json.loads(RESULTS.read_text())["hybrid"]:
        if h["ml"] == ml and h["llm"] == llm and bool(h.get("floor")) == wide:
            return h["lo"], h["hi"]
    raise SystemExit(f"no tuned thresholds for {ml} + {llm}: run `python -m hybrid_moderation.evaluate report`")


class Hybrid:
    def __init__(self, ml, llm, lo: float, hi: float):
        self.ml, self.llm, self.lo, self.hi = ml, llm, lo, hi

    def __call__(self, text: str) -> Decision:
        t0 = time.perf_counter()
        first = self.ml(text)
        if first.score <= self.lo or first.score >= self.hi:
            return Decision(first.score >= self.hi, first.score, "ml", (time.perf_counter() - t0) * 1000)
        second = self.llm(text)
        return Decision(second.flag, first.score, "llm", (time.perf_counter() - t0) * 1000, second.extra)


def build() -> Hybrid:
    from .moderators import Detox
    lo, hi = thresholds()
    return Hybrid(Detox(), LLM("gpt-oss:20b"), lo, hi)


def create_app(moderator=None):
    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    app = FastAPI(title="hybrid-moderation")
    state = {"m": moderator}

    class Comment(BaseModel):
        text: str = Field(min_length=1, max_length=5000)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.post("/moderate")
    def moderate(c: Comment):
        if state["m"] is None:
            state["m"] = build()
        d = state["m"](c.text)
        return {"flag": d.flag, "route": d.route, "ml_score": round(d.score, 4), "ms": round(d.ms, 1),
                "category": d.extra.get("category"), "reason": d.extra.get("reason")}

    return app


app = create_app()
