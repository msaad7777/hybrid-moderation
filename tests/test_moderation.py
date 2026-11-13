"""Fast tests: no model downloads, no Ollama. The LLM is faked with httpx.MockTransport."""
import json

import httpx
import numpy as np
import pandas as pd
import pytest

from hybrid_moderation import evaluate
from hybrid_moderation.data import COUNTER_NH, IDENTITY_NH, OTHER_NH, POLICY_GREY
from hybrid_moderation.moderators import LLM, THREAT, Decision, Rules
from hybrid_moderation.pipeline import Hybrid, create_app


def test_groups_do_not_overlap():
    groups = [IDENTITY_NH, COUNTER_NH, OTHER_NH, POLICY_GREY]
    flat = [f for g in groups for f in g]
    assert len(flat) == len(set(flat))
    assert all(f.endswith("_nh") for f in flat)


def test_rules_match_whole_words_and_threats(tmp_path):
    words = tmp_path / "w.txt"
    words.write_text("ass\nbadword\n")
    r = Rules(words)
    assert r("what a badword").flag
    assert not r("a classic assessment").flag          # no match inside other words
    assert r("I'm going to kill you").flag
    assert not r("this film will kill it at the box office").flag
    assert THREAT.search("we will burn your house")


def fake_ollama(content: str):
    def handler(request):
        body = json.loads(request.content)
        assert body["format"]["required"] == ["flag", "category", "reason"]
        return httpx.Response(200, json={"message": {"content": content}})
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://x")


def test_llm_parses_schema_answer():
    m = LLM("gpt-oss:20b")
    m.client = fake_ollama(json.dumps({"flag": True, "category": "insult", "reason": "name-calling"}))
    d = m("you idiot")
    assert d.flag and d.route == "llm" and d.extra["category"] == "insult"


def test_llm_bad_json_does_not_flag():
    m = LLM("qwen3:8b")
    m.client = fake_ollama("not json")
    d = m("hello")
    assert d.flag is False and "error" in d.extra


def frame(scores, flags=None, ms=1.0):
    n = len(scores)
    return pd.DataFrame({"i": range(n), "score": scores, "flag": flags if flags is not None else [False] * n,
                         "ms": [ms] * n})


def test_hybrid_routes_only_the_grey_zone():
    ml = frame([0.01, 0.5, 0.99], ms=2.0)
    llm = frame([1, 1, 1], flags=[True, False, False], ms=1000.0)
    h = evaluate.hybrid(ml, llm, lo=0.1, hi=0.9)
    assert h.to_llm.tolist() == [False, True, False]
    assert h.flag.tolist() == [False, False, True]      # clear allow, LLM says no, clear flag
    assert h.ms.tolist() == [2.0, 1002.0, 2.0]


def test_tune_respects_the_llm_cap(monkeypatch):
    rng = np.random.default_rng(0)
    y = rng.random(400) < 0.3
    ml_scores = np.clip(np.where(y, 0.7, 0.3) + rng.normal(0, 0.2, 400), 0, 1)
    tables = {"ml": frame(ml_scores), "llm": frame([1.0] * 400, flags=y.tolist(), ms=500.0)}
    monkeypatch.setattr(evaluate, "load", lambda split: pd.DataFrame({"label": y.astype(int)}))
    monkeypatch.setattr(evaluate, "scores", lambda mod, split: tables[mod])
    t = evaluate.tune("ml", "llm", cap=0.2)
    assert t["tune_llm_share"] <= 0.2
    ml_only = evaluate.f1(y, ml_scores >= 0.5)
    assert t["tune_f1"] >= ml_only                     # a perfect LLM in the grey zone can only help


def test_hatecheck_metrics_split_groups():
    hc = pd.DataFrame({"functionality": ["slur_h", "ident_neutral_nh", "counter_quote_nh", "profanity_nh"],
                       "label": [1, 0, 0, 0]})
    m = evaluate.hatecheck_metrics([True, False, True, True], hc)
    assert m["hate_recall"] == 1.0
    assert m["fp_identity"] == 0.0 and m["fp_counter_speech"] == 1.0
    assert m["flag_rate_policy_grey"] == 1.0
    assert m["accuracy"] == pytest.approx(2 / 3, abs=1e-3)   # grey cases are left out of accuracy


class Fixed:
    def __init__(self, score):
        self.score, self.calls = score, 0

    def __call__(self, text):
        self.calls += 1
        return Decision(self.score >= 0.5, self.score, "ml", 1.0)


def test_pipeline_api_calls_llm_only_in_grey_zone():
    from fastapi.testclient import TestClient
    calls = []

    def fake_llm(text):
        calls.append(text)
        return Decision(True, 1.0, "llm", 5.0, {"category": "insult", "reason": "r"})

    client = TestClient(create_app(Hybrid(Fixed(0.02), fake_llm, 0.1, 0.9)))
    r = client.post("/moderate", json={"text": "hello"}).json()
    assert r["flag"] is False and r["route"] == "ml" and not calls
    client = TestClient(create_app(Hybrid(Fixed(0.5), fake_llm, 0.1, 0.9)))
    r = client.post("/moderate", json={"text": "hmm"}).json()
    assert r["flag"] is True and r["route"] == "llm" and r["category"] == "insult" and calls == ["hmm"]
    assert client.post("/moderate", json={"text": ""}).status_code == 422


def test_tune_floor_forces_a_wider_band(monkeypatch):
    rng = np.random.default_rng(1)
    y = rng.random(400) < 0.3
    ml_scores = np.clip(np.where(y, 0.7, 0.3) + rng.normal(0, 0.2, 400), 0, 1)
    tables = {"ml": frame(ml_scores), "llm": frame([1.0] * 400, flags=y.tolist())}
    monkeypatch.setattr(evaluate, "load", lambda split: pd.DataFrame({"label": y.astype(int)}))
    monkeypatch.setattr(evaluate, "scores", lambda mod, split: tables[mod])
    t = evaluate.tune("ml", "llm", cap=0.25, floor=0.2)
    assert 0.2 <= t["tune_llm_share"] <= 0.25 and t["floor"] == 0.2


def test_pipeline_profile_picks_thresholds(tmp_path, monkeypatch):
    from hybrid_moderation import pipeline
    f = tmp_path / "results.json"
    f.write_text(json.dumps({"hybrid": [
        {"ml": "detoxify", "llm": "llm_gpt-oss", "lo": 0.3, "hi": 0.4, "floor": 0.0},
        {"ml": "detoxify", "llm": "llm_gpt-oss", "lo": 0.01, "hi": 0.4, "floor": 0.2}]}))
    monkeypatch.setattr(pipeline, "RESULTS", f)
    assert pipeline.thresholds(profile="f1") == (0.3, 0.4)
    assert pipeline.thresholds(profile="wide") == (0.01, 0.4)
