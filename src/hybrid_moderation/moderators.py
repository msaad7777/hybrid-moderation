"""Four ways to moderate a comment. Each returns a score in [0, 1] and a flag."""
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"


@dataclass
class Decision:
    flag: bool
    score: float
    route: str               # which stage decided: rules, ml, llm
    ms: float
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------ rules
THREAT = re.compile(
    r"\b(i(?:'ll| will| am going to|'m going to|'m gonna| gonna)|we(?:'ll| will)|you(?:'ll| will) be)\s+"
    r"(?:\w+\s+){0,2}(kill|murder|shoot|stab|hang|burn|hurt|beat|rape|slaughter)(?:ed)?\b", re.I)


class Rules:
    """A public bad-word list (LDNOOBW, CC BY 4.0) plus a direct-threat pattern. No model at all."""

    name = "rules"

    def __init__(self, words_file: Path = ROOT / "data" / "raw" / "ldnoobw_en.txt"):
        words = [w.strip().lower() for w in words_file.read_text().splitlines() if w.strip()]
        words.sort(key=len, reverse=True)
        self.pattern = re.compile(r"(?<![a-z])(" + "|".join(re.escape(w) for w in words) + r")(?![a-z])", re.I)

    def __call__(self, text: str) -> Decision:
        t0 = time.perf_counter()
        hit = self.pattern.search(text) or THREAT.search(text)
        return Decision(bool(hit), 1.0 if hit else 0.0, "rules", (time.perf_counter() - t0) * 1000,
                        {"match": hit.group(0).lower() if hit else None})


# ------------------------------------------------------------------ ML
class TfidfLR:
    """TF-IDF word and character n-grams with logistic regression, trained here on 150k Civil Comments."""

    name = "tfidf_lr"
    path = MODELS / "tfidf_lr.joblib"

    def __init__(self, threshold: float = 0.5):
        import joblib
        self.model = joblib.load(self.path)
        self.threshold = threshold

    @classmethod
    def train(cls, texts, labels):
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline, make_union
        feats = make_union(
            TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=200_000, sublinear_tf=True))
        model = make_pipeline(feats, LogisticRegression(C=4.0, max_iter=2000))
        model.fit(list(texts), list(labels))
        MODELS.mkdir(exist_ok=True)
        joblib.dump(model, cls.path)

    def __call__(self, text: str) -> Decision:
        t0 = time.perf_counter()
        p = float(self.model.predict_proba([text])[0, 1])
        return Decision(p >= self.threshold, p, "ml", (time.perf_counter() - t0) * 1000)


class Detox:
    """Detoxify 'unbiased' (Apache 2.0): a RoBERTa model trained with an identity-bias objective."""

    name = "detoxify"

    def __init__(self, threshold: float = 0.5):
        from detoxify import Detoxify
        self.model = Detoxify("unbiased", device="cpu")
        self.threshold = threshold

    def __call__(self, text: str) -> Decision:
        t0 = time.perf_counter()
        out = self.model.predict(text)
        p = float(out["toxicity"])
        return Decision(p >= self.threshold, p, "ml", (time.perf_counter() - t0) * 1000,
                        {k: round(float(v), 4) for k, v in out.items()})


# ------------------------------------------------------------------ LLM
POLICY = """You are a comment moderator for a news site. Decide whether a comment breaks this policy.

Flag the comment if it contains any of:
- insults, name-calling or harassment aimed at a person or a group
- hate: demeaning, dehumanising or attacking people for a protected identity (religion, race, ethnicity, nationality, gender, sexual orientation, disability, age)
- threats or calls for violence
- sexually explicit or obscene abuse

Do NOT flag:
- a neutral or positive mention of an identity ("I'm a gay woman", "Muslims in my town run the food bank")
- quoting, reporting or condemning hate in order to criticise it
- disagreement, criticism of ideas, policies, companies or public figures' actions, if it contains no insult
- swearing that is not aimed at anyone ("this weather is fucking great")

Judge the comment's meaning, including misspelt or disguised words."""

SCHEMA = {
    "type": "object",
    "properties": {
        "flag": {"type": "boolean"},
        "category": {"type": "string", "enum": ["none", "insult", "hate", "threat", "obscene"]},
        "reason": {"type": "string"},
    },
    "required": ["flag", "category", "reason"],
}


class LLM:
    """A local model through Ollama, with the policy above and a JSON schema for the answer."""

    def __init__(self, model: str = "gpt-oss:20b", host: str = "http://127.0.0.1:11434"):
        self.model, self.name = model, "llm_" + model.split(":")[0].replace(".", "")
        self.client = httpx.Client(base_url=host, timeout=180)

    def __call__(self, text: str) -> Decision:
        t0 = time.perf_counter()
        body = {"model": self.model, "stream": False, "format": SCHEMA, "options": {"temperature": 0, "num_ctx": 4096},
                "messages": [{"role": "system", "content": POLICY},
                             {"role": "user", "content": f"Comment:\n<<<\n{text}\n>>>"}]}
        if self.model.startswith("gpt-oss"):
            body["think"] = "low"
        elif self.model.startswith("qwen3"):
            body["think"] = False
        r = self.client.post("/api/chat", json=body)
        r.raise_for_status()
        raw = r.json()["message"]["content"]
        try:
            out = json.loads(raw)
            flag = bool(out["flag"])
        except (ValueError, KeyError):
            out, flag = {"error": raw[:200]}, False
        return Decision(flag, 1.0 if flag else 0.0, "llm", (time.perf_counter() - t0) * 1000, out)
