"""Score every moderator on every set, cache the per-comment results, then tune and report the hybrid.

    python -m hybrid_moderation.evaluate prepare        # build the sets, train TF-IDF
    python -m hybrid_moderation.evaluate score detoxify civil_test
    python -m hybrid_moderation.evaluate tune
    python -m hybrid_moderation.evaluate report

Scores are cached in results/scores/<moderator>__<set>.parquet, so a long LLM run can stop and resume.
The hybrid thresholds are chosen on the tune set only. civil_test and hatecheck are never used to choose anything.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import COUNTER_NH, IDENTITY_NH, OTHER_NH, POLICY_GREY, load

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SCORES = RESULTS / "scores"
SETS = ["tune", "civil_test", "hatecheck"]
LLM_SHARE_CAP = 0.25      # the hybrid may send at most this share of comments to the LLM


def make(name: str):
    from . import moderators as m
    if name == "rules":
        return m.Rules()
    if name == "tfidf_lr":
        return m.TfidfLR()
    if name == "detoxify":
        return m.Detox()
    if name.startswith("llm_"):
        return m.LLM({"llm_gpt-oss": "gpt-oss:20b", "llm_qwen3": "qwen3:8b"}[name])
    raise SystemExit(f"unknown moderator {name}")


def cache_path(mod: str, split: str) -> Path:
    return SCORES / f"{mod}__{split}.parquet"


def score(mod: str, split: str, batch: int = 25) -> pd.DataFrame:
    """Run one moderator over one set. Resumes from the cache and saves every `batch` comments."""
    SCORES.mkdir(parents=True, exist_ok=True)
    path = cache_path(mod, split)
    data = load(split)
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["i", "score", "flag", "ms", "extra"])
    todo = [i for i in range(len(data)) if i not in set(done["i"])]
    if not todo:
        return done
    moderator, rows, t0 = make(mod), [], time.time()
    for n, i in enumerate(todo, 1):
        d = moderator(data.text[i])
        rows.append({"i": i, "score": d.score, "flag": d.flag, "ms": d.ms, "extra": json.dumps(d.extra)})
        if n % batch == 0 or n == len(todo):
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if len(done) else pd.DataFrame(rows)
            done.to_parquet(path)
            rows = []
            print(f"{mod} {split}: {len(done)}/{len(data)}  {time.time() - t0:.0f}s", flush=True)
    return done


def scores(mod: str, split: str) -> pd.DataFrame:
    d = pd.read_parquet(cache_path(mod, split)).sort_values("i").reset_index(drop=True)
    if len(d) != len(load(split)):
        raise SystemExit(f"{mod} on {split} is incomplete ({len(d)} rows): run score first")
    return d


# ------------------------------------------------------------------ hybrid
def hybrid(ml: pd.DataFrame, llm: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """ML decides when it is confident (score <= lo allows, >= hi flags); the grey zone goes to the LLM."""
    grey = (ml.score > lo) & (ml.score < hi)
    flag = np.where(grey, llm.flag.astype(bool), ml.score >= hi)
    ms = ml.ms + np.where(grey, llm.ms, 0.0)
    return pd.DataFrame({"i": ml.i, "flag": flag, "ms": ms, "to_llm": grey})


def f1(y, p) -> float:
    tp = int((y & p).sum())
    return 2 * tp / max(int(y.sum()) + int(p.sum()), 1)


def tune(ml_mod: str, llm_mod: str, cap: float = LLM_SHARE_CAP, floor: float = 0.0) -> dict:
    """Grid-search lo < hi on the tune set: best F1 with `floor` <= LLM share <= `cap`."""
    y = load("tune").label.values.astype(bool)
    ml, llm = scores(ml_mod, "tune"), scores(llm_mod, "tune")
    grid = np.unique(np.quantile(ml.score, np.linspace(0, 1, 101)))
    best = None
    for a in range(len(grid)):
        for b in range(a + 1, len(grid)):
            lo, hi = float(grid[a]), float(grid[b])
            h = hybrid(ml, llm, lo, hi)
            share = h.to_llm.mean()
            if share > cap:
                break
            if share < floor:
                continue
            cand = (f1(y, h.flag.values), -share, lo, hi)
            if best is None or cand > best:
                best = cand
    return {"ml": ml_mod, "llm": llm_mod, "lo": best[2], "hi": best[3], "tune_f1": round(best[0], 3),
            "tune_llm_share": round(-best[1], 3), "cap": cap, "floor": floor}


def tune_threshold(ml_mod: str) -> float:
    """Ablation: the ML model alone, with one threshold chosen on the tune set for best F1."""
    y = load("tune").label.values.astype(bool)
    s = scores(ml_mod, "tune").score.values
    grid = np.unique(np.quantile(s, np.linspace(0, 1, 201)))
    return float(max(grid, key=lambda t: f1(y, s >= t)))


# ------------------------------------------------------------------ metrics
def civil_metrics(flag, y, ms) -> dict:
    flag, y = np.asarray(flag, bool), np.asarray(y, bool)
    tp, fp = int((flag & y).sum()), int((flag & ~y).sum())
    return {"precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / int(y.sum()), 3),
            "f1": round(f1(y, flag), 3), "ms_mean": round(float(np.mean(ms)), 1),
            "ms_p95": round(float(np.percentile(ms, 95)), 1)}


def hatecheck_metrics(flag, hc: pd.DataFrame) -> dict:
    flag = pd.Series(np.asarray(flag, bool), index=hc.index)
    hateful = hc.label == 1

    def rate(funcs):
        m = hc.functionality.isin(funcs)
        return round(float(flag[m].mean()), 3)

    non_hate = IDENTITY_NH + COUNTER_NH + OTHER_NH
    scored = hateful | hc.functionality.isin(non_hate)
    return {"hate_recall": round(float(flag[hateful].mean()), 3),
            "fp_rate_non_hate": rate(non_hate),
            "fp_identity": rate(IDENTITY_NH), "fp_counter_speech": rate(COUNTER_NH), "fp_other": rate(OTHER_NH),
            "flag_rate_policy_grey": rate(POLICY_GREY),
            "accuracy": round(float((flag[scored] == hateful[scored]).mean()), 3)}


def report() -> dict:
    out = {"civil_test": {}, "hatecheck": {}, "hybrid": [], "ml_tuned": {}, "per_functionality": {}}
    singles = [m for m in ["rules", "tfidf_lr", "detoxify", "llm_gpt-oss", "llm_qwen3"]
               if all(cache_path(m, s).exists() for s in ["civil_test", "hatecheck"])]
    civ, hc = load("civil_test"), load("hatecheck")
    systems = {m: (scores(m, "civil_test"), scores(m, "hatecheck")) for m in singles}
    for ml_mod in ["tfidf_lr", "detoxify"]:
        if all(cache_path(ml_mod, s).exists() for s in SETS):
            t = tune_threshold(ml_mod)
            out["ml_tuned"][ml_mod] = round(t, 4)
            systems[f"{ml_mod}_tuned"] = tuple(scores(ml_mod, s).assign(flag=lambda d: d.score >= t)
                                               for s in ["civil_test", "hatecheck"])
        for llm_mod in ["llm_gpt-oss", "llm_qwen3"]:
            if not all(cache_path(x, s).exists() for x in (ml_mod, llm_mod) for s in SETS):
                continue
            for suffix, floor in [("", 0.0), ("_wide", 0.20)]:     # _wide: forced to send 20-25% to the LLM
                t = tune(ml_mod, llm_mod, floor=floor)
                name = f"hybrid_{ml_mod}+{llm_mod}{suffix}"
                t["name"] = name
                out["hybrid"].append(t)
                systems[name] = tuple(hybrid(scores(ml_mod, s), scores(llm_mod, s), t["lo"], t["hi"])
                                      for s in ["civil_test", "hatecheck"])
                t["civil_llm_share"] = round(float(systems[name][0].to_llm.mean()), 3)
                t["hatecheck_llm_share"] = round(float(systems[name][1].to_llm.mean()), 3)
    for name, (c, h) in systems.items():
        out["civil_test"][name] = civil_metrics(c.flag, civ.label, c.ms)
        out["hatecheck"][name] = hatecheck_metrics(h.flag.values, hc)
        out["per_functionality"][name] = (pd.Series(h.flag.values.astype(bool), index=hc.index)
                                          .groupby(hc.functionality).mean().round(2).to_dict())
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(out, indent=2))
    return out


def main(argv=None):
    argv = argv or sys.argv[1:]
    if argv[:1] == ["prepare"]:
        from .data import build
        from .moderators import TfidfLR
        sets = build()
        t0 = time.time()
        TfidfLR.train(sets["train"].text, sets["train"].label)
        print(f"sets built, TF-IDF model trained in {time.time() - t0:.0f}s")
    elif argv[:1] == ["score"]:
        mods = argv[1].split(",")
        splits = argv[2].split(",") if len(argv) > 2 else SETS
        for m in mods:
            for s in splits:
                score(m, s)
    elif argv[:1] == ["tune"]:
        print(json.dumps(tune(argv[1] if len(argv) > 1 else "detoxify",
                              argv[2] if len(argv) > 2 else "llm_gpt-oss"), indent=2))
    elif argv[:1] == ["report"]:
        out = report()
        for split in ["civil_test", "hatecheck"]:
            print(f"\n== {split}")
            print(pd.DataFrame(out[split]).T.to_string())
        print("\n== hybrid thresholds (chosen on tune)")
        print(pd.DataFrame(out["hybrid"]).to_string())
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
