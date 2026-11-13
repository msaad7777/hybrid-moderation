"""Build the train, tuning and test sets from the raw downloads. Deterministic (fixed seeds).

Civil Comments (CC0): a comment counts as toxic when toxicity >= 0.5 (the dataset's own convention).
HateCheck (CC BY 4.0): a functional test suite, used only for testing, never for training or tuning.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SETS = ROOT / "data" / "sets"

# HateCheck functionalities grouped by what they test. Labels are about hate toward a protected group.
IDENTITY_NH = ["ident_neutral_nh", "ident_pos_nh"]          # "I'm a proud Muslim"
COUNTER_NH = ["counter_quote_nh", "counter_ref_nh"]         # quoting or denouncing hate
OTHER_NH = ["negate_neg_nh", "slur_reclaimed_nh", "slur_homonym_nh", "target_obj_nh"]
# Not hate but still abusive or profane: a toxicity policy may rightly flag them, so they are left out of
# the false-positive numbers and reported separately.
POLICY_GREY = ["profanity_nh", "target_indiv_nh", "target_group_nh"]


def civil(split: str) -> pd.DataFrame:
    files = sorted(RAW.glob(f"civil_{split}-*.parquet"))
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d["label"] = (d["toxicity"] >= 0.5).astype(int)
    d["text"] = d["text"].str.strip()
    return d[d["text"].str.len().between(5, 1500)].reset_index(drop=True)


def build() -> dict[str, pd.DataFrame]:
    SETS.mkdir(parents=True, exist_ok=True)
    train = civil("train")
    train = pd.concat([train[train.label == 1].sample(30_000, random_state=1),
                       train[train.label == 0].sample(120_000, random_state=1)]).sample(frac=1, random_state=1)
    val = civil("validation")
    tune = pd.concat([val[val.label == 1].sample(300, random_state=2),
                      val[val.label == 0].sample(700, random_state=2)]).sample(frac=1, random_state=2)
    test = civil("test")
    civil_test = pd.concat([test[test.label == 1].sample(300, random_state=3),
                            test[test.label == 0].sample(700, random_state=3)]).sample(frac=1, random_state=3)
    hc = pd.read_csv(RAW / "hatecheck.csv")
    hc["label"] = (hc["label_gold"] == "hateful").astype(int)
    hc = hc.rename(columns={"test_case": "text"})
    # up to 25 cases per functionality, so every behaviour is tested and the LLM runs stay affordable
    hc = pd.concat([g.sample(min(len(g), 25), random_state=4) for _, g in hc.groupby("functionality")])
    out = {"train": train[["text", "label"]].reset_index(drop=True),
           "tune": tune[["text", "label"]].reset_index(drop=True),
           "civil_test": civil_test[["text", "label"]].reset_index(drop=True),
           "hatecheck": hc[["case_id", "functionality", "target_ident", "text", "label"]].reset_index(drop=True)}
    for name, df in out.items():
        df.to_parquet(SETS / f"{name}.parquet")
    return out


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(SETS / f"{name}.parquet")


if __name__ == "__main__":
    for k, v in build().items():
        print(k, len(v), "positive rate", round(v.label.mean(), 3))
