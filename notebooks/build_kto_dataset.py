#!/usr/bin/env python3
"""
Build KTO dataset (HuggingFace-ready) from src/comparison_results/csv_eval_new/

Output: data/kto_dataset/train.jsonl, data/kto_dataset/eval.jsonl
Format: {"prompt": "...", "completion": "...", "label": true/false, "source": "...", "id": "..."}

Requires: GRAPHDB_MCP_URL and GRAPHDB_BEARER_TOKEN in .env (for prompt retrieval)
"""

import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from generate_mcq import _build_prompt

CSV_DIR = Path(__file__).parent.parent / "src/comparison_results/csv_eval_new"
OUT_DIR = Path(__file__).parent.parent / "data/kto_dataset"


# ---------------------------------------------------------------------------
# 1. Load and merge
# ---------------------------------------------------------------------------

files = glob.glob(str(CSV_DIR / "*.csv"))
dfs = []
for f in files:
    df = pd.read_csv(f)
    df["source"] = Path(f).stem
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
print(f"Lignes chargées : {len(df_all)}")
print(f"Modèles : {sorted(df_all['source'].unique().tolist())}")


# ---------------------------------------------------------------------------
# 2. Filter
# ---------------------------------------------------------------------------

before = len(df_all)
df_all = df_all[df_all["distractor_quality"].notna()]
df_all = df_all[df_all["correct_option"].astype(str).str.lower().isin(list("abcd"))]
df_all = df_all[df_all["question"].notna() & (df_all["question"].str.strip() != "")]
df_all = df_all[df_all["content_raw"].notna()]
df_all = df_all.reset_index(drop=True)

print(f"\nLignes supprimées : {before - len(df_all)}")
print(f"Lignes restantes  : {len(df_all)}")
print(f"\nDistribution distractor_quality :\n{df_all['distractor_quality'].value_counts()}")


# ---------------------------------------------------------------------------
# 3. Build prompts (GraphDB retrieval)
# ---------------------------------------------------------------------------

print("\nConstruction des prompts (retrieval GraphDB)...")
df_all["prompt"] = df_all["content_raw"].apply(_build_prompt)

enriched = df_all["prompt"].str.contains("OBJECTIFS DE CONNAISSANCE").sum()
print(f"Enrichis : {enriched} / {len(df_all)} ({100*enriched/len(df_all):.1f}%)")
print(f"Fallback : {len(df_all) - enriched}")


# ---------------------------------------------------------------------------
# 4. Build completions (JSON)
# ---------------------------------------------------------------------------

def build_completion(row):
    return json.dumps({
        "question":         str(row["question"]),
        "question_comment": str(row.get("question_comment") or ""),
        "option_a":         str(row["option_a"]),
        "option_a_comment": str(row.get("option_a_comment") or ""),
        "option_b":         str(row["option_b"]),
        "option_b_comment": str(row.get("option_b_comment") or ""),
        "option_c":         str(row["option_c"]),
        "option_c_comment": str(row.get("option_c_comment") or ""),
        "option_d":         str(row["option_d"]),
        "option_d_comment": str(row.get("option_d_comment") or ""),
        "correct_option":   str(row["correct_option"]).lower(),
    }, ensure_ascii=False)

df_all["completion"] = df_all.apply(build_completion, axis=1)


# ---------------------------------------------------------------------------
# 5. Label
# ---------------------------------------------------------------------------

df_all["label"] = df_all["distractor_quality"].astype(bool)
print(f"\nPositifs (True)  : {df_all['label'].sum()}")
print(f"Négatifs (False) : {(~df_all['label']).sum()}")


# ---------------------------------------------------------------------------
# 6. Deduplicate
# ---------------------------------------------------------------------------

before = len(df_all)
df_all = df_all.drop_duplicates(subset=["completion"]).reset_index(drop=True)
print(f"\nDoublons supprimés : {before - len(df_all)}")
print(f"Dataset finale     : {len(df_all)} lignes")


# ---------------------------------------------------------------------------
# 7. Train / eval split (80/20 stratified)
# ---------------------------------------------------------------------------

train, eval_ = train_test_split(
    df_all,
    test_size=0.2,
    stratify=df_all["label"],
    random_state=42,
)

print(f"\nTrain : {len(train)}  (positifs: {train['label'].sum()}, négatifs: {(~train['label']).sum()})")
print(f"Eval  : {len(eval_)}  (positifs: {eval_['label'].sum()}, négatifs: {(~eval_['label']).sum()})")


# ---------------------------------------------------------------------------
# 8. Save JSONL
# ---------------------------------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)
cols = ["prompt", "completion", "label", "source", "id"]

train[cols].to_json(OUT_DIR / "train.jsonl", orient="records", lines=True, force_ascii=False)
eval_[cols].to_json(OUT_DIR / "eval.jsonl",  orient="records", lines=True, force_ascii=False)

print(f"\nSauvegardé dans {OUT_DIR}/")
print(f"  train.jsonl : {len(train)} lignes")
print(f"  eval.jsonl  : {len(eval_)} lignes")


# ---------------------------------------------------------------------------
# 9. HuggingFace verification
# ---------------------------------------------------------------------------

from datasets import load_dataset

dataset = load_dataset("json", data_files={
    "train": str(OUT_DIR / "train.jsonl"),
    "eval":  str(OUT_DIR / "eval.jsonl"),
})

print(f"\n{dataset}")
print(f"\nFeatures : {dataset['train'].features}")

pos = next(x for x in dataset["train"] if x["label"])
neg = next(x for x in dataset["train"] if not x["label"])
print(f"\nExemple positif — prompt[:80]: {pos['prompt'][:80]}")
print(f"Exemple négatif — prompt[:80]: {neg['prompt'][:80]}")
