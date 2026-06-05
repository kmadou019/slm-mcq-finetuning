#!/usr/bin/env python3
"""Collect all generated MCQ CSVs from multiple git sources into one CSV."""

import io
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "data" / "all_mcqs.csv"
MINIMAL_COLS = ["id", "question", "option_a", "option_b", "option_c", "option_d", "correct_option"]


def read_local_csv(path: Path, model: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, usecols=lambda c: c in MINIMAL_COLS)
        df["model"] = model
        return df[["model"] + MINIMAL_COLS]
    except Exception as e:
        print(f"  [err] {path}: {e}", file=sys.stderr)
        return None


def read_git_csv(ref: str, git_path: str, model: str) -> pd.DataFrame | None:
    result = subprocess.run(
        ["git", "show", f"{ref}:{git_path}"],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    if result.returncode != 0:
        print(f"  [skip] {ref}:{git_path} — {result.stderr.strip()}", file=sys.stderr)
        return None
    try:
        df = pd.read_csv(io.StringIO(result.stdout), usecols=lambda c: c in MINIMAL_COLS)
        df["model"] = model
        return df[["model"] + MINIMAL_COLS]
    except Exception as e:
        print(f"  [err] {ref}:{git_path}: {e}", file=sys.stderr)
        return None


def nm(name: str) -> str:
    return name.lower().replace(".", "_").replace("-", "_").replace(" ", "_")


sources: list[tuple] = []

# ── A. Working tree (gpu branch) ─────────────────────────────────────────────
for csv_path in sorted((REPO_ROOT / "data" / "dataset_with_quality").glob("*.csv")):
    sources.append(("local", str(csv_path), nm(csv_path.stem)))

# ── B. HEAD: fichiers supprimés du working tree mais toujours committés ───────
for m in ["eurollm_9b", "gemma2_9b", "gemma4_31b", "mistral_7b"]:
    sources.append(("git", "HEAD", f"data/dataset_with_quality/{m}.csv", nm(m)))

# ── C. origin/main — modèles absents du gpu ───────────────────────────────────
MAIN_EXTRA = [
    "apertus_8B", "openbiollm_8b", "qwen3_0.6b", "qwen3_1_7b",
    "qwen3_4b", "qwen3_4b_pdapt_slerp", "qwen3_8b_pdapt_slerp",
]
for m in MAIN_EXTRA:
    sources.append(("git", "origin/main", f"data/dataset_with_quality/{m}.csv", nm(m)))

# ── D. 42c7056^ — dataset_with_quality/instruct/ ──────────────────────────────
INSTRUCT_COMMIT = "42c7056^"
for m in ["apertus_8B", "llama3_1_8b", "openbiollm_8b"]:
    sources.append(("git", INSTRUCT_COMMIT, f"data/dataset_with_quality/instruct/all__{m}.csv", nm(f"instruct_{m}")))
for m in ["eurollm_9b", "gemma2_9b", "medGemma_27b", "medGemma_4b",
          "mistral_7b", "qwen3_0.6b", "qwen3_1_7b", "qwen3_4b", "qwen3_8b"]:
    sources.append(("git", INSTRUCT_COMMIT, f"data/dataset_with_quality/instruct/{m}.csv", nm(f"instruct_{m}")))

# ── E. d8fedcb — base_models/ standardisés (gpt4, llama3.x, phi3.5, mistral) ─
BM_COMMIT = "d8fedcb"
BM_FILES = [
    ("data/base_models/gpt4/temp0.1.csv",           "gpt4_temp0_1"),
    ("data/base_models/gpt4/temp0.5.csv",           "gpt4_temp0_5"),
    ("data/base_models/gpt4/temp0.7.csv",           "gpt4_temp0_7"),
    ("data/base_models/llama3.1_8b/temp0.1.csv",    "llama3_1_8b_temp0_1"),
    ("data/base_models/llama3.2_1b/temp0.1.csv",    "llama3_2_1b_temp0_1"),
    ("data/base_models/llama3.2_1b/temp0.5.csv",    "llama3_2_1b_temp0_5"),
    ("data/base_models/llama3.2_1b/temp0.7.csv",    "llama3_2_1b_temp0_7"),
    ("data/base_models/llama3.2_3b/temp0.1.csv",    "llama3_2_3b_temp0_1"),
    ("data/base_models/llama3.3_3b/temp0.1.csv",    "llama3_3_3b_temp0_1"),
    ("data/base_models/llama3.3_3b/temp0.5.csv",    "llama3_3_3b_temp0_5"),
    ("data/base_models/llama3.3_3b/temp0.7.csv",    "llama3_3_3b_temp0_7"),
    ("data/base_models/llama3.3_70b/temp0.1.csv",   "llama3_3_70b_temp0_1"),
    ("data/base_models/mistral3_24b/temp0.1.csv",   "mistral3_24b_temp0_1"),
    ("data/base_models/phi3.5/temp0.1.HF.csv",      "phi3_5_temp0_1_hf"),
    ("data/base_models/phi3.5/temp0.5.ollama.csv",  "phi3_5_temp0_5_ollama"),
    ("data/base_models/phi3.5/temp0.7.HF.csv",      "phi3_5_temp0_7_hf"),
]
for git_path, model in BM_FILES:
    sources.append(("git", BM_COMMIT, git_path, model))

# ── F. cbf2ba4 — Slava + ancien llama1b (3 températures) ─────────────────────
SLAVA_COMMIT = "cbf2ba4"
SLAVA_FILES = [
    ("data/base_models/llama1b/temp0.1.csv",                  "llama1b_temp0_1"),
    ("data/base_models/llama1b/temp0.5.csv",                  "llama1b_temp0_5"),
    ("data/base_models/llama1b/temp0.7.csv",                  "llama1b_temp0_7"),
    ("data/base_models/llama3b_Slava/llama_0.1_Ollama.csv",   "llama3b_slava_temp0_1"),
    ("data/base_models/llama3b_Slava/llama_0.5_Ollama.csv",   "llama3b_slava_temp0_5"),
    ("data/base_models/llama3b_Slava/llama_0.7_Ollama.csv",   "llama3b_slava_temp0_7"),
    ("data/base_models/phi3.5_Slava/phi3_0.1_temp_HF.csv",    "phi3_5_slava_temp0_1"),
    ("data/base_models/phi3.5_Slava/phi3_0.7_temp_HF.csv",    "phi3_5_slava_temp0_7"),
]
for git_path, model in SLAVA_FILES:
    sources.append(("git", SLAVA_COMMIT, git_path, model))

# ── G. 381987f — base_models plats (llama1b, llama8b, gemma9b, phi3.5) ───────
FLAT_COMMIT = "381987f"
FLAT_FILES = [
    ("data/base_models/llama1b.csv",   "llama1b_flat"),
    ("data/base_models/llama8b.csv",   "llama8b_flat"),
    ("data/base_models/gemma_9b.csv",  "gemma_9b_flat"),
    ("data/base_models/phi3.5.csv",    "phi3_5_flat"),
]
for git_path, model in FLAT_FILES:
    sources.append(("git", FLAT_COMMIT, git_path, model))


def load_source(src: tuple) -> pd.DataFrame | None:
    kind = src[0]
    if kind == "local":
        _, path, model = src
        print(f"  local  {model}: {path}")
        return read_local_csv(Path(path), model)
    else:
        _, ref, git_path, model = src
        print(f"  git    {model}: {ref}:{git_path}")
        return read_git_csv(ref, git_path, model)


print(f"Loading {len(sources)} sources…")
frames = []
for src in sources:
    df = load_source(src)
    if df is not None and not df.empty:
        frames.append(df)
        print(f"    → {len(df):,} rows")

print("\nConcatenating…")
combined = pd.concat(frames, ignore_index=True)
print(f"  Total before dedup: {len(combined):,} rows")

combined.dropna(subset=["question", "option_a", "option_b", "option_c", "option_d"], inplace=True)
combined.drop_duplicates(subset=["model", "question"], inplace=True)
print(f"  Total after dedup:  {len(combined):,} rows")

print(f"\nSaving → {OUTPUT_PATH}")
combined.to_csv(OUTPUT_PATH, index=False)
print("Done.")
