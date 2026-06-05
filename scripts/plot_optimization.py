#!/usr/bin/env python3
"""
Plot B3 distractor score evolution — simple view.

One figure per model: average B3 score vs cumulative distractor attempt.
Vertical dashed lines mark fiche boundaries, labelled F1 F2 F3…
Line color: green = fiche passed, red = failed.

Usage:
  python scripts/plot_optimization.py
  python scripts/plot_optimization.py --model gemma4_31b
  python scripts/plot_optimization.py --trace data/optimized_prompts/trace.jsonl
  python scripts/plot_optimization.py --out-dir results/
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR   = SCRIPT_DIR.parent

DEFAULT_TRACE   = ROOT_DIR / "data" / "optimized_prompts" / "trace.jsonl"
DEFAULT_OUT_DIR = ROOT_DIR / "data" / "optimized_prompts"


# ─────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────

def load_trace(path: Path) -> list[dict]:
    events = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def build_model_data(events: list[dict]) -> dict[str, dict]:
    """
    Per model: ordered list of fiches, each with per-attempt avg B3 scores.

    Returns:
        {model_name: {"fiches": [{"label": "F1", "scores": [float, ...], "passed": bool}]}}
    """
    # Pass 1: map (model, sheet_idx) → sheet_id from sheet_start events
    sheet_ids: dict[tuple, str] = {}
    for ev in events:
        if ev.get("event") == "sheet_start":
            sheet_ids[(ev["save_name"], ev["sheet_idx"])] = ev.get("sheet_id", str(ev["sheet_idx"]))

    # Pass 2: collect b3_eval scores per (model, sheet_idx), preserving order
    models: dict[str, OrderedDict] = {}
    for ev in events:
        name = ev.get("save_name")
        if not name or ev.get("event") != "b3_eval":
            continue
        if name not in models:
            models[name] = OrderedDict()

        sheet_idx = ev["sheet_idx"]
        if sheet_idx not in models[name]:
            models[name][sheet_idx] = {"scores": [], "passed": False}

        scores = ev.get("scores", [])
        avg = sum(scores) / len(scores) if scores else None
        if avg is not None:
            models[name][sheet_idx]["scores"].append(avg)
        if ev.get("passes"):
            models[name][sheet_idx]["passed"] = True

    # Build result with F1/F2/… labels
    result: dict[str, dict] = {}
    for name, sheets in models.items():
        fiches = []
        for i, (sheet_idx, sh) in enumerate(sheets.items(), start=1):
            fiches.append({
                "label":  f"F{i}",
                "scores": sh["scores"],
                "passed": sh["passed"],
            })
        result[name] = {"fiches": fiches}
    return result


# ─────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────

def _draw_model(ax: plt.Axes, name: str, fiches: list, label_fontsize: int = 7) -> None:
    """Draw one model's curve onto an existing Axes."""
    x = 0
    for idx, fiche in enumerate(fiches):
        scores = fiche["scores"]
        xs = list(range(x, x + len(scores)))
        color = "#27ae60" if fiche["passed"] else "#e74c3c"

        ax.plot(xs, scores, color=color, linewidth=2,
                marker="o", markersize=4, zorder=3)

        mid = (xs[0] + xs[-1]) / 2
        ax.text(mid, 5.25, fiche["label"],
                ha="center", va="bottom", fontsize=label_fontsize, color="#444444")

        x += len(scores)

        if idx < len(fiches) - 1:
            ax.axvline(x - 0.5, color="#bdc3c7", linewidth=1,
                       linestyle="--", zorder=1)

    threshold = ax.axhline(4, color="#95a5a6", linewidth=0.8,
                           linestyle=":", zorder=1, label="Seuil B3 (≥ 4)")

    n_passed = sum(1 for f in fiches if f["passed"])
    n_total  = len(fiches)
    ax.set_title(f"{name}  —  {n_passed}/{n_total} fiches passées",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Tentative (cumulatif)", fontsize=8)
    ax.set_ylabel("Score moyen B3", fontsize=8)
    ax.set_ylim(0.5, 5.6)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    pass_patch = mpatches.Patch(color="#27ae60", label="Fiche passée")
    fail_patch = mpatches.Patch(color="#e74c3c", label="Fiche non passée")
    ax.legend(handles=[pass_patch, fail_patch, threshold],
              fontsize=6, loc="upper right")


def plot_model(name: str, data: dict, out_path: Path) -> None:
    fiches = [f for f in data["fiches"] if f["scores"]]
    if not fiches:
        print(f"  {name}: aucune donnée b3_eval, ignoré.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    _draw_model(ax, name, fiches, label_fontsize=7)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def plot_all(models: dict, out_path: Path) -> None:
    """One subplot per model, all on the same figure."""
    names = [n for n, d in models.items() if any(f["scores"] for f in d["fiches"])]
    if not names:
        print("  Aucune donnée à tracer.")
        return

    ncols = 2
    nrows = (len(names) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(14, 4 * nrows),
                             squeeze=False)
    fig.suptitle("Score moyen B3 — tous les modèles",
                 fontsize=13, fontweight="bold")

    for i, name in enumerate(names):
        ax = axes[i // ncols][i % ncols]
        fiches = [f for f in models[name]["fiches"] if f["scores"]]
        _draw_model(ax, name, fiches, label_fontsize=6)

    for i in range(len(names), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot B3 score evolution from trace.jsonl"
    )
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE,
                        help="Chemin vers trace.jsonl")
    parser.add_argument("--model", type=str, default=None,
                        help="Filtrer un modèle spécifique (ex: gemma4_31b)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Répertoire de sortie (un PNG par modèle dans out-dir/model/)")
    args = parser.parse_args()

    if not args.trace.exists():
        raise FileNotFoundError(f"trace.jsonl introuvable : {args.trace}")

    events = load_trace(args.trace)
    models = build_model_data(events)

    if args.model:
        if args.model not in models:
            raise ValueError(
                f"Modèle '{args.model}' introuvable dans la trace. "
                f"Disponibles : {list(models.keys())}"
            )
        models = {args.model: models[args.model]}

    print(f"✓ {len(models)} modèle(s) à tracer")
    for name, data in models.items():
        out = args.out_dir / name / "plot_simple.png"
        plot_model(name, data, out)

    if not args.model:
        plot_all(models, args.out_dir / "plot_all.png")


if __name__ == "__main__":
    main()
