#!/usr/bin/env python3
"""
Plot B3 distractor score evolution over all attempts, per model.

X : cumulative attempt index (across all LISA sheets, in order)
Y : average distractor score (mean of 3 scores, scale 1–5)

Each point = one generation. Green = B3 passed, red = failed.
Dashed vertical lines mark fiche boundaries.
Orange annotations mark large prompt cuts by Claude.

Usage:
  python scripts/plot_optimization.py
  python scripts/plot_optimization.py --trace data/optimized_prompts/trace.jsonl
  python scripts/plot_optimization.py --out results/b3_plot.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent

DEFAULT_TRACE = ROOT_DIR / "data" / "optimized_prompts" / "trace.jsonl"
DEFAULT_OUT   = ROOT_DIR / "data" / "optimized_prompts" / "plot_b3.png"


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
    For each model returns:
      attempts        — list of {x, sheet_idx, avg_score, min_score, passes}
      improve_events  — list of {x, chars_diff}  (after corresponding b3_eval)
      sheet_boundaries — list of x where a new sheet starts (for vertical lines)
    """
    models: dict[str, dict] = {}

    for ev in events:
        name = ev.get("save_name")
        if not name:
            continue
        if name not in models:
            models[name] = {
                "attempts": [],
                "improve_events": [],
                "sheet_boundaries": [],
                "_x": 0,
                "_last_sheet": None,
            }
        m = models[name]

        if ev["event"] == "b3_eval":
            sheet_idx = ev["sheet_idx"]
            if sheet_idx != m["_last_sheet"]:
                if m["_last_sheet"] is not None:
                    m["sheet_boundaries"].append(m["_x"])
                m["_last_sheet"] = sheet_idx

            scores = ev.get("scores", [])
            avg = sum(scores) / len(scores) if scores else None
            mn  = min(scores) if scores else None
            m["attempts"].append({
                "x":         m["_x"],
                "sheet_idx": sheet_idx,
                "avg_score": avg,
                "min_score": mn,
                "passes":    ev["passes"],
            })
            m["_x"] += 1

        elif ev["event"] == "claude_improve":
            # attach to the attempt that just failed (x - 1)
            m["improve_events"].append({
                "x":          m["_x"] - 1,
                "chars_diff": ev["chars_diff"],
            })

    # clean up internal counters
    for m in models.values():
        del m["_x"], m["_last_sheet"]

    return models


# ─────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────

def plot(models: dict[str, dict], out_path: Path) -> None:
    n = len(models)
    if n == 0:
        print("Aucun modèle trouvé dans la trace.")
        return

    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(14, 4 * nrows),
        squeeze=False,
    )
    fig.suptitle(
        "Évolution du score moyen des distracteurs — toutes tentatives LISA",
        fontsize=13, fontweight="bold",
    )

    for i, (name, data) in enumerate(models.items()):
        ax = axes[i // ncols][i % ncols]
        attempts = data["attempts"]

        if not attempts:
            ax.set_visible(False)
            continue

        xs  = [a["x"]         for a in attempts]
        ys  = [a["avg_score"] for a in attempts]
        pas = [a["passes"]    for a in attempts]

        # ── scatter: one dot per generation ──────────────────────
        colors = ["#27ae60" if p else "#e74c3c" for p in pas]
        ax.scatter(xs, ys, c=colors, s=45, zorder=3, linewidths=0)

        # ── rolling mean (window = 5) ─────────────────────────────
        window = min(5, len(ys))
        if len(ys) >= window:
            kernel  = np.ones(window) / window
            rolling = np.convolve(ys, kernel, mode="valid")
            rx = xs[window - 1:]
            ax.plot(rx, rolling, color="#2980b9", linewidth=1.8,
                    zorder=2, label=f"moy. glissante ({window})")

        # ── fiche boundaries ──────────────────────────────────────
        for bx in data["sheet_boundaries"]:
            ax.axvline(bx - 0.5, color="#bdc3c7", linewidth=0.8,
                       linestyle="--", zorder=1)

        # ── B3 threshold hint ─────────────────────────────────────
        ax.axhline(4, color="#95a5a6", linewidth=0.7, linestyle=":",
                   zorder=1, label="seuil score ≥ 4")

        # ── annotate large Claude prompt cuts ────────────────────
        for imp in data["improve_events"]:
            diff = imp["chars_diff"]
            if diff <= -5000:
                ax.annotate(
                    f"{diff:+,}",
                    xy=(imp["x"] + 0.5, 0.4),
                    fontsize=6, color="#e67e22", ha="center",
                    rotation=90,
                )

        ax.set_ylim(0.5, 5.5)
        ax.set_xlim(-0.5, max(xs) + 0.5)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Tentative (index cumulatif)", fontsize=9)
        ax.set_ylabel("Score moyen distracteurs (1–5)", fontsize=9)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    # hide unused subplots
    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    # shared legend
    pass_patch = mpatches.Patch(color="#27ae60", label="B3 passé")
    fail_patch = mpatches.Patch(color="#e74c3c", label="B3 échoué")
    fig.legend(
        handles=[pass_patch, fail_patch],
        loc="lower center", ncol=2, fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✓ Graphique sauvegardé : {out_path}")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot B3 score evolution from trace.jsonl"
    )
    parser.add_argument(
        "--trace", type=Path, default=DEFAULT_TRACE,
        help="Chemin vers trace.jsonl",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help="Chemin de sortie du graphique (.png)",
    )
    args = parser.parse_args()

    if not args.trace.exists():
        raise FileNotFoundError(f"trace.jsonl introuvable : {args.trace}")

    events = load_trace(args.trace)
    models = build_model_data(events)
    print(f"✓ {len(models)} modèle(s) : {list(models.keys())}")
    for name, data in models.items():
        n_att = len(data["attempts"])
        n_ok  = sum(1 for a in data["attempts"] if a["passes"])
        print(f"  {name:20s}  {n_ok}/{n_att} tentatives passées")

    plot(models, args.out)


if __name__ == "__main__":
    main()
