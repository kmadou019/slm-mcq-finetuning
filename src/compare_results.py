#!/usr/bin/env python3
"""
Affiche un tableau comparatif entre deux fichiers distribution.output.

Usage:
    python3 compare_results.py distribution_old.output distribution_new.output
"""

import sys
import re
from pathlib import Path


def parse_distribution_file(path: str) -> dict:
    """
    Parse un fichier distribution.output et retourne un dict:
        { model_name: {"true_pct": float, "true_n": int, "false_pct": float, "false_n": int, "total": int} }

    Format attendu (nouveau format avec totaux) :
        llama3_1_8b.csv quality distribution:
          Mauvaise qualité (False): 91.65% (n=549/599)
          Bonne qualité (True): 8.35% (n=50/599)

    Format ancien (fallback si pas encore migré) :
        distractor_quality
        False    91.650672
        True      8.349328
    """
    results = {}
    current_model = None

    model_re = re.compile(r"^(\S+?)\.csv quality distribution:")
    new_line_re = re.compile(
        r"^\s+(Bonne qualité|Mauvaise qualité) \((True|False)\): ([\d.]+)% \(n=(\d+)/(\d+)\)"
    )
    old_bool_re = re.compile(r"^\s*(True|False)\s+([\d.]+)")

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip()

        m = model_re.match(line)
        if m:
            current_model = m.group(1)
            results[current_model] = {}
            continue

        if current_model is None:
            continue

        # Nouveau format
        m = new_line_re.match(line)
        if m:
            is_true = m.group(2) == "True"
            pct = float(m.group(3))
            count = int(m.group(4))
            total = int(m.group(5))
            key = "true" if is_true else "false"
            results[current_model][f"{key}_pct"] = pct
            results[current_model][f"{key}_n"] = count
            results[current_model]["total"] = total
            continue

        # Ancien format (compat)
        m = old_bool_re.match(line)
        if m:
            is_true = m.group(1) == "True"
            pct = float(m.group(2))
            key = "true" if is_true else "false"
            results[current_model][f"{key}_pct"] = pct
            continue

    # Nettoyer les modèles sans données
    return {k: v for k, v in results.items() if v}


def fmt_cell(data: dict) -> str:
    pct = data.get("true_pct", 0.0)
    n = data.get("true_n")
    total = data.get("total")
    if n is not None and total is not None:
        return f"{pct:6.2f}% (n={n}/{total})"
    return f"{pct:6.2f}%"


def delta_arrow(old_pct: float, new_pct: float) -> str:
    diff = new_pct - old_pct
    if diff > 0:
        return f"↑ +{diff:.2f}pp"
    elif diff < 0:
        return f"↓ {diff:.2f}pp"
    return "  ±0.00pp"


def main():
    if len(sys.argv) != 3:
        print("Usage: compare_results.py <old_output> <new_output>")
        sys.exit(1)

    old_path, new_path = sys.argv[1], sys.argv[2]

    old_data = parse_distribution_file(old_path)
    new_data = parse_distribution_file(new_path)

    all_models = sorted(set(old_data) | set(new_data))

    if not all_models:
        print("Aucun résultat trouvé dans les fichiers.")
        sys.exit(0)

    # Largeurs de colonnes
    col_model = max(20, max(len(m) for m in all_models) + 2)
    col_sys = 28
    col_delta = 16

    sep = (
        "+"
        + "-" * (col_model + 2)
        + "+"
        + "-" * (col_sys + 2)
        + "+"
        + "-" * (col_sys + 2)
        + "+"
        + "-" * (col_delta + 2)
        + "+"
    )

    header = (
        f"| {'Modèle':<{col_model}} "
        f"| {'Ancien (fallback)':<{col_sys}} "
        f"| {'Nouveau (retrieval)':<{col_sys}} "
        f"| {'Δ Bonne qualité':<{col_delta}} |"
    )

    print(sep)
    print(header)
    print(sep)

    for model in all_models:
        old = old_data.get(model, {})
        new = new_data.get(model, {})

        old_cell = fmt_cell(old) if old else "—"
        new_cell = fmt_cell(new) if new else "—"

        old_pct = old.get("true_pct", 0.0)
        new_pct = new.get("true_pct", 0.0)
        delta = delta_arrow(old_pct, new_pct) if (old and new) else "  —"

        print(
            f"| {model:<{col_model}} "
            f"| {old_cell:<{col_sys}} "
            f"| {new_cell:<{col_sys}} "
            f"| {delta:<{col_delta}} |"
        )

    print(sep)
    print()
    print("Légende : Bonne qualité = True (≥2 distracteurs avec score ≥4/5)")
    print(f"Ancien système : {Path(old_path).name}")
    print(f"Nouveau système : {Path(new_path).name}")


if __name__ == "__main__":
    main()
