"""
Cohen's kappa: human checkbox B3 vs GPT-4o B3 PASS/FAIL
---------------------------------------------------------
Extracts from the validations DB all rows where B3 has been filled,
then computes linear Cohen's kappa between:
  - GPT-4o B3 result  : PASS=1 / FAIL=0
  - Human B3 checkbox : validated=1 / not_checked=0

Usage:
    python kappa_human_vs_gpt.py
    python kappa_human_vs_gpt.py --check B1   # run on a different check
    python kappa_human_vs_gpt.py --all-checks  # run on every check present
"""
import sys, json, argparse
sys.path.insert(0, '../src/page/backend')

import pandas as pd
from sklearn.metrics import cohen_kappa_score
from database import SessionLocal
from api.models.db_models import Validation


# ── helpers ──────────────────────────────────────────────────────────────────

def extract_check(section_b_json: str, check_id: str) -> dict | None:
    try:
        checks = json.loads(section_b_json)
        for c in checks:
            if c.get("check_id") == check_id:
                return c
    except Exception:
        pass
    return None


def result_to_int(result: str) -> int | None:
    """PASS -> 1, FAIL -> 0, anything else -> None (excluded)."""
    if result == "PASS":
        return 1
    if result == "FAIL":
        return 0
    return None


def status_to_int(status: str) -> int | None:
    """validated -> 1, not_checked -> 0, anything else -> None (excluded)."""
    if status == "validated":
        return 1
    if status == "not_checked":
        return 0
    return None


# ── main ─────────────────────────────────────────────────────────────────────

def compute_kappa_for_check(check_id: str, rows: list[Validation]) -> dict:
    records = []
    for row in rows:
        if not row.section_b_checks:
            continue
        check = extract_check(row.section_b_checks, check_id)
        if check is None:
            continue

        gpt = result_to_int(check.get("result", ""))
        human = status_to_int(check.get("status", ""))

        if gpt is None or human is None:
            continue

        records.append({
            "mcq_id": row.mcq_id,
            "user_id": row.user_id,
            "model": row.model,
            "gpt_result": gpt,
            "human_result": human,
            "decision": row.decision,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return {"check_id": check_id, "n": 0, "kappa": None, "error": "no data"}

    n = len(df)
    gpt_arr = df["gpt_result"].tolist()
    human_arr = df["human_result"].tolist()

    # Agreement rate
    agreement = sum(g == h for g, h in zip(gpt_arr, human_arr)) / n

    # Kappa (needs at least 2 classes in one of the arrays)
    try:
        kappa = cohen_kappa_score(gpt_arr, human_arr, weights="linear")
    except ValueError as e:
        kappa = None
        error = str(e)
    else:
        error = None

    # Confusion breakdown
    tp = sum(g == 1 and h == 1 for g, h in zip(gpt_arr, human_arr))
    tn = sum(g == 0 and h == 0 for g, h in zip(gpt_arr, human_arr))
    fp = sum(g == 1 and h == 0 for g, h in zip(gpt_arr, human_arr))  # GPT PASS, human not_checked
    fn = sum(g == 0 and h == 1 for g, h in zip(gpt_arr, human_arr))  # GPT FAIL, human validated

    return {
        "check_id": check_id,
        "n": n,
        "agreement": round(agreement, 3),
        "kappa": round(kappa, 3) if kappa is not None else None,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "error": error,
        "df": df,
    }


def interpret_kappa(k: float) -> str:
    if k < 0.20: return "Slight"
    if k < 0.40: return "Fair"
    if k < 0.60: return "Moderate"
    if k < 0.80: return "Substantial"
    return "Almost perfect"


def print_result(res: dict):
    print(f"\n{'='*55}")
    print(f"  Check: {res['check_id']}   N={res['n']}")
    print(f"{'='*55}")
    if res.get("error") and res["n"] == 0:
        print(f"  ⚠ {res['error']}")
        return
    print(f"  Agreement  : {res['agreement']:.1%}")
    if res["kappa"] is not None:
        print(f"  Cohen's κ  : {res['kappa']:.3f}  ({interpret_kappa(res['kappa'])})")
    else:
        print(f"  Cohen's κ  : N/A — {res.get('error', '')}")
    print(f"  TP={res['TP']}  TN={res['TN']}  FP={res['FP']}  FN={res['FN']}")
    print(f"  (FP = GPT PASS but human not_checked)")
    print(f"  (FN = GPT FAIL but human validated)")

    if res["kappa"] is not None and res["kappa"] < 0.6:
        print(f"\n  ⚠ κ < 0.6 — below substantial agreement threshold.")
        print(f"    → Review FP/FN cases to understand systematic divergence.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", default="B3", help="Check ID to evaluate (default: B3)")
    parser.add_argument("--all-checks", action="store_true", help="Run on all check IDs found in DB")
    parser.add_argument("--export", action="store_true", help="Export detailed CSV")
    args = parser.parse_args()

    db = SessionLocal()
    rows = db.query(Validation).filter(Validation.section_b_checks != None).all()
    db.close()

    print(f"Found {len(rows)} validations with section_b_checks in DB.")

    if args.all_checks:
        # Discover all check IDs
        check_ids = set()
        for row in rows:
            try:
                for c in json.loads(row.section_b_checks):
                    if "check_id" in c:
                        check_ids.add(c["check_id"])
            except Exception:
                pass
        print(f"Check IDs found: {sorted(check_ids)}\n")
        for cid in sorted(check_ids):
            res = compute_kappa_for_check(cid, rows)
            print_result(res)
    else:
        res = compute_kappa_for_check(args.check, rows)
        print_result(res)

        if args.export and res.get("df") is not None and not res["df"].empty:
            out = f"kappa_{args.check}_details.csv"
            res["df"].to_csv(out, index=False)
            print(f"\n  Exported: {out}")


if __name__ == "__main__":
    main()
