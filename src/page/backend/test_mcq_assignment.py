#!/usr/bin/env python3
"""
Script de test pour vérifier le système d'assignation séquentielle des MCQs
"""
import sys
import json
from pathlib import Path
import pandas as pd

# Configuration des chemins (identique à mcq.py)
BACKEND_DATA_DIR = Path(__file__).parent / "data"  # Données app web (backend/)
CSV_DIR = BACKEND_DATA_DIR / "mcqs"
GLOBAL_TRACKER_PATH = BACKEND_DATA_DIR / "global_assignment_tracker.json"

def load_global_tracker():
    """Charger le tracker global"""
    if not GLOBAL_TRACKER_PATH.exists():
        return {}
    try:
        with open(GLOBAL_TRACKER_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading global tracker: {e}")
        return {}

def save_global_tracker(tracker):
    """Sauvegarder le tracker global"""
    GLOBAL_TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_TRACKER_PATH, 'w') as f:
        json.dump(tracker, f, indent=2)

def test_csv_loading():
    """Test 1: Chargement du CSV"""
    print("=" * 60)
    print("TEST 1: Chargement du CSV")
    print("=" * 60)

    csv_path = CSV_DIR / "qwen3_8b_pdapt_slerp.csv"
    print(f"CSV Path: {csv_path}")
    print(f"Exists: {csv_path.exists()}")

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"✅ CSV chargé avec succès")
        print(f"   Total MCQs: {len(df)}")
        print(f"   Colonnes: {list(df.columns[:5])}...")
        return df
    else:
        print(f"❌ CSV non trouvé")
        return None

def test_tracker_loading():
    """Test 2: Chargement du tracker global"""
    print("\n" + "=" * 60)
    print("TEST 2: Tracker global")
    print("=" * 60)

    tracker = load_global_tracker()
    print(f"Tracker path: {GLOBAL_TRACKER_PATH}")
    print(f"Exists: {GLOBAL_TRACKER_PATH.exists()}")

    if tracker:
        print(f"✅ Tracker chargé:")
        print(json.dumps(tracker, indent=2))
    else:
        print(f"ℹ️  Tracker vide (normal pour la première exécution)")

    return tracker

def test_assignment_simulation(model="qwen3_8b_pdapt_slerp", count=20):
    """Test 3: Simulation d'assignation"""
    print("\n" + "=" * 60)
    print(f"TEST 3: Simulation d'assignation ({count} MCQs)")
    print("=" * 60)

    # Charger le CSV
    csv_path = CSV_DIR / f"{model}.csv"
    df = pd.read_csv(csv_path)
    total_mcqs = len(df)

    # Charger le tracker
    tracker = load_global_tracker()

    # Initialiser si nécessaire
    if model not in tracker:
        tracker[model] = {
            "last_assigned_index": -1,
            "total_available": total_mcqs,
            "assigned_count": 0
        }
        print(f"ℹ️  Initialisation du tracker pour '{model}'")

    # Calculer les indices
    last_index = tracker[model]["last_assigned_index"]
    start_index = last_index + 1

    # Vérifier qu'il reste assez de MCQs
    remaining = total_mcqs - start_index

    if remaining <= 0:
        print(f"❌ Plus de MCQs disponibles!")
        print(f"   Total: {total_mcqs}, Déjà assignés: {start_index}")
        return None

    actual_count = min(count, remaining)
    indices = list(range(start_index, start_index + actual_count))
    mcq_ids = [f"MCQ-{i+1:06d}" for i in indices]

    print(f"✅ Assignation simulée:")
    print(f"   Modèle: {model}")
    print(f"   Total MCQs dans CSV: {total_mcqs}")
    print(f"   Dernier index assigné: {last_index}")
    print(f"   Prochain index: {start_index}")
    print(f"   Nombre demandé: {count}")
    print(f"   Nombre assigné: {actual_count}")
    print(f"   Premier MCQ: {mcq_ids[0]} (index {indices[0]})")
    print(f"   Dernier MCQ: {mcq_ids[-1]} (index {indices[-1]})")
    print(f"   MCQs restants après: {total_mcqs - (start_index + actual_count)}")

    return {
        "model": model,
        "mcq_ids": mcq_ids,
        "indices": indices,
        "new_last_index": start_index + actual_count - 1
    }

def test_multiple_assignments():
    """Test 4: Assignations multiples séquentielles"""
    print("\n" + "=" * 60)
    print("TEST 4: Assignations multiples")
    print("=" * 60)

    model = "qwen3_8b_pdapt_slerp"

    # Réinitialiser le tracker pour le test
    tracker = {
        model: {
            "last_assigned_index": -1,
            "total_available": 360,
            "assigned_count": 0
        }
    }

    print("Simulation de 3 assignations consécutives:")

    for i, count in enumerate([10, 20, 15], 1):
        last_index = tracker[model]["last_assigned_index"]
        start_index = last_index + 1
        end_index = start_index + count - 1

        mcq_ids = [f"MCQ-{idx+1:06d}" for idx in range(start_index, start_index + count)]

        print(f"\n  Assignation {i} ({count} MCQs):")
        print(f"    Indices: {start_index} -> {end_index}")
        print(f"    MCQ IDs: {mcq_ids[0]} -> {mcq_ids[-1]}")

        # Mettre à jour le tracker
        tracker[model]["last_assigned_index"] = end_index
        tracker[model]["assigned_count"] += count

    print(f"\n✅ État final du tracker:")
    print(f"   Total assigné: {tracker[model]['assigned_count']} MCQs")
    print(f"   Dernier index: {tracker[model]['last_assigned_index']}")
    print(f"   Restant: {360 - tracker[model]['assigned_count']} MCQs")

if __name__ == "__main__":
    print("🧪 TEST DU SYSTÈME D'ASSIGNATION SÉQUENTIELLE DES MCQs")
    print()

    # Test 1: Chargement du CSV
    df = test_csv_loading()

    # Test 2: Tracker global
    tracker = test_tracker_loading()

    # Test 3: Simulation d'assignation
    if df is not None:
        result = test_assignment_simulation(count=20)

    # Test 4: Assignations multiples
    test_multiple_assignments()

    print("\n" + "=" * 60)
    print("✅ TOUS LES TESTS SONT TERMINÉS")
    print("=" * 60)
