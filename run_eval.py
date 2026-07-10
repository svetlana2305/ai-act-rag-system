"""Orchestrierung der IR-Evaluation auf dem Goldstandard.

nutzbar:
- als Modul: `run_full_eval()` liefert eine Liste von Dicts zurück (für Streamlit)
- als Skript: `python run_eval.py` schreibt CSV (für lokales Debuggen)
"""
import json
import csv
from pathlib import Path
from database_manager import MODELS, STRATEGIES, get_client, get_status, retrieve_chunks
from eval_ir import evaluate_query

K = 5
HYBRID_ALPHA = 0.5
METHODS = ["bm25", "semantic", "hybrid"]
GOLD_PATH = Path("eval/goldstandard.json")


def load_goldstandard(path: Path = GOLD_PATH) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["fragen"]


def aggregate(metrics_list: list[dict]) -> dict:
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in keys}


def run_full_eval(client=None, progress_callback=None) -> list[dict]:
    """Führt die volle Eval-Matrix aus und liefert eine Liste von Result-Dicts.

    progress_callback: optionale Funktion(fraction: float, message: str)
    """
    own_client = client is None
    if own_client:
        client = get_client()

    status = get_status(client)
    fragen = load_goldstandard()
    rows = []

    total = sum(
        1 for m in MODELS for s in STRATEGIES for _ in METHODS
        if status.get((m, s), {}).get("documents", 0) > 0
    )
    done = 0

    for model_key in MODELS:
        for strategy_key in STRATEGIES:
            if status.get((model_key, strategy_key), {}).get("documents", 0) == 0:
                continue
            for method in METHODS:
                per_query = []
                for q in fragen:
                    hits = retrieve_chunks(
                        client, model_key, strategy_key,
                        q["frage"], top_k=K, method=method, hybrid_alpha=HYBRID_ALPHA
                    )
                    retrieved_ids = [f"{h.get('source_type', 'artikel')}_{h.get('article_number')}" for h in hits]
                    per_query.append(
                        evaluate_query(retrieved_ids, q["relevante_artikel"], k=K)
                    )

                agg = aggregate(per_query)
                rows.append({
                    "model": model_key,
                    "strategy": strategy_key,
                    "method": method,
                    **agg,
                })

                done += 1
                if progress_callback:
                    progress_callback(
                        done / total,
                        f"{model_key}/{strategy_key}/{method} fertig"
                    )

    if own_client:
        client.close()
    return rows


def main():
    rows = run_full_eval(progress_callback=lambda f, m: print(f"  {m}"))
    if not rows:
        print("Keine Ergebnisse: Datenbank leer?")
        return
    out = Path("results_baseline.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "strategy", "method", "precision@5", "mrr@5", "ndcg@5"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Fertig: {len(rows)} Konfigurationen in {out}")


if __name__ == "__main__":
    main()
