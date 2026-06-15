"""IR-Metriken Precision@K, MRR@K, NDCG@K für die Goldstandard-Evaluation.

"""
from math import log2
from typing import Iterable, List, Set


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
    """Anteil relevanter Treffer in den Top-K."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for doc_id in top_k if doc_id in relevant_ids) / k


def mrr_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
    """Mean Reciprocal Rank: bewertet den Rang des ersten relevanten Treffers."""
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
    """Discounted Cumulative Gain: gewichtet relevante Treffer nach Position."""
    score = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        relevance = 1 if doc_id in relevant_ids else 0
        score += relevance / log2(rank + 1)
    return score


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
    """Normalized DCG: DCG geteilt durch idealen DCG (perfektes Ranking)."""
    ideal_n = min(len(relevant_ids), k)
    if ideal_n == 0:
        return 0.0
    ideal_dcg = sum(1 / log2(rank + 1) for rank in range(1, ideal_n + 1))
    return dcg_at_k(retrieved_ids, relevant_ids, k) / ideal_dcg


def evaluate_query(retrieved_ids: List[str], relevant_ids: Iterable[str], k: int = 5) -> dict:
    """Berechnet alle drei Metriken für eine einzelne Frage.

    String-Konversion automatisch, damit int-Artikelnummern und str-IDs gleich behandelt werden.
    """
    relevant_set = set(str(r) for r in relevant_ids)
    retrieved_strs = [str(r) for r in retrieved_ids]
    return {
        f"precision@{k}": precision_at_k(retrieved_strs, relevant_set, k),
        f"mrr@{k}": mrr_at_k(retrieved_strs, relevant_set, k),
        f"ndcg@{k}": ndcg_at_k(retrieved_strs, relevant_set, k),
    }
