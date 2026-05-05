"""
evaluate.py
===========
End-to-end evaluation pipeline for the HMLP link-prediction project.

Answers the professor's questions concretely:
  - How many relation types?   -> relation_type_summary()
  - Can you predict the type?  -> run_relation_classifier()
  - What type of edge?         -> Anime->Anime typed triples (SEQUEL, PREQUEL, etc.)
  - What method?               -> Jaccard, Adamic-Adar, Preferential Attachment
                                  + LogReg relation-type classifier

Usage
-----
    python evaluate.py                  # uses default data/anime_data.json
    python evaluate.py --min-weight 3   # stricter projection threshold
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from typing import Optional

# Project imports
from graph_builder import load_graph, EVAL_RELATION_TYPES, RELATION_GROUPS
from link_prediction import (
    build_all_attribute_sets,
    build_attr_frequency,
    jaccard,
    adamic_adar,
    preferential_attachment,
    RelationClassifier,
)


# Train / test split
def train_test_split_triples(
    relation_triples: list[dict],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Stratified split of typed relation triples into train/test sets.

    Stratified by relation type so every type appears in both splits
    even when counts are low.

    Returns
    -------
    train_triples, test_triples
    """
    by_type: dict[str, list[dict]] = defaultdict(list)
    for t in relation_triples:
        by_type[t["relation"]].append(t)

    rng = random.Random(seed)
    train, test = [], []

    for rel_type, triples in by_type.items():
        rng.shuffle(triples)
        n_test = max(1, int(len(triples) * test_ratio))
        # If only 1 sample, put it in train to avoid empty test class
        if len(triples) == 1:
            train.extend(triples)
        else:
            test.extend(triples[:n_test])
            train.extend(triples[n_test:])

    print(f"[evaluate] Train triples: {len(train)} | Test triples: {len(test)}")
    print(f"[evaluate] Split type: random stratified (no temporal ordering by seasonYear)")
    return train, test


# Negative sampling
def sample_negatives(
    anime_ids: list[int],
    positive_pairs: set[frozenset],
    n_negatives: int,
    seed: int = 42,
) -> list[frozenset]:
    """
    Sample random anime pairs that have NO known relation (negatives).

    Negatives are used to compute precision, recall, and AUC-ROC
    rather than just precision@K on positives.
    """
    rng = random.Random(seed)
    negatives: list[frozenset] = []
    attempts = 0
    max_attempts = n_negatives * 20

    while len(negatives) < n_negatives and attempts < max_attempts:
        a = rng.choice(anime_ids)
        b = rng.choice(anime_ids)
        attempts += 1
        if a == b:
            continue
        pair = frozenset([a, b])
        if pair in positive_pairs or pair in {frozenset(n) for n in negatives}:
            continue
        negatives.append(pair)

    return negatives


# Scoring helpers
def score_pair(
    a_id: int,
    b_id: int,
    attr_sets: dict[int, frozenset],
    attr_freq: dict[tuple, int],
) -> dict[str, float]:
    a_attrs = attr_sets[a_id]
    b_attrs = attr_sets[b_id]
    return {
        "jaccard":                 jaccard(a_attrs, b_attrs),
        "adamic_adar":             adamic_adar(a_attrs, b_attrs, attr_freq),
        "preferential_attachment": preferential_attachment(a_attrs, b_attrs),
    }


def score_candidates(
    candidate_ids: list[tuple[int, int]],
    attr_sets: dict[int, frozenset],
    attr_freq: dict[tuple, int],
) -> dict[str, list[tuple[int, int, float]]]:
    """
    Score a fixed list of (a_id, b_id) pairs with all three algorithms.

    Returns
    -------
    dict alg_name -> list of (a_id, b_id, score) sorted desc by score.
    """
    results: dict[str, list] = defaultdict(list)
    for a_id, b_id in candidate_ids:
        scores = score_pair(a_id, b_id, attr_sets, attr_freq)
        for alg, score in scores.items():
            results[alg].append((a_id, b_id, score))

    return {
        alg: sorted(rows, key=lambda x: x[2], reverse=True)
        for alg, rows in results.items()
    }


# Evaluation metrics
def precision_at_k(
    ranked: list[tuple[int, int, float]],
    positive_pairs: set[frozenset],
    k: int,
) -> float:
    """Fraction of top-k predictions that are true positives."""
    hits = sum(
        1 for a, b, _ in ranked[:k]
        if frozenset([a, b]) in positive_pairs
    )
    return hits / k if k else 0.0


def recall_at_k(
    ranked: list[tuple[int, int, float]],
    positive_pairs: set[frozenset],
    k: int,
) -> float:
    """Fraction of all positives recovered in the top-k predictions."""
    if not positive_pairs:
        return 0.0
    hits = sum(
        1 for a, b, _ in ranked[:k]
        if frozenset([a, b]) in positive_pairs
    )
    return hits / len(positive_pairs)

def average_precision(
    ranked: list[tuple[int, int, float]],
    positive_pairs: set[frozenset],
) -> float:
    """
    Mean Average Precision (MAP) over the full ranked list.
    AP = sum_k [ P@k * rel(k) ] / |positives|
    """
    if not positive_pairs:
        return 0.0
    hits = 0
    ap   = 0.0
    for rank, (a, b, _) in enumerate(ranked, 1):
        if frozenset([a, b]) in positive_pairs:
            hits += 1
            ap   += hits / rank
    return ap / len(positive_pairs)

def auc_roc(
    ranked: list[tuple[int, int, float]],
    positive_pairs: set[frozenset],
    negative_pairs: set[frozenset],
) -> float:
    """
    Approximate AUC-ROC via the Wilcoxon-Mann-Whitney statistic.

    AUC = P(score(pos) > score(neg))
    Estimated by counting concordant (pos > neg) pairs.
    """
    pos_scores = [s for a, b, s in ranked if frozenset([a, b]) in positive_pairs]
    neg_scores = [s for a, b, s in ranked if frozenset([a, b]) in negative_pairs]

    if not pos_scores or not neg_scores:
        return 0.5   # undefined, return chance level

    concordant = sum(1 for ps in pos_scores for ns in neg_scores if ps > ns)
    ties       = sum(1 for ps in pos_scores for ns in neg_scores if ps == ns)
    total      = len(pos_scores) * len(neg_scores)

    return (concordant + 0.5 * ties) / total


# Main evaluation run
def run_evaluation(
    anime_list: list[dict],
    proj_adj: dict[int, dict[int, int]],
    relation_triples: list[dict],
    k_values: tuple[int, ...] = (10, 25, 50),
    test_ratio: float = 0.2,
    neg_multiplier: int = 5,
    seed: int = 42,
) -> dict:
    """
    Full evaluation pipeline.

    1. Build attribute sets and frequency index.
    2. Stratified train/test split of typed relation triples.
    3. Sample negatives equal to neg_multiplier * |test positives|.
    4. Score all (test positives + negatives) with three heuristics.
    5. Compute Precision@K, Recall@K, MAP, AUC-ROC per algorithm.

    Returns a results dict consumed by analyse.py and visualize.py.
    """
    t_start = time.perf_counter()
    attr_sets  = build_all_attribute_sets(anime_list)
    attr_freq  = build_attr_frequency(anime_list)
    t_attr = time.perf_counter()
    anime_ids  = [a["id"] for a in anime_list]

    train_triples, test_triples = train_test_split_triples(
        relation_triples, test_ratio=test_ratio, seed=seed
    )

    # Positive pairs in test set
    test_positive_pairs = {
        frozenset([t["head"], t["tail"]]) for t in test_triples
    }

    # All known positive pairs (train + test) to avoid them as negatives
    all_positive_pairs = {
        frozenset([t["head"], t["tail"]]) for t in relation_triples
    }

    # Sample negatives
    n_neg = len(test_positive_pairs) * neg_multiplier
    neg_list = sample_negatives(anime_ids, all_positive_pairs, n_neg, seed=seed)
    negative_pairs = set(neg_list)

    # Candidate list = test positives + negatives
    candidates: list[tuple[int, int]] = (
        [(a, b) for fs in test_positive_pairs for a, b in [tuple(sorted(fs))]]
        + [(a, b) for fs in negative_pairs     for a, b in [tuple(sorted(fs))]]
    )

    # Score candidates
    t_score_start = time.perf_counter()
    ranked = score_candidates(candidates, attr_sets, attr_freq)
    t_score_end = time.perf_counter()

    # Compute metrics per algorithm
    metrics: dict[str, dict] = {}
    for alg, rows in ranked.items():
        alg_metrics: dict = {}
        for k in k_values:
            alg_metrics[f"precision@{k}"] = round(
                precision_at_k(rows, test_positive_pairs, k), 4
            )
            alg_metrics[f"recall@{k}"] = round(
                recall_at_k(rows, test_positive_pairs, k), 4
            )
        alg_metrics["MAP"]     = round(average_precision(rows, test_positive_pairs), 4)
        alg_metrics["AUC-ROC"] = round(auc_roc(rows, test_positive_pairs, negative_pairs), 4)
        metrics[alg] = alg_metrics

    runtime = {
        "attr_build_s":  round(t_attr - t_start, 3),
        "scoring_s":     round(t_score_end - t_score_start, 3),
        "total_s":       round(t_score_end - t_start, 3),
    }

    return {
        "metrics":             metrics,
        "train_triples":       train_triples,
        "test_triples":        test_triples,
        "test_positive_pairs": test_positive_pairs,
        "negative_pairs":      negative_pairs,
        "ranked":              ranked,
        "attr_sets":           attr_sets,
        "attr_freq":           attr_freq,
        "runtime":             runtime,
    }


# Relation type summary
def relation_type_summary(relation_triples: list[dict]) -> dict:
    """
    Summarise the relation type distribution in the full dataset.
    Directly answers the professor's question: "How many different relations?"
    """
    type_counts:  dict[str, int] = defaultdict(int)
    group_counts: dict[str, int] = defaultdict(int)

    for t in relation_triples:
        type_counts[t["relation"]] += 1
        group_counts[t["group"]]   += 1

    return {
        "n_relation_types":  len(type_counts),
        "type_counts":       dict(type_counts),
        "group_counts":      dict(group_counts),
        "total_eval_triples": len(relation_triples),
    }


# Relation classifier evaluation
def run_relation_classifier(
    anime_list: list[dict],
    train_triples: list[dict],
    test_triples: list[dict],
    attr_freq: dict[tuple, int],
) -> dict:
    """
    Train and evaluate the relation-type classifier.
    Answers: "Can you predict the relation type?"
    """
    clf = RelationClassifier()
    clf.train(anime_list, train_triples, attr_freq)

    if not clf._trained:
        return {"trained": False, "message": "sklearn not available or insufficient data"}

    report = clf.evaluate(anime_list, test_triples, attr_freq)
    return {"trained": True, "classification_report": report, "classifier": clf}


# Per-relation-type breakdown
def per_relation_type_metrics(
    ranked: dict[str, list],
    test_triples: list[dict],
    k: int = 25,
) -> dict:
    """
    Break down Precision@K and Recall@K by relation type.

    For each relation type, treat only its triples as positives.
    """
    per_type: dict[str, dict[str, dict]] = {}

    # Group test triples by type
    by_type: dict[str, set[frozenset]] = defaultdict(set)
    for t in test_triples:
        by_type[t["relation"]].add(frozenset([t["head"], t["tail"]]))

    for rel_type, pos_pairs in by_type.items():
        per_type[rel_type] = {}
        for alg, rows in ranked.items():
            per_type[rel_type][alg] = {
                f"precision@{k}": round(precision_at_k(rows, pos_pairs, k), 4),
                f"recall@{k}":    round(recall_at_k(rows, pos_pairs, k), 4),
                "MAP":            round(average_precision(rows, pos_pairs), 4),
            }

    return per_type


# CLI entry point
def main():
    parser = argparse.ArgumentParser(description="HMLP Link Prediction Evaluator")
    parser.add_argument("--graph",      default=None, help="Path to graph_data.json (default: data/graph_data.json)")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--neg-mult",   type=int, default=5,
                        help="Negative samples = neg_mult * |test positives|")
    args = parser.parse_args()

    anime_list, proj_adj, relation_triples, id_to_title = load_graph(args.graph)

    # Relation type summary
    print("\n=== Relation Type Summary ===")
    summary = relation_type_summary(relation_triples)
    print(f"  Distinct relation types : {summary['n_relation_types']}")
    print(f"  Total eval triples      : {summary['total_eval_triples']}")
    for rtype, cnt in sorted(summary["type_counts"].items(), key=lambda x: -x[1]):
        group = RELATION_GROUPS.get(rtype, "other")
        print(f"    {rtype:<18} {cnt:>4}  (group: {group})")

    # Main link prediction evaluation
    print("\n=== Link Prediction Evaluation ===")
    eval_results = run_evaluation(
        anime_list, proj_adj, relation_triples,
        k_values=(10, 25, 50),
        test_ratio=args.test_ratio,
        neg_multiplier=args.neg_mult,
        seed=args.seed,
    )

    rt = eval_results["runtime"]
    print(f"\n[evaluate] Practical runtime:")
    print(f"  Attribute set build : {rt['attr_build_s']:.3f}s")
    print(f"  Candidate scoring   : {rt['scoring_s']:.3f}s")
    print(f"  Total pipeline      : {rt['total_s']:.3f}s")

    for alg, m in eval_results["metrics"].items():
        print(f"\n  {alg}")
        for metric, val in m.items():
            print(f"    {metric:<18} {val}")

    # Per-relation-type breakdown
    print("\n=== Per-Relation-Type Precision@25 ===")
    per_type = per_relation_type_metrics(
        eval_results["ranked"], eval_results["test_triples"], k=25
    )
    for rel_type, alg_metrics in sorted(per_type.items()):
        print(f"\n  {rel_type}")
        for alg, metrics in alg_metrics.items():
            print(f"    {alg:<30} P@25={metrics['precision@25']}  MAP={metrics['MAP']}")

    # Relation type classifier
    print("\n=== Relation Type Classifier ===")
    clf_results = run_relation_classifier(
        anime_list,
        eval_results["train_triples"],
        eval_results["test_triples"],
        eval_results["attr_freq"],
    )
    if clf_results["trained"]:
        print(f"  Trained on {len(eval_results['train_triples'])} triples")
        report = clf_results["classification_report"]
        for cls, m in report.items():
            if isinstance(m, dict):
                print(f"  {cls:<18}  precision={m.get('precision',0):.3f}  "
                      f"recall={m.get('recall',0):.3f}  "
                      f"f1={m.get('f1-score',0):.3f}")
    else:
        print("  Classifier skipped: sklearn is not installed.")
        print("  Install it with:  pip install scikit-learn")

    # Save results JSON for visualize.py
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eval_results.json")

    # Serialise only the plain-data parts (no frozensets/objects)
    save_data = {
        "metrics":           eval_results["metrics"],
        "relation_summary":  summary,
        "per_relation_type": per_type,
        "runtime":           eval_results["runtime"],
        "classifier":        {
            "trained": clf_results["trained"],
            "report":  clf_results.get("classification_report", {}),
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[evaluate] Results saved to {out_path}")


if __name__ == "__main__":
    main()