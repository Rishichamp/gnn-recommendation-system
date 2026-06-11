# =============================================================================
# evaluate.py
# GNN-Based Recommendation System — Full Evaluation Pipeline
# =============================================================================
# WHAT THIS FILE DOES:
#   1. Loads the best saved model checkpoint
#   2. Runs evaluation on the held-out TEST set
#   3. Computes Recall@K and NDCG@K for multiple values of K
#   4. Prints a formatted results table
#   5. Shows human-readable top-10 recommendations for sample users
#   6. Saves all results to a JSON report file
#
# HOW TO RUN (after training):
#   python src/evaluate.py
#   python src/evaluate.py --checkpoint results/best_model.pt --top_k 20
#
# METRICS EXPLAINED:
#   Recall@K  — Of all items the user liked, what fraction did we recommend?
#   NDCG@K    — Same but position-aware: hitting rank 1 scores more than rank 10
# =============================================================================

import os
import sys
import json
import math
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_data
from src.model import LightGCN


# =============================================================================
# SECTION 1: CONFIG
# =============================================================================

def get_eval_config():
    parser = argparse.ArgumentParser(description="Evaluate LightGCN Model")
    parser.add_argument("--checkpoint", type=str,  default="results/best_model.pt")
    parser.add_argument("--top_k",      type=int,  default=20)
    parser.add_argument("--min_rating", type=float,default=4.0)
    parser.add_argument("--report_dir", type=str,  default="results")
    parser.add_argument("--sample_users", type=int, default=5)
    args, _ = parser.parse_known_args()
    return args


# =============================================================================
# SECTION 2: CORE METRIC FUNCTIONS
# =============================================================================

def recall_at_k(recommended_items, ground_truth_items, k):
    """
    Recall@K: fraction of ground-truth items that appear in top-K.

    FORMULA:
        Recall@K = |top_K_recommended ∩ ground_truth| / |ground_truth|

    RANGE: 0.0 (no hits) to 1.0 (all ground truth items recommended)

    Example:
        User liked movies: {A, B, C, D, E}
        We recommended top-10: {A, X, Y, B, Z, ...}
        Hits = {A, B} → Recall@10 = 2/5 = 0.40

    Args:
        recommended_items (list): Ranked list of recommended item IDs.
        ground_truth_items (set): Items the user actually interacted with.
        k (int): Cutoff rank.

    Returns:
        float: Recall@K score between 0 and 1.
    """
    if not ground_truth_items:
        return 0.0

    top_k_set = set(recommended_items[:k])
    hits      = len(top_k_set & ground_truth_items)
    return hits / len(ground_truth_items)


def ndcg_at_k(recommended_items, ground_truth_items, k):
    """
    NDCG@K: Normalized Discounted Cumulative Gain at rank K.

    WHY NDCG ON TOP OF RECALL?
        Recall treats all positions equally — a hit at rank 1 and a hit at
        rank 20 both count the same. But in practice, users mostly click
        the top few results. NDCG penalizes hits that appear lower in the
        list using a logarithmic discount.

    FORMULA:
        DCG@K  = sum_{i=1}^{K}  rel_i / log2(i + 1)
        IDCG@K = DCG of perfect ranking (all hits at the top)
        NDCG@K = DCG@K / IDCG@K

        rel_i = 1 if item at rank i is in ground truth, else 0

    RANGE: 0.0 (no hits or all hits at the bottom) to 1.0 (perfect ranking)

    Example:
        Ground truth: {A, B}
        Recommended: [X, A, Y, Z, B, ...]   (A at rank 2, B at rank 5)
        DCG@5  = 0/log2(2) + 1/log2(3) + 0 + 0 + 1/log2(6) = 0.631+0.387 = 1.018 (simplified)
        IDCG@5 = 1/log2(2) + 1/log2(3)  = 1.0 + 0.631 = 1.631 (perfect: hits at 1 & 2)
        NDCG@5 = 1.018 / 1.631 ≈ 0.624

    Args:
        recommended_items (list): Ranked list of recommended item IDs.
        ground_truth_items (set): Items the user actually interacted with.
        k (int): Cutoff rank.

    Returns:
        float: NDCG@K score between 0 and 1.
    """
    if not ground_truth_items:
        return 0.0

    top_k = recommended_items[:k]

    # DCG: sum relevance / log2(rank + 1) for each position
    dcg = 0.0
    for rank, item in enumerate(top_k, start=1):
        if item in ground_truth_items:
            dcg += 1.0 / math.log2(rank + 1)

    # IDCG: perfect DCG — all hits placed at the very top
    n_hits    = min(len(ground_truth_items), k)
    idcg      = sum(1.0 / math.log2(rank + 1) for rank in range(1, n_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(recommended_items, ground_truth_items, k):
    """
    Precision@K: fraction of top-K recommendations that are relevant.

    FORMULA:
        Precision@K = |top_K_recommended ∩ ground_truth| / K

    Less commonly reported than Recall/NDCG for rec systems but useful
    to understand how "precise" our top recommendations are.

    Args:
        recommended_items (list): Ranked list of recommended item IDs.
        ground_truth_items (set): Items the user actually interacted with.
        k (int): Cutoff rank.

    Returns:
        float: Precision@K score between 0 and 1.
    """
    if not ground_truth_items:
        return 0.0
    top_k_set = set(recommended_items[:k])
    hits      = len(top_k_set & ground_truth_items)
    return hits / k


# =============================================================================
# SECTION 3: FULL TEST-SET EVALUATION
# =============================================================================

def evaluate_full(model, edge_index, test_dict, exclude_dict,
                  num_items, k_values=(10, 20, 50)):
    """
    Runs full evaluation over ALL test users for multiple K values.

    This is the proper evaluation you report in your portfolio/README.
    Every user in the test set gets top-K recommendations generated,
    and all three metrics are computed and averaged.

    Args:
        model        (LightGCN): Trained model (loaded from checkpoint).
        edge_index   (Tensor):   Training graph edges.
        test_dict    (dict):     {user_id: set(item_ids)} — ground truth
        exclude_dict (dict):     {user_id: set(item_ids)} — items to hide
                                 (train + val interactions for each user)
        num_items    (int):      Total number of items.
        k_values     (tuple):    List of K cutoffs to evaluate at.

    Returns:
        dict: Nested results — {K: {"recall": float, "ndcg": float,
                                    "precision": float, "n_users": int}}
    """
    model.eval()

    # Only evaluate users who have ground truth test items
    test_users = [u for u in test_dict if test_dict[u]]
    print(f"[evaluate] Evaluating on {len(test_users):,} test users ...")

    # ── Score all test users in one matrix multiply ───────────────────────
    # This is much faster than scoring one user at a time.
    test_users_tensor = torch.tensor(test_users, dtype=torch.long)

    with torch.no_grad():
        # scores shape: [num_test_users, num_items]
        scores = model.predict(edge_index, user_ids=test_users_tensor)

    # ── Per-user metric accumulation ─────────────────────────────────────
    # Initialize result containers
    results = {
        k: {"recall": [], "ndcg": [], "precision": []}
        for k in k_values
    }

    for idx, user_id in enumerate(test_users):
        user_scores = scores[idx].clone()   # [num_items]

        # Mask out items the user interacted with in train+val
        for item_id in exclude_dict.get(user_id, set()):
            if item_id < num_items:
                user_scores[item_id] = float("-inf")

        # Get full ranked item list (sorted by score descending)
        ranked_items = user_scores.argsort(descending=True).tolist()

        # Ground truth: what this user liked in the test set
        ground_truth = test_dict[user_id]

        # Compute all metrics for all K values
        for k in k_values:
            results[k]["recall"].append(
                recall_at_k(ranked_items, ground_truth, k))
            results[k]["ndcg"].append(
                ndcg_at_k(ranked_items, ground_truth, k))
            results[k]["precision"].append(
                precision_at_k(ranked_items, ground_truth, k))

    # ── Average across all users ──────────────────────────────────────────
    final = {}
    for k in k_values:
        final[k] = {
            "recall"    : float(np.mean(results[k]["recall"])),
            "ndcg"      : float(np.mean(results[k]["ndcg"])),
            "precision" : float(np.mean(results[k]["precision"])),
            "n_users"   : len(test_users),
        }

    return final


# =============================================================================
# SECTION 4: PRINT RESULTS TABLE
# =============================================================================

def print_results_table(results, model_name="LightGCN"):
    """
    Prints a clean formatted results table to the console.

    This is what you screenshot for your portfolio / README.

    Example output:
    ╔══════════════════════════════════════════════════════╗
    ║         LightGCN — Test Set Evaluation               ║
    ╠══════════╦═══════════╦═══════════╦═══════════════════╣
    ║    K     ║ Recall@K  ║  NDCG@K   ║  Precision@K      ║
    ╠══════════╬═══════════╬═══════════╬═══════════════════╣
    ║   @10    ║  0.1423   ║  0.0987   ║  0.0142           ║
    ║   @20    ║  0.1891   ║  0.1134   ║  0.0095           ║
    ║   @50    ║  0.2654   ║  0.1298   ║  0.0053           ║
    ╚══════════╩═══════════╩═══════════╩═══════════════════╝
    """
    k_values = sorted(results.keys())

    # Header
    print()
    print("╔" + "═"*54 + "╗")
    print(f"║  {model_name:^50s}  ║")
    print(f"║  {'Test Set Evaluation Results':^50s}  ║")
    print("╠" + "═"*10 + "╦" + "═"*13 + "╦" + "═"*13 + "╦" + "═"*14 + "╣")
    print(f"║{'  K':^10}║{'Recall@K':^13}║{'NDCG@K':^13}║{'Precision@K':^14}║")
    print("╠" + "═"*10 + "╬" + "═"*13 + "╬" + "═"*13 + "╬" + "═"*14 + "╣")

    for k in k_values:
        r  = results[k]["recall"]
        n  = results[k]["ndcg"]
        p  = results[k]["precision"]
        nu = results[k]["n_users"]
        print(f"║{'@'+str(k):^10}║{r:^13.4f}║{n:^13.4f}║{p:^14.4f}║")

    print("╠" + "═"*10 + "╩" + "═"*13 + "╩" + "═"*13 + "╩" + "═"*14 + "╣")
    print(f"║  Users evaluated: {nu:<35d}║")
    print("╚" + "═"*54 + "╝")
    print()

    # ── Benchmark comparison ──────────────────────────────────────────────
    # Show how your model compares to known baselines on MovieLens-100K.
    # (Full dataset numbers — mock data will be lower, that's expected)
    best_k = max(k_values)
    your_recall = results[best_k]["recall"]

    print("  📊 BENCHMARK COMPARISON (MovieLens-100K, full dataset)")
    print(f"  {'Model':<25} {'Recall@20':>10} {'NDCG@20':>10}")
    print("  " + "─"*47)
    print(f"  {'Random Baseline':<25} {'~0.010':>10} {'~0.005':>10}")
    print(f"  {'Matrix Factorization':<25} {'~0.150':>10} {'~0.110':>10}")
    print(f"  {'LightGCN (paper)':<25} {'~0.187':>10} {'~0.133':>10}")
    marker = " ← you" if 20 in results else ""
    print(f"  {'Your LightGCN':<25} {results.get(20,results[best_k])['recall']:>10.4f}"
          f" {results.get(20,results[best_k])['ndcg']:>10.4f}{marker}")
    print()

    if your_recall < 0.05:
        print("  ℹ️  Note: Low scores are expected on mock/small data.")
        print("     On the full MovieLens-100K, target Recall@20 ≈ 0.15–0.19")
    print()


# =============================================================================
# SECTION 5: HUMAN-READABLE RECOMMENDATIONS
# =============================================================================

def show_recommendations(model, edge_index, data, user_ids, top_k=10):
    """
    Prints human-readable top-K movie recommendations for specific users.

    This is the most visually impressive part of your project — seeing
    actual movie names come out of your GNN model.

    Args:
        model      (LightGCN): Trained model.
        edge_index (Tensor):   Training graph.
        data       (dict):     Full data dict from load_data().
        user_ids   (list):     List of user IDs to show recommendations for.
        top_k      (int):      Number of recommendations per user.
    """
    movie_titles  = data["movie_titles"]
    train_dict    = data["train_dict"]
    val_dict      = data["val_dict"]
    test_dict     = data["test_dict"]

    print("=" * 62)
    print(f"  🎬  TOP-{top_k} MOVIE RECOMMENDATIONS")
    print("=" * 62)

    for user_id in user_ids:
        # Items to exclude = everything seen in train + val
        seen_items = (
            train_dict.get(user_id, set()) |
            val_dict.get(user_id, set())
        )

        # Generate recommendations
        recs = model.recommend(
            edge_index = edge_index,
            user_id    = user_id,
            seen_items = seen_items,
            top_k      = top_k,
        )

        # What did this user actually like? (test ground truth)
        liked = test_dict.get(user_id, set())

        print(f"\n  User {user_id}  "
              f"(trained on {len(seen_items)} movies, "
              f"test truth: {len(liked)} movie(s))")
        print("  " + "─"*58)

        # Print recommendations with hit marker
        for rank, item_id in enumerate(recs, start=1):
            title  = movie_titles.get(item_id, f"Movie {item_id}")
            hit    = "✅ HIT" if item_id in liked else ""
            print(f"  {rank:>2}. [{item_id:>4}] {title:<38} {hit}")

        # Print the test ground truth so you can see what was expected
        if liked:
            print(f"\n  Ground truth (test item):")
            for item_id in liked:
                title = movie_titles.get(item_id, f"Movie {item_id}")
                print(f"       [{item_id:>4}] {title}")

    print("\n" + "=" * 62)


# =============================================================================
# SECTION 6: SAVE RESULTS REPORT
# =============================================================================

def save_report(results, config, report_dir):
    """
    Saves the evaluation results to a JSON file.

    Useful for:
      - Comparing runs with different hyperparameters
      - Including exact numbers in your README
      - Keeping a record of experiment results

    Args:
        results    (dict): Evaluation metrics from evaluate_full().
        config     (Namespace): Eval config (checkpoint path, K, etc.)
        report_dir (str): Directory to save the report.
    """
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "eval_results.json")

    # Convert int keys to strings for JSON serialization
    serializable = {str(k): v for k, v in results.items()}
    report = {
        "checkpoint"  : config.checkpoint,
        "top_k"       : config.top_k,
        "results"     : serializable,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[evaluate] Report saved → {report_path}")
    return report_path


# =============================================================================
# SECTION 7: MASTER EVALUATE FUNCTION
# =============================================================================

def evaluate(config=None):
    """
    Full evaluation pipeline — loads model, evaluates, prints and saves results.

    Args:
        config: argparse Namespace. If None, uses get_eval_config() defaults.

    Returns:
        dict: Full evaluation results by K value.
    """
    if config is None:
        config = get_eval_config()

    print("=" * 62)
    print("  LightGCN EVALUATION — MovieLens 100K")
    print("=" * 62)
    print(f"  Checkpoint : {config.checkpoint}")
    print(f"  Top-K      : {config.top_k}")
    print()

    # ── Step 1: Load data ─────────────────────────────────────────────────
    data       = load_data(min_rating=config.min_rating)
    num_users  = data["num_users"]
    num_items  = data["num_items"]
    edge_index = data["graph"].edge_index

    # ── Step 2: Load model from checkpoint ───────────────────────────────
    if not os.path.exists(config.checkpoint):
        print(f"[evaluate] ❌ Checkpoint not found: {config.checkpoint}")
        print("  Run train.py first to generate a checkpoint.")
        sys.exit(1)

    model = LightGCN.load(config.checkpoint)
    model.eval()

    # ── Step 3: Full test evaluation ──────────────────────────────────────
    # Exclude BOTH train and val interactions when generating test recs.
    # We must not recommend things the user already saw.
    results = evaluate_full(
        model        = model,
        edge_index   = edge_index,
        test_dict    = data["test_dict"],
        exclude_dict = data["all_train_dict"],   # train + val combined
        num_items    = num_items,
        k_values     = (10, 20, 50),
    )

    # ── Step 4: Print results table ───────────────────────────────────────
    print_results_table(results)

    # ── Step 5: Show recommendations for sample users ─────────────────────
    # Pick users who have test ground truth and at least some training data
    sample_users = [
        u for u in data["test_dict"]
        if data["test_dict"][u] and len(data["train_dict"].get(u, set())) >= 3
    ][:config.sample_users]

    if sample_users:
        show_recommendations(
            model      = model,
            edge_index = edge_index,
            data       = data,
            user_ids   = sample_users,
            top_k      = 10,
        )

    # ── Step 6: Save JSON report ──────────────────────────────────────────
    save_report(results, config, config.report_dir)

    return results


# =============================================================================
# SECTION 8: ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    evaluate()
