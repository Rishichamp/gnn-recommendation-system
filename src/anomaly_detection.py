# =============================================================================
# anomaly_detection.py
# GNN-Based Recommendation System — Graph Anomaly Detection Module
# =============================================================================
# WHAT THIS FILE DOES:
#   After training LightGCN, every user has a learned embedding vector.
#   Genuine users behave consistently with their neighbourhood —
#   their embedding is similar to the embeddings of users who
#   watched the same movies. Bots and fake accounts interact
#   randomly, so their embeddings are far from their neighbours.
#
#   This module exploits that signal to flag anomalous users using
#   THREE complementary methods:
#
#   Method 1 — Neighbourhood Distance (primary)
#       Compute how far each user's embedding is from the mean
#       of their graph neighbours. High distance = anomaly.
#
#   Method 2 — Isolation Forest on embeddings
#       Classical ML anomaly detector run on embedding vectors.
#       Flags users whose embeddings are in sparse regions.
#
#   Method 3 — Degree + Entropy Score
#       Users with very low interaction counts OR who interact
#       with only the most popular items (no personal taste)
#       are flagged as suspicious.
#
#   Final score = weighted combination of all three methods.
#
# HOW TO RUN:
#   python src/anomaly_detection.py
# =============================================================================

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_data
from src.model       import LightGCN


# =============================================================================
# SECTION 1: EXTRACT USER EMBEDDINGS
# =============================================================================

def get_user_embeddings(model, edge_index):
    """
    Runs the trained LightGCN forward pass and returns the final
    user embedding matrix.

    These embeddings encode each user's taste based on:
      - Their own interactions (layer 0)
      - Interactions of similar users (layers 1, 2, 3)

    Genuine users cluster together in embedding space.
    Anomalous users are isolated outliers.

    Args:
        model      (LightGCN): Trained model.
        edge_index (Tensor):   Training graph.

    Returns:
        user_emb (np.ndarray): Shape [num_users, embedding_dim].
                               Detached from PyTorch graph, on CPU.
    """
    model.eval()
    with torch.no_grad():
        user_emb, _ = model(edge_index)
    return user_emb.cpu().numpy()


# =============================================================================
# SECTION 2: BUILD NEIGHBOUR LOOKUP FROM GRAPH
# =============================================================================

def build_user_neighbours(edge_index, num_users):
    """
    Builds a dictionary mapping each user to the set of item node IDs
    they are directly connected to in the graph.

    Used to identify each user's immediate graph neighbourhood.
    In a bipartite user-item graph, a user's neighbours are
    the items they interacted with (and through them, other users).

    Args:
        edge_index (Tensor): Shape [2, num_edges]. Bipartite graph.
        num_users  (int):    Number of user nodes.

    Returns:
        dict: {user_id (int): set of neighbour node IDs (int)}
              Includes only user → item connections.
    """
    neighbours = defaultdict(set)
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    for s, d in zip(src, dst):
        if s < num_users:          # source is a user
            neighbours[s].add(d)   # destination is an item node
        elif d < num_users:        # destination is a user
            neighbours[d].add(s)   # source is an item node

    return dict(neighbours)


# =============================================================================
# SECTION 3: METHOD 1 — NEIGHBOURHOOD DISTANCE SCORE
# =============================================================================

def neighbourhood_distance_scores(user_emb, edge_index, num_users):
    """
    Computes how different each user's embedding is from the average
    embedding of their immediate graph neighbours.

    INTUITION:
        If user A watched the same movies as users B, C, D — then A's
        learned embedding should be close to the average of B, C, D.
        A genuine user fits in with their neighbourhood.

        A bot that watches random movies will have a neighbourhood of
        dissimilar users. Its embedding won't match any of them.
        → High distance from neighbourhood mean = anomaly signal.

    SCORE:
        For each user u:
          neighbours_u = all nodes directly connected to u
          mean_emb     = average embedding of neighbours_u
          score_u      = L2 distance between user_emb[u] and mean_emb

    Args:
        user_emb   (np.ndarray): Shape [num_users, dim].
        edge_index (Tensor):     Graph edges.
        num_users  (int):        Number of user nodes.

    Returns:
        np.ndarray: Shape [num_users]. Raw distance scores.
                    Higher = more anomalous.
    """
    print("[anomaly] Computing neighbourhood distance scores ...")

    all_emb       = user_emb                 # shape [num_users, dim]
    num_all_nodes = edge_index.max().item() + 1

    # Pad embedding array to include item nodes (for neighbour lookups)
    # Item nodes don't have embeddings here — we'll handle them separately
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()

    scores = np.zeros(num_users)

    for user_id in range(num_users):
        # Find all nodes this user is connected to
        mask           = src == user_id
        neighbour_ids  = dst[mask]

        if len(neighbour_ids) == 0:
            # Isolated user — no neighbours — give neutral score
            scores[user_id] = 0.0
            continue

        # Only include neighbours that are also users (for comparison)
        # These are indirect user-user connections via shared items
        user_neighbour_ids = neighbour_ids[neighbour_ids < num_users]

        if len(user_neighbour_ids) == 0:
            # No user-neighbours found via this direction;
            # use the user's own embedding distance from global mean
            global_mean    = all_emb.mean(axis=0)
            scores[user_id] = np.linalg.norm(all_emb[user_id] - global_mean)
            continue

        # Mean embedding of user's neighbours
        neighbour_embs  = all_emb[user_neighbour_ids]   # [n_neighbours, dim]
        neighbour_mean  = neighbour_embs.mean(axis=0)   # [dim]

        # L2 distance from user embedding to neighbour mean
        scores[user_id] = np.linalg.norm(all_emb[user_id] - neighbour_mean)

    return scores


# =============================================================================
# SECTION 4: METHOD 2 — ISOLATION FOREST ON EMBEDDINGS
# =============================================================================

def isolation_forest_scores(user_emb, contamination=0.05):
    """
    Applies Isolation Forest anomaly detection to the user embeddings.

    WHAT IS ISOLATION FOREST?
        It builds random decision trees that split the data. Points
        that are easy to isolate (need fewer splits to separate) are
        anomalies — they sit in sparse, unusual regions of embedding space.

        This is a well-studied, reliable anomaly detection algorithm
        that works well in high-dimensional spaces like embeddings.

    WHY USE IT ALONGSIDE NEIGHBOURHOOD DISTANCE?
        Neighbourhood distance catches users who don't fit their graph
        local structure. Isolation Forest catches users whose embeddings
        are globally unusual — even if they happen to be near some
        neighbours by distance.

        Together they catch different kinds of anomalies.

    Args:
        user_emb      (np.ndarray): Shape [num_users, dim].
        contamination (float):      Expected fraction of anomalies (default 5%).

    Returns:
        np.ndarray: Shape [num_users]. Anomaly scores.
                    Higher = more anomalous (we flip sklearn's sign convention).
    """
    print("[anomaly] Running Isolation Forest on embeddings ...")

    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        print("[anomaly] scikit-learn not found. Skipping Isolation Forest.")
        return np.zeros(len(user_emb))

    clf = IsolationForest(
        n_estimators  = 100,
        contamination = contamination,
        random_state  = 42,
        n_jobs        = -1,
    )
    clf.fit(user_emb)

    # sklearn returns -1 for anomaly, 1 for normal
    # decision_function returns raw score — more negative = more anomalous
    raw_scores = clf.decision_function(user_emb)   # shape [num_users]

    # Flip so higher = more anomalous (consistent with other methods)
    return -raw_scores


# =============================================================================
# SECTION 5: METHOD 3 — DEGREE + POPULARITY ENTROPY SCORE
# =============================================================================

def degree_entropy_scores(train_dict, num_users, num_items):
    """
    Computes a suspicion score based on two behavioural signals:

    Signal A — Very low degree:
        Genuine users interact with multiple items over time.
        A user with only 1–2 interactions is suspicious — they
        may be a throwaway account or an injection attack.

    Signal B — Low entropy (only popular items):
        Real users have varied, personal tastes. Bots and
        click-farms tend to only interact with the most
        popular items (because it's easy and cheap).
        Entropy of the item popularity distribution captures this:
        low entropy = only popular items = suspicious.

    Score = (1 / degree) × (1 / (entropy + ε))

    Args:
        train_dict (dict): {user_id: set(item_ids)} from data_loader.
        num_users  (int):  Total number of users.
        num_items  (int):  Total number of items.

    Returns:
        np.ndarray: Shape [num_users]. Higher = more anomalous.
    """
    print("[anomaly] Computing degree + entropy scores ...")

    # ── Compute item popularity (how many users interacted with each item) ─
    item_counts = defaultdict(int)
    for items in train_dict.values():
        for item_id in items:
            item_counts[item_id] += 1

    total_interactions = sum(item_counts.values()) + 1e-9
    item_popularity    = {
        item: count / total_interactions
        for item, count in item_counts.items()
    }

    scores = np.zeros(num_users)

    for user_id in range(num_users):
        user_items = train_dict.get(user_id, set())
        degree     = len(user_items)

        if degree == 0:
            # No interactions at all — maximum suspicion
            scores[user_id] = 10.0
            continue

        # Entropy of the popularity of items this user interacted with
        # Low entropy = user only watches very popular items = suspicious
        probs   = np.array([item_popularity.get(i, 1e-9) for i in user_items])
        probs   = probs / (probs.sum() + 1e-9)
        entropy = -np.sum(probs * np.log(probs + 1e-9))

        # Combine: penalise low degree and low entropy
        degree_score  = 1.0 / (degree + 1)       # higher for fewer interactions
        entropy_score = 1.0 / (entropy + 1e-3)   # higher for lower entropy

        scores[user_id] = degree_score * entropy_score

    return scores


# =============================================================================
# SECTION 6: COMBINE SCORES + NORMALISE
# =============================================================================

def combine_scores(nd_scores, if_scores, de_scores,
                   w_nd=0.5, w_if=0.3, w_de=0.2):
    """
    Combines the three anomaly score arrays into one final score.

    Each method is first normalised to [0, 1] using min-max scaling,
    then combined as a weighted sum.

    WHY DIFFERENT WEIGHTS?
        - Neighbourhood distance (0.5) — most directly tied to the
          graph structure that LightGCN was trained on. Most reliable.
        - Isolation Forest (0.3)       — strong classical method but
          doesn't use graph structure. Good second signal.
        - Degree + Entropy (0.2)       — useful heuristic but noisy
          on small datasets.

    Args:
        nd_scores (np.ndarray): Neighbourhood distance scores.
        if_scores (np.ndarray): Isolation Forest scores.
        de_scores (np.ndarray): Degree + Entropy scores.
        w_nd, w_if, w_de (float): Weights (must sum to 1.0).

    Returns:
        np.ndarray: Shape [num_users]. Final anomaly score in [0, 1].
    """

    def minmax(arr):
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-9:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    nd_norm = minmax(nd_scores)
    if_norm = minmax(if_scores)
    de_norm = minmax(de_scores)

    combined = w_nd * nd_norm + w_if * if_norm + w_de * de_norm
    return combined


# =============================================================================
# SECTION 7: FLAG ANOMALIES + THRESHOLD
# =============================================================================

def flag_anomalies(final_scores, threshold_percentile=95):
    """
    Applies a percentile-based threshold to the final scores
    to produce binary anomaly flags.

    THRESHOLD STRATEGY:
        Rather than picking a fixed score threshold (which is
        dataset-dependent), we flag the top X% most anomalous users.
        Default: top 5% (95th percentile).

        This is interpretable: "these are the 5% most suspicious users
        in our dataset."

    Args:
        final_scores         (np.ndarray): Combined anomaly scores.
        threshold_percentile (int):        Flag users above this percentile.

    Returns:
        tuple:
            flags     (np.ndarray bool): True = flagged as anomaly.
            threshold (float):          The score threshold used.
    """
    threshold = np.percentile(final_scores, threshold_percentile)
    flags     = final_scores >= threshold
    return flags, threshold


# =============================================================================
# SECTION 8: VISUALISE ANOMALY SCORES
# =============================================================================

def plot_anomaly_scores(final_scores, flags, threshold, plots_dir):
    """
    Saves three plots to help you understand and present the detection results.

    Plot 1 — Score distribution histogram:
        Shows the spread of anomaly scores across all users.
        The red line marks where the threshold is.
        Anomalous users appear to the right of the line.

    Plot 2 — Top-50 anomalous users bar chart:
        Shows the most suspicious users ranked by score.
        Good for explaining "which specific users were flagged."

    Plot 3 — Normal vs Anomalous score comparison (box plot):
        Shows that flagged users have clearly higher scores
        than normal users — visually validating the method.

    Args:
        final_scores (np.ndarray): Combined anomaly scores.
        flags        (np.ndarray): Boolean anomaly flags.
        threshold    (float):      Score threshold used.
        plots_dir    (str):        Output directory.
    """
    os.makedirs(plots_dir, exist_ok=True)

    normal_scores   = final_scores[~flags]
    anomaly_scores  = final_scores[flags]
    n_flagged       = flags.sum()

    # ── Plot 1: Score distribution ────────────────────────────────────────
    plt.figure(figsize=(9, 4))
    plt.hist(final_scores, bins=60, color="#3B82F6", alpha=0.7,
             edgecolor="white", label="All users")
    plt.axvline(threshold, color="#EF4444", linewidth=2,
                linestyle="--", label=f"Threshold (p95) = {threshold:.3f}")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Number of Users")
    plt.title(f"User Anomaly Score Distribution  "
              f"(flagged: {n_flagged} / {len(final_scores)})")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "anomaly_score_distribution.png")
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"[anomaly] Plot saved → {p1}")

    # ── Plot 2: Top-50 anomalous users ────────────────────────────────────
    top_n        = min(50, n_flagged)
    top_indices  = np.argsort(final_scores)[::-1][:top_n]
    top_scores   = final_scores[top_indices]

    plt.figure(figsize=(12, 4))
    bars = plt.bar(range(top_n), top_scores,
                   color="#EF4444", alpha=0.8, edgecolor="white")
    plt.axhline(threshold, color="#1D4ED8", linewidth=1.5,
                linestyle="--", label=f"Threshold = {threshold:.3f}")
    plt.xlabel("Rank (most anomalous → least)")
    plt.ylabel("Anomaly Score")
    plt.title(f"Top-{top_n} Most Anomalous Users")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(plots_dir, "top_anomalous_users.png")
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"[anomaly] Plot saved → {p2}")

    # ── Plot 3: Normal vs anomalous box plot ──────────────────────────────
    plt.figure(figsize=(6, 5))
    plt.boxplot(
        [normal_scores, anomaly_scores],
        labels=["Normal users", "Flagged anomalies"],
        patch_artist=True,
        boxprops     = dict(facecolor="#BFDBFE", color="#1D4ED8"),
        medianprops  = dict(color="#1D4ED8", linewidth=2),
        flierprops   = dict(marker="o", markerfacecolor="#EF4444",
                            markersize=4, alpha=0.5),
    )
    plt.ylabel("Anomaly Score")
    plt.title("Score Distribution: Normal vs Flagged")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p3 = os.path.join(plots_dir, "anomaly_boxplot.png")
    plt.savefig(p3, dpi=150)
    plt.close()
    print(f"[anomaly] Plot saved → {p3}")


# =============================================================================
# SECTION 9: PRINT ANOMALY REPORT
# =============================================================================

def print_anomaly_report(final_scores, flags, threshold,
                         train_dict, movie_titles, top_n=10):
    """
    Prints a formatted anomaly detection report to the console.

    Shows:
      - Summary statistics
      - Top-N most suspicious users with their scores and interaction counts
      - What those users watched (to intuitively verify the detection)

    Args:
        final_scores  (np.ndarray): Combined anomaly scores.
        flags         (np.ndarray): Boolean anomaly flags.
        threshold     (float):      Score threshold used.
        train_dict    (dict):       {user_id: set(item_ids)}.
        movie_titles  (dict):       {item_id: title}.
        top_n         (int):        How many top anomalies to display.
    """
    n_total   = len(final_scores)
    n_flagged = flags.sum()
    top_ids   = np.argsort(final_scores)[::-1][:top_n]

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║           ANOMALY DETECTION REPORT                      ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Total users        : {n_total:<34d}║")
    print(f"║  Flagged anomalies  : {n_flagged:<34d}║")
    print(f"║  Flag rate          : {100*n_flagged/n_total:<33.1f}%║")
    print(f"║  Score threshold    : {threshold:<34.4f}║")
    print(f"║  Score mean         : {final_scores.mean():<34.4f}║")
    print(f"║  Score std          : {final_scores.std():<34.4f}║")
    print("╚══════════════════════════════════════════════════════════╝")

    print(f"\n  🚨  TOP-{top_n} MOST SUSPICIOUS USERS")
    print("  " + "─"*60)
    print(f"  {'Rank':<5} {'UserID':<8} {'Score':<8} "
          f"{'# Movies':<10} {'Sample Watched'}")
    print("  " + "─"*60)

    for rank, user_id in enumerate(top_ids, start=1):
        score      = final_scores[user_id]
        user_items = train_dict.get(user_id, set())
        n_items    = len(user_items)

        # Show up to 2 movie titles for context
        sample_titles = []
        for item_id in list(user_items)[:2]:
            t = movie_titles.get(item_id, f"Movie {item_id}")
            sample_titles.append(t[:20])
        sample_str = ", ".join(sample_titles) if sample_titles else "—"

        flag_marker = " 🚩" if flags[user_id] else ""
        print(f"  {rank:<5} {user_id:<8} {score:<8.4f} "
              f"{n_items:<10} {sample_str}{flag_marker}")

    print()
    print("  WHY THESE USERS ARE FLAGGED:")
    print("  ─────────────────────────────────────────────────────────")
    print("  • Their learned embeddings are far from their graph")
    print("    neighbourhood — they don't 'fit in' with similar users.")
    print("  • Isolation Forest found them in sparse regions of")
    print("    embedding space — globally unusual behaviour.")
    print("  • Low interaction count or low taste diversity")
    print("    (only popular items, no personal preference signal).")
    print()


# =============================================================================
# SECTION 10: SAVE RESULTS
# =============================================================================

def save_anomaly_results(final_scores, flags, threshold, report_dir):
    """
    Saves anomaly detection results to a JSON file.

    Saved fields:
      - List of flagged user IDs
      - Their anomaly scores
      - Threshold used
      - Summary statistics

    Args:
        final_scores (np.ndarray): Combined anomaly scores.
        flags        (np.ndarray): Boolean anomaly flags.
        threshold    (float):      Score threshold.
        report_dir   (str):        Output directory.

    Returns:
        str: Path to saved JSON file.
    """
    os.makedirs(report_dir, exist_ok=True)

    flagged_ids    = np.where(flags)[0].tolist()
    flagged_scores = final_scores[flags].tolist()

    report = {
        "threshold"      : float(threshold),
        "n_total_users"  : int(len(final_scores)),
        "n_flagged"      : int(flags.sum()),
        "flag_rate_pct"  : float(100 * flags.sum() / len(final_scores)),
        "score_mean"     : float(final_scores.mean()),
        "score_std"      : float(final_scores.std()),
        "flagged_users"  : [
            {"user_id": uid, "anomaly_score": round(sc, 6)}
            for uid, sc in zip(flagged_ids, flagged_scores)
        ]
    }

    path = os.path.join(report_dir, "anomaly_results.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[anomaly] Results saved → {path}")
    return path


# =============================================================================
# SECTION 11: MASTER FUNCTION
# =============================================================================

def detect_anomalies(checkpoint   = "results/best_model.pt",
                     min_rating   = 4.0,
                     percentile   = 95,
                     plots_dir    = "results/plots",
                     report_dir   = "results"):
    """
    Full anomaly detection pipeline.

    Steps:
        1. Load trained LightGCN model from checkpoint
        2. Extract user embeddings via forward pass
        3. Compute Method 1: neighbourhood distance scores
        4. Compute Method 2: Isolation Forest scores
        5. Compute Method 3: degree + entropy scores
        6. Combine into final score
        7. Apply percentile threshold to flag anomalies
        8. Plot score distributions
        9. Print report
       10. Save JSON results

    Args:
        checkpoint (str):  Path to saved model checkpoint.
        min_rating (float):Minimum rating filter for data loading.
        percentile (int):  Percentile threshold for flagging (default 95).
        plots_dir  (str):  Directory to save plots.
        report_dir (str):  Directory to save JSON report.

    Returns:
        dict with keys:
            final_scores, flags, threshold, flagged_user_ids
    """
    print("=" * 62)
    print("  GNN ANOMALY DETECTION")
    print("=" * 62)
    print(f"  Checkpoint  : {checkpoint}")
    print(f"  Threshold   : top {100 - percentile}% (p{percentile})")
    print()

    # ── Step 1: Load data ─────────────────────────────────────────────────
    data       = load_data(min_rating=min_rating)
    num_users  = data["num_users"]
    num_items  = data["num_items"]
    edge_index = data["graph"].edge_index
    train_dict = data["train_dict"]
    titles     = data["movie_titles"]

    # ── Step 2: Load model ────────────────────────────────────────────────
    if not os.path.exists(checkpoint):
        print(f"❌ Checkpoint not found: {checkpoint}")
        print("   Run:  python main.py --mode train  first.")
        sys.exit(1)

    model = LightGCN.load(checkpoint)

    # ── Step 3: Extract embeddings ────────────────────────────────────────
    print("[anomaly] Extracting user embeddings from trained model ...")
    user_emb = get_user_embeddings(model, edge_index)
    print(f"[anomaly] Embedding matrix shape: {user_emb.shape}")

    # ── Step 4: Method 1 — Neighbourhood distance ─────────────────────────
    nd_scores = neighbourhood_distance_scores(user_emb, edge_index, num_users)

    # ── Step 5: Method 2 — Isolation Forest ──────────────────────────────
    if_scores = isolation_forest_scores(user_emb, contamination=0.05)

    # ── Step 6: Method 3 — Degree + Entropy ──────────────────────────────
    de_scores = degree_entropy_scores(train_dict, num_users, num_items)

    # ── Step 7: Combine scores ────────────────────────────────────────────
    print("[anomaly] Combining scores (nd=0.5, if=0.3, de=0.2) ...")
    final_scores = combine_scores(nd_scores, if_scores, de_scores)

    # ── Step 8: Flag anomalies ────────────────────────────────────────────
    flags, threshold = flag_anomalies(final_scores, threshold_percentile=percentile)
    print(f"[anomaly] Flagged {flags.sum()} / {num_users} users "
          f"(threshold = {threshold:.4f})")

    # ── Step 9: Print report ──────────────────────────────────────────────
    print_anomaly_report(final_scores, flags, threshold,
                         train_dict, titles, top_n=10)

    # ── Step 10: Plot ─────────────────────────────────────────────────────
    plot_anomaly_scores(final_scores, flags, threshold, plots_dir)

    # ── Step 11: Save JSON ────────────────────────────────────────────────
    save_anomaly_results(final_scores, flags, threshold, report_dir)

    flagged_ids = np.where(flags)[0].tolist()
    print(f"\n✅ Anomaly detection complete.")
    print(f"   Flagged {len(flagged_ids)} suspicious users.")
    print(f"   Plots  → {plots_dir}/anomaly_*.png")
    print(f"   Report → {report_dir}/anomaly_results.json\n")

    return {
        "final_scores"    : final_scores,
        "flags"           : flags,
        "threshold"       : threshold,
        "flagged_user_ids": flagged_ids,
    }


# =============================================================================
# SECTION 12: ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    detect_anomalies()
