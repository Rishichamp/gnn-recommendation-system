# =============================================================================
# data_loader.py
# GNN-Based Recommendation System — MovieLens 100K
# =============================================================================
# WHAT THIS FILE DOES:
#   1. Downloads the MovieLens 100K dataset (if not already present)
#   2. Loads and cleans the raw ratings data
#   3. Filters only positive interactions (rating >= 4)
#   4. Splits data into train / validation / test sets
#   5. Builds a bipartite graph (users <-> movies) for PyTorch Geometric
#   6. Returns everything your model and trainer need
# =============================================================================

import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from collections import defaultdict


# =============================================================================
# SECTION 1: DOWNLOAD DATASET
# =============================================================================

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RATINGS_FILE  = os.path.join(DATA_DIR, "ml-100k", "u.data")
MOVIES_FILE   = os.path.join(DATA_DIR, "ml-100k", "u.item")


def download_movielens(force=False):
    """
    Downloads and extracts the MovieLens 100K dataset into data/raw/.

    Args:
        force (bool): If True, re-downloads even if already present.

    The dataset contains:
        u.data  — 100,000 ratings (user_id, item_id, rating, timestamp)
        u.item  — Movie metadata (movie_id, title, genres, ...)
        u.user  — User metadata (user_id, age, gender, occupation, zip)
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "ml-100k.zip")

    if os.path.exists(RATINGS_FILE) and not force:
        print("[data_loader] Dataset already exists. Skipping download.")
        return

    print("[data_loader] Downloading MovieLens 100K ...")
    urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
    print("[data_loader] Download complete. Extracting ...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    os.remove(zip_path)
    print(f"[data_loader] Dataset ready at: {os.path.abspath(DATA_DIR)}")


# =============================================================================
# SECTION 2: LOAD & CLEAN RATINGS
# =============================================================================

def load_ratings(min_rating=4.0):
    """
    Loads the raw ratings file and filters for positive interactions.

    Args:
        min_rating (float): Minimum rating to consider as a positive
                            interaction. Default is 4.0 (out of 5).

    Returns:
        pd.DataFrame with columns: [user_id, item_id, rating, timestamp]
        All IDs are 0-indexed.

    WHY min_rating=4?
        We're building an implicit feedback model — we only care whether
        a user liked something, not the exact score. Ratings of 4 and 5
        clearly indicate a positive experience. Lower ratings are ignored.
    """
    download_movielens()

    print("[data_loader] Loading ratings ...")
    df = pd.read_csv(
        RATINGS_FILE,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python"
    )

    print(f"[data_loader] Raw ratings: {len(df):,}")

    # Convert to 0-indexed (original data starts at 1)
    df["user_id"] = df["user_id"] - 1
    df["item_id"] = df["item_id"] - 1

    # Keep only positive interactions
    df = df[df["rating"] >= min_rating].reset_index(drop=True)
    print(f"[data_loader] After filtering (rating >= {min_rating}): {len(df):,} interactions")

    # Sort by timestamp so we can do temporal split later
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# =============================================================================
# SECTION 3: LOAD MOVIE METADATA (optional, for display)
# =============================================================================

def load_movie_titles():
    """
    Loads movie titles for human-readable output.

    Returns:
        dict: {item_id (0-indexed): movie_title (str)}

    Used when printing recommendations so you see movie names, not just IDs.
    """
    download_movielens()

    movies = {}
    with open(MOVIES_FILE, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                movie_id  = int(parts[0]) - 1   # 0-indexed
                title     = parts[1]
                movies[movie_id] = title

    print(f"[data_loader] Loaded {len(movies):,} movie titles.")
    return movies


# =============================================================================
# SECTION 4: TRAIN / VALIDATION / TEST SPLIT
# =============================================================================

def split_data(df, val_ratio=0.1, test_ratio=0.1):
    """
    Splits interaction data into train, validation, and test sets.

    Strategy: LEAVE-ONE-OUT (per user)
        - For each user, the LAST interaction (by timestamp) → test set
        - The SECOND-TO-LAST interaction              → validation set
        - All remaining interactions                  → training set

    This is the standard evaluation protocol for rec system papers.

    Args:
        df (pd.DataFrame): Filtered ratings dataframe.
        val_ratio  (float): Unused here (kept for interface consistency).
        test_ratio (float): Unused here (kept for interface consistency).

    Returns:
        train_df (pd.DataFrame)
        val_df   (pd.DataFrame)
        test_df  (pd.DataFrame)
    """
    print("[data_loader] Splitting into train / val / test ...")

    train_data = []
    val_data   = []
    test_data  = []

    # Group by user and split per user
    for user_id, group in df.groupby("user_id"):
        # Already sorted by timestamp globally
        interactions = group.reset_index(drop=True)
        n = len(interactions)

        if n < 3:
            # Not enough interactions to split — keep all in train
            train_data.append(interactions)
            continue

        # Last row → test, second-to-last → val, rest → train
        test_data.append(interactions.iloc[[-1]])
        val_data.append(interactions.iloc[[-2]])
        train_data.append(interactions.iloc[:-2])

    train_df = pd.concat(train_data).reset_index(drop=True)
    val_df   = pd.concat(val_data).reset_index(drop=True)
    test_df  = pd.concat(test_data).reset_index(drop=True)

    print(f"[data_loader] Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    return train_df, val_df, test_df


# =============================================================================
# SECTION 5: BUILD GRAPH (PyTorch Geometric format)
# =============================================================================

def build_graph(train_df, num_users, num_items):
    """
    Converts the training interactions into a PyTorch Geometric Data object
    representing a bipartite user-item graph.

    HOW THE GRAPH IS STRUCTURED:
        - Nodes 0 ... num_users-1           → User nodes
        - Nodes num_users ... num_users+num_items-1 → Item nodes
        - An edge exists between user u and item i if u rated i (in train)
        - The graph is UNDIRECTED → each edge stored in both directions

    WHY BIPARTITE?
        In a recommendation graph, users only connect to items and
        items only connect to users. There are no user-user or
        item-item edges. This structure is called bipartite.

    Args:
        train_df (pd.DataFrame): Training interactions.
        num_users (int): Total number of unique users.
        num_items (int): Total number of unique items.

    Returns:
        torch_geometric.data.Data object with:
            .edge_index — shape [2, 2*num_edges] (both directions)
            .num_nodes  — total nodes (users + items)
    """
    print("[data_loader] Building bipartite graph ...")

    user_ids  = torch.tensor(train_df["user_id"].values, dtype=torch.long)
    # Shift item IDs so they don't overlap with user IDs in node space
    item_ids  = torch.tensor(train_df["item_id"].values + num_users, dtype=torch.long)

    # Build undirected edges: user→item AND item→user
    src = torch.cat([user_ids, item_ids])   # sources
    dst = torch.cat([item_ids, user_ids])   # destinations

    edge_index = torch.stack([src, dst], dim=0)  # shape: [2, 2*E]

    graph = Data(
        edge_index = edge_index,
        num_nodes  = num_users + num_items
    )

    print(f"[data_loader] Graph built: {graph.num_nodes:,} nodes | "
          f"{edge_index.shape[1]:,} directed edges "
          f"({edge_index.shape[1]//2:,} unique interactions)")

    return graph


# =============================================================================
# SECTION 6: BUILD USER INTERACTION LOOKUP (for training & evaluation)
# =============================================================================

def build_interaction_dict(df, num_users):
    """
    Builds a dictionary mapping each user to their set of interacted item IDs.

    Used for:
        1. Negative sampling during training (don't sample items user liked)
        2. Evaluation (exclude already-seen items from recommendations)

    Args:
        df (pd.DataFrame): Any split of interactions.
        num_users (int): Total users (unused here, kept for consistency).

    Returns:
        dict: {user_id (int): set of item_ids (int)}
    """
    interaction_dict = defaultdict(set)
    for _, row in df.iterrows():
        interaction_dict[int(row["user_id"])].add(int(row["item_id"]))
    return dict(interaction_dict)


# =============================================================================
# SECTION 7: NEGATIVE SAMPLING (for BPR training)
# =============================================================================

def sample_negative(user_id, all_interacted_items, num_items, n_samples=1):
    """
    Randomly samples item(s) that a user has NOT interacted with.

    This is called 'negative sampling' — we assume items the user
    never touched are things they don't prefer.

    Args:
        user_id (int): The user we're sampling negatives for.
        all_interacted_items (set): Items this user HAS interacted with.
        num_items (int): Total number of items.
        n_samples (int): How many negative items to sample.

    Returns:
        list of int: Sampled negative item IDs.

    NOTE: Uses rejection sampling — keeps trying until it finds
          an item not in the user's history. For dense users this
          could be slow; for MovieLens-100K it's fine.
    """
    negatives = []
    while len(negatives) < n_samples:
        neg = np.random.randint(0, num_items)
        if neg not in all_interacted_items:
            negatives.append(neg)
    return negatives


# =============================================================================
# SECTION 8: MASTER LOAD FUNCTION (call this from train.py)
# =============================================================================

def load_data(min_rating=4.0):
    """
    Master function — call this from train.py to get everything at once.

    Returns a dictionary with all the data your training pipeline needs:

    {
        "train_df"       : pd.DataFrame — training interactions
        "val_df"         : pd.DataFrame — validation interactions
        "test_df"        : pd.DataFrame — test interactions
        "graph"          : torch_geometric.data.Data — bipartite graph
        "train_dict"     : dict — {user_id: set(item_ids)} for train
        "val_dict"       : dict — {user_id: set(item_ids)} for val
        "test_dict"      : dict — {user_id: set(item_ids)} for test
        "all_train_dict" : dict — train+val combined (used at test time)
        "num_users"      : int
        "num_items"      : int
        "movie_titles"   : dict — {item_id: title}
    }

    Example usage in train.py:
        from src.data_loader import load_data
        data = load_data()
        graph      = data["graph"]
        num_users  = data["num_users"]
        num_items  = data["num_items"]
        train_dict = data["train_dict"]
    """
    # Step 1: Load & clean ratings
    df = load_ratings(min_rating=min_rating)

    # Step 2: Get dataset statistics
    # IMPORTANT: use max+1 (not nunique) so node IDs don't go out of bounds.
    # IDs may not be contiguous after filtering, so max+1 gives the full range.
    num_users = int(df["user_id"].max()) + 1
    num_items = int(df["item_id"].max()) + 1
    print(f"[data_loader] Users: {num_users} | Items: {num_items}")

    # Step 3: Split into train / val / test
    train_df, val_df, test_df = split_data(df)

    # Step 4: Build the graph from training edges ONLY
    #         (we must not leak val/test edges into the graph)
    graph = build_graph(train_df, num_users, num_items)

    # Step 5: Build interaction lookup dictionaries
    train_dict     = build_interaction_dict(train_df, num_users)
    val_dict       = build_interaction_dict(val_df,   num_users)
    test_dict      = build_interaction_dict(test_df,  num_users)

    # Combined train+val dict = all known interactions at test time
    combined_df    = pd.concat([train_df, val_df])
    all_train_dict = build_interaction_dict(combined_df, num_users)

    # Step 6: Load movie titles for readable output
    movie_titles = load_movie_titles()

    print("[data_loader] ✅ Data loading complete.\n")

    return {
        "train_df"       : train_df,
        "val_df"         : val_df,
        "test_df"        : test_df,
        "graph"          : graph,
        "train_dict"     : train_dict,
        "val_dict"       : val_dict,
        "test_dict"      : test_dict,
        "all_train_dict" : all_train_dict,
        "num_users"      : num_users,
        "num_items"      : num_items,
        "movie_titles"   : movie_titles,
    }


# =============================================================================
# SECTION 9: QUICK TEST — run this file directly to verify everything works
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  TESTING data_loader.py")
    print("=" * 60)

    data = load_data()

    # ── Print summary stats ──────────────────────────────────────
    print("\n📊 DATASET SUMMARY")
    print(f"  Users       : {data['num_users']}")
    print(f"  Items       : {data['num_items']}")
    print(f"  Train edges : {len(data['train_df'])}")
    print(f"  Val edges   : {len(data['val_df'])}")
    print(f"  Test edges  : {len(data['test_df'])}")

    # ── Print graph info ─────────────────────────────────────────
    g = data["graph"]
    print(f"\n🔗 GRAPH INFO")
    print(f"  Nodes       : {g.num_nodes}")
    print(f"  Edges (dir) : {g.edge_index.shape[1]}")
    print(f"  Edge index shape: {g.edge_index.shape}")

    # ── Show a sample user's training interactions ────────────────
    sample_user = 0
    sample_items = data["train_dict"].get(sample_user, set())
    titles = data["movie_titles"]
    print(f"\n🎬 SAMPLE — User {sample_user}'s training movies:")
    for item_id in list(sample_items)[:5]:
        print(f"  item {item_id:4d} → {titles.get(item_id, 'Unknown')}")

    # ── Show a sample negative ────────────────────────────────────
    neg = sample_negative(sample_user, sample_items, data["num_items"])
    print(f"\n➖ Sample negative item for User {sample_user}: "
          f"item {neg[0]} → {titles.get(neg[0], 'Unknown')}")

    print("\n✅ data_loader.py is working correctly!")
