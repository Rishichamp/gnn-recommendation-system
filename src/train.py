# =============================================================================
# train.py
# GNN-Based Recommendation System — Full Training Pipeline
# =============================================================================
# WHAT THIS FILE DOES:
#   1. Loads data using data_loader.py
#   2. Initializes the LightGCN model from model.py
#   3. Runs a full training loop with mini-batch BPR loss
#   4. Validates after every epoch using Recall@20
#   5. Implements early stopping to prevent overfitting
#   6. Saves the best model checkpoint automatically
#   7. Logs and plots the loss + metric curves
#
# HOW TO RUN:
#   python src/train.py
#   python src/train.py --epochs 100 --lr 0.001 --embedding_dim 64
# =============================================================================

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (works without a display)
import matplotlib.pyplot as plt

# Add project root to path so imports work from anywhere
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_data, sample_negative
from src.model import LightGCN


# =============================================================================
# SECTION 1: HYPERPARAMETERS & CONFIG
# =============================================================================

def get_config():
    """
    All training hyperparameters in one place.

    WHY THESE DEFAULTS?
        lr=0.001         — Adam's default; works well for LightGCN
        embedding_dim=64 — balance between expressiveness and speed
        num_layers=3     — standard for LightGCN on MovieLens-100K
        batch_size=1024  — large enough for stable gradients, fits in RAM
        epochs=200       — enough for convergence; early stopping will cut short
        patience=20      — stop if no improvement for 20 consecutive epochs
        reg_lambda=1e-4  — light L2 regularization; prevents overfitting
        top_k=20         — Recall@20 is the standard benchmark metric
    """
    parser = argparse.ArgumentParser(description="Train LightGCN Recommendation Model")

    parser.add_argument("--embedding_dim", type=int,   default=64)
    parser.add_argument("--num_layers",    type=int,   default=3)
    parser.add_argument("--lr",            type=float, default=0.001)
    parser.add_argument("--batch_size",    type=int,   default=1024)
    parser.add_argument("--epochs",        type=int,   default=200)
    parser.add_argument("--patience",      type=int,   default=20)
    parser.add_argument("--reg_lambda",    type=float, default=1e-4)
    parser.add_argument("--top_k",         type=int,   default=20)
    parser.add_argument("--min_rating",    type=float, default=4.0)
    parser.add_argument("--checkpoint",    type=str,   default="results/best_model.pt")
    parser.add_argument("--plots_dir",     type=str,   default="results/plots")
    parser.add_argument("--seed",          type=int,   default=42)

    # Parse known args only (safe to call from notebooks too)
    args, _ = parser.parse_known_args()
    return args


# =============================================================================
# SECTION 2: BATCH GENERATOR
# =============================================================================

def generate_batch(train_dict, num_users, num_items, batch_size):
    """
    Generates one mini-batch of (user, positive_item, negative_item) triples
    for BPR training.

    WHY TRIPLES?
        BPR loss needs three things per sample:
          - A user
          - An item that user LIKED (positive)
          - An item that user NEVER touched (negative)
        The model learns: score(user, positive) > score(user, negative)

    HOW NEGATIVE SAMPLING WORKS:
        We randomly pick an item the user has NOT interacted with.
        We assume they don't prefer it (implicit feedback assumption).
        This is approximate — the user might actually like it, but
        statistically most unrated items are genuinely not preferred.

    Args:
        train_dict  (dict): {user_id: set(item_ids)} from data_loader
        num_users   (int):  Total number of users
        num_items   (int):  Total number of items
        batch_size  (int):  How many triples per batch

    Returns:
        users     (Tensor): [batch_size] user IDs
        pos_items (Tensor): [batch_size] positive item IDs
        neg_items (Tensor): [batch_size] negative item IDs
    """
    users_list     = []
    pos_items_list = []
    neg_items_list = []

    # Randomly sample batch_size users (with replacement)
    sampled_users = np.random.randint(0, num_users, batch_size)

    for user_id in sampled_users:
        user_items = train_dict.get(int(user_id), set())

        if not user_items:
            # User has no training interactions — skip
            # Use a dummy sample so batch size stays consistent
            users_list.append(0)
            pos_items_list.append(0)
            neg_items_list.append(1)
            continue

        # Randomly pick one item this user liked
        pos_item = int(np.random.choice(list(user_items)))

        # Sample one item this user has NOT seen
        neg_item = sample_negative(user_id, user_items, num_items, n_samples=1)[0]

        users_list.append(int(user_id))
        pos_items_list.append(pos_item)
        neg_items_list.append(neg_item)

    return (
        torch.tensor(users_list,     dtype=torch.long),
        torch.tensor(pos_items_list, dtype=torch.long),
        torch.tensor(neg_items_list, dtype=torch.long),
    )


# =============================================================================
# SECTION 3: VALIDATION — Recall@K
# =============================================================================

def evaluate_recall(model, edge_index, eval_dict, train_dict,
                    num_items, top_k=20, max_users=500):
    """
    Computes Recall@K on the validation or test set.

    WHAT IS RECALL@K?
        For each user, we generate top-K recommendations (excluding
        items already seen in training). We then check: did any of the
        user's held-out items appear in our top-K list?

        Recall@K = |recommended_top_K ∩ ground_truth| / |ground_truth|

        Averaged across all users.

    WHY NOT EVALUATE ON ALL USERS?
        For speed during training, we evaluate on max_users users.
        For final test evaluation, set max_users=None to use all.

    Args:
        model      (LightGCN): The model to evaluate.
        edge_index (Tensor):   Training graph.
        eval_dict  (dict):     {user_id: set(item_ids)} — ground truth
        train_dict (dict):     {user_id: set(item_ids)} — to exclude from recs
        num_items  (int):      Total items.
        top_k      (int):      K for Recall@K.
        max_users  (int):      Max users to evaluate (None = all).

    Returns:
        float: Average Recall@K across evaluated users.
    """
    model.eval()

    # Get users who have ground truth labels
    eval_users = [u for u in eval_dict if eval_dict[u]]
    if max_users:
        eval_users = eval_users[:max_users]

    if not eval_users:
        return 0.0

    eval_users_tensor = torch.tensor(eval_users, dtype=torch.long)

    with torch.no_grad():
        # Score matrix: [num_eval_users, num_items]
        scores = model.predict(edge_index, user_ids=eval_users_tensor)

    recall_scores = []

    for idx, user_id in enumerate(eval_users):
        user_scores = scores[idx].clone()

        # Mask out items seen during training (already interacted with)
        seen = train_dict.get(user_id, set())
        for item_id in seen:
            if item_id < num_items:
                user_scores[item_id] = float("-inf")

        # Get top-K recommended items
        top_k_items = set(user_scores.argsort(descending=True)[:top_k].tolist())

        # Ground truth items for this user
        ground_truth = eval_dict.get(user_id, set())

        # Recall = intersection / ground truth size
        hits = len(top_k_items & ground_truth)
        recall = hits / len(ground_truth) if ground_truth else 0.0
        recall_scores.append(recall)

    return float(np.mean(recall_scores))


# =============================================================================
# SECTION 4: PLOT TRAINING CURVES
# =============================================================================

def plot_curves(train_losses, val_recalls, plots_dir, top_k):
    """
    Saves training loss and validation Recall@K plots to disk.

    These plots are essential for your portfolio — they show:
      1. That the model actually learned (loss decreasing)
      2. How well it recommends (Recall@K increasing)
      3. Where early stopping kicked in

    Args:
        train_losses (list): Loss value per epoch.
        val_recalls  (list): Recall@K per epoch.
        plots_dir    (str):  Directory to save plots.
        top_k        (int):  K value for the recall label.
    """
    os.makedirs(plots_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # ── Plot 1: Training Loss ──────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, color="#2563EB", linewidth=2, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BPR Loss")
    plt.title("Training Loss (LightGCN)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(plots_dir, "training_loss.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"[train] Loss curve saved → {loss_path}")

    # ── Plot 2: Validation Recall@K ───────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, val_recalls, color="#16A34A", linewidth=2,
             label=f"Val Recall@{top_k}")
    plt.xlabel("Epoch")
    plt.ylabel(f"Recall@{top_k}")
    plt.title(f"Validation Recall@{top_k} (LightGCN)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    recall_path = os.path.join(plots_dir, f"val_recall_at_{top_k}.png")
    plt.savefig(recall_path, dpi=150)
    plt.close()
    print(f"[train] Recall curve saved → {recall_path}")

    # ── Plot 3: Combined (for README / portfolio) ─────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.plot(epochs, train_losses, color="#2563EB", linewidth=2)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BPR Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, val_recalls, color="#16A34A", linewidth=2)
    ax2.set_title(f"Validation Recall@{top_k}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(f"Recall@{top_k}")
    ax2.grid(alpha=0.3)

    plt.suptitle("LightGCN Training — MovieLens 100K", fontsize=13, y=1.02)
    plt.tight_layout()
    combined_path = os.path.join(plots_dir, "training_summary.png")
    plt.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[train] Combined curve saved → {combined_path}")


# =============================================================================
# SECTION 5: EARLY STOPPING HELPER
# =============================================================================

class EarlyStopping:
    """
    Monitors a metric and stops training if it doesn't improve.

    WHY EARLY STOPPING?
        Without it, the model will eventually memorize the training data
        (overfitting) and perform worse on new users and items.
        Early stopping keeps the best-performing version of the model.

    HOW IT WORKS:
        - After each epoch, check validation Recall@K
        - If it improves → save the model, reset patience counter
        - If it doesn't improve → increment counter
        - If counter reaches `patience` → stop training

    Args:
        patience  (int):   How many epochs to wait before stopping.
        min_delta (float): Minimum improvement to count as "better".
        checkpoint(str):   Where to save the best model.
    """

    def __init__(self, patience=20, min_delta=1e-4, checkpoint="results/best_model.pt"):
        self.patience    = patience
        self.min_delta   = min_delta
        self.checkpoint  = checkpoint
        self.best_score  = -float("inf")
        self.counter     = 0
        self.best_epoch  = 0
        self.should_stop = False

    def step(self, score, model, epoch):
        """
        Call once per epoch with the current validation score.

        Args:
            score (float): Current epoch's Recall@K.
            model (LightGCN): The model to save if this is the best epoch.
            epoch (int): Current epoch number (for logging).

        Returns:
            bool: True if a new best was found, False otherwise.
        """
        if score > self.best_score + self.min_delta:
            # New best — save model and reset counter
            self.best_score = score
            self.best_epoch = epoch
            self.counter    = 0
            model.save(self.checkpoint)
            return True   # Improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False  # No improvement


# =============================================================================
# SECTION 6: MAIN TRAINING LOOP
# =============================================================================

def train(config=None):
    """
    Full training pipeline — data loading → model init → train → evaluate.

    Args:
        config: argparse Namespace with hyperparameters.
                If None, uses defaults from get_config().

    Returns:
        dict: Training results summary containing:
              best_recall, best_epoch, train_losses, val_recalls
    """
    if config is None:
        config = get_config()

    # ── Reproducibility ───────────────────────────────────────────────────
    # Setting seeds ensures your results are the same every run.
    # Essential when reporting results in a portfolio or paper.
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print("=" * 60)
    print("  LightGCN TRAINING — MovieLens 100K")
    print("=" * 60)
    print(f"  Embedding dim : {config.embedding_dim}")
    print(f"  GCN layers    : {config.num_layers}")
    print(f"  Learning rate : {config.lr}")
    print(f"  Batch size    : {config.batch_size}")
    print(f"  Max epochs    : {config.epochs}")
    print(f"  Patience      : {config.patience}")
    print(f"  Eval metric   : Recall@{config.top_k}")
    print()

    # ── Step 1: Load data ─────────────────────────────────────────────────
    data = load_data(min_rating=config.min_rating)

    num_users  = data["num_users"]
    num_items  = data["num_items"]
    graph      = data["graph"]
    train_dict = data["train_dict"]
    val_dict   = data["val_dict"]
    edge_index = graph.edge_index

    # ── Step 2: Build model ───────────────────────────────────────────────
    print()
    model = LightGCN(
        num_users     = num_users,
        num_items     = num_items,
        embedding_dim = config.embedding_dim,
        num_layers    = config.num_layers,
    )

    # ── Step 3: Optimizer ─────────────────────────────────────────────────
    # Adam: adaptive learning rate optimizer — standard choice for GNNs.
    # It adjusts the learning rate for each parameter individually,
    # leading to faster and more stable convergence than plain SGD.
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # ── Step 4: Early stopping tracker ───────────────────────────────────
    early_stopper = EarlyStopping(
        patience   = config.patience,
        checkpoint = config.checkpoint,
    )

    # ── Step 5: Training history ──────────────────────────────────────────
    train_losses = []
    val_recalls  = []

    print("\n" + "─" * 60)
    print(f"{'Epoch':>6} {'Loss':>10} {'Recall@'+str(config.top_k):>12} "
          f"{'Best':>8} {'Time':>8}")
    print("─" * 60)

    # ── Step 6: Epoch loop ────────────────────────────────────────────────
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        model.train()

        # ── Mini-batch training ───────────────────────────────────────
        # We do multiple gradient steps per epoch to process the full
        # dataset approximately once (like one "pass" through the data).
        # steps_per_epoch ≈ num_training_interactions / batch_size
        steps_per_epoch = max(1, len(train_dict) * 5 // config.batch_size)
        epoch_loss = 0.0

        for _ in range(steps_per_epoch):
            # Sample a batch of (user, pos_item, neg_item) triples
            users, pos_items, neg_items = generate_batch(
                train_dict, num_users, num_items, config.batch_size
            )

            optimizer.zero_grad()

            # Forward pass: get embeddings
            user_emb, item_emb = model(edge_index)

            # Compute BPR loss
            loss = model.bpr_loss(
                user_emb, item_emb,
                users, pos_items, neg_items,
                reg_lambda=config.reg_lambda
            )

            # Backward pass: compute gradients
            loss.backward()

            # Update weights
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / steps_per_epoch
        train_losses.append(avg_loss)

        # ── Validation ────────────────────────────────────────────────
        # Evaluate every epoch to track progress and trigger early stopping.
        val_recall = evaluate_recall(
            model, edge_index, val_dict, train_dict,
            num_items, top_k=config.top_k, max_users=500
        )
        val_recalls.append(val_recall)

        # ── Early stopping check ──────────────────────────────────────
        improved = early_stopper.step(val_recall, model, epoch)
        best_marker = " ◀ best" if improved else ""

        epoch_time = time.time() - epoch_start

        print(f"{epoch:>6} {avg_loss:>10.4f} {val_recall:>12.4f} "
              f"{early_stopper.best_score:>8.4f} "
              f"{epoch_time:>6.1f}s{best_marker}")

        # ── Stop if no improvement for `patience` epochs ──────────────
        if early_stopper.should_stop:
            print(f"\n[train] Early stopping at epoch {epoch}. "
                  f"Best was epoch {early_stopper.best_epoch} "
                  f"(Recall@{config.top_k}={early_stopper.best_score:.4f})")
            break

    print("─" * 60)

    # ── Step 7: Plot training curves ──────────────────────────────────────
    plot_curves(train_losses, val_recalls, config.plots_dir, config.top_k)

    # ── Step 8: Final summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best Recall@{config.top_k} : {early_stopper.best_score:.4f}")
    print(f"  Best epoch       : {early_stopper.best_epoch}")
    print(f"  Model saved to   : {config.checkpoint}")
    print(f"{'='*60}\n")

    return {
        "best_recall"  : early_stopper.best_score,
        "best_epoch"   : early_stopper.best_epoch,
        "train_losses" : train_losses,
        "val_recalls"  : val_recalls,
        "num_users"    : num_users,
        "num_items"    : num_items,
    }


# =============================================================================
# SECTION 7: ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = train()
