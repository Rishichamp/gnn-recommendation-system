# =============================================================================
# model.py
# GNN-Based Recommendation System — LightGCN Architecture
# =============================================================================
# WHAT THIS FILE DOES:
#   1. Defines the LightGCN model (He et al., 2020)
#   2. Initializes learnable embeddings for every user and item
#   3. Implements multi-layer graph convolution (message passing)
#   4. Computes BPR loss for pairwise ranking
#   5. Provides a predict() method for generating recommendations
#   6. Provides a save/load checkpoint utility
#
# PAPER: "LightGCN: Simplifying and Powering Graph Convolution Network
#         for Recommendation" — He et al., SIGIR 2020
#   Key insight: Remove feature transformation (weight matrices) and
#   non-linear activations from GCN. For recommendation, pure
#   neighborhood aggregation outperforms the full GCN.
# =============================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LGConv


# =============================================================================
# SECTION 1: LIGHTGCN MODEL
# =============================================================================

class LightGCN(nn.Module):
    """
    LightGCN: Light Graph Convolutional Network for Recommendation.

    HOW IT WORKS (step by step):
    ─────────────────────────────
    1. Every user and item starts with a random learnable embedding vector
       of size `embedding_dim` (e.g. 64 numbers per user/item).

    2. During the forward pass, we run K layers of graph convolution.
       Each layer does ONE thing: replace each node's embedding with the
       AVERAGE of its neighbors' embeddings (no weights, no activation).

         Layer 0: raw learned embeddings
         Layer 1: user sees avg of movies it watched
         Layer 2: user sees avg of users who share movies with it
         Layer 3: even richer neighborhood patterns

    3. The final embedding for each node is the MEAN across all K+1 layers.
       This multi-layer average captures structure at different distances.

    4. To predict how much user u likes item i:
         score(u, i) = dot_product(user_embedding_u, item_embedding_i)
       High score → strong recommendation.

    Args:
        num_users     (int): Total number of users in the dataset.
        num_items     (int): Total number of items in the dataset.
        embedding_dim (int): Size of each embedding vector. Default: 64.
        num_layers    (int): Number of graph convolution layers. Default: 3.

    Architecture note:
        Node IDs 0 ... num_users-1           → user nodes
        Node IDs num_users ... num_users+num_items-1 → item nodes
        (Must match the ID scheme used in data_loader.py)
    """

    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super(LightGCN, self).__init__()

        self.num_users     = num_users
        self.num_items     = num_items
        self.embedding_dim = embedding_dim
        self.num_layers    = num_layers

        # ── Embedding table ───────────────────────────────────────────────
        # One row per node (user + item), each row = embedding_dim floats.
        # This is the ONLY set of learnable parameters in LightGCN.
        # Shape: [num_users + num_items, embedding_dim]
        self.embedding = nn.Embedding(
            num_embeddings = num_users + num_items,
            embedding_dim  = embedding_dim
        )

        # ── Graph convolution layers ──────────────────────────────────────
        # LGConv = LightGCN Convolution (built into PyTorch Geometric)
        # It does: x_i = sum_j ( 1/sqrt(deg_i * deg_j) * x_j )
        # which is just a degree-normalized sum of neighbor embeddings.
        self.convs = nn.ModuleList([
            LGConv() for _ in range(num_layers)
        ])

        # ── Weight initialization ─────────────────────────────────────────
        # Xavier uniform keeps initial values in a good range so gradients
        # don't vanish or explode at the start of training.
        nn.init.xavier_uniform_(self.embedding.weight)

        print(f"[model] LightGCN initialized")
        print(f"  Users         : {num_users}")
        print(f"  Items         : {num_items}")
        print(f"  Embedding dim : {embedding_dim}")
        print(f"  GCN layers    : {num_layers}")
        print(f"  Total params  : {self.count_parameters():,}")


    # ── FORWARD PASS ─────────────────────────────────────────────────────────

    def forward(self, edge_index):
        """
        Runs the full LightGCN forward pass to produce final embeddings.

        Args:
            edge_index (Tensor): Shape [2, num_edges]. The bipartite graph
                                 from data_loader.py (both directions).

        Returns:
            user_emb (Tensor): Shape [num_users, embedding_dim]
            item_emb (Tensor): Shape [num_items, embedding_dim]

        Process:
            - Start with raw embeddings (layer 0)
            - Apply K graph conv layers, collecting output of each
            - Final embedding = mean of all K+1 layer outputs
            - Split into user and item halves
        """
        # Start: raw learned embeddings for all nodes
        # Shape: [num_users + num_items, embedding_dim]
        x = self.embedding.weight

        # Collect embeddings at each layer (including layer 0 = raw)
        all_layer_embeddings = [x]

        for conv in self.convs:
            # Each conv layer aggregates neighbor embeddings
            # x_new[i] = mean of neighbors of node i
            x = conv(x, edge_index)
            all_layer_embeddings.append(x)

        # Stack all layers: shape [num_nodes, num_layers+1, embedding_dim]
        stacked = torch.stack(all_layer_embeddings, dim=1)

        # Mean pooling across layers: shape [num_nodes, embedding_dim]
        final_embeddings = stacked.mean(dim=1)

        # Split into user and item embeddings
        user_emb = final_embeddings[:self.num_users]
        item_emb = final_embeddings[self.num_users:]

        return user_emb, item_emb


    # ── BPR LOSS ──────────────────────────────────────────────────────────────

    def bpr_loss(self, user_emb, item_emb, users, pos_items, neg_items, reg_lambda=1e-4):
        """
        Bayesian Personalized Ranking (BPR) loss.

        WHY BPR and not MSE?
            We don't want to predict exact ratings. We want the model to
            RANK items the user liked HIGHER than items they didn't.
            BPR does this directly.

        How it works:
            For each (user, positive_item, negative_item) triple:
            1. Compute score for the positive item (user liked it)
            2. Compute score for the negative item (user never touched it)
            3. Maximize: score(pos) - score(neg)
            4. Use -log(sigmoid(...)) as a smooth loss

        Args:
            user_emb  (Tensor): Full user embedding matrix [num_users, dim]
            item_emb  (Tensor): Full item embedding matrix [num_items, dim]
            users     (Tensor): Batch of user IDs          [batch_size]
            pos_items (Tensor): Positive item IDs          [batch_size]
            neg_items (Tensor): Negative item IDs          [batch_size]
            reg_lambda (float): L2 regularization strength (prevents overfitting)

        Returns:
            loss (Tensor): Scalar BPR loss value.
        """
        # Look up embeddings for this batch
        u_emb   = user_emb[users]       # [batch, dim]
        pos_emb = item_emb[pos_items]   # [batch, dim]
        neg_emb = item_emb[neg_items]   # [batch, dim]

        # Dot product scores (how much does user "match" each item?)
        pos_score = (u_emb * pos_emb).sum(dim=-1)   # [batch]
        neg_score = (u_emb * neg_emb).sum(dim=-1)   # [batch]

        # BPR loss: -mean( log( sigmoid( score_pos - score_neg ) ) )
        # We want pos_score >> neg_score, so their difference should be large
        bpr = -F.logsigmoid(pos_score - neg_score).mean()

        # L2 regularization on the RAW (layer-0) embeddings only
        # This prevents embeddings from growing too large (overfitting)
        raw_u   = self.embedding(users)
        raw_pos = self.embedding(pos_items + self.num_users)
        raw_neg = self.embedding(neg_items + self.num_users)
        reg = reg_lambda * (
            raw_u.norm(2).pow(2) +
            raw_pos.norm(2).pow(2) +
            raw_neg.norm(2).pow(2)
        ) / users.shape[0]

        return bpr + reg


    # ── PREDICT SCORES ────────────────────────────────────────────────────────

    def predict(self, edge_index, user_ids=None):
        """
        Generates recommendation scores for users.

        Args:
            edge_index (Tensor): The training graph.
            user_ids   (Tensor or None): Specific user IDs to score.
                        If None, scores ALL users against ALL items.

        Returns:
            scores (Tensor):
                If user_ids given → shape [len(user_ids), num_items]
                If None           → shape [num_users, num_items]

        Each row = one user, each column = one item, value = match score.
        Higher score = stronger recommendation.
        """
        self.eval()
        with torch.no_grad():
            user_emb, item_emb = self.forward(edge_index)

            if user_ids is not None:
                user_emb = user_emb[user_ids]

            # Matrix multiply: [num_users, dim] x [dim, num_items]
            # → [num_users, num_items] score matrix
            scores = torch.matmul(user_emb, item_emb.T)

        return scores


    # ── TOP-K RECOMMENDATIONS ─────────────────────────────────────────────────

    def recommend(self, edge_index, user_id, seen_items, top_k=10):
        """
        Returns top-K recommended item IDs for a single user.

        Args:
            edge_index (Tensor): The training graph.
            user_id    (int): The user to generate recommendations for.
            seen_items (set): Item IDs the user already interacted with.
                              These will be excluded from recommendations.
            top_k      (int): Number of recommendations to return.

        Returns:
            list of int: Top-K item IDs (0-indexed), ranked by score.
        """
        scores = self.predict(edge_index, user_ids=torch.tensor([user_id]))
        scores = scores.squeeze(0)  # shape: [num_items]

        # Mask out items the user has already seen
        for item_id in seen_items:
            scores[item_id] = float("-inf")

        # Get top-K indices sorted by descending score
        top_items = scores.argsort(descending=True)[:top_k].tolist()
        return top_items


    # ── UTILITIES ─────────────────────────────────────────────────────────────

    def count_parameters(self):
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path):
        """
        Saves model weights and config to disk.

        Args:
            path (str): File path to save to (e.g. 'results/best_model.pt')
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "state_dict"    : self.state_dict(),
            "num_users"     : self.num_users,
            "num_items"     : self.num_items,
            "embedding_dim" : self.embedding_dim,
            "num_layers"    : self.num_layers,
        }, path)
        print(f"[model] Saved checkpoint → {path}")

    @classmethod
    def load(cls, path, device="cpu"):
        """
        Loads a saved model from disk.

        Args:
            path   (str): Path to the saved .pt file.
            device (str): 'cpu' or 'cuda'.

        Returns:
            LightGCN model with loaded weights.

        Example:
            model = LightGCN.load("results/best_model.pt")
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            num_users     = checkpoint["num_users"],
            num_items     = checkpoint["num_items"],
            embedding_dim = checkpoint["embedding_dim"],
            num_layers    = checkpoint["num_layers"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        print(f"[model] Loaded checkpoint ← {path}")
        return model


# =============================================================================
# SECTION 2: QUICK TEST — run this file directly to verify the model
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  TESTING model.py")
    print("=" * 60)

    # ── Fake data to test without needing real dataset ────────────────
    NUM_USERS     = 100
    NUM_ITEMS     = 200
    EMBEDDING_DIM = 64
    NUM_LAYERS    = 3
    BATCH_SIZE    = 32

    # Create a small random graph (bipartite: users 0-99, items 100-299)
    num_edges = 500
    src_users  = torch.randint(0, NUM_USERS, (num_edges,))
    dst_items  = torch.randint(NUM_USERS, NUM_USERS + NUM_ITEMS, (num_edges,))
    edge_index = torch.stack([
        torch.cat([src_users, dst_items]),
        torch.cat([dst_items, src_users])
    ], dim=0)

    # ── Build model ───────────────────────────────────────────────────
    print()
    model = LightGCN(NUM_USERS, NUM_ITEMS, EMBEDDING_DIM, NUM_LAYERS)

    # ── Forward pass ──────────────────────────────────────────────────
    print("\n[test] Running forward pass ...")
    user_emb, item_emb = model(edge_index)
    print(f"  user_emb shape : {user_emb.shape}")   # [100, 64]
    print(f"  item_emb shape : {item_emb.shape}")   # [200, 64]
    assert user_emb.shape == (NUM_USERS, EMBEDDING_DIM)
    assert item_emb.shape == (NUM_ITEMS, EMBEDDING_DIM)
    print("  ✅ Forward pass correct")

    # ── BPR loss ──────────────────────────────────────────────────────
    print("\n[test] Computing BPR loss ...")
    users     = torch.randint(0, NUM_USERS, (BATCH_SIZE,))
    pos_items = torch.randint(0, NUM_ITEMS, (BATCH_SIZE,))
    neg_items = torch.randint(0, NUM_ITEMS, (BATCH_SIZE,))
    loss = model.bpr_loss(user_emb, item_emb, users, pos_items, neg_items)
    print(f"  BPR loss value : {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    print("  ✅ BPR loss correct")

    # ── Predict ───────────────────────────────────────────────────────
    print("\n[test] Generating predictions ...")
    scores = model.predict(edge_index, user_ids=torch.tensor([0, 1, 2]))
    print(f"  Score matrix shape : {scores.shape}")  # [3, 200]
    assert scores.shape == (3, NUM_ITEMS)
    print("  ✅ Predict correct")

    # ── Recommend ─────────────────────────────────────────────────────
    print("\n[test] Top-10 recommendations for User 0 ...")
    seen = {5, 12, 33, 78}
    recs = model.recommend(edge_index, user_id=0, seen_items=seen, top_k=10)
    print(f"  Recommendations : {recs}")
    assert len(recs) == 10
    assert not any(r in seen for r in recs), "Should not recommend seen items"
    print("  ✅ Recommend correct (no seen items in results)")

    # ── Save & load ───────────────────────────────────────────────────
    print("\n[test] Save & load checkpoint ...")
    model.save("results/test_checkpoint.pt")
    loaded = LightGCN.load("results/test_checkpoint.pt")
    user_emb2, _ = loaded(edge_index)
    assert torch.allclose(user_emb, user_emb2), "Loaded model should give same output"
    print("  ✅ Save/load correct")

    print("\n✅ model.py is working correctly!")
