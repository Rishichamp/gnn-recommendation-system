# =============================================================================
# main.py
# GNN-Based Recommendation System — Master Entry Point
# =============================================================================
# WHAT THIS FILE DOES:
#   This is the single file you run to do ANYTHING with your project.
#   It ties data_loader → model → train → evaluate into one clean pipeline.
#
# MODES (pick one when running):
#   python main.py --mode train          → train the model from scratch
#   python main.py --mode evaluate       → evaluate saved checkpoint on test set
#   python main.py --mode recommend      → recommend movies for a specific user
#   python main.py --mode demo           → interactive recommendation terminal
#   python main.py --mode pipeline       → train + evaluate in one shot (default)
#
# EXAMPLES:
#   python main.py                                    ← full pipeline (train+eval)
#   python main.py --mode recommend --user_id 42      ← top-10 for user 42
#   python main.py --mode demo                        ← interactive terminal
#   python main.py --mode train --epochs 200 --lr 0.001
#   python main.py --mode evaluate --checkpoint results/best_model.pt
# =============================================================================

import os
import sys
import json
import argparse
import types

# ── Make sure src/ is importable from the project root ───────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data
from src.model       import LightGCN
from src.train       import train     as run_train
from src.evaluate    import evaluate  as run_evaluate, show_recommendations


# =============================================================================
# SECTION 1: ARGUMENT PARSER
# =============================================================================

def parse_args():
    """
    Parses all command-line arguments for every mode in one place.

    WHY ONE FILE FOR EVERYTHING?
        Clean projects have a single entry point. Recruiters and reviewers
        clone your repo and run ONE command. If they have to figure out
        which file to run first, they lose interest fast.

        python main.py          → shows what this project does
        python main.py --help   → shows every available option
    """
    parser = argparse.ArgumentParser(
        description="GNN-Based Movie Recommendation System (LightGCN)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
EXAMPLES:
  python main.py                              # full train + evaluate pipeline
  python main.py --mode train                 # train only
  python main.py --mode evaluate              # evaluate saved checkpoint
  python main.py --mode recommend --user_id 5 # recommendations for user 5
  python main.py --mode demo                  # interactive terminal
        """
    )

    # ── Mode ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--mode", type=str, default="pipeline",
        choices=["pipeline", "train", "evaluate", "recommend", "demo"],
        help=(
            "pipeline  : train then evaluate (default)\n"
            "train     : train from scratch\n"
            "evaluate  : evaluate saved checkpoint\n"
            "recommend : show top-K for one user\n"
            "demo      : interactive recommendation terminal"
        )
    )

    # ── Model hyperparameters ─────────────────────────────────────────────
    parser.add_argument("--embedding_dim", type=int,   default=64,
                        help="Embedding vector size (default: 64)")
    parser.add_argument("--num_layers",    type=int,   default=3,
                        help="Number of GCN layers (default: 3)")

    # ── Training settings ─────────────────────────────────────────────────
    parser.add_argument("--lr",         type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--batch_size", type=int,   default=1024,
                        help="Training batch size (default: 1024)")
    parser.add_argument("--epochs",     type=int,   default=200,
                        help="Max training epochs (default: 200)")
    parser.add_argument("--patience",   type=int,   default=20,
                        help="Early stopping patience (default: 20)")
    parser.add_argument("--reg_lambda", type=float, default=1e-4,
                        help="L2 regularization strength (default: 1e-4)")
    parser.add_argument("--seed",       type=int,   default=42,
                        help="Random seed for reproducibility (default: 42)")

    # ── Evaluation settings ───────────────────────────────────────────────
    parser.add_argument("--top_k",      type=int,   default=20,
                        help="K for Recall@K / NDCG@K (default: 20)")

    # ── File paths ────────────────────────────────────────────────────────
    parser.add_argument("--checkpoint", type=str,
                        default="results/best_model.pt",
                        help="Path to save/load model checkpoint")
    parser.add_argument("--plots_dir",  type=str,
                        default="results/plots",
                        help="Directory for training curve plots")
    parser.add_argument("--report_dir", type=str,
                        default="results",
                        help="Directory for evaluation JSON report")

    # ── Data settings ─────────────────────────────────────────────────────
    parser.add_argument("--min_rating", type=float, default=4.0,
                        help="Minimum rating to treat as positive (default: 4.0)")

    # ── Recommend mode ────────────────────────────────────────────────────
    parser.add_argument("--user_id",      type=int, default=0,
                        help="User ID for --mode recommend (default: 0)")
    parser.add_argument("--sample_users", type=int, default=5,
                        help="Number of sample users to show in evaluate mode")

    return parser.parse_args()


# =============================================================================
# SECTION 2: BANNER
# =============================================================================

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║      GNN-Based Movie Recommendation System                   ║
║      Architecture : LightGCN  (He et al., SIGIR 2020)        ║
║      Dataset      : MovieLens 100K                           ║
║      Framework    : PyTorch Geometric                        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


# =============================================================================
# SECTION 3: TRAIN MODE
# =============================================================================

def mode_train(args):
    """
    Trains the LightGCN model from scratch.

    Calls train.py's train() function with all hyperparameters
    forwarded from the command line. When done, the best checkpoint
    is saved and training curves are plotted.

    Args:
        args: Parsed argparse Namespace.

    Returns:
        dict: Training results (best_recall, best_epoch, losses, recalls)
    """
    print("── MODE: TRAIN ─────────────────────────────────────────────\n")

    # Forward all args to the training config namespace
    train_cfg = types.SimpleNamespace(
        embedding_dim = args.embedding_dim,
        num_layers    = args.num_layers,
        lr            = args.lr,
        batch_size    = args.batch_size,
        epochs        = args.epochs,
        patience      = args.patience,
        reg_lambda    = args.reg_lambda,
        top_k         = args.top_k,
        min_rating    = args.min_rating,
        checkpoint    = args.checkpoint,
        plots_dir     = args.plots_dir,
        seed          = args.seed,
    )

    results = run_train(train_cfg)

    print(f"\n✅ Training complete.")
    print(f"   Best Recall@{args.top_k} : {results['best_recall']:.4f}")
    print(f"   Best epoch       : {results['best_epoch']}")
    print(f"   Checkpoint       : {args.checkpoint}")
    print(f"   Plots            : {args.plots_dir}/\n")

    return results


# =============================================================================
# SECTION 4: EVALUATE MODE
# =============================================================================

def mode_evaluate(args):
    """
    Evaluates a saved model checkpoint on the held-out test set.

    Prints Recall@K, NDCG@K, Precision@K for K = 10, 20, 50.
    Shows sample recommendations with human-readable movie titles.
    Saves a JSON report to results/eval_results.json.

    Args:
        args: Parsed argparse Namespace.

    Returns:
        dict: Evaluation results by K value.
    """
    print("── MODE: EVALUATE ──────────────────────────────────────────\n")

    eval_cfg = types.SimpleNamespace(
        checkpoint    = args.checkpoint,
        top_k         = args.top_k,
        min_rating    = args.min_rating,
        report_dir    = args.report_dir,
        sample_users  = args.sample_users,
    )

    results = run_evaluate(eval_cfg)
    return results


# =============================================================================
# SECTION 5: RECOMMEND MODE
# =============================================================================

def mode_recommend(args):
    """
    Generates and prints top-K recommendations for one specific user.

    Useful for quickly checking what your model recommends for any user.
    Loads the saved checkpoint and shows:
      - Movies the user trained on
      - Top-K recommendations (with movie titles)
      - The held-out test ground truth

    Args:
        args: Parsed argparse Namespace.

    Returns:
        list: Top-K recommended item IDs.
    """
    print(f"── MODE: RECOMMEND  (User {args.user_id}) ──────────────────────\n")

    # ── Load data ─────────────────────────────────────────────────────────
    data       = load_data(min_rating=args.min_rating)
    edge_index = data["graph"].edge_index
    num_users  = data["num_users"]

    # ── Validate user_id ──────────────────────────────────────────────────
    if args.user_id >= num_users:
        print(f"❌ user_id {args.user_id} is out of range.")
        print(f"   Valid range: 0 to {num_users - 1}")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("   Run:  python main.py --mode train  first.")
        sys.exit(1)

    model = LightGCN.load(args.checkpoint)

    # ── Show recommendations ──────────────────────────────────────────────
    show_recommendations(
        model      = model,
        edge_index = edge_index,
        data       = data,
        user_ids   = [args.user_id],
        top_k      = args.top_k,
    )

    # ── Also return the raw list ──────────────────────────────────────────
    seen = (
        data["train_dict"].get(args.user_id, set()) |
        data["val_dict"].get(args.user_id, set())
    )
    recs = model.recommend(edge_index, args.user_id, seen, top_k=args.top_k)
    return recs


# =============================================================================
# SECTION 6: INTERACTIVE DEMO MODE
# =============================================================================

def mode_demo(args):
    """
    Launches an interactive terminal where you can type a user ID
    and instantly see their top-10 movie recommendations.

    This is your 'wow factor' feature for portfolio demos and interviews.
    You can show this live — type a number, see personalised recommendations.

    HOW TO USE:
        python main.py --mode demo
        → Enter a user ID (0–942) and press Enter
        → Type 'q' or 'quit' to exit
        → Type 'random' for a random user
        → Type 'info <user_id>' to see that user's training history

    Args:
        args: Parsed argparse Namespace.
    """
    print("── MODE: INTERACTIVE DEMO ──────────────────────────────────\n")

    # ── Load everything once ──────────────────────────────────────────────
    print("Loading data and model...")
    data       = load_data(min_rating=args.min_rating)
    edge_index = data["graph"].edge_index
    num_users  = data["num_users"]
    num_items  = data["num_items"]
    titles     = data["movie_titles"]
    train_dict = data["train_dict"]
    val_dict   = data["val_dict"]
    test_dict  = data["test_dict"]

    if not os.path.exists(args.checkpoint):
        print(f"\n❌ No checkpoint found at: {args.checkpoint}")
        print("   Train first:  python main.py --mode train")
        sys.exit(1)

    model = LightGCN.load(args.checkpoint)
    model.eval()

    # ── Welcome message ───────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════╗
║        🎬  MOVIE RECOMMENDATION DEMO                 ║
║                                                      ║
║  Dataset : MovieLens 100K                            ║
║  Model   : LightGCN ({data['num_users']} users, {data['num_items']} items)          ║
║                                                      ║
║  Commands:                                           ║
║    <number>         → recommendations for that user  ║
║    random           → random user                    ║
║    info <user_id>   → show user's watch history      ║
║    compare <u1> <u2>→ compare two users              ║
║    q / quit         → exit                           ║
╚══════════════════════════════════════════════════════╝
""")

    import random as rnd

    while True:
        try:
            user_input = input("  Enter user ID (0–{:d}): ".format(num_users - 1)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        # ── Quit ─────────────────────────────────────────────────────────
        if user_input.lower() in ("q", "quit", "exit"):
            print("\nGoodbye! 👋")
            break

        # ── Random user ───────────────────────────────────────────────────
        if user_input.lower() == "random":
            user_id = rnd.randint(0, num_users - 1)
            print(f"\n  → Random user selected: {user_id}")
        
        # ── Info command ─────────────────────────────────────────────────
        elif user_input.lower().startswith("info"):
            parts = user_input.split()
            if len(parts) < 2:
                print("  Usage: info <user_id>")
                continue
            try:
                uid = int(parts[1])
            except ValueError:
                print("  Invalid user ID.")
                continue
            if uid >= num_users:
                print(f"  User ID must be 0–{num_users - 1}")
                continue
            _print_user_info(uid, train_dict, val_dict, test_dict, titles)
            continue

        # ── Compare command ───────────────────────────────────────────────
        elif user_input.lower().startswith("compare"):
            parts = user_input.split()
            if len(parts) < 3:
                print("  Usage: compare <user_id_1> <user_id_2>")
                continue
            try:
                u1, u2 = int(parts[1]), int(parts[2])
            except ValueError:
                print("  Invalid user IDs.")
                continue
            _compare_users(model, edge_index, u1, u2, data, top_k=10)
            continue

        # ── Specific user ─────────────────────────────────────────────────
        else:
            try:
                user_id = int(user_input)
            except ValueError:
                print(f"  ❌ Not a valid command or user ID: '{user_input}'")
                print("     Type a number (0–{}) or 'random' or 'q'".format(num_users - 1))
                continue

            if user_id < 0 or user_id >= num_users:
                print(f"  ❌ User ID must be between 0 and {num_users - 1}")
                continue

        # ── Generate and show recommendations ─────────────────────────────
        show_recommendations(
            model      = model,
            edge_index = edge_index,
            data       = data,
            user_ids   = [user_id],
            top_k      = 10,
        )


def _print_user_info(user_id, train_dict, val_dict, test_dict, titles):
    """Prints a user's full interaction history across all splits."""
    train_items = train_dict.get(user_id, set())
    val_items   = val_dict.get(user_id, set())
    test_items  = test_dict.get(user_id, set())

    print(f"\n  📋 USER {user_id} — INTERACTION HISTORY")
    print("  " + "─"*50)

    if train_items:
        print(f"\n  Training movies ({len(train_items)}):")
        for i, item_id in enumerate(list(train_items)[:8]):
            print(f"    [{item_id:>4}] {titles.get(item_id, 'Unknown')}")
        if len(train_items) > 8:
            print(f"    ... and {len(train_items) - 8} more")

    if val_items:
        print(f"\n  Validation movie ({len(val_items)}):")
        for item_id in val_items:
            print(f"    [{item_id:>4}] {titles.get(item_id, 'Unknown')}")

    if test_items:
        print(f"\n  Test movie — held out for evaluation ({len(test_items)}):")
        for item_id in test_items:
            print(f"    [{item_id:>4}] {titles.get(item_id, 'Unknown')}")

    print()


def _compare_users(model, edge_index, u1, u2, data, top_k=10):
    """
    Compares top-K recommendations for two users side by side.

    Great for demonstrating PERSONALISATION in your demo —
    two different users get different recommendations from the same model.
    """
    titles     = data["movie_titles"]
    train_dict = data["train_dict"]
    val_dict   = data["val_dict"]
    num_users  = data["num_users"]

    if u1 >= num_users or u2 >= num_users:
        print(f"  User IDs must be 0–{num_users - 1}")
        return

    seen1 = train_dict.get(u1, set()) | val_dict.get(u1, set())
    seen2 = train_dict.get(u2, set()) | val_dict.get(u2, set())

    recs1 = model.recommend(edge_index, u1, seen1, top_k=top_k)
    recs2 = model.recommend(edge_index, u2, seen2, top_k=top_k)

    print(f"\n  🔀 COMPARISON — User {u1}  vs  User {u2}")
    print("  " + "─"*70)
    print(f"  {'Rank':<5} {'User ' + str(u1) + ' Recommendations':<33} "
          f"{'User ' + str(u2) + ' Recommendations':<33}")
    print("  " + "─"*70)

    for rank, (i1, i2) in enumerate(zip(recs1, recs2), start=1):
        t1 = titles.get(i1, f"Movie {i1}")[:28]
        t2 = titles.get(i2, f"Movie {i2}")[:28]
        same = "← same" if i1 == i2 else ""
        print(f"  {rank:<5} {t1:<33} {t2:<33} {same}")

    overlap = set(recs1) & set(recs2)
    print(f"\n  Overlap: {len(overlap)}/{top_k} items in common "
          f"({'personalised!' if len(overlap) < top_k // 2 else 'similar tastes'})\n")


# =============================================================================
# SECTION 7: PIPELINE MODE (train + evaluate in one shot)
# =============================================================================

def mode_pipeline(args):
    """
    Runs the full pipeline: train → evaluate → show sample recommendations.

    This is the default mode. Running:
        python main.py
    will train from scratch, then immediately evaluate on the test set
    and show you a results table + sample recommendations.

    Args:
        args: Parsed argparse Namespace.

    Returns:
        dict: {'train': train_results, 'eval': eval_results}
    """
    print("── MODE: FULL PIPELINE (train → evaluate) ──────────────────\n")

    # ── Train ─────────────────────────────────────────────────────────────
    train_results = mode_train(args)

    print("\n" + "─"*62 + "\n")

    # ── Evaluate ──────────────────────────────────────────────────────────
    eval_results = mode_evaluate(args)

    # ── Final summary ─────────────────────────────────────────────────────
    recall_20 = eval_results.get(20, {}).get("recall", 0)
    ndcg_20   = eval_results.get(20, {}).get("ndcg", 0)

    print("\n" + "═"*62)
    print("  🏁  PIPELINE COMPLETE — FINAL SUMMARY")
    print("═"*62)
    print(f"  Training")
    print(f"    Best Recall@{args.top_k} (val) : {train_results['best_recall']:.4f}")
    print(f"    Best epoch             : {train_results['best_epoch']}")
    print(f"  Test Set Results")
    print(f"    Recall@20              : {recall_20:.4f}")
    print(f"    NDCG@20                : {ndcg_20:.4f}")
    print(f"  Outputs")
    print(f"    Checkpoint   : {args.checkpoint}")
    print(f"    Plots        : {args.plots_dir}/")
    print(f"    Eval report  : {args.report_dir}/eval_results.json")
    print("═"*62)
    print()
    print("  Next steps:")
    print("    Try demo mode  : python main.py --mode demo")
    print("    Specific user  : python main.py --mode recommend --user_id 42")
    print("    Re-evaluate    : python main.py --mode evaluate")
    print()

    return {"train": train_results, "eval": eval_results}


# =============================================================================
# SECTION 8: ENTRY POINT
# =============================================================================

def main():
    print_banner()
    args = parse_args()

    # Create output directories upfront
    os.makedirs("results",       exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # ── Route to the correct mode ─────────────────────────────────────────
    if args.mode == "pipeline":
        mode_pipeline(args)

    elif args.mode == "train":
        mode_train(args)

    elif args.mode == "evaluate":
        mode_evaluate(args)

    elif args.mode == "recommend":
        mode_recommend(args)

    elif args.mode == "demo":
        mode_demo(args)

    else:
        print(f"Unknown mode: {args.mode}")
        print("Valid modes: pipeline, train, evaluate, recommend, demo")
        sys.exit(1)


if __name__ == "__main__":
    main()
