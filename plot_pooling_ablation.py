"""
plot_pooling_ablation.py
Member 1: Spatial & Pooling Dynamics Track
Step-INSPECTOR project (AI60213)

Plots the Mean vs. Last-token vs. Global pooling ablation across all 24
layers of Qwen2.5-0.5B-Instruct, producing a figure for the ICLR report
(Sections 3.1 & 4.1) and Slide 2 (Data Pipeline & Token Mapping).

Reuses the pooling ablation logic from spatial_pooling.py:

    python plot_pooling_ablation.py
"""

import matplotlib.pyplot as plt
import torch

from spatial_pooling import run_pooling_ablation


def main():
    # Load cached representations produced by extract_and_cache.py
    try:
        train_cached = torch.load("data/train_cached.pt", map_location="cpu", weights_only=True)
        test_cached = torch.load("data/test_cached.pt", map_location="cpu", weights_only=True)
    except FileNotFoundError:
        print(
            "Error: data/*.pt files not found. Run extract_and_cache.py first "
            "to generate the cached tensors."
        )
        return

    strategies = ("mean", "last", "global")
    results = run_pooling_ablation(train_cached, test_cached, strategies=strategies)

    # Organise results by strategy
    n_layers = train_cached["step_embeddings_mean"].shape[1]
    layers = list(range(1, n_layers + 1))

    data_by_strategy = {s: {"fe_acc": [], "verdict_acc": [], "joint_acc": []} for s in strategies}
    for r in results:
        data_by_strategy[r.strategy]["fe_acc"].append(r.first_error_acc)
        data_by_strategy[r.strategy]["verdict_acc"].append(r.verdict_acc)
        data_by_strategy[r.strategy]["joint_acc"].append(r.joint_acc)

    # Compute exact random step-guess baseline on faulty test examples: E[1 / (num_steps + 1)]
    faulty_mask = test_cached["gold_first_errors"] > 0
    num_valid_steps = (test_cached["step_token_lengths"][faulty_mask] > 0).sum(dim=1).float()
    random_baseline = (1.0 / (num_valid_steps + 1.0)).mean().item() * 100

    # ---- Plot: First-Error Accuracy across layers ----
    plt.figure(figsize=(9, 4.5))
    colors = {"mean": "#1f77b4", "last": "#ff7f0e", "global": "#2ca02c"}
    labels = {"mean": "Mean Pooling", "last": "Last-Token Pooling", "global": "Global (Whole-Response)"}

    for strategy in strategies:
        plt.plot(
            layers,
            data_by_strategy[strategy]["fe_acc"],
            marker="o",
            color=colors[strategy],
            lw=2.2,
            label=labels[strategy],
        )

    plt.axhline(
        y=random_baseline,
        color="crimson",
        linestyle=":",
        lw=1.6,
        label=f"Random Step Guess ({random_baseline:.1f}%)",
    )
    plt.title(
        "Step-Level Error Detectability by Pooling Strategy Across Model Depth",
        fontsize=12,
        pad=10,
    )
    plt.xlabel("Transformer Layer Index", fontsize=11)
    plt.ylabel("First-Error Accuracy (%)", fontsize=11)
    plt.xticks(range(1, n_layers + 1, 2))
    plt.ylim(0, 50)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    plt.savefig("pooling_ablation.png", dpi=300)
    print("\nSaved pooling ablation figure as 'pooling_ablation.png'")

    # ---- Print summary table ----
    print("\nPooling Ablation Summary (Test Split):")
    print(
        f"{'Strategy':<25} | {'Best Layer':<10} | {'Best FE-Acc (%)':<15} | "
        f"{'Verdict Acc (%)':<15} | {'Joint Acc (%)':<13} | {'Mean FE-Acc (%)':<15}"
    )
    print("-" * 105)
    for strategy in strategies:
        fe_accs = data_by_strategy[strategy]["fe_acc"]
        v_accs = data_by_strategy[strategy]["verdict_acc"]
        j_accs = data_by_strategy[strategy]["joint_acc"]
        best_idx = int(max(range(len(fe_accs)), key=lambda i: fe_accs[i]))
        print(
            f"{labels[strategy]:<25} | {best_idx + 1:<10} | "
            f"{fe_accs[best_idx]:<15.1f} | {v_accs[best_idx]:<15.1f} | "
            f"{j_accs[best_idx]:<13.1f} | {sum(fe_accs) / len(fe_accs):<15.1f}"
        )


if __name__ == "__main__":
    main()