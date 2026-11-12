"""
plot_pooling_ablation.py
Member 1: Spatial & Pooling Dynamics Track
Step-INSPECTOR project (AI60213)
"""

import matplotlib.pyplot as plt
import torch

from spatial_pooling import run_pooling_ablation


def main():
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

    n_layers = train_cached["step_embeddings_mean"].shape[1]
    model_label = "Qwen3-0.6B" if n_layers == 28 else "Qwen2.5-0.5B"
    layers = list(range(1, n_layers + 1))

    data_by_strategy = {s: {"fe_acc": [], "verdict_acc": [], "joint_acc": []} for s in strategies}
    for r in results:
        data_by_strategy[r.strategy]["fe_acc"].append(r.first_error_acc)
        data_by_strategy[r.strategy]["verdict_acc"].append(r.verdict_acc)
        data_by_strategy[r.strategy]["joint_acc"].append(r.joint_acc)

    faulty_mask = test_cached["gold_first_errors"] > 0
    num_valid_steps = (test_cached["step_token_lengths"][faulty_mask] > 0).sum(dim=1).float()
    random_baseline = (1.0 / (num_valid_steps + 1.0)).mean().item() * 100

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
        f"Step-Level Error Detectability by Pooling Strategy Across {model_label} Depth",
        fontsize=12,
        pad=10,
    )
    plt.xlabel("Transformer Layer Index", fontsize=11)
    plt.ylabel("First-Error Accuracy (%)", fontsize=11)
    plt.xticks(range(1, n_layers + 1, 2))
    max_fe = max(max(data_by_strategy[s]["fe_acc"]) for s in strategies)
    plt.ylim(0, max(50, int(max_fe) + 10))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    plt.savefig("pooling_ablation.png", dpi=300)
    print("\nSaved pooling ablation figure as 'pooling_ablation.png'")

    print(f"\nPooling Ablation Summary ({model_label}, Test Split):")
    print(
        f"{'Strategy':<25} | {'Best Layer':<10} | {'Best FE-Acc (%)':<15} | "
        f"{'Verdict Acc (%)':<15} | {'Joint Acc (%)':<13} | {'Mean FE-Acc (%)':<15}"
    )
    for strategy in strategies:
        fe_accs = data_by_strategy[strategy]["fe_acc"]
        v_accs = data_by_strategy[strategy]["verdict_acc"]
        j_accs = data_by_strategy[strategy]["joint_acc"]
        # Break ties on FE Acc using Joint Acc and then Verdict Acc
        best_idx = int(max(range(len(fe_accs)), key=lambda i: (fe_accs[i], j_accs[i], v_accs[i])))
        print(
            f"{labels[strategy]:<25} | {best_idx + 1:<10} | "
            f"{fe_accs[best_idx]:<15.1f} | {v_accs[best_idx]:<15.1f} | "
            f"{j_accs[best_idx]:<13.1f} | {sum(fe_accs) / len(fe_accs):<15.1f}"
        )


if __name__ == "__main__":
    main()