
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models import LinearProbe, MLPProbe
from run_role3_depth import (
    train_and_evaluate,
    evaluate_predictions,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 123, 2026]
LAYER_NUM = 22
LAYER_IDX = LAYER_NUM - 1

EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 32


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(path):
    """
    Load only the selected transformer layer
    and the tensors needed for training/evaluation.
    """
    cache = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    embeddings = cache["step_embeddings_mean"]

    if LAYER_IDX >= embeddings.shape[1]:
        raise ValueError(
            f"Layer {LAYER_NUM} unavailable in {path}"
        )

    X = embeddings[:, LAYER_IDX].float().to(DEVICE)

    y = cache["step_targets"].float().to(DEVICE)
    masks = cache["step_masks"].float().to(DEVICE)

    gold = cache["gold_first_errors"].long().to(DEVICE)
    lengths = cache["step_token_lengths"].long().to(DEVICE)

    return X, y, masks, gold, lengths


def load_data(data_dir):
    train = load_split(data_dir / "train_cached.pt")
    val = load_split(data_dir / "val_cached.pt")
    test = load_split(data_dir / "test_cached.pt")

    return train, val, test


def make_train_loader(train_data, seed):
    X, y, masks, _, _ = train_data

    dataset = TensorDataset(X, y, masks)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def compute_diagnostics(model, test_data, threshold=0.5):
    """
    Evaluate solution-level error detection.

    Positive class: solution contains an error.
    """
    X, _, _, gold, lengths = test_data

    model.eval()

    with torch.inference_mode():
        logits = model(X)
        probabilities = torch.sigmoid(logits)

    valid_steps = lengths > 0
    predicted_errors = (probabilities > threshold) & valid_steps

    # First predicted error, using 1-based indexing.
    # Zero indicates a completely correct solution.
    n_steps = logits.shape[1]

    step_indices = torch.arange(
        1, n_steps + 1,
        device=logits.device
    ).unsqueeze(0).expand_as(logits)

    first_errors = torch.where(
        predicted_errors,
        step_indices,
        n_steps + 1
    ).min(dim=1).values

    first_errors = torch.where(
        first_errors == n_steps + 1,
        0,
        first_errors
    )

    actual_has_error = gold > 0
    predicted_has_error = first_errors > 0

    TP = (actual_has_error & predicted_has_error).sum().item()
    TN = (~actual_has_error & ~predicted_has_error).sum().item()
    FP = (~actual_has_error & predicted_has_error).sum().item()
    FN = (actual_has_error & ~predicted_has_error).sum().item()

    precision = TP / (TP + FP) if TP + FP else 0.0
    recall = TP / (TP + FN) if TP + FN else 0.0
    fpr = FP / (FP + TN) if FP + TN else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    exact_localizations = (
        (first_errors == gold) & actual_has_error
    ).sum().item()

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "precision": precision * 100,
        "recall": recall * 100,
        "fpr": fpr * 100,
        "f1": f1 * 100,
        "exact_localizations": exact_localizations
    }


def save_csv(path, rows, columns):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(rows)


def evaluate_saved_checkpoint(checkpoint_path, test_data):
    """
    Reference evaluation only.
    This does not train the MLP.
    """
    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    hidden_dim = int(ckpt["hidden_dim"])

    if int(ckpt["best_layer_idx"]) != LAYER_IDX:
        raise ValueError("Checkpoint layer mismatch")

    if hidden_dim != test_data[0].shape[-1]:
        raise ValueError("Checkpoint hidden dimension mismatch")

    model = MLPProbe(hidden_dim).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    X, _, _, gold, lengths = test_data

    with torch.inference_mode():
        logits = model(X)

        scores = evaluate_predictions(
            logits, gold, lengths
        )

    return scores, ckpt.get("model_label", "unknown")


def run_experiments(data_dir, output_dir, checkpoint=None):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)
    print("Data directory:", data_dir)
    print("Transformer layer:", LAYER_NUM)

    train, val, test = load_data(data_dir)

    hidden_dim = train[0].shape[-1]

    print("Hidden dimension:", hidden_dim)
    print("Training shape:", tuple(train[0].shape))

    results = []
    diagnostic_results = []

    architectures = {
        "Linear": LinearProbe,
        "MLP": MLPProbe,
    }

    for seed in SEEDS:
        for probe_name, probe_class in architectures.items():
            set_seed(seed)

            model = probe_class(
                hidden_dim=hidden_dim
            ).to(DEVICE)

            train_loader = make_train_loader(
                train, seed
            )

            print(
                f"\nTraining {probe_name}, seed={seed}"
            )

            verdict, first_error, joint = train_and_evaluate(
                model,
                train_loader,
                val,
                test,
                epochs=EPOCHS,
                lr=LR
            )

            # The training function has restored the best
            # validation checkpoint into 'model'.
            diagnostics = compute_diagnostics(
                model,
                test,
                threshold=0.5
            )

            diagnostic_results.append({
                "seed": seed,
                "probe": probe_name,
                **diagnostics
            })

            # Verify consistency with the original evaluator.
            reconstructed_verdict = (
                (diagnostics["TP"] + diagnostics["TN"])
                / sum(diagnostics[k] for k in ["TP", "TN", "FP", "FN"])
                * 100
            )

            assert abs(reconstructed_verdict - verdict) < 1e-3

            print(
                f"TP={diagnostics['TP']}, "
                f"TN={diagnostics['TN']}, "
                f"FP={diagnostics['FP']}, "
                f"FN={diagnostics['FN']}"
            )

            print(
                f"Recall={diagnostics['recall']:.2f}% | "
                f"FPR={diagnostics['fpr']:.2f}% | "
                f"Precision={diagnostics['precision']:.2f}% | "
                f"F1={diagnostics['f1']:.2f}%"
            )

            result = {
                "seed": seed,
                "probe": probe_name,
                "layer": LAYER_NUM,
                "parameters": count_parameters(model),
                "verdict": verdict,
                "first_error": first_error,
                "joint": joint
            }

            results.append(result)

            print(
                f"Verdict={verdict:.2f}% | "
                f"FirstError={first_error:.2f}% | "
                f"Joint={joint:.2f}%"
            )

    columns = [
        "seed", "probe", "layer", "parameters",
        "verdict", "first_error", "joint"
    ]

    save_csv(
        output_dir / "runs.csv",
        results,
        columns
    )


    diagnostic_columns = [
        "seed",
        "probe",
        "TP",
        "TN",
        "FP",
        "FN",
        "precision",
        "recall",
        "fpr",
        "f1",
        "exact_localizations"
    ]

    save_csv(
        output_dir / "diagnostics.csv",
        diagnostic_results,
        diagnostic_columns
    )

    print("\n=== DIAGNOSTIC SUMMARY ===")

    for probe_name in architectures:
        probe_rows = [
                r for r in diagnostic_results
                if r["probe"] == probe_name
        ]

        print(f"\n{probe_name}")

        for metric in ["recall", "fpr", "precision", "f1"]:
                values = np.array([
                    r[metric] for r in probe_rows
                ])

                print(
                    f"{metric}: "
                    f"{values.mean():.2f} +/- "
                    f"{values.std(ddof=1):.2f}%"
                )


    summary = []

    for probe_name in architectures:
        probe_runs = [
            r for r in results
            if r["probe"] == probe_name
        ]

        row = {"probe": probe_name}

        for metric in ["verdict", "first_error", "joint"]:
            values = np.array(
                [r[metric] for r in probe_runs]
            )

            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))

        summary.append(row)

    summary_columns = [
        "probe",
        "verdict_mean", "verdict_std",
        "first_error_mean", "first_error_std",
        "joint_mean", "joint_std",
    ]

    save_csv(
        output_dir / "summary.csv",
        summary,
        summary_columns
    )

    print("\n=== SUMMARY ===")

    for row in summary:
        print(f"\n{row['probe']}")

        for metric in ["verdict", "first_error", "joint"]:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]

            print(f"{metric}: {mean:.2f} +/- {std:.2f}%")

    print("\n=== PAIRED MLP - LINEAR DIFFERENCES ===")

    for metric in ["verdict", "first_error", "joint"]:
        differences = []

        for seed in SEEDS:
            linear = next(
                r for r in results
                if r["seed"] == seed and r["probe"] == "Linear"
            )
            mlp = next(
                r for r in results
                if r["seed"] == seed and r["probe"] == "MLP"
            )

            differences.append(
                mlp[metric] - linear[metric]
            )

        differences = np.array(differences)

        print(
            f"{metric}: "
            f"{differences.mean():+.2f} +/- "
            f"{differences.std(ddof=1):.2f} pp"
        )

    if checkpoint is not None:
        scores, model_label = evaluate_saved_checkpoint(
            checkpoint,
            test
        )

        print("\n=== SAVED CHECKPOINT ===")
        print("Model:", model_label)
        print(
            "Verdict: %.2f%% | FirstError: %.2f%% | Joint: %.2f%%"
            % scores
        )

    print("\nResults saved to:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data")
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/role2/qwen25")
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None
    )

    args = parser.parse_args()

    run_experiments(
        args.data_dir,
        args.output_dir,
        args.checkpoint
    )
