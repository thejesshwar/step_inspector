
import torch

KEYS = [
    "step_targets",
    "step_masks",
    "gold_first_errors",
    "step_token_lengths",
]

for split in ["train", "val", "test"]:
    q25 = torch.load(
        f"data/{split}_cached.pt",
        map_location="cpu",
        weights_only=True
    )

    q3 = torch.load(
        f"data/qwen3/{split}_cached.pt",
        map_location="cpu",
        weights_only=True
    )

    print(f"\n{split.upper()}")

    for key in KEYS:
        a, b = q25[key], q3[key]

        same = (
            a.shape == b.shape
            and torch.equal(a, b)
        )

        print(f"{key}: {'MATCH' if same else 'DIFFERENT'}")

    print(
        "Qwen2.5:",
        q25["step_embeddings_mean"].shape
    )
    print(
        "Qwen3:",
        q3["step_embeddings_mean"].shape
    )
