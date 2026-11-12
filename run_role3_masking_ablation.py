"""
run_role3_masking_ablation.py
Course: Foundation of Large Language Models (AI60213)
Project: Step-INSPECTOR
Track: Member 3 (Causal Masking & Loss Dynamics)

Executes a controlled ablation comparing:
1. Proposed Causal Masking: Steps after the first error are masked out (m = 0).
2. Naive Unmasked (Target = 0): Post-error steps are supervised as correct (target = 0).
3. Naive Unmasked (Target = 1): Post-error steps are supervised as broken (target = 1).
"""

import os
import copy
import torch
from torch.utils.data import TensorDataset, DataLoader

from run_role3_depth import StepProbe, get_optimizer, evaluate_predictions, prep_data, DEVICE
from custom_loss import MaskedStepBCELoss

def train_ablation(X_train, y_train, mask_train, lens_train, val_tup, test_tup, mode="masked", layer=21, epochs=20):
    X_tr = X_train[:, layer, :, :]
    hidden_dim = X_tr.shape[-1]
    valid_steps = (lens_train > 0).float()
    
    if mode == "unmasked_target_0":
        train_mask = valid_steps.clone()
        train_targets = y_train.clone()
    elif mode == "unmasked_target_1":
        train_mask = valid_steps.clone()
        train_targets = torch.cummax(y_train, dim=-1).values * valid_steps
    else:
        train_mask = mask_train.clone()
        train_targets = y_train.clone()

    g = torch.Generator(device="cpu").manual_seed(42)
    dataset = TensorDataset(X_tr, train_targets, train_mask)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, generator=g)
    
    torch.manual_seed(42)
    model = StepProbe(hidden_dim=hidden_dim).to(DEVICE)
    optimizer = get_optimizer(model, lr=1e-3)
    loss_fn = MaskedStepBCELoss()

    best_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())

    X_val = val_tup[0][:, layer, :, :]
    y_val = val_tup[1]
    mask_val = val_tup[2]

    for epoch in range(epochs):
        model.train()
        for x_b, y_b, m_b in loader:
            optimizer.zero_grad()
            logits = model(x_b)
            loss = loss_fn(logits, y_b, m_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = loss_fn(val_logits, y_val, mask_val)
            if val_loss.item() < best_loss:
                best_loss = val_loss.item()
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        X_test = test_tup[0][:, layer, :, :]
        test_logits = model(X_test)
        return evaluate_predictions(test_logits, test_tup[3], test_tup[4])

def main():
    print(f"Loading cached tensors on {DEVICE}")
    try:
        train = torch.load("data/train_cached.pt", map_location=DEVICE, weights_only=True)
        val = torch.load("data/val_cached.pt", map_location=DEVICE, weights_only=True)
        test = torch.load("data/test_cached.pt", map_location=DEVICE, weights_only=True)
    except FileNotFoundError:
        print("Error: Cached data files not found in 'data/'. Please ensure 'train_cached.pt' exists.")
        return

    X_train, y_train, mask_train, gold_train, lens_train = prep_data(train, DEVICE)
    val_tup = prep_data(val, DEVICE)
    test_tup = prep_data(test, DEVICE)

    if os.path.exists("data/best_step_probe.pt"):
        ckpt = torch.load("data/best_step_probe.pt", map_location="cpu", weights_only=True)
        best_layer_idx = int(ckpt["best_layer_idx"])
        model_label = ckpt["model_label"]
    else:
        best_layer_idx = X_train.shape[1] - 3
        model_label = "Qwen3-0.6B" if X_train.shape[1] == 28 else "Qwen2.5-0.5B"
    print(f"MEMBER 3 ABLATION: CAUSAL MASKING VS. NAIVE SUPERVISION ({model_label}, LAYER {best_layer_idx+1})")
    print(f"{'Supervision Masking Scheme':<30} | {'Verdict Acc':<13} | {'First-Error Acc':<16} | {'Joint Acc':<10}")

    ablation_modes = [
        ("masked", "Proposed Causal Masking"),
        ("unmasked_target_0", "Naive Unmasked (Target = 0)"),
        ("unmasked_target_1", "Naive Unmasked (Target = 1)")
    ]

    results = []
    for mode_key, mode_name in ablation_modes:
        v_acc, fe_acc, j_acc = train_ablation(
            X_train, y_train, mask_train, lens_train, val_tup, test_tup, mode=mode_key, layer=best_layer_idx, epochs=20
        )
        results.append((mode_name, v_acc, fe_acc, j_acc))
        print(f"{mode_name:<30} | {v_acc:>10.1f}%  | {fe_acc:>14.1f}%  | {j_acc:>8.1f}%")
    
    print("\nLaTeX Table Snippet for ICLR Report (Section 4.3)")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Supervision Scheme & Verdict Acc (\%) & First-Error Acc (\%) & Joint Acc (\%) \\")
    print(r"\midrule")
    for name, v, fe, j in results:
        print(f"{name} & {v:.1f} & {fe:.1f} & {j:.1f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(f"\\caption{{Ablation of Causal Loss Masking against naive supervision strategies on {model_label} (Layer {best_layer_idx+1}).}}")
    print(r"\label{tab:masking_ablation}")
    print(r"\end{table}")

if __name__ == "__main__":
    main()