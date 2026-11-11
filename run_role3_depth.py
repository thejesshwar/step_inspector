import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import copy
import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class StepProbe(nn.Module):
    def __init__(self, hidden_dim=896):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )
    def forward(self, x):
        return self.head(x).squeeze(-1)

class CrossLayerAttentionProbe(nn.Module):
    def __init__(self, num_layers=24, hidden_dim=896):
        super().__init__()
        # Layer Positional Encodings
        self.layer_pos_embed = nn.Embedding(num_layers, hidden_dim)
        
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * (hidden_dim ** -0.5))
        
        # Attention
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.classifier = StepProbe(hidden_dim)

    def forward(self, all_layers_x):
        B, L, M, D = all_layers_x.shape
        
        # Inject positional awareness across model depth
        positions = torch.arange(L, device=all_layers_x.device)
        pos_emb = self.layer_pos_embed(positions).view(1, L, 1, D)
        all_layers_x = all_layers_x + pos_emb
        
        # Reshape for multi head attention math
        x_reshaped = all_layers_x.permute(0, 2, 1, 3).reshape(B * M, L, D) 
        
        Q = self.W_q(self.query.expand(B * M, -1, -1)) 
        K = self.W_k(x_reshaped)                       
        V = self.W_v(x_reshaped) 
        
        # Scaled Dot Product Attention
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(D) 
        alpha = torch.softmax(scores, dim=-1)                  
        
        fused = torch.bmm(alpha, V).squeeze(1)        
        fused_steps = fused.reshape(B, M, D)                   
        
        logits = self.classifier(fused_steps)                  
        return logits, alpha.reshape(B, M, L).mean(dim=1)      

def get_optimizer(model, lr=1e-3, weight_decay=1e-2):
    """Prevents weight decay on LayerNorm and bias parameters."""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optim.AdamW(optimizer_grouped_parameters, lr=lr)

def evaluate_predictions(logits, gold_first_errors, step_token_lengths, threshold=0.5):
    """Computes exact metrics without masking valid steps."""
    valid_step_mask = (step_token_lengths > 0)
    probs = torch.sigmoid(logits) * valid_step_mask.float()
    
    is_error = probs > threshold
    pred_first_errors = []
    for b in range(probs.size(0)):
        err_indices = torch.where(is_error[b])[0]
        pred_first_errors.append(err_indices[0].item() + 1 if len(err_indices) > 0 else 0)
            
    pred_first_errors = torch.tensor(pred_first_errors, device=logits.device)
    
    pred_verdict = (pred_first_errors > 0)
    gold_verdict = (gold_first_errors > 0)
    verdict_acc = (pred_verdict == gold_verdict).float().mean().item() * 100
    
    faulty_mask = (gold_first_errors > 0)
    if faulty_mask.sum() > 0:
        first_error_acc = (pred_first_errors[faulty_mask] == gold_first_errors[faulty_mask]).float().mean().item() * 100
    else:
        first_error_acc = 0.0
        
    joint_acc = (pred_first_errors == gold_first_errors).float().mean().item() * 100
    return verdict_acc, first_error_acc, joint_acc

def train_and_evaluate(model, train_loader, val_data, test_data, epochs=20, lr=1e-3, is_cross_layer=False):
    optimizer = get_optimizer(model, lr=lr)
    
    X_val, y_val, mask_val, gold_val, lens_val = val_data
    X_test, y_test, mask_test, gold_test, lens_test = test_data
    
    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        for X_b, y_b, mask_b in train_loader:
            optimizer.zero_grad()
            logits = model(X_b)[0] if is_cross_layer else model(X_b)
            
            bce = nn.functional.binary_cross_entropy_with_logits(logits, y_b, reduction='none')
            loss = (bce * mask_b).sum() / (mask_b.sum() + 1e-8)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)[0] if is_cross_layer else model(X_val)
            val_bce = nn.functional.binary_cross_entropy_with_logits(val_logits, y_val, reduction='none')
            val_loss = (val_bce * mask_val).sum() / (mask_val.sum() + 1e-8)
            
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)[0] if is_cross_layer else model(X_test)
        return evaluate_predictions(test_logits, gold_test, lens_test)

def prep_data(data_dict, device):
    return (
        data_dict["step_embeddings_mean"].to(device).float(),
        data_dict["step_targets"].to(device),
        data_dict["step_masks"].to(device),
        data_dict["gold_first_errors"].to(device),
        data_dict["step_token_lengths"].to(device)
    )

def main():
    print(f"Loading cached tensors on {DEVICE}")
    try:
        train = torch.load("data/train_cached.pt", map_location=DEVICE, weights_only=True)
        val = torch.load("data/val_cached.pt", map_location=DEVICE, weights_only=True)
        test = torch.load("data/test_cached.pt", map_location=DEVICE, weights_only=True)
    except FileNotFoundError:
        print("Error: Cached tensors not found. Ensure 'train_cached.pt' is in 'data/'.")
        return

    X_train, y_train, mask_train, gold_train, lens_train = prep_data(train, DEVICE)
    val_tup = prep_data(val, DEVICE)
    test_tup = prep_data(test, DEVICE)

    num_layers = X_train.shape[1]
    print(f"{'Layer Depth':<14} | {'Verdict Acc (%)':<16} | {'First-Error Acc (%)':<20} | {'Joint Acc (%)':<14}")

    layer_fe_accs = []

    for layer in range(num_layers):
        dataset = TensorDataset(X_train[:, layer, :, :], y_train, mask_train)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        val_layer_tup = (val_tup[0][:, layer, :, :], val_tup[1], val_tup[2], val_tup[3], val_tup[4])
        test_layer_tup = (test_tup[0][:, layer, :, :], test_tup[1], test_tup[2], test_tup[3], test_tup[4])

        torch.manual_seed(42)
        probe = StepProbe().to(DEVICE)
        
        v_acc, fe_acc, j_acc = train_and_evaluate(
            probe, loader, val_layer_tup, test_layer_tup, epochs=20, lr=1e-3
        )

        layer_fe_accs.append(fe_acc)
        print(f"Layer {layer+1:02d}/24     | {v_acc:<16.2f} | {fe_acc:<20.2f} | {j_acc:<14.2f}")

    best_layer = np.argmax(layer_fe_accs) + 1
    best_fe = np.max(layer_fe_accs)
    print("="*70)
    print(f"Peak First Error Accuracy: Layer {best_layer} ({best_fe:.2f}%)")

    print("\nTraining Cross Layer Attention Probe")
    dataset_multi = TensorDataset(X_train, y_train, mask_train)
    loader_multi = DataLoader(dataset_multi, batch_size=32, shuffle=True)
    
    torch.manual_seed(42)
    clap = CrossLayerAttentionProbe(num_layers=num_layers).to(DEVICE)
    c_vacc, c_feacc, c_jacc = train_and_evaluate(
        clap, loader_multi, val_tup, test_tup, epochs=30, lr=5e-4, is_cross_layer=True
    )
        
    print(f"Cross-Layer Attn | {c_vacc:<16.2f} | {c_feacc:<20.2f} | {c_jacc:<14.2f}")

    # Plotting
    plt.figure(figsize=(9, 4.5))
    layers = list(range(1, num_layers + 1))
    plt.plot(layers, layer_fe_accs, marker='o', color='#1f77b4', lw=2.2, label='Single Layer Probes')
    plt.axhline(y=c_feacc, color='#2ca02c', linestyle='--', lw=2, label=f'Cross-Layer Probe ({c_feacc:.1f}%)')
    plt.axvline(x=best_layer, color='crimson', linestyle=':', lw=1.8, label=f'Optimal Depth (L{best_layer})')
    
    plt.title("Reasoning Error Localization Across Qwen2.5-0.5B Depth", fontsize=12, pad=10)
    plt.xlabel("Transformer Layer Index", fontsize=11)
    plt.ylabel("First-Error Accuracy (%)", fontsize=11)
    plt.xticks(range(1, num_layers + 1, 2))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, loc='best')
    plt.tight_layout()
    plt.savefig("layer_depth_progression.png", dpi=300)
    print("\nSaved plot as 'layer_depth_progression.png'")

if __name__ == "__main__":
    main()