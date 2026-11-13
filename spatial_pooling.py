"""
spatial_pooling.py
------------------
Course: Foundation of Large Language Models (AI60213)
Project: Step-INSPECTOR
Track: Member 1 (Spatial & Pooling Dynamics) & Member 3

Defines step-span token slicing, spatial pooling modules (Mean, Last, Global),
and provides the standalone pooling ablation benchmark.
"""

import re
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Robust step pattern matching "Step 1:", "Step 1.", "Step 1 -", etc.
STEP_PATTERN = re.compile(r"(Step\s+\d+[:\.\-].*?)(?=(?:\n\s*Step\s+\d+[:\.\-]|$))", re.DOTALL)

def parse_steps(candidate_text: str) -> List[Tuple[int, int]]:
    """Extracts start and end character offsets for each reasoning step."""
    matches = list(STEP_PATTERN.finditer(candidate_text))
    return [(m.start(), m.end()) for m in matches]

def map_char_span_to_token_indices(offset_mapping: List[Tuple[int, int]], abs_start: int, abs_end: int) -> Tuple[int, int]:
    """
    Maps character offsets in the prompt to exact token indices.
    Filters out zero-width special tokens.
    """
    tok_indices = [
        idx for idx, (ts, te) in enumerate(offset_mapping)
        if ts < abs_end and te > abs_start and (ts != te)
    ]
    if not tok_indices:
        raise ValueError(f"No tokens overlap character span [{abs_start}, {abs_end})")
    return (tok_indices[0], tok_indices[-1] + 1)

@dataclass
class StepSpan:
    step_index: int
    char_start: int
    char_end: int
    tok_start: int
    tok_end: int
    num_tokens: int

class StepSpanExtractor:
    """Parses candidate reasoning text into structured token spans."""
    def __init__(self, candidate_text: str, offset_mapping: List[Tuple[int, int]], candidate_start: int):
        self.candidate_text = candidate_text
        self.offset_mapping = offset_mapping
        self.candidate_start = candidate_start
        self.spans = self._build()

    def _build(self) -> List[StepSpan]:
        char_spans = parse_steps(self.candidate_text)
        spans = []
        for i, (c_start, c_end) in enumerate(char_spans):
            abs_start = self.candidate_start + c_start
            abs_end = self.candidate_start + c_end
            tok_start, tok_end = map_char_span_to_token_indices(self.offset_mapping, abs_start, abs_end)
            spans.append(StepSpan(i, c_start, c_end, tok_start, tok_end, tok_end - tok_start))
        return spans
        
    def __iter__(self):
        return iter(self.spans)
        
    def __len__(self):
        return len(self.spans)

# Pooling Modules

class MeanStepPooling(nn.Module):
    """Mean pooling over token representations in a reasoning step."""
    def forward(self, step_hidden: torch.Tensor) -> torch.Tensor:
        return step_hidden.mean(dim=1)

class LastTokenStepPooling(nn.Module):
    """Picks the final boundary token representation of a reasoning step."""
    def forward(self, step_hidden: torch.Tensor) -> torch.Tensor:
        return step_hidden[:, -1, :]

# Evaluation Protocol

@dataclass
class PoolingResult:
    strategy: str
    layer: int
    verdict_acc: float
    first_error_acc: float
    joint_acc: float

def get_strategy_repr(cached: dict, strategy: str, layer: int) -> torch.Tensor:
    """Retrieves step representations for the specified strategy and layer."""
    embeddings_mean = cached["step_embeddings_mean"]
    embeddings_last = cached["step_embeddings_last"]
    lens = cached["step_token_lengths"]
    n_steps = embeddings_mean.shape[2]

    if strategy == "mean":
        return embeddings_mean[:, layer, :, :]
    elif strategy == "last":
        return embeddings_last[:, layer, :, :]
    elif strategy == "global":
        # Global response average: Simulates base INSPECTOR whole response baseline
        layer_repr = embeddings_mean[:, layer, :, :] 
        pad_mask = (lens > 0).float().unsqueeze(-1)
        sum_repr = (layer_repr * pad_mask).sum(dim=1, keepdim=True)
        valid_counts = pad_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return (sum_repr / valid_counts).expand(-1, n_steps, -1) 
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def evaluate_first_error_metrics(probs_tensor: torch.Tensor, gold_first_errors: torch.Tensor, 
                                 step_token_lengths: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float, float]:
    """
    Computes Verdict Accuracy, First Error Accuracy, and Joint Accuracy.
    Includes cross device assertion to eliminate CUDA/CPU mismatch bugs.
    """
    device = probs_tensor.device
    gold_first_errors = gold_first_errors.to(device)
    step_token_lengths = step_token_lengths.to(device)

    valid_step_mask = (step_token_lengths > 0)
    probs = probs_tensor * valid_step_mask.float()
    
    is_error = probs > threshold
    pred_first_errors = []
    
    for b in range(probs.size(0)):
        err_indices = torch.where(is_error[b])[0]
        # Step indices are 1 based
        pred_first_errors.append(err_indices[0].item() + 1 if len(err_indices) > 0 else 0)
            
    pred_first_errors = torch.tensor(pred_first_errors, device=device)
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

def run_pooling_ablation(train_cache: dict, test_cache: dict, strategies: tuple = ("mean", "last", "global")) -> List[PoolingResult]:
    """Runs a complete linear probe ablation over layers and pooling strategies."""
    results = []
    n_layers = train_cache["step_embeddings_mean"].shape[1]
    
    targets_train, masks_train = train_cache["step_targets"], train_cache["step_masks"]
    gold_test, lens_test = test_cache["gold_first_errors"], test_cache["step_token_lengths"]

    for strategy in strategies:
        print(f"\nEvaluating Strategy: {strategy.upper()}")
        for layer in range(n_layers):
            step_repr_train = get_strategy_repr(train_cache, strategy, layer)
            step_repr_test = get_strategy_repr(test_cache, strategy, layer)
            B_test, M_test, D = step_repr_test.shape

            valid = masks_train.bool().reshape(-1)
            
            # .float() ensures compatibility with StandardScaler and LogisticRegression
            X_train = step_repr_train.reshape(-1, D)[valid].float().detach().cpu().numpy()
            y_train = targets_train.reshape(-1)[valid].detach().cpu().numpy().astype(int)

            if X_train.shape[0] == 0 or len(np.unique(y_train)) < 2:
                continue

            clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
            clf.fit(X_train, y_train)

            X_test_all = step_repr_test.reshape(-1, D).float().detach().cpu().numpy()
            probs_all = clf.predict_proba(X_test_all)[:, 1]
            probs_tensor = torch.tensor(probs_all, dtype=torch.float32).reshape(B_test, M_test)

            v_acc, fe_acc, j_acc = evaluate_first_error_metrics(probs_tensor, gold_test, lens_test)
            results.append(PoolingResult(strategy, layer, v_acc, fe_acc, j_acc))
            print(f"Strat: {strategy:<6} | Layer: {layer+1:02d} | V-Acc: {v_acc:>5.1f}% | FE-Acc: {fe_acc:>5.1f}% | J-Acc: {j_acc:>5.1f}%")
            
    return results

if __name__ == "__main__":
    try:
        train_c = torch.load("backup_qwen25/train_cached.pt", map_location="cpu", weights_only=True)
        test_c = torch.load("backup_qwen25/test_cached.pt", map_location="cpu", weights_only=True)
        print(f"\nExecuting Spatial Pooling Ablation (Train -> Test)")
        run_pooling_ablation(train_c, test_c)
    except FileNotFoundError:
        print("Error: data/*.pt files not found. Run extract_and_cache.py first.")