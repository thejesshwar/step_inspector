"""
spatial_pooling.py
------------------
Member 1: Spatial & Pooling Dynamics Track
Step-INSPECTOR project (AI60213, IIT Kharagpur)

Deliverable for Sections 3.1 & 4.1 (Step-Span Slicing & Spatial Pooling Ablation)
and Slide 2 (Data Pipeline & Token Mapping).

This module provides:

  1. Step-span slicing via regex (verified 650/650 on MR-GSM8K)
  2. Tokenizer offset-mapping utilities (char spans -> token indices)
  3. PyTorch pooling operators (nn.Modules):
       - MeanStepPooling      : h_m = (1 / |s_m|) * sum_{t in s_m} h_t
       - LastTokenStepPooling : h_m = h_end(m)  (step boundary / last token)
       - GlobalResponsePooling: whole-response pooling (base INSPECTOR baseline)
  4. A pooling-ablation runner that compares how well each strategy makes
     step-level errors linearly separable across layers, using cached
     multi-layer representations.

The pooling operators are standalone nn.Modules so they can be imported and
reused by other members' probe code (models.py / evaluate.py) without
duplicating the offset-mapping logic.
"""

import glob
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Step-span slicing & offset mapping
# ---------------------------------------------------------------------------

STEP_PATTERN = re.compile(
    r"(Step\s+\d+[:\.\-].*?)(?=(?:\n\s*Step\s+\d+[:\.\-]|$))",
    re.DOTALL,
)


def parse_steps(candidate_text: str) -> List[Tuple[int, int]]:
    """
    Extract character (start, end) spans for each numbered step in a candidate
    solution. Matches 'Step 1:', 'Step 1.', 'Step 1 -', etc.

    Returns a list of (char_start, char_end) tuples.

    Verified on 650/650 MR-GSM8K examples (see verify_steps.py).
    """
    matches = list(STEP_PATTERN.finditer(candidate_text))
    return [(m.start(), m.end()) for m in matches]


def map_char_span_to_token_indices(
    offset_mapping: List[Tuple[int, int]],
    abs_start: int,
    abs_end: int,
) -> Tuple[int, int]:
    """
    Map a character span [abs_start, abs_end) to contiguous token indices using
    the tokenizer's offset mapping (return_offsets_mapping=True).

    Returns (tok_start, tok_end_exclusive).
    Raises ValueError if no tokens overlap the span.
    """
    tok_indices = [
        idx
        for idx, (ts, te) in enumerate(offset_mapping)
        if ts < abs_end and te > abs_start and (ts != te)  # exclude zero-width
    ]
    if not tok_indices:
        raise ValueError(
            f"No tokens overlap char span [{abs_start}, {abs_end})"
        )
    return (tok_indices[0], tok_indices[-1] + 1)


@dataclass
class StepSpan:
    """Represents a single reasoning step with both char and token spans."""
    step_index: int            # 0-based index within the candidate
    char_start: int            # char start within candidate text
    char_end: int              # char end (exclusive) within candidate text
    tok_start: int             # token start (inclusive) in the full prompt
    tok_end: int               # token end (exclusive) in the full prompt
    num_tokens: int            # tok_end - tok_start


class StepSpanExtractor:
    """
    Combines regex step parsing with tokenizer offset mapping to produce
    per-step token spans for a candidate solution.

    Usage:
        extractor = StepSpanExtractor(candidate, offset_mapping,
                                      candidate_start, num_steps_expected)
        for span in extractor:
            step_hidden = all_layers[:, span.tok_start:span.tok_end, :]
    """

    def __init__(
        self,
        candidate_text: str,
        offset_mapping: List[Tuple[int, int]],
        candidate_start: int,
        num_steps_expected: Optional[int] = None,
    ):
        self.candidate_text = candidate_text
        self.offset_mapping = offset_mapping
        self.candidate_start = candidate_start
        self.num_steps_expected = num_steps_expected
        self.spans: List[StepSpan] = self._build()

    def _build(self) -> List[StepSpan]:
        char_spans = parse_steps(self.candidate_text)
        if (
            self.num_steps_expected is not None
            and len(char_spans) != self.num_steps_expected
        ):
            raise ValueError(
                f"Step count mismatch: expected {self.num_steps_expected}, "
                f"found {len(char_spans)}"
            )
        spans: List[StepSpan] = []
        for i, (c_start, c_end) in enumerate(char_spans):
            abs_start = self.candidate_start + c_start
            abs_end = self.candidate_start + c_end
            tok_start, tok_end = map_char_span_to_token_indices(
                self.offset_mapping, abs_start, abs_end
            )
            spans.append(
                StepSpan(
                    step_index=i,
                    char_start=c_start,
                    char_end=c_end,
                    tok_start=tok_start,
                    tok_end=tok_end,
                    num_tokens=tok_end - tok_start,
                )
            )
        return spans

    def __len__(self) -> int:
        return len(self.spans)

    def __iter__(self):
        return iter(self.spans)


# ---------------------------------------------------------------------------
# 2. PyTorch pooling operators (nn.Modules)
# ---------------------------------------------------------------------------

class MeanStepPooling(nn.Module):
    """
    Mean pooling across the tokens of a single step.

    h_m = (1 / |s_m|) * sum_{t in s_m} h_t

    Input  : hidden states of shape (num_layers, num_step_tokens, hidden_dim)
    Output : (num_layers, hidden_dim)
    """

    def __init__(self):
        super().__init__()

    def forward(self, step_hidden: torch.Tensor) -> torch.Tensor:
        # step_hidden: (L, T, D) -> mean over token dim -> (L, D)
        return step_hidden.mean(dim=1)


class LastTokenStepPooling(nn.Module):
    """
    Last-token (step boundary) pooling.

    h_m = h_{end(m)}  -- representation at the step delimiter / last token

    Input  : hidden states of shape (num_layers, num_step_tokens, hidden_dim)
    Output : (num_layers, hidden_dim)
    """

    def __init__(self):
        super().__init__()

    def forward(self, step_hidden: torch.Tensor) -> torch.Tensor:
        # step_hidden: (L, T, D) -> last token -> (L, D)
        return step_hidden[:, -1, :]


class GlobalResponsePooling(nn.Module):
    """
    Whole-response pooling (base INSPECTOR baseline).

    Pools ALL tokens of the candidate response (across all steps) into a
    single vector per response. This is the coarse whole-response baseline
    against which step-level pooling (Mean / Last-token) is compared.

    Supports two modes: 'mean' and 'last'.

    Input  : hidden states of shape (num_layers, num_response_tokens, hidden_dim)
    Output : (num_layers, hidden_dim)
    """

    def __init__(self, mode: str = "mean"):
        super().__init__()
        if mode not in ("mean", "last"):
            raise ValueError(f"mode must be 'mean' or 'last', got '{mode}'")
        self.mode = mode

    def forward(self, response_hidden: torch.Tensor) -> torch.Tensor:
        # response_hidden: (L, T, D)
        if self.mode == "mean":
            return response_hidden.mean(dim=1)  # (L, D)
        return response_hidden[:, -1, :]  # (L, D)


# ---------------------------------------------------------------------------
# 3. Pooling ablation experiment (Member 1 research question)
# ---------------------------------------------------------------------------

@dataclass
class PoolingResult:
    strategy: str          # 'mean' | 'last' | 'global'
    layer: int
    auroc: float           # step-level error detectability (AUC)
    accuracy: float        # step-level binary accuracy at midpoint threshold
    first_error_acc: Optional[float] = None  # exact first-error localization (reserved)


def load_cached_split(path: str) -> dict:
    """Load a cached .pt split produced by extract_and_cache.py."""
    return torch.load(path, map_location="cpu")


def run_pooling_ablation(
    cached: dict,
    strategies: Tuple[str, ...] = ("mean", "last", "global"),
    device: str = "cpu",
) -> List[PoolingResult]:
    """
    Compare how well each pooling strategy makes step-level errors linearly
    separable across layers.

    Reads from cached tensors produced by extract_and_cache.py:
      - step_embeddings_mean : (N, L, S, D)  mean-pooled per step
      - step_embeddings_last : (N, L, S, D)  last-token per step
      - step_targets         : (N, S)        binary step correctness
      - step_masks           : (N, S)        causal supervision mask

    For 'global', whole-response representations are computed by pooling
    across all steps and broadcast to every step (coarse baseline).

    Returns a list of PoolingResult, one per (strategy, layer).
    """
    results: List[PoolingResult] = []
    embeddings_mean = cached["step_embeddings_mean"]  # (N, L, S, D)
    embeddings_last = cached["step_embeddings_last"]  # (N, L, S, D)
    targets = cached["step_targets"]                  # (N, S)
    masks = cached["step_masks"]                      # (N, S)

    n_layers = embeddings_mean.shape[1]
    n_steps = targets.shape[1]

    for strategy in strategies:
        for layer in range(n_layers):
            if strategy == "mean":
                step_repr = embeddings_mean[:, layer, :, :]  # (N, S, D)
            elif strategy == "last":
                step_repr = embeddings_last[:, layer, :, :]  # (N, S, D)
            elif strategy == "global":
                # Coarse baseline: pool all steps into one vector per response,
                # broadcast to every step position.
                step_repr = (
                    embeddings_mean[:, layer, :, :]
                    .mean(dim=1, keepdim=True)  # (N, 1, D)
                    .expand(-1, n_steps, -1)  # (N, S, D)
                )
            else:
                raise ValueError(f"Unknown strategy: '{strategy}'")

            # Flatten to (N*S, D) and apply the causal mask.
            flat_repr = step_repr.reshape(-1, step_repr.shape[-1])      # (N*S, D)
            flat_targets = targets.reshape(-1)                          # (N*S,)
            flat_masks = masks.reshape(-1)                              # (N*S,)

            valid = flat_masks.bool()
            x = flat_repr[valid]   # (V, D)
            y = flat_targets[valid]  # (V,)

            if x.shape[0] == 0 or len(torch.unique(y)) < 2:
                results.append(PoolingResult(strategy, layer, float("nan"), float("nan")))
                continue

            # Linear separability: project onto the difference of class means
            # (a Fisher-like direction) and threshold at the midpoint.
            with torch.no_grad():
                pos_mean = x[y == 1].mean(dim=0)  # (D,)
                neg_mean = x[y == 0].mean(dim=0)  # (D,)
                direction = pos_mean - neg_mean   # (D,)
                direction_norm = direction.norm()
                if direction_norm == 0:
                    results.append(
                        PoolingResult(strategy, layer, float("nan"), float("nan"))
                    )
                    continue
                direction = direction / direction_norm
                scores = x @ direction  # (V,)

                pos_scores = scores[y == 1]
                neg_scores = scores[y == 0]
                threshold = (pos_scores.mean() + neg_scores.mean()) / 2
                preds = (scores >= threshold).long()
                acc = (preds == y).float().mean().item()
                auroc = _compute_auroc(scores, y)

            results.append(PoolingResult(strategy, layer, auroc, acc))

    return results


def _compute_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank-based AUROC (Mann-Whitney U statistic). Handles ties."""
    scores = scores.detach().cpu().double()
    labels = labels.detach().cpu().long()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos = pos.shape[0]
    n_neg = neg.shape[0]
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)  # (n_pos, n_neg)
    wins = (diff > 0).sum().item()
    ties = (diff == 0).sum().item()
    return (wins + 0.5 * ties) / (n_pos * n_neg)


def print_ablation_summary(results: List[PoolingResult]) -> None:
    """Pretty-print the pooling ablation results."""
    print(f"\n{'Strategy':<12} {'Layer':<6} {'AUROC':<9} {'Accuracy':<9}")
    print("-" * 42)
    for r in results:
        if not (r.auroc != r.auroc):  # not NaN
            print(
                f"{r.strategy:<12} {r.layer:<6} "
                f"{r.auroc:<9.4f} {r.accuracy:<9.4f}"
            )
        else:
            print(f"{r.strategy:<12} {r.layer:<6} {'nan':<9} {'nan':<9}")


# ---------------------------------------------------------------------------
# 4. CLI demo / smoke test
# ---------------------------------------------------------------------------

def main():
    import os

    cached_paths = sorted(glob.glob("data/*_cached.pt"))
    if not cached_paths:
        print(
            "No cached .pt files found in data/. "
            "Run extract_and_cache.py first to generate them."
        )
        return

    for path in cached_paths:
        print(f"\n=== Pooling ablation on {os.path.basename(path)} ===")
        cached = load_cached_split(path)
        results = run_pooling_ablation(cached)
        print_ablation_summary(results)


if __name__ == "__main__":
    main()
