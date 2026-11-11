"""
custom_loss.py
--------------
Course: Foundation of Large Language Models (AI60213)
Project: Step-INSPECTOR
Track: Member 3 (Causal Masking & Loss Dynamics)

Defines the Causal Masked Step Binary Cross-Entropy Loss module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedStepBCELoss(nn.Module):
    """
    Causal Masked Step Binary Cross Entropy Loss.
    
    Mathematical Formulation:
        L = [ \sum_{m=1}^M m_m * BCE(p_hat_m, y_m) ] / [ \sum_{m=1}^M m_m + \epsilon ]
    
    Where:
        m_m = 1.0 for valid deductive steps up to the first error (m <= k*)
        m_m = 0.0 for steps following an error (m > k*) to prevent false-premise noise.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (Batch_Size, Max_Steps) raw unnormalized model predictions.
            targets: (Batch_Size, Max_Steps) binary labels (0.0 or 1.0).
            masks:   (Batch_Size, Max_Steps) supervision mask (1.0 = supervise, 0.0 = ignore).
        Returns:
            Scalar tensor containing the step-normalized loss.
        """
        if targets.shape != logits.shape:
            targets = targets.view_as(logits)
        if masks.shape != logits.shape:
            masks = masks.view_as(logits)

        # Adapt precision to logits
        targets = targets.to(dtype=logits.dtype)
        masks = masks.to(dtype=logits.dtype)

        # Compute unreduced step wise BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Apply causal mask and normalize by valid step count
        masked_bce = bce * masks
        loss = masked_bce.sum() / (masks.sum() + self.eps)
        return loss

if __name__ == "__main__":
    # Self contained Unit Test
    loss_fn = MaskedStepBCELoss()
    dummy_logits = torch.tensor([[2.0, -1.0, 3.0], [-2.0, 1.0, -1.0]])
    dummy_targets = torch.tensor([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    dummy_masks = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]) # Step 3 masked out
    
    test_loss = loss_fn(dummy_logits, dummy_targets, dummy_masks)
    assert not torch.isnan(test_loss), "Loss computed as NaN"
    print(f"Unit Test Passed MaskedStepBCELoss successfully computed: {test_loss.item():.4f}")