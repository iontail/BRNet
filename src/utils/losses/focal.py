import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        """
        Focal Loss for binary classification

        Args:
            alpha (float): Class weighting factor (default: 1.0)
            gamma (float): Focusing parameter (default: 2.0)
            reduction (str): 'none', 'mean', or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model outputs (logits), shape [B, image_size, image_size, 1]
            targets: Ground truth labels (binary), shape [B, image_size, image_size, 1]
        Returns:
            Focal loss value
        """
        # Flatten tensors to [B, -1]
        inputs = inputs.view(inputs.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)

        # Compute BCE Loss (without sigmoid)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Compute Focal Loss weight
        probas = torch.sigmoid(inputs)  # Convert logits to probabilities
        focal_weight = (1 - probas) ** self.gamma * targets + probas ** self.gamma * (1 - targets)
        focal_loss = self.alpha * focal_weight * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss  # 'none'일 경우