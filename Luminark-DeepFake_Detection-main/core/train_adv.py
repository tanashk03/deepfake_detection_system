"""
Adversarial Training for Deepfake Detection

Implements PGD adversarial training to improve robustness.

Reference:
    Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


class PGDAttack:
    """
    Projected Gradient Descent adversarial attack.
    
    Iteratively perturbs input to maximize loss while staying
    within epsilon-ball of original input.
    """
    
    def __init__(
        self,
        model: nn.Module,
        eps: float = 0.03,         # L-inf perturbation budget
        alpha: float = 0.007,      # Step size
        steps: int = 10,           # Number of PGD steps
        random_start: bool = True, # Random initialization
    ):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
    
    def perturb(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable = nn.CrossEntropyLoss(),
    ) -> torch.Tensor:
        """
        Generate adversarial examples using PGD.
        
        Args:
            x: Input images (B, C, H, W)
            y: True labels (B,)
            loss_fn: Loss function to maximize
            
        Returns:
            Adversarial examples
        """
        x_adv = x.clone().detach()
        
        # Random start
        if self.random_start:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv, 0, 1)
        
        for _ in range(self.steps):
            x_adv.requires_grad = True
            
            # Forward pass
            outputs = self.model(x_adv)
            loss = loss_fn(outputs, y)
            
            # Backward pass
            loss.backward()
            
            # PGD step
            with torch.no_grad():
                grad_sign = x_adv.grad.sign()
                x_adv = x_adv + self.alpha * grad_sign
                
                # Project back to epsilon ball
                delta = torch.clamp(x_adv - x, -self.eps, self.eps)
                x_adv = torch.clamp(x + delta, 0, 1)
            
            x_adv = x_adv.detach()
        
        return x_adv


class FGSMAttack:
    """
    Fast Gradient Sign Method (single-step PGD).
    
    Faster but less effective than PGD.
    """
    
    def __init__(self, model: nn.Module, eps: float = 0.03):
        self.model = model
        self.eps = eps
    
    def perturb(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable = nn.CrossEntropyLoss(),
    ) -> torch.Tensor:
        """Generate FGSM adversarial examples."""
        x_adv = x.clone().detach().requires_grad_(True)
        
        outputs = self.model(x_adv)
        loss = loss_fn(outputs, y)
        loss.backward()
        
        with torch.no_grad():
            x_adv = x_adv + self.eps * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv


class AdversarialTrainer:
    """
    Adversarial training loop.
    
    Mixes clean and adversarial examples during training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        attack: Optional[PGDAttack] = None,
        adv_ratio: float = 0.5,      # Ratio of adversarial examples
        device: str = 'cpu',
    ):
        self.model = model
        self.optimizer = optimizer
        self.attack = attack or PGDAttack(model)
        self.adv_ratio = adv_ratio
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
    
    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Single training step with adversarial examples.
        
        Returns:
            (clean_loss, adv_loss)
        """
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)
        
        # Split batch
        batch_size = x.size(0)
        n_adv = int(batch_size * self.adv_ratio)
        
        x_clean = x[:batch_size - n_adv]
        y_clean = y[:batch_size - n_adv]
        x_adv_base = x[batch_size - n_adv:]
        y_adv = y[batch_size - n_adv:]
        
        # Generate adversarial examples
        with torch.enable_grad():
            x_adv = self.attack.perturb(x_adv_base, y_adv, self.loss_fn)
        
        # Combine
        x_combined = torch.cat([x_clean, x_adv], dim=0)
        y_combined = torch.cat([y_clean, y_adv], dim=0)
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(x_combined)
        loss = self.loss_fn(outputs, y_combined)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Compute separate losses for logging
        with torch.no_grad():
            clean_loss = self.loss_fn(outputs[:len(y_clean)], y_clean).item()
            adv_loss = self.loss_fn(outputs[len(y_clean):], y_adv).item()
        
        return clean_loss, adv_loss
    
    def evaluate_robustness(
        self,
        dataloader,
        attack: Optional[PGDAttack] = None,
    ) -> dict:
        """
        Evaluate model robustness against attacks.
        
        Returns:
            Dictionary with clean and adversarial accuracy
        """
        self.model.eval()
        attack = attack or self.attack
        
        clean_correct = 0
        adv_correct = 0
        total = 0
        
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            batch_size = x.size(0)
            
            # Clean accuracy
            with torch.no_grad():
                outputs = self.model(x)
                pred = outputs.argmax(dim=1)
                clean_correct += (pred == y).sum().item()
            
            # Adversarial accuracy
            x_adv = attack.perturb(x, y, self.loss_fn)
            with torch.no_grad():
                outputs_adv = self.model(x_adv)
                pred_adv = outputs_adv.argmax(dim=1)
                adv_correct += (pred_adv == y).sum().item()
            
            total += batch_size
        
        return {
            'clean_accuracy': clean_correct / total,
            'adversarial_accuracy': adv_correct / total,
            'robustness_gap': (clean_correct - adv_correct) / total,
        }


def create_adversarial_augmentation(
    eps_range: Tuple[float, float] = (0.01, 0.05),
    prob: float = 0.3,
) -> Callable:
    """
    Create data augmentation that randomly applies FGSM.
    
    For use during training data augmentation pipeline.
    """
    def augment(x: torch.Tensor, model: nn.Module, y: torch.Tensor) -> torch.Tensor:
        if np.random.random() > prob:
            return x
        
        eps = np.random.uniform(eps_range[0], eps_range[1])
        attack = FGSMAttack(model, eps)
        
        return attack.perturb(x, y)
    
    return augment


# Training script entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Adversarial training for Luminark")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--adv-ratio", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="weights/adv_trained.pt")
    
    args = parser.parse_args()
    
    print(f"Adversarial training with eps={args.eps}, ratio={args.adv_ratio}")
    print(f"This is a stub - implement full training loop with your dataset")
