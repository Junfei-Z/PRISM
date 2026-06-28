"""
Soft Gating Module for PRISM Framework
Implements entropy-regularized routing mechanism for privacy-aware inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from typing import Tuple, Dict, List, Optional
from enum import Enum
import logging


class RoutingMode(Enum):
    """Execution path options for PRISM routing"""
    EDGE_ONLY = 0
    COLLABORATIVE = 1
    CLOUD_ONLY = 2


# The risk score R(P) (Eq. 1) is an unbounded sum of per-entity category
# weights. We scale it into [0, 1] for the gating feature vector by dividing by
# RISK_SCALE rather than hard-clipping at 1.0, which would collapse all
# multi-entity prompts to the same value and destroy the signal that separates
# edge-only (high risk) from collaborative (moderate risk) routing.
RISK_SCALE = 5.0


class SoftGatingModule(nn.Module):
    """
    Soft gating mechanism with entropy regularization.
    Maps sensitivity indicators to routing probability distribution.
    
    Architecture:
    - Input: Feature vector z ∈ R^{1+m} (risk score + sensitivity mask)
    - Hidden: Linear transformation with ReLU activation
    - Output: Probability distribution π over 3 routing modes
    """
    
    def __init__(self, 
                 input_dim: int = 11,  # 1 risk score + 10 max entities
                 hidden_dim: int = 64,
                 dropout_rate: float = 0.1):
        """
        Initialize soft gating module.
        
        Args:
            input_dim: Dimension of input feature vector
            hidden_dim: Dimension of hidden layer
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        # Network architecture
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 3)  # 3 routing modes
        
        # Activation functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        # Initialize weights
        self._initialize_weights()
        
        self.logger = logging.getLogger("SoftGatingModule")
        
    def _initialize_weights(self):
        """Initialize network weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the gating network.
        
        Args:
            z: Feature vector [batch_size, input_dim]
            
        Returns:
            π: Routing probability distribution [batch_size, 3]
        """
        # First layer. Batch norm needs >1 sample to compute batch statistics in
        # training mode, but in eval mode it uses the running statistics and is
        # valid for any batch size. Applying it in both cases keeps the
        # single-sample inference path consistent with training.
        h1 = self.fc1(z)
        if h1.shape[0] > 1 or not self.training:
            h1 = self.bn1(h1)
        h1 = self.relu(h1)
        h1 = self.dropout(h1)

        # Second layer
        h2 = self.fc2(h1)
        if h2.shape[0] > 1 or not self.training:
            h2 = self.bn2(h2)
        h2 = self.leaky_relu(h2)
        h2 = self.dropout(h2)
        
        # Output layer
        logits = self.fc3(h2)
        
        # Apply softmax to get probability distribution
        π = F.softmax(logits, dim=-1)
        
        return π
    
    def compute_entropy(self, π: torch.Tensor) -> torch.Tensor:
        """
        Compute Shannon entropy of routing distribution.
        H(π) = -Σ π_i log(π_i)
        
        Args:
            π: Routing probability distribution
            
        Returns:
            Entropy value
        """
        # Add small epsilon to avoid log(0)
        entropy = -torch.sum(π * torch.log(π + 1e-8), dim=-1)
        return entropy
    
    def compute_loss(self, 
                     π: torch.Tensor, 
                     labels: torch.Tensor,
                     lambda_entropy: float = 0.4) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss with entropy regularization.
        L = L_task + λ * H(π)
        
        Args:
            π: Predicted routing distribution
            labels: Ground truth routing decisions
            lambda_entropy: Entropy regularization weight
            
        Returns:
            Total loss and component losses
        """
        # Task loss (cross-entropy)
        task_loss = F.cross_entropy(torch.log(π + 1e-8), labels)
        
        # Entropy regularization (negative because we want to minimize entropy)
        entropy = self.compute_entropy(π)
        entropy_loss = -lambda_entropy * torch.mean(entropy)
        
        # Total loss
        total_loss = task_loss + entropy_loss
        
        # Return losses for logging
        losses = {
            'total': total_loss.item(),
            'task': task_loss.item(),
            'entropy': entropy_loss.item(),
            'avg_entropy': torch.mean(entropy).item()
        }
        
        return total_loss, losses
    
    def predict(self, z: torch.Tensor, deterministic: bool = True) -> Tuple[int, torch.Tensor]:
        """
        Make routing prediction.
        
        Args:
            z: Feature vector
            deterministic: If True, return argmax; if False, sample from distribution
            
        Returns:
            Routing decision and probability distribution
        """
        with torch.no_grad():
            π = self.forward(z)
            
            if deterministic:
                # Take argmax for deterministic routing
                decision = torch.argmax(π, dim=-1).item()
            else:
                # Sample from distribution
                decision = torch.multinomial(π, 1).item()
            
            return decision, π


class SoftGatingPredictor:
    """
    High-level interface for soft gating predictions.
    Handles model loading, feature extraction, and routing decisions.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = 'cpu'):
        """
        Initialize predictor with pretrained model.
        
        Args:
            model_path: Path to pretrained model weights
            device: Device to run inference on
        """
        self.device = torch.device(device)
        self.model = SoftGatingModule().to(self.device)
        self.model.eval()
        self.logger = logging.getLogger("SoftGatingPredictor")
        
        # Load trained weights. A missing checkpoint is a setup error rather
        # than something to silently fall back from: random weights would
        # produce meaningless routing.
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Soft gating checkpoint not found at '{model_path}'. "
                    "Train it first with `python train_soft_gating.py`.")
            self.load_model(model_path)
        else:
            self.logger.warning("No model_path provided; using random initialization "
                                "(intended only for unit tests).")
        
    def load_model(self, model_path: str):
        """Load pretrained model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Loaded pretrained model from {model_path}")
        
        # Log training metrics if available
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            self.logger.info(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
            self.logger.info(f"Best validation accuracy: {metrics.get('best_val_acc', 'unknown'):.2%}")
            if 'routing_distribution' in metrics:
                dist = metrics['routing_distribution']
                self.logger.info(f"Routing distribution - Edge: {dist.get('edge_only', 0):.1%}, "
                               f"Collab: {dist.get('collaborative', 0):.1%}, "
                               f"Cloud: {dist.get('cloud_only', 0):.1%}")
    
    def prepare_features(self, 
                        risk_score: float,
                        sensitivity_mask: List[int],
                        max_entities: int = 10) -> torch.Tensor:
        """
        Prepare feature vector for soft gating.
        
        Args:
            risk_score: Overall privacy risk score
            sensitivity_mask: Binary mask for entity protection
            max_entities: Maximum number of entities to consider
            
        Returns:
            Feature tensor ready for model input
        """
        # Normalize risk score to [0, 1] by scaling (see RISK_SCALE).
        normalized_risk = min(max(risk_score, 0.0) / RISK_SCALE, 1.0)
        
        # Prepare feature vector
        features = [normalized_risk]
        
        # Add sensitivity mask (pad or truncate to max_entities)
        mask_features = sensitivity_mask[:max_entities]
        while len(mask_features) < max_entities:
            mask_features.append(0)
        features.extend(mask_features)
        
        # Convert to tensor
        z = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        return z
    
    def route(self, 
              risk_score: float,
              sensitivity_mask: List[int],
              return_probs: bool = False) -> Tuple[RoutingMode, Optional[Dict[str, float]]]:
        """
        Determine routing decision based on input features.
        
        Args:
            risk_score: Overall privacy risk score
            sensitivity_mask: Binary mask for entity protection
            return_probs: Whether to return probability distribution
            
        Returns:
            Routing mode and optionally probability distribution
        """
        # Prepare features
        z = self.prepare_features(risk_score, sensitivity_mask)
        
        # Get prediction
        decision_idx, π = self.model.predict(z, deterministic=True)
        
        # Map to routing mode
        routing_mode = RoutingMode(decision_idx)
        
        # Prepare probability distribution if requested
        probs = None
        if return_probs:
            probs = {
                'edge_only': π[0, 0].item(),
                'collaborative': π[0, 1].item(),
                'cloud_only': π[0, 2].item(),
                'entropy': self.model.compute_entropy(π).item()
            }
            
        self.logger.info(f"Routing decision: {routing_mode.name} "
                        f"(π=[{π[0,0]:.3f}, {π[0,1]:.3f}, {π[0,2]:.3f}])")
        
        return routing_mode, probs