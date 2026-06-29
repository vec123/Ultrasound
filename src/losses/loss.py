import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometricClusteringLoss(nn.Module):
    def __init__(self, smoothness_weight=1.0, balance_weight=0.1):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.balance_weight = balance_weight

    def forward(self, logits, edge_index, geometric_sim=None):
        """
        logits: (n, 2) tensor of scalar features
        edge_index: (2, E) tensor of neighbor indices
        geometric_sim: (E,) weights for edges (e.g., normal similarity)
        """
        probs = F.softmax(logits, dim=1)
        
        # 1. Smoothness Loss: Penalize high differences between neighbors
        # We use the edge_index to pull neighbor probabilities together
        row, col = edge_index
        diff = torch.norm(probs[row] - probs[col], p=2, dim=1)
        
        if geometric_sim is not None:
            smoothness_loss = (diff * geometric_sim).mean()
        else:
            smoothness_loss = diff.mean()
            
        # 2. Entropy Loss: Encourage high confidence (sharp clusters)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1).mean()
        
        # 3. Balance Loss: Prevent one cluster from taking over all points
        mean_probs = probs.mean(dim=0)
        balance_loss = torch.sum((mean_probs - 0.5)**2)
        
        return (self.smoothness_weight * smoothness_loss + 
                self.balance_weight * (entropy + balance_loss))

# Usage Example:
# logits = model(data)
# loss = GeometricClusteringLoss()(logits, data.edge_index)