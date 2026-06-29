import jax
import jax.numpy as jnp
import jax.nn as jnn
import math

def kl_divergence_loss(mean, log_var):
    """
    Computes the KL divergence between N(mean, exp(log_var)) 
    and N(0, 1).
    """
    # 0.5 * sum(exp(log_var) + mean^2 - 1 - log_var)
    kl_loss = -0.5 * jnp.sum(1 + log_var - jnp.square(mean) - jnp.exp(log_var))
    return kl_loss

def geometric_clustering_loss(logits, edge_index, smoothness_weight=1.0, balance_weight=0.1, entropy_weight = 0.1):
    # 1. Use log_softmax for numerical stability
    log_probs = jax.nn.log_softmax(logits, axis=1)
    probs = jnp.exp(log_probs)
    
    # 2. Entropy (using log_probs directly is more stable)
    entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=1))
    
    # 3. Smoothness Loss
    row, col = edge_index
    # Using L2 norm squared is often more stable for gradients than raw L2 norm
    diff = jnp.sum((probs[row] - probs[col])**2, axis=1)
    smoothness_loss = jnp.mean(diff)
    
    # 4. Balance Loss
    mean_probs = jnp.mean(probs, axis=0)
    # Target 0.5 for binary classification
    balance_loss = jnp.mean((mean_probs - 0.5)**2)
    
    return (smoothness_weight * smoothness_loss + 
            balance_weight * balance_loss 
            + entropy_weight* entropy)

def laplacian_loss(pred_pos):
    """
    Infers grid_size from pred_pos shape.
    pred_pos: [Batch, N_samples, 3]
    """
    n_samples = pred_pos.shape[1]
    grid_size = int(math.sqrt(n_samples))
    
    # Assert that it is a perfect square to ensure grid validity
    assert grid_size * grid_size == n_samples, f"N_samples {n_samples} is not a perfect square!"
    
    # Reshape to 2D grid structure
    grid = pred_pos.reshape(-1, grid_size, grid_size, 3)
    
    # Finite difference
    diff_u = grid[:, 1:, :, :] - grid[:, :-1, :, :]
    diff_v = grid[:, :, 1:, :] - grid[:, :, :-1, :]
    
    return jnp.mean(jnp.square(diff_u)) + jnp.mean(jnp.square(diff_v))

def soft_chamfer_loss(pred_pos, target_pos, target_mask, epsilon=0.01):
    # diff: [B, N_pred, N_target, 3]
    diff = pred_pos[:, :, None, :] - target_pos[:, None, :, :]
    dist_sq = jnp.sum(jnp.square(diff), axis=-1)
    
    # Masking: Use a large constant to ignore padding
    dist_sq = jnp.where(target_mask[:, None, :], dist_sq, 1e6)
    
    # Softmin via Log-Sum-Exp: Smooths the transition between points
    # Epsilon controls how "sharp" the selection is. 
    # Smaller epsilon = closer to hard min, but harder to train.
    dist_p_to_t = -epsilon * jax.nn.logsumexp(-dist_sq / epsilon, axis=2)
    dist_t_to_p = -epsilon * jax.nn.logsumexp(-dist_sq / epsilon, axis=1)
    
    term1 = jnp.mean(dist_p_to_t)
    term2 = jnp.sum(dist_t_to_p * target_mask) / (jnp.sum(target_mask) + 1e-6)
    
    return term1 + term2

def combined_surface_loss_(pred_pos, target_pos, target_mask, 
                                 laplacian_weight=0.1, epsilon=0.01):
    """
    A stable, soft version of the combined surface loss using Log-Sum-Exp
    for differentiable Chamfer and Laplacian regularization.
    """
    print("pred_pos: ", pred_pos)
    chamfer_loss = soft_chamfer_loss(pred_pos, target_pos, target_mask, epsilon=0.01)
    
    # 5. Stable Laplacian loss
    lap_loss = laplacian_loss(pred_pos)
    loss = chamfer_loss + (laplacian_weight * lap_loss)
    print("loss: ", loss)
    return loss


def combined_surface_loss(pred_pos, target_pos, target_mask, laplacian_weight=0.1):
    # Use a large constant instead of jnp.inf to keep values in f32 range
    LARGE_VAL = 1e6 
    
    diff = pred_pos[:, :, None, :] - target_pos[:, None, :, :]
    dist_sq = jnp.sum(jnp.square(diff), axis=-1)
    
    # Masking: Clamp values rather than using inf
    dist_sq = jnp.where(target_mask[:, None, :], dist_sq, LARGE_VAL)
    
    dist_p_to_t = jnp.min(dist_sq, axis=2)
    # Ensure we don't have remaining large values by clipping
    dist_p_to_t = jnp.where(dist_p_to_t >= LARGE_VAL, 0.0, dist_p_to_t)
    term1 = jnp.mean(dist_p_to_t)
    
    # Target -> Pred
    dist_t_to_p = jnp.min(dist_sq, axis=1)
    dist_t_to_p = jnp.where(dist_t_to_p >= LARGE_VAL, 0.0, dist_t_to_p)
    
    # Use a safer denominator
    mask_sum = jnp.sum(target_mask)
    term2 = jnp.sum(dist_t_to_p * target_mask) / (mask_sum + 1e-6)
    
    chamfer_loss = term1 + term2
    
    # Add small eps to Laplacian to avoid zero-gradient issues
    lap_loss = laplacian_loss(pred_pos)
    return chamfer_loss + (laplacian_weight * (lap_loss + 1e-8))

def combined_surface_loss_(pred_pos, target_pos, target_mask, laplacian_weight=0.1):
    """
    pred_pos: [Batch, N_pred, 3]
    target_pos: [Batch, Max_Nodes_Target, 3]
    target_mask: [Batch, Max_Nodes_Target]
    """
    # 1. Chamfer Distance (The data fidelity term)
    diff = pred_pos[:, :, None, :] - target_pos[:, None, :, :]
    dist_sq = jnp.sum(jnp.square(diff), axis=-1)
    
    # Mask padding: set to infinity so min() ignores them
    dist_sq = jnp.where(target_mask[:, None, :], dist_sq, jnp.inf)
    
    # Term 1: Pred -> Target
    dist_p_to_t = jnp.min(dist_sq, axis=2)
    dist_p_to_t = jnp.where(jnp.isinf(dist_p_to_t), 0.0, dist_p_to_t)
    term1 = jnp.mean(dist_p_to_t)
    
    # Term 2: Target -> Pred
    dist_t_to_p = jnp.min(dist_sq, axis=1)
    dist_t_to_p = jnp.where(jnp.isinf(dist_t_to_p), 0.0, dist_t_to_p)
    term2 = jnp.sum(dist_t_to_p * target_mask) / (jnp.sum(target_mask) + 1e-8)
    
    chamfer_loss = term1 + term2
    
    # 2. Laplacian Loss (The surface connectivity term)
    lap_loss = laplacian_loss(pred_pos)
    
    return chamfer_loss + (laplacian_weight * lap_loss)

def point_to_surface_loss_chamfer(pred_pos, target_pos, target_mask):
    """
    pred_pos: (Batch, N_pred, 3)
    target_pos: (Batch, Max_Nodes_Target, 3)
    target_mask: (Batch, Max_Nodes_Target)
    """
    # 1. Compute squared distance matrix
    diff = pred_pos[:, :, None, :] - target_pos[:, None, :, :]
    dist_sq = jnp.sum(jnp.square(diff), axis=-1)

    # 2. Mask target points: set distances to padded entries as infinity
    dist_sq = jnp.where(target_mask[:, None, :], dist_sq, jnp.inf)

    # 3. Term 1: Pred -> Target
    # min over target points (axis 2). Replace any resulting INF with 0
    # (this happens if a pred point has no valid target point to map to)
    dist_p_to_t = jnp.min(dist_sq, axis=2)
    dist_p_to_t = jnp.where(jnp.isinf(dist_p_to_t), 0.0, dist_p_to_t)
    term1 = jnp.mean(dist_p_to_t)

    # 4. Term 2: Target -> Pred
    # min over predicted points (axis 1).
    dist_t_to_p = jnp.min(dist_sq, axis=1)
    
    # Check for INF: If a target point has no "near" prediction (e.g. model output collapse),
    # min returns INF. We must zero these out before averaging to avoid NaNs.
    dist_t_to_p = jnp.where(jnp.isinf(dist_t_to_p), 0.0, dist_t_to_p)
    
    # Calculate weighted average using the mask
    term2 = jnp.sum(dist_t_to_p * target_mask) / (jnp.sum(target_mask) + 1e-8)

    return term1 + term2

def point_to_surface_loss_chamfer_(pred_pos, target_pos, target_mask):
    """
    pred_pos: (Batch, N_pred, 3) - Fixed size from model output
    target_pos: (Batch, Max_Nodes_Target, 3) - Padded ground truth
    target_mask: (Batch, Max_Nodes_Target) - Mask for ground truth points
    """
    # 1. Compute squared distance matrix: (Batch, N_pred, Max_Nodes_Target)
    diff = pred_pos[:, :, None, :] - target_pos[:, None, :, :]
    dist_sq = jnp.sum(jnp.square(diff), axis=-1)

    # 2. Mask target points (set padded distances to infinity)
    dist_sq = jnp.where(target_mask[:, None, :], dist_sq, jnp.inf)

    # 3. Term 1: For each pred point, find nearest GT point
    # L1: Average distance from each pred point to its nearest target point
    dist_p_to_t = jnp.min(dist_sq, axis=2)
    term1 = jnp.mean(dist_p_to_t)

    # 4. Term 2: For each GT point, find nearest pred point
    # L2: Average distance from each target point to its nearest pred point
    # We must weight this by the mask to ignore padded GT points
    dist_t_to_p = jnp.min(dist_sq, axis=1)
    term2 = jnp.sum(dist_t_to_p * target_mask) / (jnp.sum(target_mask) + 1e-8)

    # Total Chamfer Loss
    return term1 + term2
    
def point_to_surface_loss(pred_points, target_points, target_mask):
    """
    Args:
        pred_points: [batch, num_pred, 3]
        target_points: [batch, max_nodes, 3]
        target_mask: [batch, max_nodes] (Boolean: True=real, False=padding)
    """
    target_idx =  jnp.where(target_mask[:, None, :], dist_sq, 1e9)
    target_points =target_points[:,target_idx, : ]

    # 1. Standard distance matrix
    diff = pred_points[:, :, None, :] - target_points[:, None, :, :]
    dist_sq = jnp.sum(jnp.square(diff), axis=-1)
    
    # 3. Min distance
    min_dist_per_pred = jnp.min(dist_sq, axis=2)
    
    # 4. Final mean (only over the actual predicted points)
    return jnp.mean(min_dist_per_pred)