import jax.numpy as jnp


def pad_vertices(vertices_list):
    # 1. Get the global maximum
    max_nodes = max(v.shape[0] for v in vertices_list)
    padded_list = []
    masks_list = []
    for v in vertices_list:
        n = v.shape[0]
        # Create zero-padded array (Max_Nodes, 3)
        padded = jnp.zeros((max_nodes, 3))
        padded = padded.at[:n, :].set(v)
        
        # Create mask (Max_Nodes,)
        mask = jnp.zeros((max_nodes,))
        mask = mask.at[:n].set(1.0)
        
        padded_list.append(padded)
        masks_list.append(mask)

    return jnp.stack(padded_list), jnp.stack(masks_list)

def recover_original_list(padded_verts, padding_mask):
    """
    Args:
        padded_verts: (Batch, Max_Nodes, 3)
        padding_mask: (Batch, Max_Nodes) - values 1.0 for valid, 0.0 for padded
    Returns:
        A list of length Batch, where each element is (N_i, 3)
    """
    original_list = []
    # Iterate over the batch dimension
    for i in range(padded_verts.shape[0]):
        # Select the mask for this specific sample
        mask_i = padding_mask[i] == 1.0
        # Select only the rows where mask is True
        original_list.append(padded_verts[i][mask_i])
    return original_list
