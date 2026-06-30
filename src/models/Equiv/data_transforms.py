import jraph
import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import jraph


def get_y_rot(t):
    return jnp.array([
                 [jnp.cos(t),  0, jnp.sin(t)],
                    [0,           1, 0],
                    [-jnp.sin(t), 0, jnp.cos(t)]
        ])




def transform_graphs_explicit_jax(key, graph, rotations, translations, permute=False, r_max=0.1):
    n_graphs = graph.n_node.shape[0]
    
    # Pre-calculate offsets without using jnp.repeat
    # We use jnp.cumsum, which is JIT-friendly
    node_offsets = jnp.concatenate([jnp.array([0]), jnp.cumsum(graph.n_node)])
    
    def scan_fn(carry, i):
        start, end = node_offsets[i], node_offsets[i+1]
        
        # Extract slices
        p = jax.lax.dynamic_slice_in_dim(graph.nodes, start, end - start)
        R = rotations[i]
        t = translations[i]
        
        # Transformation
        p_new = jnp.dot(p, R.T) + t
        
        # Optional Permutation
        if permute:
            subkey = jax.random.fold_in(key, i)
            p_new = jax.random.permutation(subkey, p_new, axis=0)
            
        return carry, p_new

    # Scan over the range of graphs instead of using repeat/einsum with dynamic arrays
    _, nodes_list = jax.lax.scan(scan_fn, None, jnp.arange(n_graphs))
    
    # Flatten the list of nodes back into a single array
    # Note: If n_node varies per graph, you'll need to pad 
    # to a fixed max_nodes per graph to keep this JIT-stable.
    return graph._replace(nodes=jnp.concatenate(nodes_list))

def transform_graphs_explicit(key, graph, rotations, translations, permute=False, r_max= 0.1):
    """
    Applies rotations, translations, and optional permutations to a batched graph.
    
    Args:
        key: JAX PRNGKey.
        graph: Batched jraph.GraphsTuple.
        rotations: Array of shape [n_graphs, 3, 3].
        translations: Array of shape [n_graphs, 3].
        permute: Boolean, whether to shuffle node order.
    """
    n_graphs = graph.n_node.shape[0]
    keys = jax.random.split(key, n_graphs)
    
    # Calculate slice indices for each graph in the batch
    node_offsets = jnp.concatenate([jnp.array([0]), jnp.cumsum(graph.n_node)])
    new_graphs = []

    for i in range(n_graphs):
        # Extract nodes for this specific graph
        start, end = node_offsets[i], node_offsets[i+1]
        p = graph.nodes[start:end]
        
        # Apply transformation: p' = p @ R^T + t
        p = jnp.dot(p, rotations[i].T) + translations[i]
        
        # Random Permutation
        if permute:
            p = jax.random.permutation(keys[i], p, axis=0)

        # Recalculate edges based on transformed positions
        senders, receivers = e3nn.radius_graph(p, r_max)
        
        new_graphs.append(
            jraph.GraphsTuple(
                nodes=p, 
                edges=None, 
                globals=None,
                senders=senders, 
                receivers=receivers,
                n_node=graph.n_node[i:i+1], # Use original count for this graph
                n_edge=jnp.array([len(senders)]), # Updated count based on new connectivity
            )
        )
        
    return jraph.batch(new_graphs)