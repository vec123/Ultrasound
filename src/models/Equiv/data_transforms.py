import jraph
import jax
import jax.numpy as jnp
import e3nn_jax as e3nn

def get_y_rot(t):
    return jnp.array([
                 [jnp.cos(t),  0, jnp.sin(t)],
                    [0,           1, 0],
                    [-jnp.sin(t), 0, jnp.cos(t)]
        ])

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
                globals=graph.globals[i:i+1],
                senders=senders, 
                receivers=receivers,
                n_node=graph.n_node[i:i+1], # Use original count for this graph
                n_edge=jnp.array([len(senders)]), # Updated count based on new connectivity
            )
        )
        
    return jraph.batch(new_graphs)