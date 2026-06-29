import jraph
import jax.numpy as jnp
import e3nn_jax as e3nn
from e3nn_jax import IrrepsArray
import jax
from functools import partial

@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7)) 
def make_single_graph_jit(nodes, key, mask, r_max, dropout_rate, noise_std, max_nodes, max_edges):
    # Now this function has 8 arguments (indices 0 to 7)
    # Argument 3: r_max
    # Argument 4: dropout_rate
    # Argument 5: noise_std
    # Argument 6: max_nodes
    # Argument 7: max_edges
    
    k1, k2, k3 = jax.random.split(key, 3)
    
    # Apply Dropout
    drop_mask = jax.random.uniform(k1, shape=(nodes.shape[0],)) > dropout_rate
    final_mask = mask * drop_mask.astype(jnp.float32)
    
    # Apply Gaussian Noise
    noise = jax.random.normal(k2, shape=nodes.shape) * noise_std
    nodes = nodes + (noise * final_mask[..., jnp.newaxis])
    
    # Radius Graph calculation
    dist = jnp.sum((nodes[:, None, :] - nodes[None, :, :])**2, axis=-1)
    adj = (dist < r_max**2) & (final_mask[:, None] > 0) & (final_mask[None, :] > 0)
    adj = adj.at[jnp.diag_indices(max_nodes)].set(False)

    # Use fixed size for JIT compilation
    senders, receivers = jnp.where(adj, size=max_edges, fill_value=-1)
    
    return jraph.GraphsTuple(
        nodes=nodes, 
        edges=None, 
        globals=jnp.array([]),
        senders=senders, 
        receivers=receivers,
        n_node=jnp.array([max_nodes]), 
        n_edge=jnp.array([max_edges]) 
    ), final_mask

def make_graphs_from_vertices(
    vertices_list: list, 
    key: jax.random.PRNGKey,
    r_max: float = 1.1,
    dropout_rate: float = 0.1,
    noise_std: float = 0.05
) -> jraph.GraphsTuple:
    """
    Constructs a batched graph with vertex dropout and Gaussian noise.
    """
    graphs = []
    keys = jax.random.split(key, len(vertices_list))

    for i, nodes in enumerate(vertices_list):
        nodes = jnp.array(nodes, dtype=jnp.float32)
        k1, k2 = jax.random.split(keys[i])

        # 1. Random Dropout (Uniformly sampling a subset of nodes)
        if dropout_rate > 0:
            probs = jax.random.uniform(k1, shape=(nodes.shape[0],))
            mask = probs > dropout_rate
            nodes = nodes[mask]

        # 2. Add Gaussian Noise to node positions
        if noise_std > 0:
            noise = jax.random.normal(k2, shape=nodes.shape) * noise_std
            nodes = nodes + noise
        
        num_nodes = nodes.shape[0]
        senders, receivers = e3nn.radius_graph(nodes, r_max=r_max)
        
        graphs.append(
            jraph.GraphsTuple(
                nodes=nodes,
                edges=None,
                globals=jnp.array([i]),
                senders=senders,
                receivers=receivers,
                n_node=jnp.array([num_nodes]),
                n_edge=jnp.array([len(senders)]),
            )
        )

    return jraph.batch(graphs)


def get_vertices_from_graph(graph: jraph.GraphsTuple) -> list:
    """
    Splits the concatenated node features of a batched jraph.GraphsTuple
    back into a list of individual vertex arrays.
    
    Args:
        graph: The batched jraph.GraphsTuple.
        
    Returns:
        A list of arrays, where each array is of shape [num_nodes_i, feature_dim_i].
    """
    # Calculate the split points based on the number of nodes per graph
    # We use cumsum to get indices like [0, 4, 8, 12...]
    split_indices = jnp.cumsum(graph.n_node[:-1])
    
    # Use jnp.split to partition the node array
    nodes_split = jnp.split(graph.nodes, split_indices)
    
    return nodes_split
def unbatch_graphs(graph: jraph.GraphsTuple) -> list[jraph.GraphsTuple]:
    """
    Splits a batched jraph.GraphsTuple into a list of individual GraphsTuples,
    handling standard arrays and e3nn.IrrepsArray objects.
    """
    # 1. Calculate split indices
    node_split_indices = jnp.cumsum(graph.n_node[:-1])
    edge_split_indices = jnp.cumsum(graph.n_edge[:-1])
    
    # 2. Split Nodes (Handling IrrepsArray or standard jnp.ndarray)
    if isinstance(graph.nodes, IrrepsArray):
        nodes_split = [
            IrrepsArray(graph.nodes.irreps, chunk) 
            for chunk in jnp.split(graph.nodes.array, node_split_indices)
        ]
    else:
        nodes_split = jnp.split(graph.nodes, node_split_indices)
    
    # 3. Split Edges (senders and receivers)
    senders_split = jnp.split(graph.senders, edge_split_indices)
    receivers_split = jnp.split(graph.receivers, edge_split_indices)
    
    # 4. Handle globals
    if graph.globals is not None:
        # If globals is a single array, split it; otherwise assume it's already a list/array
        globals_split = jnp.split(graph.globals, len(graph.n_node))
    else:
        globals_split = [None] * len(graph.n_node)

    # 5. Reconstruct individual GraphsTuples
    individual_graphs = []
    for i in range(len(graph.n_node)):
        # Calculate local offset to adjust indices (senders/receivers must be 0-indexed per graph)
        offset = jnp.sum(graph.n_node[:i])
        
        individual_graph = jraph.GraphsTuple(
            nodes=nodes_split[i],
            edges=None,  # Adjust if your graph has edge features
            globals=globals_split[i],
            senders=senders_split[i] - offset,
            receivers=receivers_split[i] - offset,
            n_node=graph.n_node[i:i+1],
            n_edge=graph.n_edge[i:i+1]
        )
        individual_graphs.append(individual_graph)
        
    return individual_graphs