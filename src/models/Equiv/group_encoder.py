
import jax
import jax.numpy as jnp
import flax.linen as nn
import e3nn_jax as e3nn
from src.models.Equiv.layers import Layer
from src.models.Equiv.layers import EquiLayer, EquiLayerCone
import jraph

from src.models.Equiv.data import make_graphs_from_vertices, get_vertices_from_graph, unbatch_graphs
from src.models.Equiv.vtk import save_graphs_as_vtp


def get_segment_ids(n_node):
    return jnp.repeat(jnp.arange(len(n_node)), n_node)

class SimpleEncoder(nn.Module):
    verbose: bool = False
    latent_dim: int = 5
    @nn.compact
    def __call__(self, graphs, break_symmetry = False):
     
        # Setup Input Features (Similar to your OHE/ones init)
        positions = e3nn.IrrepsArray("1o", graphs.nodes)
        if self.verbose:
            print(">>>>>>>>>>Encoding: ", graphs.nodes.shape)
        if break_symmetry:
          node_indices = jnp.arange(len(positions)).reshape(-1, 1)
          node_feats = e3nn.IrrepsArray("1x0e", node_indices)
          graphs = graphs._replace(nodes=node_feats)
        else:
            graphs = graphs._replace(nodes=jnp.ones((len(positions), 1)))
            node_feats = e3nn.IrrepsArray("1x0e", jnp.ones((len(positions), 1)))

        graphs = graphs._replace(nodes=node_feats)

        # Define Message Passing Layers 
        history = {}
        embedding_irreps = f"{self.latent_dim}x0e + 2x1o"
        layer_irreps = 2 * ["32x0e + 32x0o + 16x1e + 16x1o"] + [embedding_irreps]
        
        num_output_nodes_list = [80, 20, 3]

        assert len(num_output_nodes_list) == len(layer_irreps)
        variances = {}
        for i, irreps_str in enumerate(layer_irreps):
            graphs = EquiLayer(target_irreps=irreps_str,  verbose = self.verbose)(graphs, positions)
            #graphs, positions = EquiLayerCone(
            #    target_irreps=irreps_str, 
            #    num_output_nodes = num_output_nodes_list[i],
            #    verbose = self.verbose )(graphs, positions)
            history[f"layer_{i}"] = graphs
            scalars = graphs.nodes.filtered("0e").array
            variances[f"layer_{i}"] = jnp.var(scalars, axis=0)
            if self.verbose:
                print(f">>>>>>>>>>Layer {i} produced encoding: ", graphs.nodes.shape, graphs.nodes.irreps)

        if self.verbose:
            print("Variances per layer:", variances)
            print(f"----------- Finished-----------")
        
        
        # Global Pooling
        embeddings = graphs.nodes
        v_out = e3nn.flax.Linear(embedding_irreps, force_irreps_out=False)(embeddings)
        scalars = v_out.filtered("0e").array
        vectors = v_out.filtered(keep="1o").array   


        #vector_norms = jnp.linalg.norm(vectors, axis=-1, keepdims=True) - Produces Nans if 0 vectors
        invariant_features = scalars #jnp.concatenate([scalars], axis=-1)

        if self.verbose:
            print("scalars (latent_dim + num_vectors)", scalars.shape)
            print("vectors", vectors.shape)
            print("invariant_features", invariant_features.shape)
        
        mu_nodes = nn.Dense(self.latent_dim)(invariant_features)
        weights = nn.Dense(1)(invariant_features)
        weights = jax.nn.softmax(weights) # Softmax ensures nodes sum to 1 per graph
        mu = e3nn.scatter_mean(weights*mu_nodes, nel=graphs.n_node)

        var_nodes = nn.softplus(nn.Dense(self.latent_dim)(invariant_features))
        weights = nn.Dense(1)(invariant_features)
        weights = jax.nn.softmax(weights) # Softmax ensures nodes sum to 1 per graph
        z_var_graph = e3nn.scatter_mean(weights*var_nodes, nel=graphs.n_node)
        logvar = jnp.log(z_var_graph + 1e-8)
        
        if self.verbose:
            print("mu_nodes: ", mu_nodes)
            print("mu_nodes.shape", mu_nodes.shape)
            print("mu: ", mu)
            print("mu.shape", mu.shape)
            print("var_nodes.shape", var_nodes.shape)
            print("logvar: ", logvar)
            print("logvar.shape", logvar.shape)

        vectors_ir = graphs.nodes.filtered("1o")
        weights = nn.Dense(1)(invariant_features)
        weights = jax.nn.softmax(weights)
        weighted_vectors = vectors_ir * weights
        vector_graph = e3nn.scatter_sum(weighted_vectors, nel=graphs.n_node)
        vout_raw = vector_graph.array 


        vout = vout_raw.reshape((-1, 2, 3)) 
        v1 = vout[:, 0, :]
        v2 = vout[:, 1, :]
        rot_matrix = self.get_rotation_matrix_from_two_vector(v1, v2)

        if self.verbose:
            print("vectors.shape: ", vectors.shape)
            print("vector_graph.shape: ", vector_graph.shape)
            print("vout.shape: ", vout.shape)
            print("v1.shape: ", v1.shape)
            print("v2.shape: ", v2.shape)

        transl_out = e3nn.scatter_mean(positions.array, nel=graphs.n_node)
         
        if self.verbose:
            print(">>>>>>>>>>Encoder final output: ", mu.shape, logvar.shape, rot_matrix.shape, vout.shape, transl_out.shape  )
        return (mu, logvar), rot_matrix, vout, transl_out

    def get_rotation_matrix_from_two_vector(self, v1, v2):
        """Analogous to your PyTorch helper function"""
        # Normalize first vector (y1)
        u = v1 / (jnp.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8)

        # Gram-Schmidt for second vector (y2)
        dot = jnp.einsum('bi,bi->b', u, v2)[..., None]
        w_raw = v2 - dot * u
        w = w_raw / (jnp.linalg.norm(w_raw, axis=-1, keepdims=True) + 1e-8)

        # Fallback logic to avoid nans at initialization
        _, fallback_w = self.get_orthogonal_basis(u)
        is_degenerate = jnp.linalg.norm(w_raw, axis=-1) < 1e-4
        w = jnp.where(is_degenerate[..., None], fallback_w, w)

        # Third vector via cross product (y3)
        last_v = jnp.cross(u, w)

        # Stack to form rotation matrix R = [u | w | last_v]
        return jnp.stack([u, w, last_v], axis=-1)

    def get_orthogonal_basis(self, v):
        """Fixed axis fallback for stability"""
        condition = jnp.abs(v[..., 0]) < 0.9
        helper = jnp.where(condition[..., None], jnp.array([1., 0., 0.]), jnp.array([0., 1., 0.]))
        u = jnp.cross(v, helper)
        u = u / (jnp.linalg.norm(u, axis=-1, keepdims=True) + 1e-8)
        w = jnp.cross(v, u)
        return u, w
