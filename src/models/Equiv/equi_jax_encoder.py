import flax
import jraph
import haiku as hk
import jax
import jax.numpy as jnp
import e3nn_jax as e3nn

from src.models.Equiv.modules import EquivariantLayerNorm, SelfInteraction, SpatialConvolution
from src.models.Equiv.layers import EquiLayer


class EquiDeepNetwork(hk.Module):
    def __init__(self, L=4, input_irreps="32x0e + 16x1o", internal_irreps="32x0e + 16x1o", 
                 output_irreps="16x0e", distance_cutoff=10.0, max_edges = 100, name=None, verbose = True):
        super().__init__(name=name)
        self.L = L
        
        self.input_irreps = e3nn.Irreps(input_irreps)
        self.internal_irreps = e3nn.Irreps(internal_irreps)
        self.output_irreps = e3nn.Irreps(output_irreps)

        self.distance_cutoff = distance_cutoff
        self.max_edges = max_edges
        self.verbose = verbose
        
    def __call__(self, node_features: e3nn.IrrepsArray, 
                 positions:  e3nn.IrrepsArray,
                 senders: jnp.ndarray,
                 receivers: jnp.ndarray,
                 tau: jnp.ndarray = None):
        
        if tau is not None:
            # Broadcast tau to [num_nodes, 1] and treat as 1x0e
            tau_val = jnp.broadcast_to(tau, (node_features.shape[0], 1))
            tau_irreps = e3nn.IrrepsArray("1x0e", tau_val)
            node_features = e3nn.concatenate([node_features, tau_irreps], axis=-1)
            
        # --- Internal Graph Construction ---
        num_nodes = positions.shape[0]

        #senders, receivers = e3nn.radius_graph(
        #    pos=positions, 
        #    r_max=self.distance_cutoff,
        #    size = self.max_edges
        #)
        # Create the jraph structure
        h_jraph = jraph.GraphsTuple(
            nodes=node_features,
            edges=None, 
            senders=senders,
            receivers=receivers,
            n_node=jnp.array([num_nodes]),
            n_edge=jnp.array([len(senders)]),
            globals=None
        )

        # --- Forward Pass ---
        # Initial Embedding
        h_nodes = e3nn.haiku.Linear(self.input_irreps, name="init_embed")(h_jraph.nodes)
        h = h_jraph._replace(nodes=h_nodes)
        
        history = [h.nodes]

        # First Self-Interaction
        h_0_nodes = SelfInteraction(target_irreps=self.internal_irreps, verbose = self.verbose)(h.nodes)
        history.append(h_0_nodes)
        h = h._replace(nodes=h_0_nodes)

        # Iterative Message Passing
        for l in range(self.L):
            h_l_nodes = EquiLayer(
                target_irreps=self.internal_irreps, 
                name=f"equijump_layer_{l}",
                verbose = self.verbose
            )(h, positions)
            history.append(h_l_nodes)
            h = h._replace(nodes=h_l_nodes)

        # Jumping Knowledge Aggregation
        h_agg_nodes = e3nn.concatenate(history, axis=-1)
        h_agg_nodes = e3nn.haiku.Linear(self.internal_irreps, name="agg_linear")(h_agg_nodes)
        
        print("----Last layer")
        print("last h_agg_nodes.irreps : ", h_agg_nodes.irreps)
        h_out_nodes = SelfInteraction(target_irreps=self.output_irreps,  verbose = self.verbose)(h_agg_nodes)
        print("last h_out_nodes.irreps : ", h_out_nodes.irreps)

        gate = hk.get_parameter("output_gate", shape=(), init=hk.initializers.Constant(1e-3))

        return h_out_nodes
    

class GraphLevelHead(hk.Module):
    def __init__(self, input_irreps, output_irreps="2x0e + 2x1e", name=None):
        super().__init__(name=name)
        self.input_irreps = e3nn.Irreps(input_irreps)
        self.output_irreps = e3nn.Irreps(output_irreps)

    def __call__(self, h_nodes: e3nn.IrrepsArray):
        
        h_nodes = e3nn.haiku.Linear(self.input_irreps, name="init_embed")(h_nodes)
        
        scalars = h_nodes.filtered("0e")
        # Fallback to vector norms if no scalars exist
        if scalars.irreps.dim == 0:
            scalars = e3nn.norm(h_nodes)
            
        attn_logits = hk.nets.MLP([32, 1], name="attn_mlp")(scalars.array)
        attn_weights = jax.nn.softmax(attn_logits, axis=0)
        
        weighted_nodes = h_nodes * attn_weights
        
        mesh_representation = e3nn.sum(weighted_nodes, axis=0)
        print(f"DEBUG: mesh_representation irreps: {  mesh_representation.irreps}")
      
        out =  e3nn.haiku.Linear(self.output_irreps, name="final_proj")(mesh_representation)
        print(f"DEBUG: out irreps: {  out.irreps}")
        return out