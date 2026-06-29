import jax
import jax.numpy as jnp
from flax import nnx
import e3nn_jax as e3nn
import jraph
from src.modules.LayerNorm_flax import EquivariantLayerNorm

class SelfInteraction(nnx.Module):
    def __init__(self, in_size, out_site, in_irreps, target_irreps, l_max=4, rngs: nnx.Rngs = None):
        
        self.in_size = in_size
        self.target_irreps = out_site

        self.in_irreps = e3nn.Irreps(in_irreps)
        self.target_irreps = e3nn.Irreps(target_irreps)
        
        self.l_max = l_max
        
        self.gate_mlp = nnx.MLP(
            in_size=self.in_size, 
            out_size=self.out_site, 
            hidden_sizes=[32], 
            rngs=rngs
        )
        
        self.linear = e3nn.nnx.Linear(
            irreps_in=self.in_irreps, 
            irreps_out=self.target_irreps,
            rngs=rngs
        )

    def __call__(self, node_features: e3nn.IrrepsArray):

        v_sq = e3nn.tensor_product(node_features, node_features).regroup()
        v_sq = v_sq.filter(lmax=self.l_max)

        v_intermediate = e3nn.concatenate([node_features, v_sq])

        scalars = v_intermediate.filtered("0e")
        vectors = v_intermediate.filtered("1o")
        v_lengths = e3nn.norm(vectors)
        
        gate_input = e3nn.concatenate([scalars, v_lengths], axis=-1)
        
        gate = self.gate_mlp(gate_input.array)
        
        return self.linear(v_intermediate * gate)
    

class SpatialConvolution(nnx.Module):
    def __init__(self, target_irreps, gate_in_dim, geo_irreps, sh_lmax=4, rngs: nnx.Rngs = None):
        super().__init__()
        self.target_irreps = e3nn.Irreps(target_irreps)
        self.sh_lmax = sh_lmax
        
        self.gate_mlp = nnx.MLP(
            in_size=gate_in_dim, 
            out_size=geo_irreps.num_irreps, 
            hidden_sizes=[32], 
            rngs=rngs
        )
        self.res_proj = e3nn.nnx.Linear(
            irreps_in=target_irreps, 
            irreps_out=target_irreps, 
            rngs=rngs
        )

    def __call__(self, graph: jraph.GraphsTuple, positions: e3nn.IrrepsArray):
        
        def update_edge_fn(edge_features, sender_features, receiver_features, globals):
            rel_pos = positions.array[graph.receivers] - positions.array[graph.senders]
            rel_pos = e3nn.IrrepsArray("1x1o", rel_pos)
            dist = e3nn.norm(rel_pos)
            
            Y = e3nn.spherical_harmonics(list(range(1, self.sh_lmax + 1)), rel_pos, True)
            tp_message = e3nn.tensor_product(sender_features, Y).regroup()
            geo_features = e3nn.concatenate([sender_features, tp_message])

            v_intermediate = e3nn.concatenate([receiver_features.filtered(lmax=0), 
                                               sender_features.filtered(lmax=0)])
            
            gate = self.gate_mlp(v_intermediate.array)
            return geo_features * gate

        def update_node_fn(nodes, senders, receivers, globals):
            k = nodes.filtered("0e").array[:, -1:]
            receivers = receivers.filtered(lmax=self.sh_lmax)
            normalized_messages = receivers / jnp.maximum(k, 1.0)
            
            v_current = nodes.filtered(self.target_irreps)
            
            v_residual = self.res_proj(v_current)
            
            return v_residual + normalized_messages

        return jraph.GraphNetwork(update_edge_fn, update_node_fn)(graph)
    

class EquiLayer(nnx.Module):
    def __init__(self, in_irreps, target_irreps, sh_lmax=4, rngs: nnx.Rngs = None):
        super().__init__()
        self.target_irreps = e3nn.Irreps(target_irreps)
        
        # Initialize Sub-modules
        # Note: You need to pass the required dimensions for these modules
        self.si = SelfInteraction(in_irreps, target_irreps, rngs=rngs)
        self.conv = SpatialConvolution(target_irreps, gate_in_dim=..., geo_irreps=..., sh_lmax=sh_lmax, rngs=rngs)
        
        self.res_proj = e3nn.nnx.Linear(
            irreps_in=in_irreps, 
            irreps_out=target_irreps, 
            rngs=rngs
        )
        
        # Assuming EquivariantLayerNorm is also migrated to nnx
        self.norm = EquivariantLayerNorm(target_irreps, rngs=rngs)

    def __call__(self, graph: jraph.GraphsTuple, positions: e3nn.IrrepsArray):
        in_nodes = graph.nodes
        
        # 1. Self Interaction
        h = self.si(in_nodes)
        graph = graph._replace(nodes=h)
        
        # 2. Spatial Convolution
        graph = self.conv(graph, positions)
        msg = graph.nodes
        
        # 3. Residual Connection
        # We check irreps to decide if projection is needed
        if in_nodes.irreps == msg.irreps:
            res = msg + in_nodes
        else:
            res = msg + self.res_proj(in_nodes)
            
        # 4. Normalization
        return self.norm(res)
    

class EquiDeepNetwork(nnx.Module):
    def __init__(self, L=4, input_irreps="32x0e + 16x1o", internal_irreps="32x0e + 16x1o", 
                 output_irreps="16x0e", rngs: nnx.Rngs = None):
        super().__init__()
        self.L = L
        self.input_irreps = e3nn.Irreps(input_irreps)
        self.internal_irreps = e3nn.Irreps(internal_irreps)
        self.output_irreps = e3nn.Irreps(output_irreps)

        # Layers defined in __init__
        self.init_embed = e3nn.nnx.Linear(self.input_irreps, self.input_irreps, rngs=rngs)
        self.si_first = SelfInteraction(input_irreps, internal_irreps, rngs=rngs)
        
        # Iterative layers (using a list of modules)
        self.equi_layers = [
            EquiLayer(internal_irreps, internal_irreps, rngs=rngs) for _ in range(L)
        ]
        
        # Aggregation and final output
        # History contains L+2 items, so we calculate total irreps for concatenation
        agg_in_irreps = e3nn.Irreps(internal_irreps) * (L + 2) 
        self.agg_linear = e3nn.nnx.Linear(agg_in_irreps, internal_irreps, rngs=rngs)
        self.si_out = SelfInteraction(internal_irreps, output_irreps, rngs=rngs)
        
        # Learnable gate as a parameter
        self.output_gate = nnx.Param(jnp.array(1e-3, dtype=jnp.float32))

    def __call__(self, node_features: e3nn.IrrepsArray, positions: e3nn.IrrepsArray,
                 senders: jnp.ndarray, receivers: jnp.ndarray, tau: jnp.ndarray = None):
        
        # Handle optional tau
        if tau is not None:
            tau_val = jnp.broadcast_to(tau, (node_features.shape[0], 1))
            tau_irreps = e3nn.IrrepsArray("1x0e", tau_val)
            node_features = e3nn.concatenate([node_features, tau_irreps], axis=-1)

        # Initial Embedding
        h = self.init_embed(node_features)
        history = [h, self.si_first(h)]
        h = history[-1]

        # Iterative Message Passing
        for layer in self.equi_layers:
            h = layer(jraph.GraphsTuple(nodes=h, edges=None, senders=senders, receivers=receivers, 
                                        n_node=jnp.array([h.shape[0]]), n_edge=jnp.array([len(senders)]), globals=None), 
                      positions)
            history.append(h)

        # Jumping Knowledge Aggregation
        h_agg = e3nn.concatenate(history, axis=-1)
        h_agg = self.agg_linear(h_agg)
        
        return self.si_out(h_agg) * self.output_gate.value