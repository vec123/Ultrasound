import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import jraph
import flax 
import flax.linen as nn


class EquivariantAttention(nn.Module):
    target_irreps: str
    sh_lmax: int = 1
    verbose: bool = True
    def setup(self):
        self.irreps_out = e3nn.Irreps(self.target_irreps)
        
        # Projections
        self.h_q = e3nn.flax.Linear(self.irreps_out)
        self.lin_k = e3nn.flax.Linear(self.irreps_out)
        self.lin_v = e3nn.flax.Linear(self.irreps_out)
        
        # Dot product: maps to "0e" (scalar) for attention weight
        self.lin_dot = e3nn.flax.Linear("0e")

    def __call__(self, graph: jraph.GraphsTuple, positions: e3nn.IrrepsArray):
        # 1. Geometry: Compute relative vectors and Spherical Harmonics
        rel_pos = positions.array[graph.receivers] - positions.array[graph.senders]
        rel_pos = e3nn.IrrepsArray("1x1o", rel_pos)
        edge_sh = e3nn.spherical_harmonics([1, 2], rel_pos, normalize=True)

        # 2. Queries (Per node: Receiver)
        q = self.h_q(graph.nodes)
        
        # 3. Keys and Values (Per edge: Sender -> Receiver)
        sender_f = graph.nodes[graph.senders]
        
        # Combine node features and edge spherical harmonics
        msg_k = e3nn.tensor_product(sender_f, edge_sh)
        msg_v = e3nn.tensor_product(sender_f, edge_sh)
        
        # Project to target dimension
        k = self.lin_k(msg_k)
        v = self.lin_v(msg_v)
        
        # 4. Attention mechanism
        # Q (receiver) dot K (edge) -> scalar importance score
        dot_product_input = e3nn.tensor_product(q[graph.receivers], k)
        alpha_raw = self.lin_dot(dot_product_input).array.squeeze(-1)
        
        # Softmax normalized per receiver
        alpha = jax.nn.softmax(alpha_raw, axis=0)
        
        # 5. Aggregate: weighted sum of Values (v)
        v_weighted = v * alpha[..., None]
        
        # Correct scatter function: e3nn.scatter_sum
        f_out = e3nn.scatter_sum(v_weighted, dst=graph.receivers, output_size=graph.nodes.shape[0])
        if self.verbose:
            print("--------------Attention --------------")
            print("graph.nodes.irreps: ", graph.nodes.irreps)
            print("positions.irreps: ", positions.irreps)
            print("f_out.irreps: ", f_out.irreps)
            print("--------------Finished --------------")
        return f_out, alpha
    
class EquivariantPooling(nn.Module):
    num_output_nodes: int
    target_irreps: e3nn.Irreps
    verbose: bool = True

    @nn.compact
    def __call__(self, graph, positions):
        batch_size = graph.n_node.shape[0]
        # Create segment IDs: [0, 0, ..., 1, 1, ..., N, N]
        node_segments = jnp.repeat(jnp.arange(batch_size), graph.n_node)
        
        # 1. Learn assignment matrix
        scalars = graph.nodes.filtered("0e").array
        A = nn.Dense(self.num_output_nodes, name="assign_dense")(scalars)
        
        # Segmented Softmax per graph
        def segment_softmax(x, segments):
            max_x = jax.ops.segment_max(x, segments, num_segments=batch_size)
            exp_x = jnp.exp(x - max_x[segments])
            sum_exp = jax.ops.segment_sum(exp_x, segments, num_segments=batch_size)
            return exp_x / (sum_exp[segments] + 1e-8)
        
        A = segment_softmax(A, node_segments) # Shape: (Total_Nodes, Num_Output_Nodes)

        # 2. Compute Barycenters per graph
        # normalization factor per (graph, output_node)
        norm_factor = jax.ops.segment_sum(A, node_segments, num_segments=batch_size)
        norm_A = A / (norm_factor[node_segments] + 1e-8)
        
        # Compute positions (N_out, 3) for each graph
        # We need to map positions via the assignment matrix
        pos_pooled = jax.ops.segment_sum(
            norm_A[..., None] * positions.array[:, None, :], 
            node_segments, 
            num_segments=batch_size
        ) # Shape: (batch_size, num_output_nodes, 3)

        # 3. Aggregate Features per graph
        nodes_pooled_list = []
        
        # Access the slices object
        slices = graph.nodes.irreps.slices()
        
        for i in range(len(slices)):
            irrep = graph.nodes.irreps[i]
            sl = slices[i]
            
            feat = graph.nodes.array[:, sl]
            # Sum feature weighted by assignment A
            pooled_feat = jax.ops.segment_sum(
                A[..., None] * feat[:, None, :], 
                node_segments, 
                num_segments=batch_size
            )
            nodes_pooled_list.append(pooled_feat)
        
        # Reshape to (Total_New_Nodes, Irreps_Dim) for Jraph
        nodes_flat = jnp.concatenate(nodes_pooled_list, axis=-1)
        nodes_pooled = e3nn.IrrepsArray(
            graph.nodes.irreps, 
            nodes_flat.reshape(-1, nodes_flat.shape[-1])
        )

        # 4. Construct new Clique Graph
        # Each graph in the batch becomes a clique of size M
        M = self.num_output_nodes
        indices = jnp.arange(M)
        # Create base clique indices
        senders = jnp.repeat(indices, M)
        receivers = jnp.tile(indices, M)
        
        # Offset indices for batching
        offsets = jnp.arange(batch_size) * M
        senders_batched = (senders[None, :] + offsets[:, None]).flatten()
        receivers_batched = (receivers[None, :] + offsets[:, None]).flatten()

        new_graph = jraph.GraphsTuple(
            nodes=nodes_pooled,
            senders=senders_batched,
            receivers=receivers_batched,
            n_node=jnp.full((batch_size,), M),
            n_edge=jnp.full((batch_size,), M * M),
            globals=None,
            edges=None
        )
        pos_irreps = e3nn.Irreps("1o")
        pos_pooled_array = e3nn.IrrepsArray(
            pos_irreps, 
            pos_pooled.reshape(-1, 3)
        )

        return new_graph, pos_pooled_array

class GatingBlock(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        # Implementation of your MLP gate logic
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.silu(x)
        x = nn.Dense(self.out_dim)(x)

        return x
    
class SelfInteraction(nn.Module):
    target_irreps: str
    sh_lmax: int = 4
    verbose: bool = True

    @nn.compact
    def __call__(self, node_features: e3nn.IrrepsArray):
        target_irreps = e3nn.Irreps(self.target_irreps)
       

        # Tensor Product: V_i ⊗ V_i 
        v_sq = e3nn.tensor_product(node_features, node_features).regroup()
        v_sq = v_sq.filter(lmax=self.sh_lmax)

        # Skip connection
        v_intermediate = e3nn.concatenate([node_features, v_sq])

        #  Gating
        scalars = v_intermediate.filtered("0e")
        vectors = v_intermediate.filtered("1o")
        v_lengths = e3nn.norm(vectors) 
        
        gate_input = e3nn.concatenate([scalars, v_lengths], axis=-1)
        
        # Gating MLP
        gate_net = GatingBlock(hidden_dim=32, out_dim=v_intermediate.irreps.num_irreps)
        gate = gate_net(gate_input.array)

        gated = v_intermediate * gate

        v_out = e3nn.flax.Linear(target_irreps, force_irreps_out=False)(gated)
        if self.verbose:
                print("--------------SelfInteraction --------------")
                print("target_irreps: ", self.target_irreps)
                print("in.irreps: ", node_features.irreps)
                print("v_intermediate.irreps: ", v_intermediate.irreps)
                print("v_out.irreps: ", v_out.irreps)
                print("--------------Finished --------------")
        return  v_out
    
class SpatialConvolution(nn.Module):
  
    target_irreps: str
    sh_lmax: int = 4
    verbose: bool = True

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, positions: e3nn.IrrepsArray):
        target_irreps = e3nn.Irreps(self.target_irreps)
        
        def update_edge_fn(edge_features, sender_features, receiver_features, globals):
            # rel_pos = P_alpha_j - P_alpha_i
            rel_pos = positions.array[graph.receivers] - positions.array[graph.senders]
            rel_pos = e3nn.IrrepsArray("1x1o", rel_pos)
            dist = e3nn.norm(rel_pos)
            
            # Spherical Harmonics Path
            l_list = [l for l in range(1, self.sh_lmax + 1)]
            Y = e3nn.spherical_harmonics(l_list, rel_pos, True)
            R = e3nn.soft_one_hot_linspace(dist.array, start=0.0, end=10.0, number=16, basis='gaussian', start_zero=False, end_zero = False)
            tp_message = e3nn.tensor_product(sender_features, Y).regroup()
            geo_features = e3nn.concatenate([sender_features, tp_message])

            # Gating message by distance and neighbor cloud magnitudes
            r0e, s0e, = receiver_features.filtered("0e"), receiver_features.filtered("0e")
            r0o, s0o = receiver_features.filtered("0o"),receiver_features.filtered("0o")
            R_squeezed = jnp.squeeze(R, axis=1)
            R_irreps = e3nn.IrrepsArray(f"{R_squeezed.shape[-1]}x0e", R_squeezed)    
            v_intermediate = e3nn.concatenate([ receiver_features.filtered(lmax=0),sender_features.filtered(lmax=0)])
            v_node = sender_features.filtered("1o")
            v_norm = e3nn.norm(v_node)

            #concatenate scalars for gating, optionally also concatenate r0, s0, R_irreps and v_norm
            gate_in = jnp.concatenate([v_intermediate.array], axis=-1)

            gate_net = GatingBlock(hidden_dim=32, out_dim=geo_features.irreps.num_irreps)
            gate = gate_net(gate_in)

            gated = geo_features * gate
            v_out = e3nn.flax.Linear(target_irreps, force_irreps_out=False)(gated)
            return v_out

        def update_node_fn(nodes, senders, receivers, globals):
            # 'nodes' contains [V_i, degree]
            # 'receivers' contains the sum of V_tilde
    
            # Extract the degree (k) we stored earlier
            # Assuming it's the last 0e channel
            k = nodes.filtered("0e").array[:, -1:] 
            
            # Perform the division (1/k * sum(V_tilde))
            receivers = receivers.filtered(lmax=self.sh_lmax)
            normalized_messages = receivers / (jnp.maximum(k, 1.0) +1e-6)
            
            # V = Linear(V + normalized_messages)
            # filter 'nodes' to remove the extra degree scalar before the sum
            v_current = nodes.filtered(target_irreps) 
        
            # Check if they are identical in both symmetry and dimensions
            if v_current.irreps == normalized_messages.irreps and v_current.shape == normalized_messages.shape:
                v_residual = v_current
            else:
                v_residual =e3nn.flax.Linear(normalized_messages.irreps, name="res_proj", force_irreps_out=True)(v_current)
          
            out = v_residual + normalized_messages
            if self.verbose:
                print("-------------- SpatialConvolution --------------")
                print("target_irreps: ", self.target_irreps)
                print("in.irreps: ",v_current.irreps)
                print("msg.irreps: ", normalized_messages.irreps)
                print("out.irreps: ", out.irreps)
                print("-------------- Finished --------------")
            return out.filtered(lmax=self.sh_lmax)

        return jraph.GraphNetwork(update_edge_fn, update_node_fn)(graph)
    
def safe_sqrt(x, eps):
    return jnp.sqrt(jnp.maximum(x, eps))

class EquivariantLayerNorm(flax.linen.Module):

    irreps: str
    eps: float =1e-5
    affine:  bool=True
    normalization:  str ='component'
    verbose: bool = True

    @nn.compact
    def __call__(self, x: e3nn.IrrepsArray):
        
        # Optional: verify input matches expected irreps to catch bugs early
        # assert x.irreps == self.irreps
        irreps = e3nn.Irreps(x.irreps)
        output_list = []
        # Index into the raw flat array of x
        start_flat_idx = 0 

        for i, (mul, ir) in enumerate(irreps):
            # Calculate the slice for this irrep group
            # Total values = multiplicity * dimension (2l + 1)
            num_values = mul * ir.dim
            
            # Slice the flat array: (nodes, mul * dim)
            field_data = x.array[:, start_flat_idx : start_flat_idx + num_values]
            start_flat_idx += num_values
            
            # Reshape to (nodes, mul, dim) for easy normalization and broadcasting
            field_reshaped = field_data.reshape(field_data.shape[0], mul, ir.dim)
            
            if ir.l == 0 and ir.p == 1:
                # --- SCALAR BRANCH ---
                mean = jnp.mean(field_reshaped, axis=(1, 2), keepdims=True)
                var = jnp.var(field_reshaped, axis=(1, 2), keepdims=True)
                normed = (field_reshaped - mean) / jnp.sqrt(var + self.eps)
                
                if self.affine:
                    # Shape (1, mul, 1) ensures perfect broadcasting with (N, mul, 1)
                    w = self.param(f"w_{i}", nn.initializers.ones, (1, mul, 1))
                    b = self.param(f"b_{i}", nn.initializers.zeros, (1, mul, 1))
                    normed = normed * w + b
            else:
                # --- VECTOR/TENSOR BRANCH ---
                # Equivariant RMS normalization
                sq = jnp.square(field_reshaped) # (N, mul, dim)
                if self.normalization == 'norm':
                    # sum over dim (2l+1)
                    #rms = jnp.sqrt(jnp.sum(sq, axis=-1, keepdims=True) + self.eps)
                    rms = safe_sqrt(jnp.sum(sq, axis=1, keepdims=True), self.eps)
                else:
                    # mean over dim (2l+1)
                   # rms = jnp.sqrt(jnp.mean(sq, axis=-1, keepdims=True) + self.eps)
                    rms = safe_sqrt(jnp.mean(sq, axis=1, keepdims=True), self.eps)
                
                # Normalize across the multiplicities (channels)
                layer_rms = jnp.sqrt(jnp.mean(jnp.square(rms), axis=1, keepdims=True) + self.eps)
                normed = field_reshaped / (layer_rms + self.eps)
                
                if self.affine:
                    w = self.param(f"w_{i}", nn.initializers.ones, (1, mul, 1))
                    normed = normed * w
            
            # Flatten back to (nodes, mul * dim)
            output_list.append(normed.reshape(field_data.shape[0], -1))

        # Reconstruct the IrrepsArray
        final_array = jnp.concatenate(output_list, axis=-1)
        out = e3nn.IrrepsArray(irreps, final_array)
        if self.verbose:
            print("----EquivariantLayerNorm-----")
            print("x.irreps: ", x.irreps)
            print("self.irreps: ", self.irreps)
            print("out.irreps: ", out.irreps)
            print("--------Finished-------------")
        return out