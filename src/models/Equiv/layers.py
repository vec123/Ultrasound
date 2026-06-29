
import e3nn_jax as e3nn
import jraph
import flax
import flax.linen as nn
from src.models.Equiv.modules import(
SelfInteraction, SpatialConvolution, EquivariantLayerNorm, EquivariantAttention, EquivariantPooling
) 

class Layer(nn.Module):
    target_irreps: e3nn.Irreps
    denominator: float
    sh_lmax: int = 3

    @flax.linen.compact
    def __call__(self, graphs, positions):
        target_irreps = e3nn.Irreps(self.target_irreps)

        def update_edge_fn(edge_features, sender_features, receiver_features, globals):
            sh = e3nn.spherical_harmonics(
                list(range(1, self.sh_lmax + 1)),
                positions[graphs.receivers] - positions[graphs.senders],
                True,
            )
            return e3nn.concatenate(
                [sender_features, e3nn.tensor_product(sender_features, sh)]
            ).regroup()

        def update_node_fn(node_features, sender_features, receiver_features, globals):
            node_feats = receiver_features / self.denominator
            node_feats = e3nn.flax.Linear(target_irreps, name="linear_pre")(node_feats)
            node_feats = e3nn.scalar_activation(node_feats)
            node_feats = e3nn.flax.Linear(target_irreps, name="linear_post")(node_feats)
            shortcut = e3nn.flax.Linear(
                node_feats.irreps, name="shortcut", force_irreps_out=True
            )(node_features)
            return shortcut + node_feats

        return jraph.GraphNetwork(update_edge_fn, update_node_fn)(graphs)



class EquiLayer(nn.Module):
    target_irreps:  str
    verbose: bool = True

    @nn.compact
    def __call__(self, graph, positions):
         
    
        # Self Interaction (Update V based on internal tensor cloud structure)
        in_irreps = graph.nodes
        h = SelfInteraction(
                target_irreps=self.target_irreps, 
                sh_lmax=1,
                verbose = self.verbose )(graph.nodes)
        graph = graph._replace(nodes=h)
        
        # Spatial Convolution (Update V based on neighboring points)
        graph = SpatialConvolution(
                target_irreps=self.target_irreps, 
                sh_lmax=1,
                verbose = self.verbose )(graph, positions)
        
        
        msg = graph.nodes
        if in_irreps.irreps == msg.irreps and in_irreps.shape == msg.shape:
            skip = in_irreps
        else:
            skip =e3nn.flax.Linear(msg.irreps, name="res_proj", force_irreps_out=True)(in_irreps)

        res = msg+skip
        if self.verbose:
            print("------Skip connection--------")
            print("in_irreps.irreps: ", in_irreps.irreps )
            print("msg.irreps: ", msg.irreps ) 
            print("skip.irreps: ", skip.irreps )
            print("res.irreps: ", res.irreps )
            print("-------Finished--------")

        # Apply Layer Norm
        h_norm = EquivariantLayerNorm(self.target_irreps, name=f"layer_norm", verbose=self.verbose)(res)
   
        if self.verbose:
            print("--------------Layer --------------")
            print("in.irreps : ", in_irreps.irreps)
            print("msg.irreps : ", msg.irreps)
            print("out.irreps: ", h_norm.irreps)
            print("out.shape: ", h_norm.shape)
            print("graph.nodes.shape: ", graph.nodes.shape)
            print("------------Finished-------------")
        return graph._replace(nodes=h_norm)
    


class EquiLayerCone(nn.Module):
    target_irreps:  str
    verbose: bool = True
    keep_ratio: float = 0.5
    num_output_nodes: int = 100
    @nn.compact
    def __call__(self, graph, positions):
         
        # Self Interaction (Update V based on internal tensor cloud structure)
        in_irreps = graph.nodes
        h = SelfInteraction(
                target_irreps=self.target_irreps, 
                sh_lmax=1,
                verbose = self.verbose )(graph.nodes)
        graph = graph._replace(nodes=h)
        
        # Spatial Convolution (Update V based on neighboring points)
        graph = SpatialConvolution(
                target_irreps=self.target_irreps, 
                sh_lmax=1,
                verbose = self.verbose )(graph, positions)
        
        
        if self.verbose:
            print("------skip connection--------")

        msg = graph.nodes
        if in_irreps.irreps == msg.irreps and in_irreps.shape == msg.shape:
            skip = in_irreps
        else:
            skip =e3nn.flax.Linear(msg.irreps, name="res_proj", force_irreps_out=True)(in_irreps)

        res = msg+skip
        if self.verbose:
                    print("in_irreps.irreps: ", in_irreps.irreps )
                    print("msg.irreps: ", msg.irreps ) 
                    print("skip.irreps: ", skip.irreps )
                    print("res.irreps: ", res.irreps )
                    print("-------Finished--------")
        # Apply Layer Norm
        h_norm = EquivariantLayerNorm(self.target_irreps, 
                                      name=f"layer_norm",
                                       verbose = self.verbose)(res)
        graph = graph._replace(nodes=h_norm)

        f_msg, alpha = EquivariantAttention(
            target_irreps=self.target_irreps,
            verbose = self.verbose)(graph, positions)
        
        graph_pooled, positions_pooled = EquivariantPooling(
            num_output_nodes=self.num_output_nodes,
            target_irreps = self.target_irreps,
            verbose = self.verbose)(graph, positions)

        if self.verbose:
            print("-------------- Pooling Layer --------------")
            print(f"Original Node Count: {graph.nodes.shape[0]}")
            print(f"New V_i (Features) shape: {graph_pooled.nodes.shape}")
            print(f"New P_i (Positions) shape: {positions_pooled.shape}")
            print("------------ Finished -------------")
            
        # Return the new structures
        return graph_pooled, positions_pooled