import jax
import jax.numpy as jnp
import haiku as hk
import e3nn_jax as e3nn
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
from src.models.equi_jax_encoder import EquiDeepNetwork, GraphLevelHead

def run_model_with_knn():
    # 1. Load Data
    graph_data = os.path.join(PROJECT_ROOT, "scripts", "processors", "graph_data.npz")
    data = np.load(graph_data)
    pos = jnp.array(data['positions'])
    senders = jnp.array(data['senders'])
    receivers = jnp.array(data['receivers'])
    features =  jnp.array(data['rel_distances'])
    features_reshaped = features.reshape(pos.shape[0], -1, 3)
    n_features = features_reshaped.shape[1]
    feature_irreps = features.reshape(pos.shape[0], n_features* 3)

    print("pos.shape: ", pos.shape)
    print("senders.shape: ", senders.shape)
    print("receivers.shape: ", receivers.shape)
    print("features", features.shape)
    print("features_reshaped", features_reshaped.shape)
    print("feature_irreps", feature_irreps.shape)

   
    # 2. Setup Irreps
    input_irreps = f"{n_features}x1o"
    node_features = e3nn.IrrepsArray(input_irreps, feature_irreps)
    pos = e3nn.IrrepsArray("1x1o", pos)

    # Define Transform
    def backbone_forward(nodes, p, s, r):
        model = EquiDeepNetwork(
            L=2, 
            input_irreps=input_irreps, 
            internal_irreps="32x0e + 16x1o",
            output_irreps="32x0e + 16x1o"
        )
        return model(nodes, p, s, r)

    def head_forward(node_outputs):
        model = GraphLevelHead(
            input_irreps="32x0e + 16x1o", 
            output_irreps="2x0e + 2x1o"
        )
        return model(node_outputs)
    
    backbone_transformed = hk.transform(backbone_forward)
    head_transformed = hk.transform(head_forward)

    # 4. Initialize and Apply
    print("-----init backbone")
    params_backbone = backbone_transformed.init(jax.random.PRNGKey(1), node_features, pos, senders, receivers)

    print("-----init head")
    # We use the backbone output shape to initialize the head
    # We run a single pass to get the shape for initialization
    dummy_output = backbone_transformed.apply(params_backbone, jax.random.PRNGKey(1), node_features, pos, senders, receivers)
    params_head = head_transformed.init(jax.random.PRNGKey(2), dummy_output)

    print("------apply backbone")
    jit_apply_backbone = jax.jit(backbone_transformed.apply)
    output = jit_apply_backbone(params_backbone, jax.random.PRNGKey(1), node_features, pos, senders, receivers)
    print(f"Output shape: {output.array.shape}")

    print("------apply head")
    jit_apply_head = jax.jit(head_transformed.apply)
    graph_lvl = jit_apply_head(params_head, jax.random.PRNGKey(2), output)

    print(f"Graph processed with {len(senders)} edges.")
    print(f"graph_lvl shape: {graph_lvl.array.shape}")

    scalars = graph_lvl.filtered("0e").array  # Shape: (2,)
    vectors = graph_lvl.filtered("1o").array  # Shape: (2, 3) or (6,) flattened

    print(f"Scalars: {scalars}")
    print(f"Vectors: {vectors}")

if __name__ == "__main__":
    run_model_with_knn()