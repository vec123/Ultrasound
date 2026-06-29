import jax
import jax.numpy as jnp
import haiku as hk
import optax
import e3nn_jax as e3nn
import numpy as np
import os
from dotenv import load_dotenv

# Imports from your project
from src.models.Equiv.equi_jax_encoder import EquiDeepNetwork, GraphLevelHead
from src.models.Equiv.equi_jax_decoder import HaikuShapeDecoder 

load_dotenv()
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")

def run_autoencoder_training():
    # 1. Load Data
    graph_data = os.path.join(PROJECT_ROOT, "scripts", "processors", "graph_data.npz")
    data = np.load(graph_data)
    pos = jnp.array(data['positions'])
    senders = jnp.array(data['senders'])
    receivers = jnp.array(data['receivers'])
    features = jnp.array(data['rel_distances'])
    n_features = features.reshape(pos.shape[0], -1, 3).shape[1]
    
    node_features = e3nn.IrrepsArray(f"{n_features}x1o", features.reshape(pos.shape[0], -1))
    pos_irreps = e3nn.IrrepsArray("1x1o", pos)

    # 2. Define Models
    def encoder_forward(nodes, p, s, r):
        backbone = EquiDeepNetwork(L=2, input_irreps=node_features.irreps, 
                                   internal_irreps="32x0e + 16x1o", output_irreps="32x0e + 16x1o")
        head = GraphLevelHead(input_irreps="32x0e + 16x1o", output_irreps="120x0e")
        return head(backbone(nodes, p, s, r))

    encoder_transformed = hk.transform(encoder_forward)
    decoder_transformed = hk.transform(lambda latent: HaikuShapeDecoder(output_nodes=pos.shape[0])(latent))

    # 3. Initialization
    rng = jax.random.PRNGKey(42)
    rng_enc, rng_dec, rng_train = jax.random.split(rng, 3)
    
    params = {
        'encoder': encoder_transformed.init(rng_enc, node_features, pos_irreps, senders, receivers),
        'decoder': decoder_transformed.init(rng_dec, e3nn.IrrepsArray("120x0e", jnp.zeros((1, 120)) ))
    }

    # 4. Training Components
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, nodes, pos, s, r, rng_key):
        rng_enc_key, rng_dec_key = jax.random.split(rng_key)
        
        # Encode
        print("encoding: ", nodes.shape, pos.shape, s.shape, r.shape)
        latent = encoder_transformed.apply(params['encoder'], rng_enc_key, nodes, pos, s, r)
        print("latent: ", latent.shape)
        # Decode
        reconstructed_pos = decoder_transformed.apply(params['decoder'], rng_dec_key, latent)
        print("reconstructed_pos: ", reconstructed_pos.shape)
        # Loss: Comparing reconstructed [batch, 4, 3] to target 
        # Note: Ensure pos.array is sliced to match your decoder output [batch, 4, 3]
        loss = jnp.mean((reconstructed_pos - pos.array[:pos.shape[0], :])**2)

        print("loss: ", loss)
        return loss

    @jax.jit
    def train_step(params, opt_state, batch, rng_key):
        nodes, pos, s, r = batch
        loss, grads = jax.value_and_grad(loss_fn)(params, nodes, pos, s, r, rng_key)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # 5. Training Loop
    batch = (node_features, pos_irreps, senders, receivers)
    print("Starting Training...")
    for epoch in range(101):
        rng_train, step_rng = jax.random.split(rng_train)
        params, opt_state, loss = train_step(params, opt_state, batch, step_rng)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.6f}")

if __name__ == "__main__":
    run_autoencoder_training()