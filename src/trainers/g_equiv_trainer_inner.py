import jax
import jax.numpy as jnp
import jraph
import os
import optax
from flax.training import train_state
from functools import partial
import matplotlib.pyplot as plt

from src.models.Equiv.data import make_graphs_from_vertices 
from src.utils.utils  import pad_vertices
from src.models.Equiv.data_transforms import transform_graphs_explicit, get_y_rot
from src.models.Equiv.vtk import save_graphs_as_vtp
from src.models.Equiv.data import make_single_graph_jit
from src.losses.loss_jax import combined_surface_loss, kl_divergence_loss
from src.geometry.vtk import create_polydata, save_vtp_mesh
from src.utils.utils  import recover_original_list


class SO3EquivTrainer:
    def __init__(self, encoder, decoder, max_nodes=500, learning_rate=1e-5, log_dir = "logs"):
        self.encoder = encoder
        self.decoder = decoder
        self.lr = learning_rate
        self.log_dir = log_dir
        self.max_nodes = max_nodes
        self.max_edges = max_nodes*(max_nodes-1)
        self.batch_make_graphs = jax.vmap(
                make_single_graph_jit, 
                in_axes=(0, 0, 0, None, None, None, None, None) 
            )

    def check_grads(self, grads):
        is_finite = jax.tree_util.tree_reduce(
            lambda a, b: a & b, 
            jax.tree_util.tree_map(lambda x: jnp.all(jnp.isfinite(x)), grads)
        )
        if not is_finite:
            print("ALERT: Gradients contain NaNs or Infs!")
        else:
            print("Gradients are finite.")

    def sample_latent(self, mu, logvar, key):
        """
        Performs the reparameterization trick: z = mu + sigma * epsilon
        """
        # 1. Sample epsilon from N(0, 1)
        # The shape should match the mu/logvar output
        eps = jax.random.normal(key, shape=mu.shape)
        
        # 2. Compute standard deviation from logvar
        # std = exp(0.5 * logvar)
        std = jnp.exp(0.5 * logvar)
        
        # 3. Scale and shift
        z = mu + std * eps
        return z
    
    # --- Geometric Logic ---
    def apply_group_action(self, pos_canonical, R_frame, t_frame):
        """
        Transforms canonical coordinates to world coordinates.
        Equation: x_world = (R_frame @ x_canonical) + t_frame
        """
        rotated = jnp.einsum('bij,bkj->bki', R_frame, pos_canonical)
        return rotated + t_frame[:, jnp.newaxis, :]

    # --- Training Logic ---
    def loss_fn(self, params, graph, true_verts, padding_mask, VAE=True):
        kl_gain = 0.001
        (inv_mean, inv_logvar), R_frame, _, t_frame = self.encoder.apply({'params': params['encoder']}, graph)
        inv = self.sample_latent(inv_mean, inv_logvar, jax.random.PRNGKey(42))
        inv = inv_mean
        pos_canonical = self.decoder.apply({'params': params['decoder']}, inv)
       # pred_pos = self.apply_group_action(pos_canonical, R_frame, t_frame)
        pred_pos = pos_canonical
        recon_loss = combined_surface_loss(
                pred_pos, 
                true_verts, 
                padding_mask
            )
        
        kl_loss = kl_divergence_loss(inv_mean, inv_logvar)
        loss = recon_loss  + kl_gain*kl_loss
        return loss, (pred_pos, pos_canonical, inv, R_frame, t_frame)

    @jax.jit(static_argnums=(0,))
    def train_step(self, state, graph, true_verts, padding_mask):
        grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True)
        print("Computing Grads")
        (loss, (pred_pos, pos_canonical, inv, R_pred, t_pred)), grads = grad_fn(
            state.params, 
            graph,
            true_verts,
            padding_mask
            )
        #self.check_grads(grads)
        print("Applying Grads")
        state = state.apply_gradients(grads=grads)
        return state, loss, pred_pos, pos_canonical, inv, R_pred, t_pred

    def fit(self, true_verts, padding_masks, num_steps=1000, log_every=100, plot_every=200):
        n_graphs = true_verts.shape[0]
        master_key = jax.random.PRNGKey(42)

        initial_graphs, _ = self.batch_make_graphs(
                true_verts, 
                jax.random.split(master_key, n_graphs), 
                padding_masks, 
                0.4,   # r_max
                0.9,   # dropout_rate
                0.0,   # noise_std
                self.max_nodes,
                self.max_edges
            )

        rng = jax.random.PRNGKey(0)
        rng, e_key, d_key = jax.random.split(rng, 3)
        
        encoder_vars = self.encoder.init(e_key, initial_graphs)
        (z_mu, _), _, _, _ = self.encoder.apply(encoder_vars, initial_graphs)
        decoder_vars = self.decoder.init(d_key, z_mu)
        
        state = train_state.TrainState.create(
            apply_fn=None, 
            params={'encoder': encoder_vars['params'], 'decoder': decoder_vars['params']}, 
            tx=optax.adam(self.lr)
        )

        def scan_body(state, step):
            step_key = jax.random.fold_in(master_key, step)
            graphs, _ = self.batch_make_graphs(
                true_verts, 
                jax.random.split(master_key, n_graphs), 
                padding_masks, 
                0.4,   # r_max
                0.9,   # dropout_rate
                0.0,   # noise_std
                self.max_nodes,
                self.max_edges
            )

            state, loss, preds, canon, inv, R, t = self.train_step(state, graphs, true_verts, padding_masks)
            return state, (loss, preds, canon, graphs)

        for start_step in range(0, num_steps, log_every):
            end_step = min(start_step + log_every, num_steps)
            
            # Scan a small chunk of steps
            state, (chunk_losses, chunk_preds, chunk_canon, chunk_graphs) = jax.lax.scan(
                scan_body, state, jnp.arange(start_step, end_step)
            )
            
            # --- Perform Logging (Python side) ---
            step = end_step - 1
            print(f"Step {step:4d} | Loss: {chunk_losses[-1]:.6f}")
            
            # Accessing the last element of the chunk for visualization
            orig_graphs = jraph.unbatch(chunk_graphs[-1]) # Need to handle unbatching
            
            if step % plot_every == 0 or step == num_steps - 1:
                # Setup shapes for logging
                def get_shapes(graph_tuple):
                    split_indices = jnp.cumsum(graph_tuple.n_node[:-1])
                    return jnp.split(graph_tuple.nodes, split_indices)

                target_shapes = get_shapes(chunk_graphs[-1])
                # Note: chunk_graphs[-1] is a batched object
                self.log_visualizations(
                    target_shapes, target_shapes, # Placeholder for logic
                    chunk_canon[-1], chunk_preds[-1], 
                    recover_original_list(true_verts, padding_masks), 
                    step=step
                )
                
        return state, chunk_preds[-1]

    def log_visualizations(self, original_b, target_b, canonical_b, rotated_b, gt_b, step, num_samples=3):
        sample_num = min(2, len(original_b))
        for sample_idx in range(sample_num):
            sample_dir = os.path.join(f"{self.log_dir}", f"sample_{sample_idx}")
            os.makedirs(sample_dir, exist_ok=True)

            original, target = original_b[sample_idx], target_b[sample_idx]
            canonical, rotated, gt = canonical_b[sample_idx], rotated_b[sample_idx], gt_b[sample_idx]
        
            self.save_vtp_logs(original, target, canonical, rotated, gt, step, sample_idx)
            # Check if they are exactly identical
            are_identical = jnp.allclose(canonical, rotated, atol=1e-5)
            print(f"Are canonical and rotated identical? {are_identical}")
            
                # Plotting 
            os.makedirs(self.log_dir, exist_ok=True)
            fig = plt.figure(figsize=(18, 5))
            titles = ["Input (Original)", "Input (Augmented)", "Learned Canonical", "Model Reconstruction"]

            data = [original, target, canonical, rotated]
            colors = ["gold", "royalblue", "black", "crimson"]
            markers = ['o', '^', 's', 'x']
            sizes = [50, 20, 50, 20]

            ax = fig.add_subplot(111, projection='3d')
            for i in range(4):

                verts = data[i] 
                ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], 
                        color=colors[i], label=titles[i], 
                        marker=markers[i], s=sizes[i])
            
            ax.legend()
            ax.set_title(f"Step {step}")
            plt.tight_layout()
            path = os.path.join(sample_dir, f"plot_step_{step}.png")
            plt.savefig(path)
            plt.close()
            print(f"--- Saved plot and VTPs for step {step} ---")

  
    def save_vtp_logs(self, original, target, canonical, rotated, gt, step, sample_idx):
        """
        Saves the geometric states as VTP files for visualization in ParaView.
        """
        step_dir = f"{self.log_dir}/vtk"
        os.makedirs(step_dir, exist_ok=True)
        data = {
            "original": original,
            "target": target,
            "canonical": canonical,
            "rotated": rotated,
            "gt": gt
        }
        for key, value in data.items():
            print(f"DEBUG: {key} batch shape: {value.shape}")
            d = value
            print("d.shape: ", d.shape)
            poly = create_polydata(d)
            path = os.path.join(self.log_dir, f"vtk", f"{sample_idx}_{key}_{step}.vtp")
            save_vtp_mesh(poly, path)
        
        print(f"--- Saved visualization VTPs for step {step} to {step_dir} ---")