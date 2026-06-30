import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from functools import partial
import matplotlib.pyplot as plt
from src.models.Equiv.data import make_graphs_from_vertices 
from src.utils.utils  import pad_vertices
from src.models.Equiv.data_transforms import transform_graphs_explicit, get_y_rot
from src.models.Equiv.vtk import save_graphs_as_vtp
from src.losses.loss_jax import combined_surface_loss, kl_divergence_loss
from src.geometry.vtk import create_polydata, save_vtp_mesh
from src.utils.utils  import recover_original_list
import jraph
import os


class SO3EquivTrainer:
    def __init__(self, encoder, decoder, learning_rate=1e-5, log_dir = "logs"):
        self.encoder = encoder
        self.decoder = decoder
        self.lr = learning_rate
        self.log_dir = log_dir
       
    def mask_batch_graphs(self, key, graphs, drop_prob=0.1):
        """
        Randomly drops nodes and their incident edges for each graph in a batch.
        
        Args:
            key: jax.random.PRNGKey
            graphs: jraph.GraphsTuple
            drop_prob: float (0.0 to 1.0), probability of dropping a node
        """
        n_node = graphs.n_node
        total_nodes = graphs.nodes.shape[0]
        
        # 1. Create a mask for nodes (1 = keep, 0 = drop)
        keep_node_mask = jax.random.bernoulli(key, p=1.0 - drop_prob, shape=(total_nodes,))
        
        # 2. Map nodes to their respective graph index to identify which edges to drop
        # Create an array [0, 0, 1, 1, 1, 2, 2...] representing graph IDs
        node_to_graph_id = jnp.repeat(jnp.arange(len(n_node)), n_node)
        
        # 3. An edge is dropped if either the sender OR the receiver node is dropped
        sender_mask = keep_node_mask[graphs.senders]
        receiver_mask = keep_node_mask[graphs.receivers]
        keep_edge_mask = sender_mask & receiver_mask
        
        # 4. Filter edges and nodes
        # Note: If your model requires fixed-size inputs, use a padding mask 
        # instead of physically removing them from the arrays.
        
        # For GNNs, you usually pass the mask to the message passing function
        # rather than resizing the tensors, which breaks JIT.
        return graphs._replace(
            nodes=graphs.nodes * keep_node_mask[:, None],
            edges=graphs.edges * keep_edge_mask[:, None] if graphs.edges is not None else None
        ), keep_node_mask
    
    def check_grads(self, grads):
        # Flatten the tree to easily check all leaf nodes
        is_finite = jax.tree_util.tree_reduce(
            lambda a, b: a & b, 
            jax.tree_util.tree_map(lambda x: jnp.all(jnp.isfinite(x)), grads)
        )
        
        if not is_finite:
            print("ALERT: Gradients contain NaNs or Infs!")
            # Optional: identify which parameter is causing the issue
            # jax.debug.print("Grads: {x}", x=grads) 
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
    def train_step(self, state, graph, true_verts, padding_mask, step):
        rng = jax.random.PRNGKey(step)
        masked_graphs, node_mask = self.mask_batch_graphs(rng, graph, drop_prob=0.9)
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

    def fit(self, graphs_batch, true_verts, padding_mask, num_steps=1000, log_every=100, plot_every=200):

        rng = jax.random.PRNGKey(0)
        rng, e_key, d_key = jax.random.split(rng, 3)

        key, subkey = jax.random.split(rng)
        masked_graphs, node_mask = self.mask_batch_graphs(subkey, graphs_batch, drop_prob=0.9)

        # Initialization
        print("Init Encoder")
        encoder_vars = self.encoder.init(e_key, masked_graphs)
        print("encoder apply")
        (z_inv_mu, z_inv_logvar), _, _, _ = self.encoder.apply(encoder_vars, masked_graphs)
        print("z_inv.shape: ", z_inv_mu.shape)
        print("Init Decoder")
        decoder_vars = self.decoder.init(d_key, z_inv_mu)
        print("decoder apply")
        graph = self.decoder.apply(decoder_vars, z_inv_mu)
        print("graph.shape: ", graph.shape)

        params = {'encoder': encoder_vars['params'], 'decoder': decoder_vars['params']}
        state = train_state.TrainState.create(
            apply_fn=None, params=params, tx=optax.adam(self.lr)
        )

        print("-------------------------------------------------")
        print("--                                             --")
        print("--                                             --")
        print("--                                             --")
        print("-------------------------------------------------")
        print(f"Starting training for {num_steps} steps...")
       
        for step in range(num_steps):
  
            # (Optional Augmentation (Apply Group Transform)
            # Get Group element
            rng, step_key = jax.random.split(rng)
            n_graphs = graphs_batch.n_node.shape[0]
            k1, k2 = jax.random.split(step_key)
            
            thetas = jax.random.uniform(k2, (n_graphs,), minval=0.0, maxval=0 * jnp.pi)
            rot_mats = jax.vmap(get_y_rot)(thetas)
            trans_vecs = jax.random.uniform(k1, (n_graphs, 3), minval=-0.0, maxval=0.0)

            # Apply Group element to the batch
            graphs_aug = transform_graphs_explicit(k2, graphs_batch, rot_mats, trans_vecs, permute=False)

            # Perform Training Step
            state, loss, preds, canon, inv, R_pred, t_pred = self.train_step(
                state, graphs_aug, true_verts, padding_mask, step
                )
            
            # ------------------------------------------
            # Logging & Visualization
            if step % log_every == 0 or step == num_steps - 1:
                print(f"\nStep {step:4d} | Loss: {loss:.6f}")

                if graphs_batch.n_node.shape[0] >= 2:
                    graphs_list = jraph.unbatch(graphs_batch)
                    if len(graphs_list) >= 2:
                            g1 = graphs_list[0]
                            g2 = graphs_list[1]

                            (inv_mu_1, _), _, _, _ = self.encoder.apply({'params': state.params['encoder']}, g1)
                            (inv_mu_2, _), _, _, _ = self.encoder.apply({'params': state.params['encoder']}, g2)

                            latent_dist = jnp.mean(jnp.abs(inv_mu_1 - inv_mu_2))
                            print(f"\n[DEBUG] Latent distance between graph 0 and 1: {latent_dist:.6f}")
                            print("inv_mu_1: ", inv_mu_1)
                            print("inv_mu_2: ", inv_mu_2)
               

                # Consistency Checks - test encoder on transformed and original data
                (inv_orig_mu, inv_orig_logvar), R_orig, _, t_orig = self.encoder.apply({'params': state.params['encoder']}, graphs_batch)

                # Consistency Checks - test encoder on transformed and original data
                (inv_orig_mu, inv_orig_logvar), R_orig, _, t_orig = self.encoder.apply({'params': state.params['encoder']}, graphs_batch)
                (inv_aug_mu, inv_aug_logvar), R_aug, _, t_aug = self.encoder.apply({'params': state.params['encoder']}, graphs_aug)
                inv_orig = inv_orig_mu
                inv_aug = inv_aug_mu
                inv_delta = jnp.mean(jnp.abs(inv_orig - inv_aug))

                # R_aug should be rot_mats @ R_orig
                R_expected = jnp.einsum('bij,bjk->bik', rot_mats, R_orig)
                frame_delta = jnp.mean(jnp.abs(R_aug - R_expected))

                # t_aug should be (rot_mats @ t_orig) + trans_vecs
                t_expected = jnp.einsum('bij,bj->bi', rot_mats, t_orig) + trans_vecs
                t_delta = jnp.mean(jnp.abs(t_aug - t_expected))

                
                print(f"Consistency Deltas -> Inv: {inv_delta:.2e} | Frame: {frame_delta:.2e} | Transl: {t_delta:.2e}")

                if step % plot_every == 0:
                    def get_shapes(graph):
                        split_indices = jnp.cumsum(graph.n_node[:-1])
                        return jnp.split(graph.nodes, split_indices)

                    target_shapes = get_shapes(graphs_aug)
                    orig_shapes = get_shapes(graphs_batch)

                    # Pass these to log_visualizations
                    gt =  true_verts, padding_mask
                    gt_list = recover_original_list(true_verts, padding_mask)
                    gt = gt_list[0]
                    self.log_visualizations(orig_shapes, target_shapes, canon, preds, gt_list, step=step)
        return state, preds

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