import jax
import jax.numpy as jnp
import equinox as eqx
import optax  # Optimization library for JAX

class GeometricBasis(eqx.Module):
    k_trunc: int
    sigmas: jnp.ndarray

    def __init__(self, k_trunc, dim=1, key=jax.random.PRNGKey(0)):
        super().__init__()
        self.k_trunc = k_trunc
        # Initialize sigmas as ones
        self.sigmas = jnp.ones((k_trunc, dim))

    def get_basis(self, P, evecs):
        # P: (V,) or (V, 1), evecs: (V, K)
        Phi = evecs[:, :self.k_trunc]
        if P.ndim == 1:
            P = P[:, None]
        
        # JAX broadcasting: P(V, 1), sigmas(K, dim) -> (V, K, dim)
        decay = jnp.exp(-P[:, None, :] * self.sigmas[None, :, :])
        basis = decay * Phi[:, :, None]
        return basis

    def compute_coeffs(self, f, basis):
        # f: (V, dim)
        if f.ndim == 1: f = f[:, None]
        
        # 'vkd,vjd->kj'
        gram_matrix = jnp.einsum('vkd,vjd->kj', basis, basis)
        rhs = jnp.einsum('vkd,vd->kd', basis, f)
        
        reg = 1e-3 * jnp.eye(self.k_trunc)
        # JAX equivalent to torch.linalg.lstsq
        alphas, _, _, _ = jnp.linalg.lstsq(gram_matrix + reg, rhs)
        return alphas

    def __call__(self, f, P, evecs):
        basis = self.get_basis(P, evecs)
        alphas = self.compute_coeffs(f, basis)
        # Reconstruct: 'vkd,kd->vd'
        return jnp.einsum('vkd,kd->vd', basis, alphas)

def fit_geometric_basis(f, model, P, evecs, iterations=100, lr=5):
    # Optimizer setup using optax
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(model, opt_state, f, P, evecs):
        def loss_fn(model):
            f_rec = model(f, P, evecs)
            return jnp.mean((f - f_rec)**2)
        
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        
        # Clamp sigmas (manually inside or via optax constraints)
        new_sigmas = jnp.clip(model.sigmas, -5000.0, 5000.0)
        model = eqx.tree_at(lambda m: m.sigmas, model, new_sigmas)
        
        return model, opt_state, loss

    for i in range(iterations):
        model, opt_state, loss = make_step(model, opt_state, f, P, evecs)
        if i % 20 == 0:
            print(f"Iteration {i:03d} | Loss: {loss:.6f}")
            
    return model