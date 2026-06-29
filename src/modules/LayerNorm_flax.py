import jax
import jax.numpy as jnp
from flax import nnx
import e3nn_jax as e3nn

class EquivariantLayerNorm(nnx.Module):
    def __init__(self, irreps, eps=1e-2, affine=True, normalization='component', rngs: nnx.Rngs = None):
        super().__init__()
        self.irreps = e3nn.Irreps(irreps)
        self.eps = eps
        self.affine = affine
        self.normalization = normalization

        # Initialize learnable affine parameters as nnx.Param
        self.weights = {}
        self.biases = {}
        
        if self.affine:
            for i, (mul, ir) in enumerate(self.irreps):
                # Weights: shape (1, mul, 1)
                self.weights[i] = nnx.Param(jnp.ones((1, mul, 1)))
                # Biases: only applicable to scalars (l=0)
                if ir.is_scalar:
                    self.biases[i] = nnx.Param(jnp.zeros((1, mul, 1)))

    def __call__(self, x: e3nn.IrrepsArray):
        output_list = []
        start_flat_idx = 0
        
        for i, (mul, ir) in enumerate(self.irreps):
            num_values = mul * ir.dim
            field_data = x.array[:, start_flat_idx : start_flat_idx + num_values]
            start_flat_idx += num_values
            field_reshaped = field_data.reshape(field_data.shape[0], mul, ir.dim)
            
            if ir.is_scalar:
                # --- SCALAR BRANCH ---
                mean = jnp.mean(field_reshaped, axis=(1, 2), keepdims=True)
                var = jnp.var(field_reshaped, axis=(1, 2), keepdims=True)
                normed = (field_reshaped - mean) / jnp.sqrt(var + self.eps)
                
                if self.affine:
                    normed = normed * self.weights[i].value + self.biases[i].value
            else:
                # --- VECTOR/TENSOR BRANCH ---
                sq = jnp.square(field_reshaped)
                if self.normalization == 'norm':
                    rms = jnp.sqrt(jnp.maximum(jnp.sum(sq, axis=1, keepdims=True), self.eps))
                else:
                    rms = jnp.sqrt(jnp.maximum(jnp.mean(sq, axis=1, keepdims=True), self.eps))
                
                layer_rms = jnp.sqrt(jnp.mean(jnp.square(rms), axis=1, keepdims=True) + self.eps)
                normed = field_reshaped / layer_rms
                
                if self.affine:
                    normed = normed * self.weights[i].value
            
            output_list.append(normed.reshape(field_data.shape[0], -1))

        return e3nn.IrrepsArray(self.irreps, jnp.concatenate(output_list, axis=-1))