import jax
import jax.numpy as jnp
import haiku as hk
import e3nn_jax as e3nn

def safe_sqrt(x, eps):
    return jnp.sqrt(jnp.maximum(x, eps))

class EquivariantLayerNorm(hk.Module):
    def __init__(self, irreps, eps=1e-2, affine=True, normalization='component', name=None):
        super().__init__(name=name)
        self.irreps = e3nn.Irreps(irreps)
        self.eps = eps
        self.affine = affine
        self.normalization = normalization

    def __call__(self, x: e3nn.IrrepsArray):
        # Optional: verify input matches expected irreps to catch bugs early
        # assert x.irreps == self.irreps
        
        output_list = []
        # Index into the raw flat array of x
        start_flat_idx = 0 
        
        for i, (mul, ir) in enumerate(self.irreps):
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
                    w = hk.get_parameter(f"w_{i}_l{ir.l}", shape=(1, mul, 1), init=jnp.ones)
                    b = hk.get_parameter(f"b_{i}_l{ir.l}", shape=(1, mul, 1), init=jnp.zeros)
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
                normed = field_reshaped / layer_rms
                
                if self.affine:
                    w = hk.get_parameter(f"w_{i}_l{ir.l}", shape=(1, mul, 1), init=jnp.ones)
                    normed = normed * w
            
            # Flatten back to (nodes, mul * dim)
            output_list.append(normed.reshape(field_data.shape[0], -1))

        # Reconstruct the IrrepsArray
        final_array = jnp.concatenate(output_list, axis=-1)
        return e3nn.IrrepsArray(self.irreps, final_array)