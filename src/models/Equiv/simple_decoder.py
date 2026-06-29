import flax.linen as nn
import jax.numpy as jnp
import math 

import jax
import jax.numpy as jnp
from flax import linen as nn
import math

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)

class FoldingDecoder(nn.Module):
    num_samples: int = 256
    latent_dim: int = 8
    n_freqs: int = 4
    verbose: bool = True

    @property
    def grid_size(self):
        return int(math.sqrt(self.num_samples))

    def positional_encoding(self, coords):
        freqs = 2.0 ** jnp.arange(self.n_freqs)
        scaled = coords[..., None] * freqs[None, None, None, :] * math.pi
        encoded = jnp.concatenate([jnp.sin(scaled), jnp.cos(scaled)], axis=-1)
        return encoded.reshape(coords.shape[0], coords.shape[1], -1)
    
    def setup(self):
        # Using variance_scaling for better gradient flow
        # Use a smaller scale for the final folding layers to stabilize early training
        init_dense = nn.initializers.variance_scaling(1.0, 'fan_in', 'truncated_normal')
        init_fold = nn.initializers.variance_scaling(0.01, 'fan_in', 'truncated_normal')

        self.dense1_1 = nn.Dense(128, kernel_init=init_dense)
        self.norm1_1 = nn.LayerNorm()
        self.dense1_2 = nn.Dense(128, kernel_init=init_dense)
        self.norm1_2 = nn.LayerNorm()
        self.fold1 = nn.Dense(3, kernel_init=init_fold) 
        
        self.dense2_1 = nn.Dense(128, kernel_init=init_dense)
        self.norm2_1 = nn.LayerNorm()
        self.dense2_2 = nn.Dense(128, kernel_init=init_dense)
        self.norm2_2 = nn.LayerNorm()
        self.fold2 = nn.Dense(3, kernel_init=init_fold)

    def __call__(self, latent, manual_inv=False):

        if self.verbose:
            print(">>>>>> Decoder latent.shape: ", latent.shape)

        batch_size = latent.shape[0]
        latent_raw = latent.array if hasattr(latent, 'array') else latent
        latent_raw = 0*latent_raw if manual_inv == True else latent_raw
        latent_tiled = jnp.tile(latent_raw[:, None, :], (1, self.num_samples, 1))
        
        u = jnp.linspace(-1, 1, self.grid_size)
        v = jnp.linspace(-1, 1, self.grid_size)
        uu, vv = jnp.meshgrid(u, v)
        grid = jnp.stack([uu.flatten(), vv.flatten()], axis=-1)
        grid = jnp.broadcast_to(grid, (batch_size, self.num_samples, 2))
        
        encoded_grid = self.positional_encoding(grid)
        
        # --- First Fold ---
        x = jnp.concatenate([encoded_grid, latent_tiled], axis=-1)
        # Using nn.swish for smoother gradient propagation
        h = self.norm1_1(nn.swish(self.dense1_1(x)))
        h = self.norm1_2(nn.swish(self.dense1_2(h) + h)) 
        points_coarse = self.fold1(h) 
        
        # --- Second Fold ---
        x = jnp.concatenate([encoded_grid, points_coarse, latent_tiled], axis=-1)
        h = self.norm2_1(nn.swish(self.dense2_1(x)))
        h = self.norm2_2(nn.swish(self.dense2_2(h) + h)) 
        
        # Final output
        points_final = points_coarse + self.fold2(h)

        if self.verbose:
            print(">>>>>> Decoder output: ", points_final.shape)
            
        return points_final
    
class SimpleFoldingDecoder(nn.Module):
    num_samples: int = 64
    latent_dim: int = 8
    
    @property
    def grid_size(self):
        return int(math.sqrt(self.num_samples))

    def setup(self):
        # We process the latent and the 2D grid
        # Block 1: Coarse deformation
        self.mlp1 = [nn.Dense(128), nn.relu, nn.Dense(128), nn.relu]
        self.fold1 = nn.Dense(3) # Predicts 3D displacement
        
        # Block 2: Fine-detail folding
        self.mlp2 = [nn.Dense(128), nn.relu, nn.Dense(128), nn.relu]
        self.fold2 = nn.Dense(3) # Predicts residual displacement

    def __call__(self, latent, manual_inv=True):
        batch_size = latent.shape[0]
        latent_raw = latent.array if hasattr(latent, 'array') else latent
        latent_raw = 0*latent_raw if manual_inv == True else latent_raw

        # Create static 2D grid (the "canvas")
        u = jnp.linspace(-1, 1, self.grid_size)
        v = jnp.linspace(-1, 1, self.grid_size)
        uu, vv = jnp.meshgrid(u, v)
        grid = jnp.stack([uu.flatten(), vv.flatten()], axis=-1) # [num_samples, 2]
        
        # Tile grid for batch
        grid = jnp.broadcast_to(grid, (batch_size, self.num_samples, 2))
        
        # First Fold: Map 2D -> 3D "blob"
        # Concatenate latent to every point on the grid
        x = jnp.concatenate([
            grid, 
            jnp.tile(latent_raw[:, None, :], 
            (1, self.num_samples, 1))
        ], axis=-1)
        for layer in self.mlp1: x = layer(x)
        
        # Coarse 3D points
        points_coarse = self.fold1(x) 
        
        # Second Fold: Refine the shape
        # Concatenate original grid + coarse points + latent
        x = jnp.concatenate([
            grid, 
            points_coarse, 
            jnp.tile(latent_raw[:, None, :], 
            (1, self.num_samples, 1))
            ], axis=-1)
        for layer in self.mlp2: x = layer(x)
        
        # Final output is a residual of the coarse points
        points_final = points_coarse + self.fold2(x)
        
        return points_final


class SimpleFreqGridDecoder(nn.Module):
    num_samples: int = 64
    latent_dim: int = 8
    n_freqs: int = 4  # Number of frequency bands for encoding

    @property
    def grid_size(self):
        return int(math.sqrt(self.num_samples))

    def positional_encoding(self, coords):
        """
        Encodes [batch, num_samples, 2] -> [batch, num_samples, 2 * 2 * n_freqs]
        """
        # Create frequency bands: 2^0, 2^1, ... 2^(n-1)
        freqs = 2.0 ** jnp.arange(self.n_freqs) # [n_freqs]
        # Shape: [batch, num_samples, 2, n_freqs]
        scaled = coords[..., None] * freqs[None, None, None, :] * math.pi
        
        # Interleave sin and cos: [batch, num_samples, 2, 2 * n_freqs]
        encoded = jnp.concatenate([jnp.sin(scaled), jnp.cos(scaled)], axis=-1)
        return encoded.reshape(coords.shape[0], coords.shape[1], -1)

    @nn.compact
    def __call__(self, inv=None, manual_inv=True):
        # Allow passing a manual latent if needed
        
        inv = inv.array if hasattr(inv, 'array') else inv
        inv = 0*inv if manual_inv == True else inv
        print("inv.shape: ", inv.shape)
        batch_size = inv.shape[0]
        
        # 1. Create a 2D mesh grid
        u = jnp.linspace(-1, 1, self.grid_size)
        v = jnp.linspace(-1, 1, self.grid_size)
        uu, vv = jnp.meshgrid(u, v)
        query_coords = jnp.stack([uu.flatten(), vv.flatten()], axis=-1)
        
        # 2. Encode Coordinates (The "Positional" part)
        z_encoded = self.positional_encoding(query_coords[jnp.newaxis, ...]) 
        z_tiled = jnp.broadcast_to(z_encoded, (batch_size, self.num_samples, z_encoded.shape[-1]))
        
        # 3. Prepare latents
        latent_tiled = jnp.broadcast_to(
            inv[:, jnp.newaxis, :], 
            (batch_size, self.num_samples, inv.shape[-1])
        )
        
        # 4. Concatenate: [batch, num_samples, latent_dim + encoded_coords_dim]
        x = jnp.concatenate([latent_tiled, z_tiled], axis=-1)
        
        x_in = nn.Dense(256)(x)
        norm = nn.LayerNorm()(x_in)
        x = nn.relu(norm)
        x = nn.Dense(256)(x)
        skip = x + x_in
        norm =nn.LayerNorm()(skip)
        x = nn.relu(norm)# Skip + Norm
        
        # Block 2: 256 -> 128
        # Projection to match dimensions for the residual skip
        x_in = nn.Dense(128)(x)
        norm = nn.LayerNorm()(x_in)
        x = nn.relu(norm)
        x = nn.Dense(128)(x)
        skip =x + x_in
        norm =nn.LayerNorm()(skip)
        x = nn.relu(norm) # Skip + Norm
        
        # Final head
        coords = nn.Dense(3)(x)
        return coords
    
class SimpleGridDecoder(nn.Module):
    """
    Coordinate-MLP Decoder.
    Takes latent 'inv' and a list of query points/indices to produce coordinates.
    """
    num_samples: int = 256 # Number of points to generate for the reconstruction

    @nn.compact
    def __call__(self, inv, manual_inv = True):
        # inv shape: [batch, latent_dim]
        inv = inv.array if hasattr(inv, 'array') else inv
        inv = inv.array if hasattr(inv, 'array') else inv
        inv = 0*inv if manual_inv == True else inv

        point_ids = jnp.linspace(-1, 1, self.num_samples) 
        z_tiled = jnp.tile(inv, (self.num_samples, 1))
        combined = jnp.concatenate([z_tiled, point_ids[:, None]], axis=1)

        # 1. Expand latent to be available for all sample points
        batch_size = inv.shape[0]
        # Create a coordinate grid or sampling indices as input
        # Here we use simple linear sampling to represent the "latent surface"
        z = jnp.linspace(-1, 1, self.num_samples).reshape(1, -1, 1)
        z = jnp.repeat(z, batch_size, axis=0) # [batch, num_samples, 1]
        
        # Tile the invariant features to match the number of samples
        latent = inv[:, jnp.newaxis, :].repeat(self.num_samples, axis=1) # [batch, num_samples, latent_dim]
        
        # 2. Concatenate Latent + Query Point
        x = jnp.concatenate([latent, z], axis=-1)
        
        # 3. MLP that predicts 3D coordinates from the latent + index
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        
        # Output [batch, num_samples, 3]
        coords = nn.Dense(3)(x)
        
        return coords
    
class SimpleDecoder(nn.Module):
    """Decodes latent vectors back into 4 coordinate positions [4, 3]."""

    @nn.compact
    def __call__(self, inv,manual_inv=True):
        # inv shape: [batch, 6]
        # equiv shape: [batch, 3] (the 1o part)

        # 1. Process the invariant features through an MLP
        # This determines the 'template' shape coordinates.
        inv = inv.array if hasattr(inv, 'array') else inv
        inv = 0*inv if manual_inv == True else inv
        x = nn.Dense(32)(inv)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64*2)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)

        # 2. Output 12 values (4 nodes * 3 coordinates)
        # We treat these as coordinates in a local reference frame
        coords = nn.Dense(12)(x)
        coords = coords.reshape(-1, 4, 3) # [batch, 4, 3]

        return coords
   