import flax.linen as nn
import jax.numpy as jnp
import haiku as hk
import e3nn_jax as e3nn
import jax
class ShapeDecoder(nn.Module):
    """Decodes latent vectors back into 4 coordinate positions [4, 3]."""

    @nn.compact
    def __call__(self, inv):
        # inv shape: [batch, 6]
        # equiv shape: [batch, 3] (the 1o part)
        
        # 1. Process the invariant features through an MLP
        # This determines the 'template' shape coordinates.
        inv = inv.array if hasattr(inv, 'array') else inv
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
        local_coords = nn.Dense(12)(x)
        local_coords = local_coords.reshape(-1, 4, 3) # [batch, 4, 3]

        # 3. Incorporate the Equivariant part (the orientation)
        # We can treat 'equiv' as a translation or a scale/rotation guide.
        # For a simple reconstruction, we add the equivariant vector
        # to the local template to place it in global space.
        global_coords = local_coords

        return global_coords
    
class HaikuShapeDecoder(hk.Module):
    def __init__(self, output_nodes=4, name=None):
        super().__init__(name=name)
        self.output_nodes = output_nodes

    def __call__(self, latent):
            # 1. Extract invariant features 
            # If your latent is 120x0e, 'latent' is already an array of shape [batch, 120]
            # If it is still an IrrepsArray, use latent.array
            inv = latent.array if hasattr(latent, 'array') else latent
            
            # 2. Build the MLP
            # Use 'activation' instead of 'activate'
            mlp = hk.Sequential([
                hk.nets.MLP([32, 64, 128, 64], activation=jax.nn.relu, activate_final=True),
                hk.Linear(self.output_nodes * 3)
            ])
            
            # 3. Output
            local_coords = mlp(inv)
            return local_coords.reshape(-1, self.output_nodes, 3)