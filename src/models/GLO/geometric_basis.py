
import torch
import torch.nn as nn

def MSE_loss(f,f_rec):
     loss = torch.mean((f - f_rec)**2)
     return loss

def fit_geometric_basis_old(f, model, P, evecs, iterations=100, lr=5, decay= None, decay_every= None):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for i in range(iterations):
        optimizer.zero_grad() 
        
        f_rec = model(f, P, evecs)
        loss = torch.mean((f - f_rec)**2)
        
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"Iteration {i:03d} | Loss: {loss.item():.6f} | Sigma Mean: {model.sigmas.mean().item():.4f}")
            
    return model

def fit_geometric_basis(f, model, P, evecs, iterations=100, lr=5, decay=None, decay_every=None):
    # Initialize Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Initialize Scheduler if decay parameters are provided
    scheduler = None
    if decay is not None and decay_every is not None:
        # StepLR reduces learning rate by 'decay' factor every 'decay_every' steps
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_every, gamma=decay)
    
    for i in range(iterations):
        optimizer.zero_grad() 
        
        f_rec = model(f, P, evecs)
        loss = torch.mean((f - f_rec)**2)
        
        loss.backward()
        optimizer.step()
        
        # Step the scheduler if it exists
        if scheduler:
            scheduler.step()
        
        if i % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Iteration {i:03d} | Loss: {loss.item():.6f} | "
                  f"LR: {current_lr:.6f} | Sigma Mean: {model.sigmas.mean().item():.4f}")
            
    return model

class GeometricBasis(nn.Module):
    def __init__(self, k_trunc, dim = 1, learn_sigma = True):
        super().__init__()
        self.k_trunc = k_trunc
  
        self.sigmas = nn.Parameter(torch.ones(k_trunc, dim) * 1, requires_grad=learn_sigma)
        

    def get_basis(self, P, evecs):
        """
        Constructs the geometric basis epsilon_z(x) = exp(-sigma * P) * phi_omega.
        
        Args:
            sigmas: (K, 1) - Learned spatial decay/localization parameters.
            Delta_evals: (K, 1) - Laplacian eigenvalues (used to identify frequency).
            Delta_evecs: (V, K) - Laplace-Beltrami eigenfunctions.
            P: (V, 1) - Intrinsic geometric features (e.g., HKS).
            
        Returns:
            basis: (V, K) - The localized geometric basis functions.
        """

        Phi = evecs[:, :self.k_trunc] 
        if P.dim() == 1: 
            P = P.view(-1, 1)
        evecs = evecs[:, :self.k_trunc] 
 
        decay = torch.exp(-P.unsqueeze(-1) * self.sigmas.unsqueeze(0))

        basis = decay * Phi.unsqueeze(-1)

        return basis

    def compute_coeffs_(self, f, basis):
        """
        basis: (V, K, dim)
        f: (V, 1) or (V, dim)
        """
        # 1. Force f to be (V, dim) regardless of input shape
        # If f is 1D (V,), make it (V, 1)
        if f.dim() == 1:
            f = f.view(-1, 1)
        
        # Now f is definitely (V, D). 
        # If D=1, the einsum 'vkd,vd->kd' works perfectly.
        # If D=3, it also works perfectly.

        # 2. Gram Matrix: sum over V and D
        # basis: (V, K, dim), basis: (V, K, dim)
        #gram_matrix = torch.einsum('vkd,vjd->kj', basis, basis)
        
        # 3. RHS: sum over V.
        # basis: (V, K, dim), f: (V, dim)
        #rhs = torch.einsum('vkd,vd->kd', basis, f)
        
        # 4. Solve (K, K) x (K, dim) -> (K, dim)
        #reg = 1e-1 * torch.eye(self.k_trunc, device=f.device)
        #alphas = torch.linalg.solve(gram_matrix + reg, rhs)
        norm = torch.norm(basis, dim=0, keepdim=True) # (1, K, dim)
        basis = basis / (norm + 1e-6)
        gram_matrix = torch.einsum('vkd,vjd->kj', basis, basis)
        rhs = torch.einsum('vkd,vd->kd', basis, f)
        # Replace linalg.solve with linalg.lstsq
        # rcond=None automatically handles the thresholding for singular matrices
        alphas, _, _, _ = torch.linalg.lstsq(gram_matrix + 1e-3 * torch.eye(self.k_trunc, device=f.device), rhs, rcond=None)
        
        return alphas
    
    def compute_coeffs(self, f, basis):
        """
        basis: (V, K, D)
        f: (V, D)
        """
        if f.dim() == 1: f = f.view(-1, 1)
        
        # 'vkd' is (V, K, D), 'vjd' is (V, K, D)
        # By outputting 'kj', we are summing over both V and D
        gram_matrix = torch.einsum('vkd,vjd->kj', basis, basis)
        
        # 'vkd' (V, K, D), 'vd' (V, D) -> 'kd' (K, D)
        # This sums over V only
        rhs = torch.einsum('vkd,vd->kd', basis, f)
        
        reg = 1e-3 * torch.eye(self.k_trunc, device=f.device)
        alphas, _, _, _ = torch.linalg.lstsq(gram_matrix + reg, rhs, rcond=None)
        
        return alphas

    def forward(self, f, P, evecs):
        basis = self.get_basis(P, evecs) # (V, K, dim)
        alphas = self.compute_coeffs(f, basis) # (K, dim)
        func = torch.einsum('vkd,kd->vd', basis, alphas)

        self.sigmas.data.clamp_(min=-5000, max=5000.0)
        return func
    


class GeometricBasis_shared_sigma(nn.Module):
    def __init__(self, k_trunc):
        super().__init__()
        self.k_trunc = k_trunc
        self.sigmas = nn.Parameter(torch.ones(k_trunc, 1) * 0.1)


    def get_basis(self, P, evecs):
        """
        Constructs the geometric basis epsilon_z(x) = exp(-sigma * P) * phi_omega.
        
        Args:
            sigmas: (K, 1) - Learned spatial decay/localization parameters.
            Delta_evals: (K, 1) - Laplacian eigenvalues (used to identify frequency).
            Delta_evecs: (V, K) - Laplace-Beltrami eigenfunctions.
            P: (V, 1) - Intrinsic geometric features (e.g., HKS).
            
        Returns:
            basis: (V, K) - The localized geometric basis functions.
        """

        Phi = evecs[:, :self.k_trunc] 
        if P.dim() == 1: 
            P = P.view(-1, 1)
        evecs = evecs[:, :self.k_trunc] 

        decay = torch.exp(-P @ self.sigmas.t()) 
        basis = decay * Phi
        
        return basis

    def compute_coeffs(self, f, basis):
        """
        Computes coefficients alpha via projection onto the localized basis.
        Since the localized basis is not orthonormal, we solve the 
        normal equations: (B^T @ B) * alpha = B^T * f
        """
        # Solve for alpha using least squares (Moore-Penrose pseudo-inverse)
        # This is the manifold-equivalent of the FFT projection
        # alpha = (B^T @ B)^-1 @ B^T @ f
        basis_t = basis.t()
        gram_matrix = basis_t @ basis
        rhs = basis_t @ f
        # Use torch.linalg.solve for stability
        alphas = torch.linalg.solve(gram_matrix + 1e-4 * torch.eye(self.k_trunc, device=f.device), rhs)
        return alphas


    def forward(self, f,  P, evecs):
        basis = self.get_basis( P, evecs)
        alphas = self.compute_coeffs(f, basis)
        func =  basis @ alphas
        return func