import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

class LaplaceDatasetGenerator:
    def __init__(self, nx=64, ny=64, Lx=1.0, Ly=1.0):
        """
        Initialize the Laplace equation solver for dataset generation.
        
        Parameters:
        - nx, ny: Number of grid points in x and y directions
        - Lx, Ly: Physical domain size in x and y directions
        """
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        
        # Grid spacing
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        
        # Create coordinate matrices
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Build the finite difference matrix
        self._build_laplace_matrix()
    
    def _build_laplace_matrix(self):
        """Build the finite difference matrix for the 2D Laplace equation."""
        N = self.nx * self.ny
        
        # Coefficients for the 5-point stencil
        dx2 = self.dx**2
        dy2 = self.dy**2
        
        # Main diagonal
        main_diag = -2 * (1/dx2 + 1/dy2) * np.ones(N)
        
        # Off-diagonals
        x_diag = (1/dx2) * np.ones(N-1)
        y_diag = (1/dy2) * np.ones(N-self.nx)
        
        # Handle boundary conditions in the matrix
        # Remove connections across domain boundaries
        for i in range(self.nx-1, N-1, self.nx):
            x_diag[i] = 0
            
        # Create the sparse matrix
        diagonals = [y_diag, x_diag, main_diag, x_diag, y_diag]
        offsets = [-self.nx, -1, 0, 1, self.nx]
        
        self.A = diags(diagonals, offsets, shape=(N, N), format='csr')
    
    def _apply_boundary_conditions(self, alpha1, alpha2):
        """
        Apply boundary conditions parameterized by alpha1 and alpha2.
        
        Boundary conditions:
        - Left boundary (x=0): u = alpha1 * sin(π*y/Ly)
        - Right boundary (x=Lx): u = alpha2 * sin(2*π*y/Ly)  
        - Top boundary (y=Ly): u = 0
        - Bottom boundary (y=0): u = 0
        """
        b = np.zeros(self.nx * self.ny)
        A_bc = self.A.copy()
        
        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i
                
                # Bottom boundary (y = 0)
                if j == 0:
                    A_bc[idx, :] = 0
                    A_bc[idx, idx] = 1
                    b[idx] = 0
                
                # Top boundary (y = Ly)
                elif j == self.ny - 1:
                    A_bc[idx, :] = 0
                    A_bc[idx, idx] = 1
                    b[idx] = 0
                
                # Left boundary (x = 0)
                elif i == 0:
                    A_bc[idx, :] = 0
                    A_bc[idx, idx] = 1
                    y_val = j * self.dy
                    b[idx] = alpha1 * np.sin(np.pi * y_val / self.Ly)
                
                # Right boundary (x = Lx)
                elif i == self.nx - 1:
                    A_bc[idx, :] = 0
                    A_bc[idx, idx] = 1
                    y_val = j * self.dy
                    b[idx] = alpha2 * np.sin(2 * np.pi * y_val / self.Ly)
        
        return A_bc, b
    
    def solve_laplace(self, alpha1, alpha2):
        """
        Solve the 2D Laplace equation with given parameters.
        
        Returns:
        - solution: 2D array of the solution u(x,y)
        - parameters: dict with alpha1, alpha2 values
        """
        A_bc, b = self._apply_boundary_conditions(alpha1, alpha2)
        
        # Solve the linear system
        u_flat = spsolve(A_bc, b)
        
        # Reshape to 2D grid
        solution = u_flat.reshape(self.ny, self.nx)
        
        parameters = {'alpha1': alpha1, 'alpha2': alpha2}
        
        return solution, parameters
    
    def generate_dataset(self, n_samples, alpha1_range=(-2.0, 2.0), alpha2_range=(-2.0, 2.0), 
                        random_seed=42):
        """
        Generate a dataset of Laplace equation solutions.
        
        Parameters:
        - n_samples: Number of samples to generate
        - alpha1_range, alpha2_range: Ranges for parameter sampling
        - random_seed: Random seed for reproducibility
        
        Returns:
        - solutions: Array of shape (n_samples, ny, nx)
        - parameters: Array of shape (n_samples, 2) with [alpha1, alpha2]
        """
        np.random.seed(random_seed)
        
        solutions = []
        parameters = []
        
        print(f"Generating {n_samples} samples...")
        
        for i in range(n_samples):
            # Sample parameters
            alpha1 = np.random.uniform(*alpha1_range)
            alpha2 = np.random.uniform(*alpha2_range)
            
            # Solve equation
            solution, params = self.solve_laplace(alpha1, alpha2)
            
            solutions.append(solution)
            parameters.append([alpha1, alpha2])
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{n_samples} samples")
        
        solutions = np.array(solutions)
        parameters = np.array(parameters)
        
        return solutions, parameters
    
    def save_dataset(self, solutions, parameters, filename_prefix="laplace_dataset"):
        """Save the dataset to numpy files."""
        np.save(f"{filename_prefix}_solutions.npy", solutions)
        np.save(f"{filename_prefix}_parameters.npy", parameters)
        
        print(f"Dataset saved as {filename_prefix}_solutions.npy and {filename_prefix}_parameters.npy")
        print(f"Solutions shape: {solutions.shape}")
        print(f"Parameters shape: {parameters.shape}")
    
    def visualize_samples(self, solutions, parameters, n_samples=4):
        """Visualize some samples from the dataset."""
        fig, axes = plt.subplots(int(np.sqrt(n_samples)), int(np.sqrt(n_samples)), figsize=(12, 10))
        axes = axes.ravel()
        
        indices = np.random.choice(len(solutions), min(int(np.sqrt(n_samples))**2, len(solutions)), replace=False)
        
        for i, idx in enumerate(indices):
            #if i >= 4:
            #    break
                
            im = axes[i].contourf(self.X, self.Y, solutions[idx], levels=50, cmap='viridis')
            axes[i].set_title(f'α₁={parameters[idx][0]:.2f}, α₂={parameters[idx][1]:.2f}')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('y')
            axes[i].set_aspect('equal')
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize generator with 64x64 grid
    generator = LaplaceDatasetGenerator(nx=64, ny=64, Lx=1.0, Ly=1.0)
    
    # Generate dataset
    n_samples = 1000
    solutions, parameters = generator.generate_dataset(
        n_samples=n_samples,
        alpha1_range=(-5.0, 5.0),
        alpha2_range=(-5.0, 5.0)
    )
    
    # Save dataset
    generator.save_dataset(solutions, parameters, "data/laplace/laplace_dataset")
    
    # Visualize some samples
    generator.visualize_samples(solutions, parameters, n_samples=9)
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Solution min: {solutions.min():.4f}, max: {solutions.max():.4f}")
    print(f"Alpha1 range: [{parameters[:, 0].min():.2f}, {parameters[:, 0].max():.2f}]")
    print(f"Alpha2 range: [{parameters[:, 1].min():.2f}, {parameters[:, 1].max():.2f}]")
    
    # Example: Access individual sample
    sample_idx = 0
    print(f"\nSample {sample_idx}:")
    print(f"Parameters: α₁={parameters[sample_idx, 0]:.3f}, α₂={parameters[sample_idx, 1]:.3f}")
    print(f"Solution shape: {solutions[sample_idx].shape}")
