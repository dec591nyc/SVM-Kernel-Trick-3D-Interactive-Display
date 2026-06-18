import numpy as np
from sklearn.datasets import make_circles

def generate_concentric_circles(noise=0.05, n_samples=200, random_seed=42, factor=0.4):
    """
    Generates a deterministic concentric circle dataset.
    
    Parameters:
        noise (float): Standard deviation of Gaussian noise added to the data.
        n_samples (int): Total number of points generated.
        random_seed (int): Random seed for reproducibility.
        factor (float): Scale factor between inner and outer circle.
        
    Returns:
        X (np.ndarray): Shape (n_samples, 2), containing the (x, y) coordinates.
        y (np.ndarray): Shape (n_samples,), containing class labels:
                        - 0: Inner circle (will be rendered Blue)
                        - 1: Outer circle (will be rendered Red)
    """
    X, y_raw = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=random_seed
    )
    
    # By default, make_circles assigns 1 to the inner circle and 0 to the outer circle.
    # We invert the labels so that:
    # y = 0 -> Inner circle (Blue)
    # y = 1 -> Outer circle (Red)
    y = 1 - y_raw
    
    return X, y
