import numpy as np
from sklearn.svm import SVC

def fit_svm_model(X, y, kernel='rbf', C=1.0, gamma='scale', degree=3):
    """
    Fits a Support Vector Classifier (SVC) to the data.
    
    Parameters:
        X (np.ndarray): Features, shape (n_samples, 2).
        y (np.ndarray): Labels, shape (n_samples,).
        kernel (str): Specifies the kernel type ('linear', 'rbf', 'poly').
        C (float): Regularization parameter.
        gamma (float or str): Kernel coefficient for 'rbf' and 'poly'.
        degree (int): Degree of the polynomial kernel function ('poly').
        
    Returns:
        model (SVC): Fitted scikit-learn SVC model.
    """
    # SVC in scikit-learn expects gamma to be a float, 'scale', or 'auto'.
    # If gamma is passed as a string representing a float, we convert it.
    if isinstance(gamma, str):
        if gamma.lower() not in ['scale', 'auto']:
            try:
                gamma = float(gamma)
            except ValueError:
                gamma = 'scale'
                
    model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
    model.fit(X, y)
    return model

def get_decision_grid(model, X, grid_resolution=150, padding=0.2):
    """
    Generates a 2D meshgrid and computes the decision function values.
    
    Parameters:
        model (SVC): Fitted scikit-learn SVC model.
        X (np.ndarray): Input features, shape (n_samples, 2), used to determine limits.
        grid_resolution (int): Number of points in each grid dimension.
        padding (float): Margin added around the max/min bounds of X.
        
    Returns:
        xx (np.ndarray): X coordinates of grid mesh.
        yy (np.ndarray): Y coordinates of grid mesh.
        Z (np.ndarray): Decision values for the grid mesh, shape (grid_resolution, grid_resolution).
    """
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution)
    )
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # decision_function returns the distance of each sample to the separation boundary.
    Z = model.decision_function(grid_points)
    Z = Z.reshape(xx.shape)
    
    return xx, yy, Z
