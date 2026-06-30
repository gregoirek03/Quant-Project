import numpy as np

def fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float = 1.0, fit_intercept: bool = True) -> np.ndarray:
    """
    Calcule les coefficients de la régression Ridge (Pénalité L2).
    Résolution stable via np.linalg.solve.
    """
    if fit_intercept:
        n_samples = X.shape[0]
        X_biased = np.hstack((np.ones((n_samples, 1)), X))
    else:
        X_biased = X
        
    n_features = X_biased.shape[1]
    
    # Création de la matrice identité pour la pénalité
    I = np.eye(n_features)
    if fit_intercept:
        I[0, 0] = 0  # On ne pénalise pas l'intercept constante

    # Système matriciel : (X^T @ X + lam * I) @ beta = X^T @ Y
    A = (X_biased.T @ X_biased) + (lam * I)
    b = X_biased.T @ Y
    
    return np.linalg.solve(A, b)

def predict_ridge(X: np.ndarray, beta: np.ndarray, fit_intercept: bool = True) -> np.ndarray:
    """
    Prédit les valeurs de Y à partir de X et des coefficients beta.
    """
    if fit_intercept:
        n_samples = X.shape[0]
        X_biased = np.hstack((np.ones((n_samples, 1)), X))
    else:
        X_biased = X
        
    return X_biased @ beta