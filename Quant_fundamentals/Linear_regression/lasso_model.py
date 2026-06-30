import numpy as np

def seuillage_doux(rho: float, lam: float) -> float:
    """
    Opérateur de seuillage doux (Soft Thresholding), cœur mathématique du LASSO.
    Force les coefficients non significatifs à valoir exactement 0.0.
    """
    if rho < -lam:
        return rho + lam
    elif rho > lam:
        return rho - lam
    else:
        return 0.0

def fit_lasso(X: np.ndarray, y: np.ndarray, lam: float, max_iter: int = 500, epsilon: float = 1e-6) -> tuple[float, np.ndarray]:
    """
    Calcule l'intercept et les coefficients beta par descente par coordonnées.
    Pénalité L1 pour la sélection automatique de variables (sparsité).
    """
    n_samples, n_features = X.shape
    intercept = 0.0
    beta = np.zeros(n_features)
    
    for _ in range(max_iter):
        beta_old = beta.copy()
        
        # Mise à jour vectorisée de l'intercept (moyenne des résidus)
        intercept = np.mean(y - (X @ beta))
        
        # Optimisation variable par variable
        for j in range(n_features):
            # Prédiction sans la variable j (opérateur @ pour la lisibilité)
            y_pred_without_j = intercept + (X @ beta) - (beta[j] * X[:, j])
            r = y - y_pred_without_j
            
            # Corrélation entre la variable j et le résidu
            rho = X[:, j] @ r
            
            # Seuillage doux et normalisation
            denominator = np.sum(X[:, j]**2)
            beta[j] = seuillage_doux(rho, lam * n_samples) / denominator
            
        # Condition d'arrêt par convergence de la norme L2
        if np.linalg.norm(beta - beta_old) < epsilon:
            break
            
    return intercept, beta

def predict_lasso(X: np.ndarray, intercept: float, beta: np.ndarray) -> np.ndarray:
    """Prédit les valeurs cibles à partir de l'intercept et du vecteur beta."""
    return intercept + (X @ beta)