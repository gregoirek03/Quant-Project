import numpy as np

# ==========================================
# 1. MODÈLE OLS (Moindres Carrés Ordinaires)
# ==========================================
def fit_ols(X: np.ndarray, Y: np.ndarray, fit_intercept: bool = True) -> np.ndarray:
    """Calcule les coefficients beta par la méthode OLS (stable)."""
    if fit_intercept:
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))
    else:
        X_b = X
    return np.linalg.solve(X_b.T @ X_b, X_b.T @ Y)

def predict_ols(X: np.ndarray, beta: np.ndarray, fit_intercept: bool = True) -> np.ndarray:
    """Prédit les valeurs cibles (Y_pred) pour le modèle OLS."""
    if fit_intercept:
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))
    else:
        X_b = X
    return X_b @ beta

# =============================
# 2. MODÈLE RIDGE (Pénalité L2)
# =============================
def fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float = 1.0, fit_intercept: bool = True) -> np.ndarray:
    """Calcule les coefficients de la régression Ridge (Pénalité L2)."""
    if fit_intercept:
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))
    else:
        X_b = X
        
    n_features = X_b.shape[1]
    I = np.eye(n_features)
    if fit_intercept:
        I[0, 0] = 0  # On ne pénalise pas la constante
        
    A = (X_b.T @ X_b) + (lam * I)
    b = X_b.T @ Y
    return np.linalg.solve(A, b)

def predict_ridge(X: np.ndarray, beta: np.ndarray, fit_intercept: bool = True) -> np.ndarray:
    """Prédit les valeurs cibles (Y_pred) pour le modèle Ridge."""
    if fit_intercept:
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))
    else:
        X_b = X
    return X_b @ beta

# =========================================================
# 3. MODÈLE LASSO (Pénalité L1 par Descente de Coordonnées)
# =========================================================
def seuillage_doux(rho: float, lam: float) -> float:
    """Opérateur de seuillage doux (Soft Thresholding) pour le LASSO."""
    # rho = coefficient de corrélation
    if rho < -lam:
        return rho + lam
    elif rho > lam:
        return rho - lam
    else:
        return 0.0

def fit_lasso(X: np.ndarray, y: np.ndarray, lam: float, max_iter: int = 500, epsilon: float = 1e-6) -> tuple[float, np.ndarray]:
    """Calcule l'intercept et les coefficients beta par descente par coordonnées."""
    n_samples, n_features = X.shape
    intercept = 0.0
    beta = np.zeros(n_features)
    # on initialise tous les coefficients beta à 0
    
    for _ in range(max_iter):
        beta_old = beta.copy()
        # sauvegarde des anciens beta
        intercept = np.mean(y - (X @ beta))
        # recalcul de l'intercept comme moyenne des écarts
        
        for j in range(n_features):
            y_pred_without_j = intercept + (X @ beta) - (beta[j] * X[:, j])
            # X[:, j] = colonne j de X
            r = y - y_pred_without_j
            rho = X[:, j] @ r
            # rho mesure à quel point la variable j est corrélationnelle avec l'erreur qu'il reste à combler. 
            # Si rho est grand, la variable j est très utile pour corriger l'erreur.
            
            denominator = np.sum(X[:, j]**2)
            # norme de la colonne j --> normalisation
            beta[j] = seuillage_doux(rho, lam * n_samples) / denominator
            
        if np.linalg.norm(beta - beta_old) < epsilon:
            break
            
    return intercept, beta

def predict_lasso(X: np.ndarray, intercept: float, beta: np.ndarray) -> np.ndarray:
    """Prédit les valeurs cibles (Y_pred) pour le modèle LASSO."""
    return intercept + (X @ beta)