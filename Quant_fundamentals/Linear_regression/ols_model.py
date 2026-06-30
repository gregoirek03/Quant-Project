import numpy as np

def fit_ols(X: np.ndarray, Y: np.ndarray, fit_intercept: bool = True) -> np.ndarray:
# Calcule les coefficients beta par la méthode des Moindres Carrés Ordinaires (OLS)
    
    if fit_intercept: # on inclut une constante au modèle (ordonnée à l'origine)
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))
         # np.hstack est préféré à np.c_ pour la clarté et l'auto-documentation
    else:
        X_b = X

    A = X_b.T @ X_b
    b = X_b.T @ Y
    # Utilisation de l'opérateur @ pour la lisibilité matricielle
    
    return np.linalg.solve(A, b)


def predict_ols(X: np.ndarray, beta: np.ndarray, fit_intercept: bool = True) -> np.ndarray:
#Prédit les valeurs cibles (Y_pred) à partir des caractéristiques X et des coefficients beta estimés.
    
    if fit_intercept:
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))
    else:
        X_b = X
        
    return X_b @ beta
