import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Calcule la fonction sigmoïde pour projeter les valeurs entre 0 et 1."""
    z = np.clip(z, -500, 500)  # Sécurité anti-overflow
    return 1 / (1 + np.exp(-z))

def fit_logistic(X_b: np.ndarray, Y: np.ndarray, lr: float = 0.1, num_iter: int = 1000) -> tuple[np.ndarray, list[float]]:
    """
    Entraîne le modèle par descente de gradient et enregistre l'évolution de la perte.
    Optimise la fonction de coût Cross-Entropy (Log Loss).
    """
    n_samples = X_b.shape[0]
    beta = np.zeros((X_b.shape[1], 1))
    loss_history = []
    
    for i in range(num_iter):
        Y_pred = sigmoid(X_b @ beta)
        
        # Calcul du gradient vectorisé
        gradient = (X_b.T @ (Y_pred - Y)) / n_samples
        beta -= lr * gradient
        
        # Enregistrement de la Log Loss
        if i % 200 == 0:
            loss = - (1 / n_samples) * np.sum(Y * np.log(Y_pred + 1e-15) + (1 - Y) * np.log(1 - Y_pred + 1e-15))
            loss_history.append((i, loss))
            
    return beta, loss_history

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Calcule manuellement les Vrais Positifs, Vrais Négatifs, Faux Positifs et Faux Négatifs."""
    # .ravel() aplatit les vecteurs pour faciliter la comparaison terme à terme
    t = y_true.ravel()
    p = y_pred.ravel()
    
    TN = int(np.sum((t == 0) & (p == 0)))
    FP = int(np.sum((t == 0) & (p == 1)))
    FN = int(np.sum((t == 1) & (p == 0)))
    TP = int(np.sum((t == 1) & (p == 1)))
    
    return TN, FP, FN, TP