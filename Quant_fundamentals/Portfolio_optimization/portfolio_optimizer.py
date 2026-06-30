import numpy as np
import scipy.optimize as sco

def portfolio_statistics(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> tuple[float, float, float]:
    """
    Calcule le rendement attendu, la volatilité et le ratio de Sharpe d'un portefeuille.
    """
    # On s'assure que les vecteurs ont les bonnes dimensions mathématiques
    w = np.array(weights).reshape(-1, 1)
    r_mean = np.array(mean_returns).reshape(-1, 1)
    
    # Rendement attendu : W^T @ R
    port_return = float(w.T @ r_mean)
    
    # Volatilité attendue : sqrt(W^T @ Cov @ W)
    port_volatility = float(np.sqrt(w.T @ cov_matrix @ w))
    
    # Ratio de Sharpe (taux sans risque Rf = 0)
    sharpe_ratio = port_return / port_volatility if port_volatility != 0 else 0.0
    
    return port_return, port_volatility, sharpe_ratio

def _min_func_sharpe(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Fonction objectif : on retourne (-Sharpe) car scipy cherche un minimum."""
    return -portfolio_statistics(weights, mean_returns, cov_matrix)[2]

def maximize_sharpe_ratio(mean_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """
    Optimise les poids du portefeuille pour maximiser le ratio de Sharpe 
    via l'algorithme SLSQP avec contrainte budgétaire (somme des poids = 1).
    """
    num_assets = len(mean_returns)
    initial_guess = num_assets * [1. / num_assets]
    
    # Contrainte : la somme des poids doit être égale à 1 (100% du capital)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    
    # Bornes : Poids entre 0 et 1 pour chaque actif (pas de short-selling)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    result = sco.minimize(
        fun=_min_func_sharpe,
        x0=initial_guess,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x