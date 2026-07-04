import numpy as np
import pandas as pd

# =================================
# 1. CALCUL D'UNE MOYENNE GLISSANTE
# =================================

def calculate_moving_average(prices: np.ndarray, window: int) -> pd.Series:
    """
    Calcule une moyenne mobile glissante de manière hautement vectorisée.
    Délègue les calculs aux routines optimisées en C de Pandas.
    """
    s = pd.Series(prices)
    return s.rolling(window=window).mean()

# ========================
# 2. MODÈLE DE MONTE-CARLO
# ========================

def simulate_monte_carlo(start_price: float, mu: float, sigma: float, days: int, n_simulations: int, seed: int = 42) -> np.ndarray:
    """
    Génère des trajectoires de prix d'actifs via un Mouvement Brownien Géométrique (GBM).
    Code entièrement vectorisé (aucun loop sur les simulations).
    """
    rng = np.random.default_rng(seed=seed)
    
    # Matrice de chocs gaussiens : dimensions (jours, simulations)
    shocks = rng.normal(loc=0, scale=1, size=(days, n_simulations))
    
    # Calcul des rendements journaliers selon la dynamique de Black-Scholes
    dt = 1 / 252  # Pas de temps annuel journalier
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * shocks
    
    # Calcul des prix par capitalisation continue (produit cumulé exponentiel)
    price_paths = start_price * np.exp(np.cumsum(drift + diffusion, axis=0))
    
    # On ajoute le prix de départ sur la première ligne
    initial_row = np.full((1, n_simulations), start_price)
    return np.vstack((initial_row, price_paths))

# ======================
# 3. CALCUL DU RENDEMENT
# ======================

def calculate_arithmetic_returns(prices: np.ndarray) -> np.ndarray:
    """
    Calcule les rendements arithmétiques (simples) d'une série de prix.
    Formule : (P_t - P_{t-1}) / P_{t-1}
    """
    # np.diff calcule P_t - P_{t-1}
    return np.diff(prices, axis=0) / prices[:-1]

def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Calcule les rendements logarithmiques (continus) d'une série de prix.
    Formule : ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
    """
    return np.diff(np.log(prices), axis=0)