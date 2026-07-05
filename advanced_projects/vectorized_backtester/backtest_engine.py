import numpy as np
import pandas as pd

def compute_returns(close_prices: pd.Series) -> pd.Series:
    """Calcule les rendements logarithmiques d'un actif financier."""
    return np.log(close_prices / close_prices.shift(1))

# utilité du rendement logarithmique :
# additivté temporelle
# meilleure adaptation aux modèles statistiques

def compute_strategy_returns(returns: pd.Series, signals: pd.Series) -> pd.Series:
    """
    Calcule les rendements de la stratégie.
    Le signal est décalé de 1 jour (.shift(1)) pour éviter le biais de survie
    (on utilise le signal calculé la veille au soir pour investir aujourd'hui).
    """
    return returns * signals.shift(1)

# signals vaut 1 si on est acheteur, 0 si on est neutre, -1 si on shorte

# shift(1) : on multiplie le signal d'hier soir par le rendement aujourd'hui
# Si on oublie shift(1), le modèle triche en utilisant des informations du futur

def compute_drawdown(equity_curve: pd.Series) -> tuple[pd.Series, float]:
    """
    Calcule la série temporelle du Drawdown et le Maximum Drawdown historique.
    Entièrement vectorisé en utilisant les maximums cumulés (.cummax()).
    """
    # Calcul des sommets historiques glissants de la courbe de capital
    running_max = equity_curve.cummax()
    
    # Drawdown en pourcentage du dernier sommet
    drawdown = (equity_curve - running_max) / running_max
    
    # Pire valeur (la plus négative)
    max_drawdown = float(drawdown.min())
    
    return drawdown, max_drawdown

def run_vectorized_backtest(close_prices: pd.Series, signals: pd.Series) -> dict:
    """
    Moteur principal de backtesting vectorisé.
    Prend les prix de clôture et un vecteur de signaux (-1, 0, 1) et calcule les performances.
    """
    # 1. Rendements de l'actif
    asset_returns = compute_returns(close_prices)
    
    # 2. Rendements de la stratégie
    strat_returns = compute_strategy_returns(asset_returns, signals)
    
    # 3. Courbes de capital (Base 1)
    # L'exponentielle de la somme cumulée des log returns donne la performance géométrique
    asset_equity = np.exp(asset_returns.fillna(0).cumsum())
    strat_equity = np.exp(strat_returns.fillna(0).cumsum())
    
    # 4. Risque & Drawdown
    _, max_dd = compute_drawdown(strat_equity)
    
    # 5. Métriques de performance (Annualisation brute sur 252 jours de trading)
    total_return = strat_equity.iloc[-1] - 1.0
    ann_return = np.exp(strat_returns.mean() * 252) - 1.0
    ann_vol = strat_returns.std() * np.sqrt(252)
    sharpe_ratio = ann_return / ann_vol if ann_vol > 0 else 0.0
    
    return {
        "asset_returns": asset_returns,
        "strategy_returns": strat_returns,
        "asset_equity": asset_equity,
        "strategy_equity": strat_equity,
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe_ratio
    }