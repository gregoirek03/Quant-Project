import numpy as np
import pandas as pd


def compute_returns(close_prices: pd.Series) -> pd.Series:
    """Calcule les rendements logarithmiques d'un actif financier."""
    return np.log(close_prices / close_prices.shift(1))


def compute_strategy_returns(
    returns: pd.Series, signals: pd.Series, tc: float = 0.0
) -> pd.Series:
    """Calcule les rendements de la stratégie nets de frais."""
    gross_returns = returns * signals.shift(1)
    trades = signals.diff().abs().fillna(0)
    transaction_penalties = trades * tc
    return gross_returns - transaction_penalties


def compute_drawdown(equity_curve: pd.Series) -> tuple[pd.Series, float]:
    """Calcule le Drawdown temporel et le Maximum Drawdown historique."""
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = float(drawdown.min())
    return drawdown, max_drawdown


def run_vectorized_backtest(
    close_prices: pd.Series, signals: pd.Series, transaction_costs: float = 0.0
) -> dict:
    """Moteur principal de backtesting vectorisé avec gestion des frictions."""
    # 1. Rendements de l'actif
    asset_returns = compute_returns(close_prices)

    # 2. Rendements de la stratégie (Nets de frais)
    strat_returns = compute_strategy_returns(
        asset_returns, signals, tc=transaction_costs
    )

    # 3. Courbes de capital (Base 1)
    asset_equity = np.exp(asset_returns.fillna(0).cumsum())
    strat_equity = np.exp(strat_returns.fillna(0).cumsum())

    # 4. Risque & Drawdown
    _, max_dd = compute_drawdown(strat_equity)

    # 5. Métriques de performance annualisées (252 jours)
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
        "sharpe_ratio": sharpe_ratio,
        "total_trades": int(signals.diff().abs().sum()),
    }