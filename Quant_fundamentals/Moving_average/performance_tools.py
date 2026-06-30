import numpy as np
import pandas as pd

def calculate_moving_average_vectorized(prices: np.ndarray, window: int) -> pd.Series:
    """
    Calcule une moyenne mobile glissante de manière vectorisée.
    Utilise les fonctions natives de Pandas codées en C pour maximiser les performances.
    """
    df = pd.DataFrame({'Prix': prices})
    return df['Prix'].rolling(window=window).mean()