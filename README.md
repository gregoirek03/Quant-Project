📈 Quant Finance Laboratory

Welcome to my quantitative finance research repository. This project is divided into two distinct development phases: a foundational research hub to master core mechanics, followed by production-grade financial engineering applications.

The primary objective of this Quant_fundamentals section is to deeply assimilate the mathematical and statistical foundations of quantitative asset management. To ensure a thorough, low-level understanding of numerical optimization and statistical mechanics, every single formula, algorithm, and objective function has been coded entirely from scratch using Python, avoiding any pre-existing high-level machine learning libraries (such as Scikit-Learn).

Every directory follows a production-grade layout: a standalone core engineering file (.py) containing pure vectorized functions, paired with a comprehensive research notebook (.ipynb) acting as a visual validation lab.

🚀 Project Roadmap: From Theory to Production

To build institutional-grade trading systems, one must first master the underlying mathematics. This repository mirrors that exact professional progression:

Phase 1: Quant Fundamentals (Current Hub): Deep-dive into raw implementations (Optimization, SDEs, Matrix Regressions). This acts as our proprietary math library.

Phase 2: Advanced Projects (Next Hub): Leveraging our fundamental blocks to build large-scale systems, including Vectorized Backtesting Engines (SMA/RSI strategies with real-world data), Statistical Arbitrage (Pairs Trading) frameworks, and alternative data pipelines.

🗂️ Repository Architecture

Quant_fundamentals/
├── Linear_regression/        # Continuous modeling & regularizations (L1/L2)
│   ├── linear_models.py
│   └── benchmark_linear.ipynb
├── Logistic_regression/      # Binary classification & custom optimization
│   ├── logistic_models.py
│   └── benchmark_logistic.ipynb
├── Portfolio_optimization/   # Modern Portfolio Theory (MPT) & Sharpe Max
│   ├── portfolio_optimizer.py
│   └── benchmark_portfolio.ipynb
└── Numerical_methods/        # Time-series vectorization & stochastic models
    ├── numerical_tools.py
    └── benchmark_numerical.ipynb

🧠 Technical Overview of Modules

1. Advanced Backtesting Framework (Advanced_backtesting)

Concepts: Event-free vectorized execution modeling institutional market frictions, asset class regime variations, and conditional strategy filtering.

Risk Architecture: Moves beyond traditional returns to compute higher-moment statistics (SciPy Stats) including Sortino Ratio, Downside Deviation, Skewness, and Kurtosis to detect empirical Fat Tails.

Strategies Implemented: Fast/Slow SMA Crossover, Bollinger Bands Mean Reversion, and a Trend-Filtered Hybrid Framework (triggers mean-reversion buying only during verified long-term macroeconomic uptrends).

2. Linear Regression & Regularization (Linear_regression)

Algorithms: Designed and built a custom Coordinate Descent engine from scratch. It integrates a Soft-Thresholding operator for the LASSO model to perform sparse feature selection, effectively zeroing out irrelevant coefficients to isolate true alpha signals from structural market noise.

3. Portfolio Optimization (Portfolio_optimization)

Algorithms: Expressed portfolio volatility via a vectorized quadratic form ($W^T\Sigma W$). Leveraged SciPy's non-linear optimize.minimize (SLSQP) to dynamically compute the asset allocation weights that maximize the Sharpe Ratio under strict budget and long-only constraints.

4. Numerical Methods & Performance (Numerical_methods)

Algorithms: Execution speed analysis demonstrating runtime drops via C-routine vectorization inside NumPy. Future asset price-path generator based on Geometric Brownian Motion (GBM) via stochastic discrete solution paths.

📊 Phase 2: Empirical Research & Insights

Our multi-asset simulations conducted between 2016 and 2026 across diverse asset classes (S&P 500, Bitcoin, Gold, Apple) generated crucial structural insights:

Trend vs. Mean Reversion: The Bitcoin (BTC-USD) backtest highlighted the dominance of Trend-Following models due to its historically parabolic trends, whereas standard Bollinger Bands suffered from "falling knife" syndrome without structural filters.

The Power of Conditioning: By introducing the Trend-Filtered Hybrid Strategy, we successfully neutralized false mean-reversion signals during broad market corrections on the S&P 500 (SPY) and Apple (AAPL), significantly boosting overall Sortino and Sharpe ratios.

Strategy Cross-Asset Showdown (Net of 10 bps Frictions)

![Strategy Sharpe Comparison](Advanced_backtesting/heatmap_results.png)

⚙️ Core Dependencies

Python >= 3.9

Pandas / NumPy / SciPy (Stats & Optimize)

Matplotlib / Seaborn

yfinance