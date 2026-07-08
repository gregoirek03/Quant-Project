# 📈 Quant Finance Laboratory

Welcome to my quantitative finance research repository. This project bridges pure academic foundations with production-grade financial engineering applications, structured across modular development phases.

The primary objective of this laboratory is to deeply assimilate the mathematical and statistical foundations of quantitative asset management, moving from raw numerical optimization to robust multi-asset framework execution.

Every directory follows a production-grade layout: a standalone core engineering file (`.py`) containing pure vectorized functions, paired with a comprehensive research notebook (`.ipynb`) acting as a visual validation lab.

---

## 🚀 Project Roadmap: From Theory to Production

To build institutional-grade trading systems, one must first master the underlying mathematics. This repository mirrors that exact professional progression:

- **Phase 1: Quant Fundamentals (Proprietary Math Hub):** Deep-dive into raw implementations from scratch (Optimization, SDEs, Matrix Regressions) to avoid pre-existing black-box libraries.
- **Phase 2: Advanced Production Frameworks (Current Focus):** Leveraging fundamental building blocks to deploy large-scale vectorized backtesting engines, risk-management systems, and hybrid cross-asset strategy evaluations.

---

## 🗂️ Repository Architecture

```text
Quant_Project/
├── Quant_fundamentals/
│   ├── Linear_regression/        # Continuous modeling & regularizations (L1/L2)
│   │   ├── linear_models.py
│   │   └── benchmark_linear.ipynb
│   ├── Logistic_regression/      # Binary classification & custom optimization
│   │   ├── logistic_models.py
│   │   └── benchmark_logistic.ipynb
│   ├── Portfolio_optimization/   # Modern Portfolio Theory (MPT) & Sharpe Max
│   │   ├── portfolio_optimizer.py
│   │   └── benchmark_portfolio.ipynb
│   └── Numerical_methods/        # Time-series vectorization & stochastic models
│       ├── numerical_tools.py
│       └── benchmark_numerical.ipynb
│
└── Advanced_backtesting/         # 🟢 Phase 2: Production Vectorized Engine
    ├── backtest_engine.py        # Core computational engine (Higher-moment risks)
    ├── benchmark_backtest.ipynb  # Empirical Research & Cross-Asset Lab
    └── heatmap_results.png       # Generated performance visualizer

## 🧠 Technical Overview of Modules

### 1. Advanced Backtesting Framework (`Advanced_backtesting`)

- **Concepts:** Event-free vectorized execution modeling institutional market frictions, asset class regime variations, and conditional strategy filtering.
- **Risk Architecture:** Moves beyond traditional returns to compute higher-moment statistics (`SciPy Stats`) including **Sortino Ratio, Downside Deviation, Skewness, and Kurtosis** to detect empirical *Fat Tails*.
- **Strategies Implemented:** Fast/Slow SMA Crossover, Bollinger Bands Mean Reversion, and a **Trend-Filtered Hybrid Framework** (triggers mean-reversion buying *only* during verified long-term macroeconomic uptrends).

### 2. Linear Regression & Regularization (`Linear_regression`)

- **Algorithms:** Designed and built a custom Coordinate Descent engine from scratch. It integrates a Soft-Thresholding operator for the **LASSO** model to perform sparse feature selection, effectively zeroing out irrelevant coefficients to isolate true alpha signals from structural market noise.

### 3. Portfolio Optimization (`Portfolio_optimization`)

- **Algorithms:** Expressed portfolio volatility via a vectorized quadratic form ($W^T\Sigma W$). Leveraged SciPy's non-linear `optimize.minimize` (using the **SLSQP** engine) to dynamically compute the asset allocation weights that maximize the Sharpe Ratio under strict budget and long-only constraints.

### 4. Numerical Methods & Performance (`Numerical_methods`)

- **Algorithms:** Execution speed analysis demonstrating the massive runtime drop achieved by replacing sequential Python for loops with vectorized, pre-compiled C-routines built inside **NumPy** and **Pandas**. Future asset price-path generator based on **Geometric Brownian Motion (GBM)** optimized via matrix exponentiation and cumulative products.

---

## 📊 Phase 2: Empirical Research & Insights

Our multi-asset simulations conducted between 2016 and 2026 across diverse asset classes (S&P 500, Bitcoin, Gold, Apple) generated crucial structural insights regarding market microstructures:

- **Trend vs. Mean Reversion:** The **Bitcoin (BTC-USD)** backtest highlighted the dominance of Trend-Following models due to its historically parabolic trends, whereas standard Bollinger Bands suffered from a severe "falling knife" syndrome without structural filters.
- **The Power of Conditioning:** By introducing the **Trend-Filtered Hybrid Strategy**, we successfully neutralized false mean-reversion signals during broad market corrections on the **S&P 500 (SPY)** and **Apple (AAPL)**, significantly boosting overall Sortino and Sharpe ratios while stabilizing the equity curve.

### Strategy Cross-Asset Showdown (Net of 10 bps Frictions)
![Strategy Sharpe Comparison](Advanced_backtesting/heatmap_results.png)

---

## ⚙️ Core Dependencies

- **Python** >= 3.9
- **Pandas** / **NumPy** / **SciPy** (`Stats` & `Optimize`)
- **Matplotlib** / **Seaborn**
- **yfinance**