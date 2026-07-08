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

- **Concepts:** Event-free vectorized backtesting architecture modeling institutional market frictions, asset class regime variations, and conditional strategy combinations.
- **Risk Architecture:** Quantifies tail-risk dynamics by extracting higher-moment statistical distributions to isolate empirical *Fat Tails* and asymmetrical drawdowns.
- **Strategies Implemented:** Fast/Slow SMA Crossover (Trend Following), Bollinger Bands (Mean Reversion), and a **Trend-Filtered Hybrid Framework** (triggers mean-reversion asset loading *only* under verified long-term macroeconomic uptrends).

| Strategy Model | Signal Generation Logic | Core Tactical Target |
| :--- | :--- | :--- |
| **SMA Crossover** | $\text{Signal} = 1 \text{ if } \text{SMA}_{20} > \text{SMA}_{50} \text{ else } 0$ | Structural Trend Following |
| **Bollinger Bands** | $\text{Buy if } P_t < \mu_t - 2\sigma_t \text{; Close if } P_t > \mu_t$ | Statistical Volatility Mean Reversion |
| **Trend-Filtered Hybrid** | $\text{Buy if } (P_t < \mu_t - 2\sigma_t) \land (\text{SMA}_{20} > \text{SMA}_{50})$ | Conditional Regime-Filtered Alpha |

| Risk Performance Metric | Mathematical Formulation | Target Portfolio Objective |
| :--- | :--- | :--- |
| **Downside Deviation** | $\sigma_d = \sqrt{\frac{252}{N} \sum_{t=1}^{N} \min(r_t, 0)^2}$ | Isolate Downside-Only Dispersion |
| **Sortino Ratio** | $\text{Sortino} = \frac{R_a - R_f}{\sigma_d}$ | Maximize Return per Unit of Bad Risk |
| **Skewness ($3^{\text{rd}}$ Moment)** | $S = \mathbb{E}\left[\left(\frac{X-\mu}{\sigma}\right)^3\right]$ | Measure Return Distribution Asymmetry |
| **Kurtosis ($4^{\text{th}}$ Moment)** | $K = \mathbb{E}\left[\left(\frac{X-\mu}{\sigma}\right)^4\right]$ | Quantify Extreme Tail Risk Exposure (*Fat Tails*) |

---

### 2. Linear Regression & Regularization (`Linear_regression`)

- **Concepts:** Mathematical implementation of Ordinary Least Squares (OLS) via the Normal Equation, Ridge Regression ($L_2$ regularization), and LASSO ($L_1$ regularization).
- **Algorithms:** Designed and built a custom Coordinate Descent engine from scratch. It integrates a Soft-Thresholding operator for the LASSO model to perform sparse feature selection, effectively zeroing out irrelevant coefficients to isolate true alpha signals from structural market noise.

| Model | Key Mathematical Formula | Regularization Type & Effect |
| :--- | :--- | :--- |
| **OLS** | $\min_{\beta} \|y - X\beta\|_2^2$ | None (Baseline Model) |
| **Ridge** | $\min_{\beta} \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2$ | $L_2$ Penalty (Coefficients Shrinkage) |
| **LASSO** | $\min_{\beta} \|y - X\beta\|_2^2 + \lambda \|\beta\|_1$ | $L_1$ Penalty (Sparse Feature Selection) |

---

### 3. Logistic Regression & Classification (`Logistic_regression`)

- **Concepts:** Probabilistic binary modeling optimized for quantitative trend forecasting (Up/Down market direction) and conditional credit default risk estimation.
- **Algorithms:** Vectorized Gradient Descent minimizing the Cross-Entropy loss function (Log Loss). Model performance is rigorously evaluated through a manual, raw-matrix implementation of a Confusion Matrix (tracking True/False Positives & Negatives) alongside custom accuracy metrics.

| Performance Metric | Mathematical Formula | Target Objective |
| :--- | :--- | :--- |
| **Log Loss** | $-\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right]$ | Minimize Classification Error |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Maximize Correct Directional Hits |

---

### 4. Portfolio Optimization (`Portfolio_optimization`)

- **Concepts:** Operational and tactical application of Harry Markowitz's Modern Portfolio Theory (MPT).
- **Algorithms:** Expressed portfolio volatility via a vectorized quadratic form ($W^T\Sigma W$). Leveraged SciPy's non-linear `optimize.minimize` (using the SLSQP engine) to dynamically compute the asset allocation weights that maximize the Sharpe Ratio under strict constraints.

| Optimization Parameter | Mathematical Formulation | Operational Boundary |
| :--- | :--- | :--- |
| **Portfolio Volatility** | $\sigma_p = \sqrt{W^T\Sigma W}$ | Covariance Matrix Variance Minimization |
| **Sharpe Ratio (Max)** | $\text{Sharpe} = \frac{W^T R - R_f}{\sqrt{W^T\Sigma W}}$ | Multi-Asset Target Objective ($R_f = 0$) |
| **Budget Constraint** | $\sum_{i=1}^{N} w_i = 1.0$ | Fully Invested Portfolio Capital |
| **Long-Only Bounds** | $0.0 \le w_i \le 1.0$ | Short-Selling Restrictions Enforced |

---

### 5. Numerical Methods & Performance (`Numerical_methods`)

- **Concepts:** High-performance computational processing of financial time-series data and robust stochastic simulation engines.
- **Algorithms:**
  - **Vectorization Benchmark:** Execution speed analysis demonstrating the massive runtime drop achieved by replacing sequential Python `for` loops with vectorized, pre-compiled C-routines built inside NumPy and Pandas.
  - **Monte Carlo Simulation:** Future asset price-path generator based on Geometric Brownian Motion (GBM) optimized via matrix exponentiation and cumulative products.

| Stochastic / Time-Series Engine | Stochastic Differential Equation (SDE) & Discrete Solution |
| :--- | :--- |
| **Geometric Brownian Motion (SDE)** | $dS_t = \mu S_t dt + \sigma S_t dW_t$ |
| **GBM Discrete Path (Monte Carlo)** | $S_{t+\Delta t} = S_t \exp\left( \left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}Z \right), \quad Z \sim \mathcal{N}(0,1)$ |
| **Logarithmic Returns Engine** | $r_t = \ln(P_t) - \ln(P_t-1)$ |

---

## 📊 Phase 2: Empirical Research & Insights

Our multi-asset simulations conducted between 2016 and 2026 across diverse asset classes generated crucial structural insights regarding market microstructures:

- **Trend vs. Mean Reversion:** The **Bitcoin (BTC-USD)** backtest highlighted the dominance of Trend-Following models due to its historically parabolic trends. Conversely, standard Bollinger Bands suffered from severe "falling knife" drawdowns when trying to catch bottoms without structural trend confirmations.
- **The Power of Conditioning:** By introducing the **Trend-Filtered Hybrid Strategy**, we successfully neutralized false counter-trend buy signals during macro corrections on the **S&P 500 (SPY)** and **Apple (AAPL)**. This behavioral conditioning significantly boosted both Sortino and Sharpe metrics while stabilizing the equity curve.

### Strategy Cross-Asset Showdown (Net of 10 bps Frictions)
![Strategy Sharpe Comparison](Advanced_backtesting/heatmap_results.png)

---

## ⚙️ Core Dependencies

- **Runtime Environment:** Python $\ge$ 3.9
- **Data Engineering:** `Pandas` | `NumPy`
- **Scientific Computing & Analytics:** `SciPy` (`Stats` & `Optimize`)
- **Data Visualization:** `Matplotlib` | `Seaborn`
- **Market Data Ingestion:** `yfinance`