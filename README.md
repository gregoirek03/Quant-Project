# 📈 Quant Finance Laboratory

Welcome to my quantitative finance research repository. This project is divided into two distinct development phases: a foundational research hub to master core mechanics, followed by production-grade financial engineering applications.

The primary objective of this **`Quant_fundamentals`** section is to deeply assimilate the mathematical and statistical foundations of quantitative asset management. To ensure a thorough, low-level understanding of numerical optimization and statistical mechanics, every single formula, algorithm, and objective function has been coded entirely from scratch using Python, avoiding any pre-existing high-level machine learning libraries (such as Scikit-Learn).

Every directory follows a production-grade layout: a standalone core engineering file (`.py`) containing pure vectorized functions, paired with a comprehensive research notebook (`.ipynb`) acting as a visual validation lab.

---

## 🚀 Project Roadmap: From Theory to Production

To build institutional-grade trading systems, one must first master the underlying mathematics. This repository mirrors that exact professional progression:

1. **Phase 1: Quant Fundamentals (Current Hub):** Deep-dive into raw implementations (Optimization, SDEs, Matrix Regressions). This acts as our proprietary math library.
2. **Phase 2: Advanced Projects (Next Hub):** Leveraging our fundamental blocks to build large-scale systems, including **Vectorized Backtesting Engines** (SMA/RSI strategies with real-world data), **Statistical Arbitrage (Pairs Trading)** frameworks, and alternative data pipelines.

---

## 🗂️ Repository Architecture

```text
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

```

---

## 🧠 Technical Overview of Modules

---

### 1. Linear Regression & Regularization (`Linear_regression`)

* **Concepts:** Mathematical implementation of Ordinary Least Squares (OLS) via the Normal Equation, Ridge Regression ($L_2$ regularization), and LASSO ($L_1$ regularization).
* **Algorithms:** Designed and built a custom **Coordinate Descent** engine from scratch. It integrates a **Soft-Thresholding** operator for the LASSO model to perform sparse feature selection, effectively zeroing out irrelevant coefficients to isolate true alpha signals from structural market noise.

| Model | Key Mathematical Formula | Regularization |
| :--- | :--- | :--- |
| **OLS** | $$\min_{\beta}  \|y - X\beta\|_2^2$$ | None |
| **Ridge** | $$\min_{\beta}  \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2$$ | $L_2$ (Coefficients Shrinkage) |
| **LASSO** | $$\min_{\beta}  \|y - X\beta\|_2^2 + \lambda \|\beta\|_1$$ | $L_1$ (Feature Selection) |

---

### 2. Logistic Regression & Classification (`Logistic_regression`)

* **Concepts:** Probabilistic binary modeling optimized for quantitative trend forecasting (Up/Down market direction) and conditional credit default risk estimation.
* **Algorithms:** Vectorized Gradient Descent minimizing the *Cross-Entropy* loss function (Log Loss). Model performance is rigorously evaluated through a manual, raw-matrix implementation of a **Confusion Matrix** (tracking True/False Positives & Negatives) alongside custom accuracy metrics.

| Performance Metric | Mathematical Formula | Target Objective |
| :--- | :--- | :--- |
| **Log Loss** | $$-\frac{1}{n}\sum_{i=1}^n \left[ y_i\ln(\hat{y}_i) + (1-y_i)\ln(1-\hat{y}_i) \right]$$ | Minimize Error |
| **Accuracy** | $$\frac{TP + TN}{TP + TN + FP + FN}$$ | Maximize Correct Hits |

---

### 3. Portfolio Optimization (`Portfolio_optimization`)

* **Concepts:** Operational and tactical application of Harry Markowitz's Modern Portfolio Theory (MPT).
* **Algorithms:** Expressed portfolio volatility via a vectorized quadratic form ($W^T \Sigma W$). Leveraged SciPy's non-linear `optimize.minimize` (using the SLSQP engine) to dynamically compute the asset allocation weights that maximize the **Sharpe Ratio** under strict constraints.

| Optimization Parameter | Mathematical Formulation | Operational Boundary |
| :--- | :--- | :--- |
| **Portfolio Volatility** | $$\sigma_p = \sqrt{W^T \Sigma W}$$ | Risk Matrix |
| **Sharpe Ratio (Max)** | $$\text{Sharpe} = \frac{W^T R - R_f}{\sqrt{W^T \Sigma W}}$$ | Target Objective ($R_f = 0$) |
| **Budget Constraint** | $$\sum_{i=1}^N w_i = 1.0$$ | Fully Invested |
| **Long-Only Bounds** | $$0.0 \le w_i \le 1.0$$ | No Short-Selling |

---

### 4. Numerical Methods & Performance (`Numerical_methods`)

* **Concepts:** High-performance computational processing of financial time-series data and robust stochastic simulation engines.
* **Algorithms:**
    * **Vectorization Benchmark:** Execution speed analysis demonstrating the massive runtime drop achieved by replacing sequential Python `for` loops with vectorized, pre-compiled C-routines built inside NumPy and Pandas.
    * **Monte Carlo Simulation:** Future asset price-path generator based on **Geometric Brownian Motion (GBM)** optimized via matrix exponentiation and cumulative products.
    * **Returns Engine:** Multi-dimensional analytical functions computing arithmetic and logarithmic returns, capitalizing on time-additivity properties for asset modeling.

| Stochastic / Time-Series Engine | Stochastic Differential Equation (SDE) & Analytical Solution |
| :--- | :--- |
| **Geometric Brownian Motion (SDE)** | $$dS_t = \mu S_t dt + \sigma S_t dW_t$$ |
| **GBM Discrete Solution (Monte Carlo)** | $$S_{t+\Delta t} = S_t \exp\left( \left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}Z \right), \quad Z \sim \mathcal{N}(0,1)$$ |
| **Logarithmic Returns** | $$r_t = \ln(P_t) - \ln(P_{t-1})$$ |