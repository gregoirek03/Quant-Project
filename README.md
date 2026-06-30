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