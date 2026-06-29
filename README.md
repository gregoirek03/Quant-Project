# Quant-Project
Quantitative Finance playground: Implementations of statistical models, machine learning algorithms, and vectorized backtesting strategies.


# Project Overview
This repository documents my intensive technical preparation for upcoming Quantitative Finance gap-year internship interviews. The core objective is twofold: mastering the mathematical and statistical foundations of quantitative models, and developing a strong obsession with performance and code optimization (vectorization).

---

# My Learning Roadmap (The Story)

# Step 1: Statistical Foundations & Machine Learning
Before manipulating financial portfolios, I re-implemented core statistical bricks from scratch to master model behavior and the bias-variance trade-off.
* **[Linear Regression](./Quant_fundamentals/Linear_regression):** Implementation of Ordinary Least Squares (OLS), Ridge, and Lasso regularizations to prevent overfitting.
* **[Logistic Regression](./Quant_fundamentals/Logistic_regression):** Binary classification and signal generation (buy/sell prediction boundaries).

# Step 2: The Obsession with Performance (NumPy Vectorization)
In quantitative production pipelines, standard loops (`for`, `while`) are prohibitive. This stage was entirely dedicated to rewriting and optimizing classic math and data algorithms using pure vectorized operations:
* **[Vectorization](./Quant_fundamentals/Vectorization):** Highly efficient market data normalization, large-scale correlation matrix computations, and vectorized Gradient Descent implementation.
* **Simulations:** Introduction to stochastic processes via Monte Carlo simulations.

# Step 3: Applied Quantitative Finance & Portfolio Management
Applying the previous concepts to real-world financial time-series data:
* **[Returns](./Quant_fundamentals/Returns) & [Moving Averages](./Quant_fundamentals/Moving_average):** Logarithmic returns calculation, handling look-ahead bias, and generating momentum signals using Pandas rolling windows.
* **[Portfolio Optimization](./Quant_fundamentals/Portfolio_optimization):** Modeling Markowitz Efficient Frontier and implementing optimization algorithms to maximize the Sharpe Ratio.
* **Portfolio Optimization Preview:**
![Efficient Frontier](./Quant_fundamentals/Portfolio_optimization/portfolio.png)

---

# Step 4: Production-Grade Quantitative Projects (Next Steps)
Moving away from isolated academic exercises, I am now focusing on building comprehensive, structured quantitative frameworks inside the `Advanced_projects` directory. These upcoming modules will focus on modular architecture (`src/`), advanced risk management, and rigorous backtesting pipelines.
* **[Advanced Projects Directory](./Advanced_projects/):** *Project pipeline under construction.*

---

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Core Libraries:** 
  * **NumPy:** Vectorized linear algebra & matrix operations.
  * **Pandas:** Financial time-series analysis and data manipulation.
  * **Scikit-Learn:** Model validation and regression baseline comparisons.
  * **SciPy:** Constrained optimization algorithms.