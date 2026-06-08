# SABR Stochastic Volatility Pricing Engine

A production-grade, object-oriented Python library designed to calibrate the SABR stochastic volatility model on live market data and price path-dependent options using a vectorized Monte Carlo simulation engine.

## Project Architecture
- `sabr_pricing/`
  - `market_data.py`: Automated yfinance retrieval and implied volatility extraction via a custom Newton-Raphson solver.
  - `calibrator.py`: Objective function optimization using Hagan's analytical approximation and SciPy's `L-BFGS-B` algorithm. Returns parameters via a structured `SABRParams` dataclass.
  - `simulator.py`: 100% vectorized path generation using an Euler-Maruyama discretization scheme with Itô's lemma log-normal correction.
  - `pricer.py`: Evaluation of Asian/Vanilla Call options and finite-difference calculation of market Greeks ($\Delta$, $\Gamma$, $\mathcal{V}$) using Common Random Numbers (CRN) for variance reduction.
- `demo_sabr.ipynb`: Comprehensive Jupyter Notebook walking through the mathematical derivations, calibration plots, and model sanity checks.

## Setup & Execution
1. Install dependencies: `pip install -r requirements.txt`
2. Run the interactive notebook: `jupyter notebook demo_sabr.ipynb`
