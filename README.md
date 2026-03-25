# AI Investment Portfolio Risk Manager

A Streamlit app for analyzing portfolio risk metrics and generating AI-guided rebalancing advice.

## Features

- Fetches historical prices via `yfinance`
- Computes returns, volatility, Sharpe ratio, VaR, CVaR
- Classifies risk profile (Conservative / Balanced / Aggressive)
- Suggests rebalancing actions
- Stress test for downside shocks

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## GitHub

- Add this repo to GitHub (e.g., `git init`, `git add .`, `git commit -m "initial"`, `git remote add origin ...`, `git push -u origin main`).
- Include this project README and code in your GitHub repository.

## Disclaimer

Not financial advice. Educational tool only.
