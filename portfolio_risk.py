import numpy as np
import pandas as pd
import yfinance as yf


def fetch_price_history(tickers, period='1y', interval='1d'):
    data = yf.download(tickers, period=period, interval=interval, progress=False)
    if data.empty:
        raise ValueError('No data fetched. Check ticker symbols and internet connection.')
    if isinstance(tickers, str):
        prices = data['Close'].to_frame(tickers)
    else:
        prices = data['Close']
    prices.dropna(how='all', inplace=True)
    return prices


def compute_returns(prices):
    returns = prices.pct_change().dropna()
    return returns


def portfolio_metrics(returns, weights, risk_free_rate=0.0):
    weights = np.array(weights, dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        weights = weights / weights.sum()

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    portfolio_return = float(np.dot(mean_returns, weights))
    portfolio_volatility = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))

    daily_port_return = returns.dot(weights)
    sharpe_ratio = float((portfolio_return - risk_free_rate) / portfolio_volatility) if portfolio_volatility > 0 else np.nan

    var_95 = -np.percentile(daily_port_return, 5) * np.sqrt(252)
    cvar_95 = -daily_port_return[daily_port_return <= np.percentile(daily_port_return, 5)].mean() * np.sqrt(252)

    return {
        'expected_annual_return': portfolio_return,
        'annual_volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'weights': weights,
    }


def risk_profile(metrics):
    vol = metrics['annual_volatility']
    sharpe = metrics['sharpe_ratio']
    if vol < 0.1 and sharpe > 1:
        level = 'Conservative'
    elif vol < 0.18 and sharpe > 0.8:
        level = 'Balanced'
    else:
        level = 'Aggressive'

    return {
        'risk_level': level,
        'notes': (
            'Diversification is good, continue monitoring. ' if level == 'Balanced' else
            'Consider trimming high-volatility positions and adding bonds/cash equivalents.' if level == 'Aggressive' else
            'Your portfolio is low volatility; you may accept lower return in exchange for stability.'
        )
    }


def optimize_portfolio(returns, target='min_variance'):
    import scipy.optimize as sco

    n = returns.shape[1]
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    def portfolio_stats(weights):
        weights = np.array(weights)
        port_return = np.dot(mean_returns, weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return port_return, port_vol

    def min_variance(weights):
        return portfolio_stats(weights)[1]

    def neg_sharpe(weights):
        ret, vol = portfolio_stats(weights)
        return -(ret / vol) if vol > 0 else 1e6

    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    init = np.ones(n) / n

    if target == 'max_sharpe':
        optimization = sco.minimize(neg_sharpe, init, method='SLSQP', bounds=bounds, constraints=cons)
    else:
        optimization = sco.minimize(min_variance, init, method='SLSQP', bounds=bounds, constraints=cons)

    if not optimization.success:
        raise RuntimeError('Optimization failed: ' + optimization.message)

    weights_opt = optimization.x
    return weights_opt


def suggest_rebalance(tickers, weights, metrics):
    # Simple suggestion: reduce allocation of asset with largest weight when risk is high
    highest_weight_index = int(np.argmax(weights))
    highest_ticker = tickers[highest_weight_index]
    highest_percent = weights[highest_weight_index] * 100

    advice = []
    if metrics['annual_volatility'] > 0.18:
        advice.append('Your portfolio volatility is elevated; consider shifting to lower-volatility assets.')
    if metrics['cvar_95'] > 0.2:
        advice.append('Expected tail risk (CVaR) is high; consider adding diversification or hedges.')

    if not advice:
        advice.append('Risk metrics are reasonable but continue to monitor regularly.')

    suggestion = (
        f"Top allocation is {highest_ticker} ({highest_percent:.1f}%). "
        "Consider trimming large positions and rebalancing to target weights. "
        + ' '.join(advice)
    )
    return suggestion
