import streamlit as st
import pandas as pd
import numpy as np
from portfolio_risk import fetch_price_history, compute_returns, portfolio_metrics, risk_profile, suggest_rebalance, optimize_portfolio

st.set_page_config(page_title='AI Investment Portfolio Risk Manager', layout='wide')

st.title('AI Investment Portfolio Risk Manager')
st.write('Analyze and manage portfolio risk with data-driven metrics and AI-enabled guidance.')

with st.sidebar:
    st.header('Portfolio Input')
    tickers_input = st.text_area('Tickers (comma-separated)', 'AAPL, MSFT, GOOG, TSLA')
    weights_input = st.text_area('Weights (comma-separated, sum to 1)', '0.25,0.25,0.25,0.25')
    lookback_period = st.selectbox('Price history period', ['1y', '2y', '5y'])
    submit = st.button('Analyze Portfolio')

if submit:
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    weights = [float(w.strip()) for w in weights_input.split(',') if w.strip()]

    if len(tickers) == 0 or len(weights) == 0 or len(tickers) != len(weights):
        st.error('Number of tickers and weights must match and be > 0.')
    else:
        try:
            prices = fetch_price_history(tickers, period=lookback_period)
            st.subheader('Price History (Close)')
            st.line_chart(prices)

            returns = compute_returns(prices)
            metrics = portfolio_metrics(returns, weights)
            profile = risk_profile(metrics)

            col1, col2, col3 = st.columns(3)
            col1.metric('Expected Annual Return', f"{metrics['expected_annual_return']*100:.2f}%")
            col2.metric('Annual Volatility', f"{metrics['annual_volatility']*100:.2f}%")
            col3.metric('Sharpe Ratio', f"{metrics['sharpe_ratio']:.2f}")

            col1.metric('VaR 95', f"{metrics['var_95']*100:.2f}%")
            col2.metric('CVaR 95', f"{metrics['cvar_95']*100:.2f}%")
            col3.metric('Risk Profile', profile['risk_level'])

            st.subheader('Holdings & Weights')
            st.dataframe(pd.DataFrame({'Ticker': tickers, 'Weight': weights}))

            st.subheader('Risk Insights')
            st.write(profile['notes'])

            st.subheader('AI Rebalance Suggestion')
            st.write(suggest_rebalance(tickers, weights, metrics))

            st.subheader('Optimized Portfolio')
            try:
                optimized_weights = optimize_portfolio(returns, target='min_variance')
                opt_df = pd.DataFrame({'Ticker': tickers, 'Current Weight': weights, 'Min Variance Weight': optimized_weights})
                st.dataframe(opt_df)
            except Exception as e:
                st.warning(f'Optimization unavailable: {e}')

            st.subheader('Stress Test: Flexible Shock Analysis')
            shock_pct = st.slider('Uniform downside shock (%)', 1, 30, 10)
            shocked_returns = returns * (1 - shock_pct / 100)
            shocked_metrics = portfolio_metrics(shocked_returns, weights)
            st.write(f"Under {shock_pct}% shock: Volatility {shocked_metrics['annual_volatility']*100:.2f}%, CVaR {shocked_metrics['cvar_95']*100:.2f}%")

            st.warning('Disclaimer: This tool is educational and not financial advice. Invest at your own risk.')

            st.markdown('---')
            st.markdown('## GitHub Repository')
            st.markdown('[📦 View code on GitHub](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME)')
        except Exception as e:
            st.error(f'Error: {e}')
