# Black-Scholes Option Pricing Platform

Interactive platform for pricing European options using the Black-Scholes model.

## Features
- Closed-form BS pricing with real-time parameter adjustment
- Monte Carlo simulation (50,000 paths) with BS comparison
- Full Greeks dashboard — Delta, Gamma, Vega, Theta, Rho (analytical)
- Price heatmap across Spot × Implied Volatility
- P&L at expiry with profit/loss zones

## Run locally
pip install -r requirements.txt
streamlit run streamlit_app.py
