import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from numpy import log, sqrt, exp
import matplotlib.pyplot as plt
import seaborn as sns

#######################
# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
main.block-container {
    max-width: 1200px;
}

/* Conteneur CALL / PUT */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 14px;
    width: 100%;
    border-radius: 12px;
    font-family: 'Segoe UI', sans-serif;
    color: white;
    font-weight: 600;
    font-size: 1.4rem;
    box-shadow: 0 3px 6px rgba(0,0,0,0.15);
}

/* CALL = vert */
.metric-call {
    background-color: #4CAF50;
}

/* PUT = rouge */
.metric-put {
    background-color: #E53935;
}

.metric-label {
    font-size: 0.9rem;
    opacity: 0.9;
    color: white;
}

.metric-value {
    font-size: 1.9rem;
    font-weight: 800;
    margin-top: 4px;
    color: white;
}
</style>
""", unsafe_allow_html=True)


#############################################
# Black-Scholes Class
#############################################

class BlackScholes:
    def __init__(self, time_to_maturity, strike, current_price, volatility, interest_rate):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def _compute_d1_d2(self):
        S = self.current_price
        K = self.strike
        T = self.time_to_maturity
        sigma = self.volatility
        r = self.interest_rate

        d1 = (log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        return d1, d2

    def calculate_prices(self):
        S = self.current_price
        K = self.strike
        T = self.time_to_maturity
        sigma = self.volatility
        r = self.interest_rate

        if T <= 0 or sigma <= 0:
            self.call_price = max(S-K, 0)
            self.put_price = max(K-S, 0)
            return self.call_price, self.put_price

        d1, d2 = self._compute_d1_d2()

        # Prix
        self.call_price = S * norm.cdf(d1) - K * exp(-r*T) * norm.cdf(d2)
        self.put_price  = K * exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        # Greeks
        self.call_delta = norm.cdf(d1)
        self.put_delta  = norm.cdf(d1) - 1  # formule correcte

        self.call_gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
        self.put_gamma  = self.call_gamma

        self.vega = S * norm.pdf(d1) * sqrt(T)

        self.call_theta = -(S * norm.pdf(d1) * sigma) / (2*sqrt(T)) - r*K*exp(-r*T)*norm.cdf(d2)
        self.put_theta  = -(S * norm.pdf(d1) * sigma) / (2*sqrt(T)) + r*K*exp(-r*T)*norm.cdf(-d2)

        return self.call_price, self.put_price


#############################################
# Heatmap function
#############################################

def plot_heatmap(bs_model, spot_range, vol_range, strike):
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))

    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            model = BlackScholes(bs_model.time_to_maturity, strike, spot, vol, bs_model.interest_rate)
            c, p = model.calculate_prices()
            call_prices[i][j] = c
            put_prices[i][j]  = p

    fig_call, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(call_prices, annot=True, cmap="viridis",
                xticklabels=np.round(spot_range,2), yticklabels=np.round(vol_range,2),
                fmt=".2f", ax=ax)
    ax.set_title("CALL Price Heatmap")
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Volatility")

    fig_put, ax2 = plt.subplots(figsize=(10,8))
    sns.heatmap(put_prices, annot=True, cmap="viridis",
                xticklabels=np.round(spot_range,2), yticklabels=np.round(vol_range,2),
                fmt=".2f", ax=ax2)
    ax2.set_title("PUT Price Heatmap")
    ax2.set_xlabel("Spot Price")
    ax2.set_ylabel("Volatility")

    return fig_call, fig_put



#############################################
# Sidebar
#############################################

with st.sidebar:
    st.header("Model Parameters")

    current_price = st.number_input("Current Asset Price", value=100.0)
    strike = st.number_input("Strike Price", value=100.0)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0)
    volatility = st.number_input("Volatility (σ)", value=0.2)
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05)

    st.markdown("---")
    st.subheader("Heatmap Settings")

    spot_min = st.number_input("Min Spot Price", value=current_price*0.8, min_value=0.01)
    spot_max = st.number_input("Max Spot Price", value=current_price*1.2, min_value=0.01)

    vol_min = st.slider("Min Vol (heatmap)", 0.01, 1.0, volatility*0.5)
    vol_max = st.slider("Max Vol (heatmap)", 0.01, 1.0, volatility*1.5)

    spot_range = np.linspace(spot_min, spot_max, 12)
    vol_range = np.linspace(vol_min, vol_max, 12)


#############################################
# Main Page
#############################################

st.title("Black-Scholes Option Pricing Model")
st.markdown("Compute option prices, Greeks, payoffs and heatmaps.")

# Compute
bs = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
call_price, put_price = bs.calculate_prices()

#############################################
# Input summary
#############################################

st.subheader("Input Summary")
st.table(pd.DataFrame({
    "Current Price":[current_price],
    "Strike":[strike],
    "Maturity":[time_to_maturity],
    "Volatility":[volatility],
    "Rate":[interest_rate]
}))


#############################################
# CALL & PUT Buttons
#############################################

st.subheader("Option Prices")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div style="text-align:center;">
                <div class="metric-label">CALL Price</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div style="text-align:center;">
                <div class="metric-label">PUT Price</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


#############################################
# Greeks table
#############################################

st.subheader("Greeks")

st.table(pd.DataFrame({
    "Greek":["Delta","Gamma","Vega","Theta"],
    "Call":[bs.call_delta, bs.call_gamma, bs.vega, bs.call_theta],
    "Put":[bs.put_delta, bs.put_gamma, bs.vega, bs.put_theta],
}))


#############################################
# Payoff plot
#############################################

st.subheader("Payoff at Maturity")

S_range = np.linspace(0.5*strike, 1.5*strike, 200)
call_payoff = np.maximum(S_range - strike, 0)
put_payoff  = np.maximum(strike - S_range, 0)

fig, ax = plt.subplots()
ax.plot(S_range, call_payoff, label="Call Payoff")
ax.plot(S_range, put_payoff,  label="Put Payoff")
ax.axvline(strike, linestyle="--")
ax.set_xlabel("Underlying Price")
ax.set_ylabel("Payoff")
ax.legend()

st.pyplot(fig)


#############################################
# Price vs spot
#############################################

st.subheader("Option Price vs Spot")

spot_grid = np.linspace(spot_min, spot_max, 80)
call_curve, put_curve = [], []

for s in spot_grid:
    m = BlackScholes(time_to_maturity, strike, s, volatility, interest_rate)
    c, p = m.calculate_prices()
    call_curve.append(c)
    put_curve.append(p)

fig2, ax2 = plt.subplots()
ax2.plot(spot_grid, call_curve, label="Call")
ax2.plot(spot_grid, put_curve, label="Put")
ax2.legend()
ax2.set_xlabel("Spot Price")
ax2.set_ylabel("Option Price")

st.pyplot(fig2)


#############################################
# Heatmaps
#############################################

st.subheader("Heatmaps")

colh1, colh2 = st.columns(2)

with colh1:
    fig_call, _ = plot_heatmap(bs, spot_range, vol_range, strike)
    st.pyplot(fig_call)

with colh2:
    _, fig_put = plot_heatmap(bs, spot_range, vol_range, strike)
    st.pyplot(fig_put)
