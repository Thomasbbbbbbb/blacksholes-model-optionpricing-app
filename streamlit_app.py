import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Black-Scholes Option Pricing",
    page_icon="📈",
    layout="wide"
)

# ─── Black-Scholes core functions ─────────────────────────────────────────────

def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def bs_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes closed-form price for European options."""
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * norm.cdf(_d1) - K * np.exp(-r * T) * norm.cdf(_d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-_d2) - S * norm.cdf(-_d1)

# ─── Greeks ───────────────────────────────────────────────────────────────────

def greeks(S, K, T, r, sigma, option_type="call"):
    """Compute Delta, Gamma, Vega, Theta, Rho."""
    if T <= 0:
        return {"Delta": np.nan, "Gamma": np.nan, "Vega": np.nan, "Theta": np.nan, "Rho": np.nan}
    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(_d1)

    # Delta
    if option_type == "call":
        delta = norm.cdf(_d1)
    else:
        delta = norm.cdf(_d1) - 1

    # Gamma (same for call and put)
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))

    # Vega (same for call and put) — expressed per 1% move in vol
    vega = S * pdf_d1 * np.sqrt(T) / 100

    # Theta — expressed per calendar day
    if option_type == "call":
        theta = (
            - (S * pdf_d1 * sigma) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(_d2)
        ) / 365
    else:
        theta = (
            - (S * pdf_d1 * sigma) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-_d2)
        ) / 365

    # Rho — expressed per 1% move in rate
    if option_type == "call":
        rho = K * T * np.exp(-r * T) * norm.cdf(_d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-_d2) / 100

    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}

# ─── Monte Carlo pricer ───────────────────────────────────────────────────────

def mc_price(S, K, T, r, sigma, option_type="call", n_sims=50_000):
    """Monte Carlo price via GBM simulation."""
    np.random.seed(42)
    Z = np.random.standard_normal(n_sims)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    return np.exp(-r * T) * np.mean(payoffs)

# ─── Sidebar: inputs ──────────────────────────────────────────────────────────

st.sidebar.header("Option Parameters")

S     = st.sidebar.number_input("Spot Price (S)", min_value=1.0, value=100.0, step=1.0)
K     = st.sidebar.number_input("Strike Price (K)", min_value=1.0, value=100.0, step=1.0)
T     = st.sidebar.number_input("Time to Expiry (T, years)", min_value=0.01, value=1.0, step=0.05)
r     = st.sidebar.number_input("Risk-Free Rate (r, %)", min_value=0.0, value=5.0, step=0.1) / 100
sigma = st.sidebar.number_input("Volatility (σ, %)", min_value=0.1, value=20.0, step=0.5) / 100
opt   = st.sidebar.radio("Option Type", ["call", "put"])

# ─── Main layout ──────────────────────────────────────────────────────────────

st.title("📈 Black-Scholes Option Pricing Platform")
st.caption("European options | Closed-form BS · Greeks · Monte Carlo · Heatmaps")

col1, col2, col3 = st.columns(3)

price_bs = bs_price(S, K, T, r, sigma, opt)
price_mc = mc_price(S, K, T, r, sigma, opt)
g = greeks(S, K, T, r, sigma, opt)

with col1:
    st.metric("BS Price", f"${price_bs:.4f}")
with col2:
    st.metric("Monte Carlo Price (50k sims)", f"${price_mc:.4f}")
with col3:
    st.metric("MC vs BS Error", f"${abs(price_mc - price_bs):.4f}")

st.divider()

# ─── Greeks display ───────────────────────────────────────────────────────────

st.subheader("Option Greeks")

greek_cols = st.columns(5)
greek_labels = {
    "Delta": "Δ  Sensitivity to S",
    "Gamma": "Γ  Rate of Δ change",
    "Vega":  "ν  Sensitivity to σ (per 1%)",
    "Theta": "Θ  Time decay (per day)",
    "Rho":   "ρ  Sensitivity to r (per 1%)"
}
for i, (name, label) in enumerate(greek_labels.items()):
    greek_cols[i].metric(label, f"{g[name]:.5f}")

st.divider()

# ─── Plots ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Price vs Spot", "Greeks vs Spot", "Price Heatmap"])

spot_range = np.linspace(max(S * 0.5, 1), S * 1.5, 200)

with tab1:
    prices = [bs_price(s, K, T, r, sigma, opt) for s in spot_range]
    intrinsic = [max(s - K, 0) if opt == "call" else max(K - s, 0) for s in spot_range]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(spot_range, prices, color="#1F3B73", linewidth=2, label="BS Price")
    ax.plot(spot_range, intrinsic, color="#AAAAAA", linewidth=1, linestyle="--", label="Intrinsic Value")
    ax.axvline(S, color="red", linestyle=":", linewidth=1, label=f"Current S = {S}")
    ax.axvline(K, color="orange", linestyle=":", linewidth=1, label=f"Strike K = {K}")
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Option Price")
    ax.set_title(f"{opt.capitalize()} Price vs Spot")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with tab2:
    delta_vals = [greeks(s, K, T, r, sigma, opt)["Delta"] for s in spot_range]
    gamma_vals = [greeks(s, K, T, r, sigma, opt)["Gamma"] for s in spot_range]
    vega_vals  = [greeks(s, K, T, r, sigma, opt)["Vega"]  for s in spot_range]
    theta_vals = [greeks(s, K, T, r, sigma, opt)["Theta"] for s in spot_range]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    pairs = [
        (axes[0, 0], delta_vals, "Delta (Δ)", "#1F3B73"),
        (axes[0, 1], gamma_vals, "Gamma (Γ)", "#E87722"),
        (axes[1, 0], vega_vals,  "Vega (ν)",  "#2E8B57"),
        (axes[1, 1], theta_vals, "Theta (Θ)", "#CC2222"),
    ]
    for ax, vals, title, color in pairs:
        ax.plot(spot_range, vals, color=color, linewidth=2)
        ax.axvline(S, color="gray", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Spot Price")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

with tab3:
    st.markdown("**Option Price as a function of Spot × Volatility**")

    spot_grid  = np.linspace(S * 0.7, S * 1.3, 40)
    sigma_grid = np.linspace(0.05, 0.60, 40)
    Z = np.array([[bs_price(s, K, T, r, sv, opt) for s in spot_grid] for sv in sigma_grid])

    fig, ax = plt.subplots(figsize=(9, 5))
    cp = ax.contourf(spot_grid, sigma_grid * 100, Z, levels=25, cmap="YlOrRd")
    plt.colorbar(cp, ax=ax, label="Option Price ($)")
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title(f"{opt.capitalize()} Price Heatmap (Spot × Vol)")
    ax.axvline(S, color="white", linestyle="--", linewidth=1.5, label=f"S = {S}")
    ax.axhline(sigma * 100, color="cyan", linestyle="--", linewidth=1.5, label=f"σ = {sigma*100:.0f}%")
    ax.legend()
    st.pyplot(fig)

# ─── Payoff at expiry ─────────────────────────────────────────────────────────

st.divider()
st.subheader("Payoff at Expiry")

payoff = [max(s - K, 0) if opt == "call" else max(K - s, 0) for s in spot_range]
pnl    = [p - price_bs for p in payoff]

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(spot_range, pnl, color="#1F3B73", linewidth=2, label="P&L at expiry")
ax.axhline(0, color="gray", linewidth=0.8)
ax.axhline(-price_bs, color="red", linestyle="--", linewidth=1, label=f"Max loss = ${price_bs:.2f} (premium)")
ax.fill_between(spot_range, pnl, 0, where=[p > 0 for p in pnl], alpha=0.15, color="green", label="Profit zone")
ax.fill_between(spot_range, pnl, 0, where=[p < 0 for p in pnl], alpha=0.15, color="red", label="Loss zone")
ax.set_xlabel("Spot Price at Expiry")
ax.set_ylabel("P&L ($)")
ax.set_title(f"Long {opt.capitalize()} P&L at Expiry")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

# ─── Model assumptions footer ─────────────────────────────────────────────────

with st.expander("Model assumptions & methodology"):
    st.markdown("""
**Black-Scholes (1973) closed-form solution**
- Underlying follows Geometric Brownian Motion: $dS = \\mu S\\,dt + \\sigma S\\,dW_t$
- Constant volatility and risk-free rate over the option's life
- No dividends, frictionless markets, European-style exercise

**Monte Carlo simulation**
- 50,000 paths generated under the risk-neutral measure
- Terminal stock price: $S_T = S_0 \\exp\\!\\left[(r - \\tfrac{1}{2}\\sigma^2)T + \\sigma\\sqrt{T}\\,Z\\right]$, $Z \\sim \\mathcal{N}(0,1)$
- Discounted expected payoff: $e^{-rT}\\,\\mathbb{E}^\\mathbb{Q}[\\max(S_T - K,\\,0)]$

**Greeks** are computed analytically from the BS formula.  
Vega and Rho are expressed per 1% change in σ and r respectively.  
Theta is expressed per calendar day.
""")
