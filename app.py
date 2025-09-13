import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Sidebar controls
st.sidebar.title("Simulation Parameters")
initial_price = st.sidebar.number_input("Initial Stock Price", value=100.0)
volatility = st.sidebar.slider("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.2)
days = st.sidebar.number_input("Number of Days", value=252)
simulations = st.sidebar.slider("Number of Simulations", min_value=1, max_value=100, value=10)
risk_free_rate = st.sidebar.slider("Risk-Free Rate (r)", min_value=0.0, max_value=0.1, value=0.01)

# Geometric Brownian Motion simulation
def simulate_gbm(S0, mu, sigma, T, N, M):
    dt = T / N
    t = np.linspace(0, T, N)
    paths = np.zeros((M, N))
    paths[:, 0] = S0
    for i in range(1, N):
        z = np.random.standard_normal(M)
        paths[:, i] = paths[:, i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return t, paths

# Run simulation
time, simulated_paths = simulate_gbm(initial_price, risk_free_rate, volatility, days, days, simulations)

# Convert to DataFrame
df = pd.DataFrame(simulated_paths.T)
df['Day'] = time
df.set_index('Day', inplace=True)

# Plotly visualization
fig = go.Figure()
for col in df.columns:
    fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=f'Sim {col+1}'))
fig.update_layout(title="Stock Price Simulation (GBM)", xaxis_title="Day", yaxis_title="Price")

# Streamlit layout
st.title("📈 Stock Simulation App")
st.plotly_chart(fig, use_container_width=True)

# Download CSV
csv = df.to_csv().encode('utf-8')
st.download_button("Download Simulation Data", data=csv, file_name="stock_simulation.csv", mime="text/csv")
