# Investigate - Frontier Portfolio Engine

Investigate is a robust, interactive portfolio analysis and optimization engine built for the Indian equities market (NSE). By leveraging Modern Portfolio Theory (MPT), it provides an institutional-grade toolkit to visualize efficient frontiers, compute optimal asset allocations, and perform historical backtesting.

## Features

- **Efficient Frontier Generation:** Automatically calculate and visualize the efficient frontier for any custom selection of NSE assets.
- **Dynamic Asset Universe:** Includes a built-in universe of NIFTY 50 equities, gold, and debt ETFs, with the ability to inject any valid Yahoo Finance NSE ticker instantly.
- **Optimization Presets:** Quickly select mathematically optimal portfolios including Max Sharpe, Min Volatility, and customized Balanced or Aggressive profiles.
- **Historical Backtesting:** Run rigorous historical simulations to compare the performance of optimized portfolios against the NIFTY 50 benchmark, analyzing drawdowns, CAGR, and Sharpe ratios.
- **Rebalancing Guide:** Input your current holdings to generate an actionable, priority-sorted rebalancing plan to align with your targeted allocation.

## Architecture

The application is structured into four primary layers:
1. **Data Layer (data.py):** Handles market data ingestion from Yahoo Finance with a highly resilient local caching system.
2. **Optimization Engine (optimize.py):** Uses PyPortfolioOpt to calculate covariance matrices, expected returns, and mathematically optimal frontier points.
3. **Backtesting (backtest.py):** Simulates daily rebalanced portfolio trajectories against historical price action.
4. **Presentation (app.py):** A responsive, rich Streamlit frontend providing interactive Plotly charting and dynamic state management.

## Installation

Ensure you have Python 3.10 or newer installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/investigate.git
   cd investigate
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Start the Streamlit development server:
```bash
streamlit run app.py
```
The application will be available at http://localhost:8501.

## Disclaimer

This application is intended strictly for educational and informational purposes. It does not constitute financial advice. All calculations are based on historical data, and past performance does not guarantee future results. Consult a registered financial advisor before making any investment decisions.
