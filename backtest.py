import yfinance as yf
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas_ta as ta

# --- THIS IS THE FIX ---
# A helper function that converts the data back to a pandas Series
# before calling the indicator.
def SMA_func(data, period):
    """
    Converts data to pd.Series and calculates SMA.
    'data' will be a NumPy array passed by the backtesting library.
    """
    return pd.Series(data).rolling(period).mean()
# --- END OF FIX ---

def get_data(ticker):
    """Fetches and prepares data for backtesting."""
    df = yf.download(ticker, start='2020-01-01', end='2025-10-30')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # We DON'T need to calculate SMAs here anymore,
    # the strategy will do it.
    df = df.dropna()
    return df

class SmaCross(Strategy):
    """
    A simple Moving Average Crossover strategy.
    """
    n1 = 20 # Fast SMA
    n2 = 50 # Slow SMA
    
    def init(self):
        # --- THIS IS THE FIX ---
        # We call our new helper function instead of ta.sma directly.
        self.sma1 = self.I(SMA_func, self.data.Close, self.n1)
        self.sma2 = self.I(SMA_func, self.data.Close, self.n2)
        # --- END OF FIX ---

    def next(self):
        # If the fast SMA crosses above the slow SMA, buy
        if crossover(self.sma1, self.sma2):
            self.buy()
        # If the fast SMA crosses below the slow SMA, sell
        elif crossover(self.sma2, self.sma1):
            self.sell()

def run_backtest(ticker):
    print(f"Running backtest for {ticker}...")
    data = get_data(ticker)
    
    # Configure the backtest
    bt = Backtest(data, SmaCross,
                  cash=100000, # Starting with $100,000
                  commission=.002, # 0.2% commission
                  exclusive_orders=True)

    # Run the backtest
    stats = bt.run()
    
    print("Backtest complete. Results:")
    print(stats)
    
    # Save the report to an HTML file
    report_filename = "backtest_report.html"
    bt.plot(filename=report_filename, open_browser=False)
    print(f"Report saved to {report_filename}")

if __name__ == "__main__":
    # You can change this ticker to backtest different assets
    run_backtest(ticker="^NSEBANK")