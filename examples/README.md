# Kalshi Latency Bot Examples

This directory contains example scripts demonstrating how to use the Kalshi Latency Bot.

## Examples

### backtest_example.py

Demonstrates how to use the backtesting framework:

```bash
cd examples
python backtest_example.py
```

This script:
- Generates synthetic trading signals
- Runs a backtest simulation
- Displays performance metrics (win rate, Sharpe ratio, max drawdown)
- Shows example trades

## Creating Your Own Examples

You can use these examples as templates for your own trading strategies and backtests.

### Custom Backtest

```python
from kalshi_bot.backtester import Backtester
from kalshi_bot.models import TradingSignal, OrderSide

# Load your historical data
signals = load_your_signals()
outcomes = load_your_outcomes()

# Create backtester
backtester = Backtester(
    initial_capital=10000,
    max_daily_loss=1000,
    max_position_size=100,
    kelly_fraction=0.25
)

# Run backtest
result = backtester.run_backtest(signals, outcomes)
result.print_summary()
```

### Custom Signal Generator

```python
from kalshi_bot.trading import SignalGenerator
from kalshi_bot.models import AggregatedPrice, MarketData

# Create signal generator
signal_gen = SignalGenerator(edge_threshold=0.03)

# Generate signals
signal = signal_gen.generate_signal(
    aggregated_price,
    market_data,
    time_to_expiry_minutes=15
)
```
