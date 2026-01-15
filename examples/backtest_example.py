#!/usr/bin/env python3
"""Example script showing how to use the backtester."""

import asyncio
from datetime import datetime, timedelta
import random

from kalshi_bot.backtester import Backtester, BacktestResult
from kalshi_bot.models import TradingSignal, OrderSide


def generate_example_signals(num_signals=100):
    """Generate example trading signals for demonstration."""
    signals = []
    outcomes = []
    
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(num_signals):
        # Generate random signal
        edge = random.uniform(0.03, 0.15)  # 3-15% edge
        probability = random.uniform(0.55, 0.75)  # 55-75% win probability
        
        signal = TradingSignal(
            symbol=random.choice(['BTC', 'ETH']),
            side=random.choice([OrderSide.BUY, OrderSide.SELL]),
            edge=edge,
            probability=probability,
            kalshi_price=random.uniform(0.3, 0.7),
            fair_value=probability,
            timestamp=base_time + timedelta(hours=i),
            confidence=random.uniform(0.6, 0.9)
        )
        
        signals.append(signal)
        
        # Simulate outcome based on probability
        outcome = random.random() < probability
        outcomes.append(outcome)
    
    return signals, outcomes


def main():
    """Run example backtest."""
    print("=" * 60)
    print("Kalshi Latency Bot - Backtest Example")
    print("=" * 60)
    print()
    
    # Generate example data
    print("Generating example signals...")
    signals, outcomes = generate_example_signals(100)
    print(f"Generated {len(signals)} signals")
    print()
    
    # Create backtester
    backtester = Backtester(
        initial_capital=10000,
        max_daily_loss=1000,
        max_position_size=100,
        kelly_fraction=0.25
    )
    
    # Run backtest
    print("Running backtest...")
    result = backtester.run_backtest(signals, outcomes)
    
    # Print results
    result.print_summary()
    
    # Print some example trades
    print("Example Trades:")
    print("-" * 60)
    for i, trade in enumerate(result.trades[:5]):
        print(f"Trade {i+1}:")
        print(f"  Symbol: {trade['symbol']}")
        print(f"  Side: {trade['side']}")
        print(f"  Quantity: {trade['quantity']}")
        print(f"  Price: ${trade['price']:.2f}")
        print(f"  PnL: ${trade['pnl']:.2f}")
        print(f"  Outcome: {'Win' if trade['outcome'] else 'Loss'}")
        print()


if __name__ == '__main__':
    main()
