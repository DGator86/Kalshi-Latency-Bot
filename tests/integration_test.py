#!/usr/bin/env python3
"""Integration test for the Kalshi Latency Bot."""

import sys
sys.path.insert(0, 'src')

import asyncio
from datetime import datetime
from kalshi_bot.models import (
    PriceData, AggregatedPrice, MarketData, TradingSignal,
    OrderSide, VolatilityData
)
from kalshi_bot.trading import ProbabilityEngine, SignalGenerator, RiskManager
from kalshi_bot.websockets import PriceAggregator
from kalshi_bot.backtester import Backtester
from kalshi_bot.config import Config


def test_probability_engine():
    """Test probability engine."""
    print("Testing ProbabilityEngine...")
    
    engine = ProbabilityEngine()
    
    # Add some price history
    for i in range(100):
        engine.update_price_history('BTC', 50000 + i * 10, datetime.now())
    
    # Calculate volatility
    vol_data = engine.get_volatility_data('BTC')
    assert vol_data is not None, "Volatility data should not be None"
    assert vol_data.historical_vol > 0, "Historical volatility should be positive"
    
    # Calculate probability
    prob = engine.calculate_probability(
        current_price=50000,
        strike_price=50500,
        time_to_expiry_minutes=15,
        volatility=0.5,
        direction="above"
    )
    assert 0 <= prob <= 1, "Probability should be between 0 and 1"
    
    print(f"✓ ProbabilityEngine: volatility={vol_data.historical_vol:.4f}, prob={prob:.4f}")


def test_signal_generator():
    """Test signal generator."""
    print("Testing SignalGenerator...")
    
    signal_gen = SignalGenerator(edge_threshold=0.03)
    
    # Update price history
    for i in range(100):
        signal_gen.update_price_history('BTC', 50000 + i * 10, datetime.now())
    
    # Create aggregated price
    agg_price = AggregatedPrice(
        symbol='BTC',
        vwap=50000,
        prices={'binance': 50000, 'coinbase': 50010, 'kraken': 49990},
        volumes={'binance': 10, 'coinbase': 8, 'kraken': 7},
        timestamp=datetime.now()
    )
    
    # Create market data
    market_data = MarketData(
        market_id='BTC-15MIN-50500',
        symbol='BTC',
        yes_bid=40,
        yes_ask=45,
        no_bid=55,
        no_ask=60,
        volume=1000,
        timestamp=datetime.now()
    )
    
    # Generate signal
    signal = signal_gen.generate_signal(agg_price, market_data, 15.0)
    
    if signal:
        print(f"✓ SignalGenerator: edge={signal.edge:.3%}, prob={signal.probability:.3f}")
    else:
        print(f"✓ SignalGenerator: no signal generated (edge below threshold)")


def test_risk_manager():
    """Test risk manager."""
    print("Testing RiskManager...")
    
    risk_mgr = RiskManager(
        max_daily_loss=1000,
        max_position_size=100,
        kelly_fraction=0.25
    )
    
    # Create a signal
    signal = TradingSignal(
        symbol='BTC',
        side=OrderSide.BUY,
        edge=0.05,
        probability=0.6,
        kalshi_price=0.45,
        fair_value=0.5,
        timestamp=datetime.now(),
        confidence=0.7
    )
    
    # Calculate risk metrics
    risk_metrics = risk_mgr.calculate_risk_metrics(signal, 10000)
    
    assert risk_metrics.kelly_size >= 0, "Kelly size should be non-negative"
    assert risk_metrics.max_position <= 100, "Max position should respect limit"
    
    # Check if we can trade
    can_trade = risk_mgr.can_trade(signal, risk_metrics)
    
    print(f"✓ RiskManager: kelly_size={risk_metrics.kelly_size}, max_pos={risk_metrics.max_position}, can_trade={can_trade}")


def test_backtester():
    """Test backtester."""
    print("Testing Backtester...")
    
    backtester = Backtester(
        initial_capital=10000,
        max_daily_loss=1000,
        max_position_size=100,
        kelly_fraction=0.25
    )
    
    # Create test signals
    signals = []
    outcomes = []
    
    for i in range(10):
        signal = TradingSignal(
            symbol='BTC',
            side=OrderSide.BUY,
            edge=0.05,
            probability=0.6,
            kalshi_price=0.45,
            fair_value=0.5,
            timestamp=datetime.now(),
            confidence=0.7
        )
        signals.append(signal)
        outcomes.append(i % 2 == 0)  # Alternate wins/losses
    
    result = backtester.run_backtest(signals, outcomes)
    
    assert result.total_trades > 0, "Should have executed some trades"
    assert result.win_rate >= 0 and result.win_rate <= 1, "Win rate should be valid"
    
    print(f"✓ Backtester: trades={result.total_trades}, win_rate={result.win_rate:.2%}, pnl=${result.total_pnl:.2f}")


def test_config():
    """Test configuration."""
    print("Testing Config...")
    
    config = Config('config.example.json')
    
    assert config.get('trading.edge_threshold') == 0.03, "Edge threshold should be 0.03"
    assert config.get('trading.kelly_fraction') == 0.25, "Kelly fraction should be 0.25"
    
    print(f"✓ Config: loaded successfully")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Kalshi Latency Bot - Integration Tests")
    print("=" * 60)
    print()
    
    try:
        test_config()
        test_probability_engine()
        test_signal_generator()
        test_risk_manager()
        test_backtester()
        
        print()
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
