#!/usr/bin/env python3
"""Command-line interface for the Kalshi Latency Bot."""

import argparse
import sys

from kalshi_bot.bot import TradingBot, main as run_bot
from kalshi_bot.backtester import Backtester, BacktestResult
from kalshi_bot.models import TradingSignal, OrderSide
from datetime import datetime


def run_live(args):
    """Run the bot in live trading mode."""
    print(f"Starting Kalshi Latency Bot with config: {args.config}")
    bot = TradingBot(args.config)
    
    try:
        import asyncio
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_backtest(args):
    """Run a backtest."""
    print("Running backtest mode...")
    print(f"Initial capital: ${args.capital}")
    
    # This is a placeholder - in real usage, you'd load historical data
    print("\nNote: Backtest requires historical signal and outcome data.")
    print("Please implement data loading in your backtest script.")
    
    # Example backtest structure
    backtester = Backtester(
        initial_capital=args.capital,
        max_daily_loss=args.max_loss,
        max_position_size=args.max_position,
        kelly_fraction=args.kelly_fraction
    )
    
    # Load your signals and outcomes here
    # signals = load_signals_from_file(args.data_file)
    # outcomes = load_outcomes_from_file(args.data_file)
    
    # result = backtester.run_backtest(signals, outcomes)
    # result.print_summary()
    
    print("\nBacktest mode is ready. Integrate your historical data source.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Kalshi Latency Bot - Async crypto trading bot'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Live trading command
    live_parser = subparsers.add_parser('live', help='Run live trading bot')
    live_parser.add_argument(
        '--config',
        default='config.json',
        help='Path to configuration file'
    )
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument(
        '--config',
        default='config.json',
        help='Path to configuration file'
    )
    backtest_parser.add_argument(
        '--capital',
        type=float,
        default=10000,
        help='Initial capital'
    )
    backtest_parser.add_argument(
        '--max-loss',
        type=float,
        default=1000,
        help='Maximum daily loss'
    )
    backtest_parser.add_argument(
        '--max-position',
        type=int,
        default=100,
        help='Maximum position size'
    )
    backtest_parser.add_argument(
        '--kelly-fraction',
        type=float,
        default=0.25,
        help='Kelly criterion fraction'
    )
    
    args = parser.parse_args()
    
    if args.command == 'live':
        run_live(args)
    elif args.command == 'backtest':
        run_backtest(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
