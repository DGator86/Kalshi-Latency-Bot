"""Backtester for historical strategy validation."""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np

from ..models import TradingSignal, Order, OrderSide, OrderStatus
from ..trading import RiskManager

logger = logging.getLogger(__name__)


class BacktestResult:
    """Results from a backtest run."""
    
    def __init__(self):
        self.trades: List[dict] = []
        self.daily_pnl: Dict[str, float] = {}
        self.total_pnl: float = 0.0
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.max_drawdown: float = 0.0
        self.sharpe_ratio: float = 0.0
        self.win_rate: float = 0.0
        
    def calculate_metrics(self):
        """Calculate performance metrics."""
        if not self.trades:
            return
            
        self.total_trades = len(self.trades)
        self.winning_trades = sum(1 for t in self.trades if t['pnl'] > 0)
        self.losing_trades = sum(1 for t in self.trades if t['pnl'] < 0)
        
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
            
        # Calculate total PnL
        self.total_pnl = sum(t['pnl'] for t in self.trades)
        
        # Calculate daily PnL
        for trade in self.trades:
            date = trade['timestamp'].date().isoformat()
            if date not in self.daily_pnl:
                self.daily_pnl[date] = 0.0
            self.daily_pnl[date] += trade['pnl']
            
        # Calculate max drawdown
        cumulative_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        
        for trade in sorted(self.trades, key=lambda x: x['timestamp']):
            cumulative_pnl += trade['pnl']
            peak = max(peak, cumulative_pnl)
            drawdown = peak - cumulative_pnl
            max_dd = max(max_dd, drawdown)
            
        self.max_drawdown = max_dd
        
        # Calculate Sharpe ratio
        if len(self.daily_pnl) > 1:
            daily_returns = list(self.daily_pnl.values())
            avg_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            
            if std_return > 0:
                # Annualized Sharpe (assuming 365 trading days)
                self.sharpe_ratio = (avg_return / std_return) * np.sqrt(365)
            else:
                self.sharpe_ratio = 0.0
        else:
            self.sharpe_ratio = 0.0
            
    def print_summary(self):
        """Print backtest summary."""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Total Trades: {self.total_trades}")
        print(f"Winning Trades: {self.winning_trades}")
        print(f"Losing Trades: {self.losing_trades}")
        print(f"Win Rate: {self.win_rate:.2%}")
        print(f"Total PnL: ${self.total_pnl:.2f}")
        print(f"Max Drawdown: ${self.max_drawdown:.2f}")
        print(f"Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print("="*60 + "\n")


class Backtester:
    """Backtest trading strategies."""
    
    def __init__(
        self,
        initial_capital: float = 10000,
        max_daily_loss: float = 1000,
        max_position_size: int = 100,
        kelly_fraction: float = 0.25
    ):
        self.initial_capital = initial_capital
        self.risk_manager = RiskManager(
            max_daily_loss=max_daily_loss,
            max_position_size=max_position_size,
            kelly_fraction=kelly_fraction
        )
        self.capital = initial_capital
        
    def simulate_trade(
        self,
        signal: TradingSignal,
        actual_outcome: bool,
        execution_price: Optional[float] = None
    ) -> dict:
        """
        Simulate a trade based on signal and actual outcome.
        
        Args:
            signal: Trading signal
            actual_outcome: True if prediction was correct, False otherwise
            execution_price: Actual execution price (defaults to signal price)
            
        Returns:
            Trade result dictionary
        """
        if execution_price is None:
            execution_price = signal.kalshi_price
            
        # Calculate risk metrics
        risk_metrics = self.risk_manager.calculate_risk_metrics(signal, self.capital)
        
        # Check if we can trade
        if not self.risk_manager.can_trade(signal, risk_metrics):
            return {
                'timestamp': signal.timestamp,
                'symbol': signal.symbol,
                'side': signal.side.value,
                'quantity': 0,
                'price': execution_price,
                'pnl': 0.0,
                'executed': False,
                'reason': 'Risk limits exceeded'
            }
            
        # Execute trade
        quantity = risk_metrics.max_position
        
        if quantity <= 0:
            return {
                'timestamp': signal.timestamp,
                'symbol': signal.symbol,
                'side': signal.side.value,
                'quantity': 0,
                'price': execution_price,
                'pnl': 0.0,
                'executed': False,
                'reason': 'Position size zero'
            }
            
        # Calculate PnL
        # For binary options: payout is $1 if correct, $0 if wrong
        # Cost is execution_price * quantity
        cost = execution_price * quantity
        
        if actual_outcome:
            # Win: get $1 per contract
            payout = quantity * 1.0
            pnl = payout - cost
        else:
            # Loss: lose cost
            pnl = -cost
            
        # Update capital and risk manager
        self.capital += pnl
        
        order = Order(
            order_id=f"backtest_{signal.timestamp.timestamp()}",
            symbol=signal.symbol,
            side=signal.side,
            quantity=quantity,
            price=execution_price,
            status=OrderStatus.FILLED,
            timestamp=signal.timestamp,
            filled_quantity=quantity
        )
        
        self.risk_manager.record_trade(order, pnl)
        
        return {
            'timestamp': signal.timestamp,
            'symbol': signal.symbol,
            'side': signal.side.value,
            'quantity': quantity,
            'price': execution_price,
            'cost': cost,
            'pnl': pnl,
            'capital': self.capital,
            'executed': True,
            'outcome': actual_outcome
        }
        
    def run_backtest(
        self,
        signals: List[TradingSignal],
        outcomes: List[bool]
    ) -> BacktestResult:
        """
        Run backtest on historical signals.
        
        Args:
            signals: List of historical trading signals
            outcomes: List of actual outcomes (True/False) matching signals
            
        Returns:
            BacktestResult with performance metrics
        """
        if len(signals) != len(outcomes):
            raise ValueError("Signals and outcomes must have same length")
            
        result = BacktestResult()
        
        logger.info(f"Starting backtest with {len(signals)} signals")
        logger.info(f"Initial capital: ${self.initial_capital:.2f}")
        
        for signal, outcome in zip(signals, outcomes):
            trade_result = self.simulate_trade(signal, outcome)
            if trade_result['executed']:
                result.trades.append(trade_result)
                
        result.calculate_metrics()
        
        logger.info(f"Backtest complete. Final capital: ${self.capital:.2f}")
        logger.info(f"Total PnL: ${result.total_pnl:.2f}")
        
        return result
        
    def reset(self):
        """Reset backtester to initial state."""
        self.capital = self.initial_capital
        self.risk_manager = RiskManager(
            max_daily_loss=self.risk_manager.max_daily_loss,
            max_position_size=self.risk_manager.max_position_size,
            kelly_fraction=self.risk_manager.kelly_fraction
        )
