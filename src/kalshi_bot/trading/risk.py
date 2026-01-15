"""Risk manager with Kelly sizing and daily limits."""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
import numpy as np

from ..models import TradingSignal, RiskMetrics, Position, Order

logger = logging.getLogger(__name__)


class RiskManager:
    """Manage risk with position sizing and limits."""
    
    def __init__(
        self,
        max_daily_loss: float = 1000,
        max_position_size: int = 100,
        kelly_fraction: float = 0.25
    ):
        self.max_daily_loss = max_daily_loss
        self.max_position_size = max_position_size
        self.kelly_fraction = kelly_fraction
        
        self.positions: Dict[str, Position] = {}
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.last_reset: datetime = datetime.now()
        self.orders_today: list = []
        
    def reset_daily_metrics(self):
        """Reset daily metrics if new day."""
        now = datetime.now()
        if now.date() > self.last_reset.date():
            logger.info(f"Resetting daily metrics. Previous PnL: ${self.daily_pnl:.2f}")
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.orders_today = []
            self.last_reset = now
            
    def calculate_kelly_size(
        self,
        edge: float,
        probability: float,
        price: float,
        bankroll: float
    ) -> int:
        """
        Calculate optimal position size using Kelly criterion.
        
        Args:
            edge: Expected edge (e.g., 0.05 for 5%)
            probability: Win probability
            price: Entry price
            bankroll: Available capital
            
        Returns:
            Position size in contracts
        """
        if probability <= 0 or probability >= 1:
            return 0
            
        # Kelly formula for binary outcomes
        # f = (p * b - q) / b
        # where p = probability, q = 1-p, b = odds
        
        # Calculate odds from price
        # For binary option: if price is 0.4, payout is 1, so odds = (1-0.4)/0.4 = 1.5
        if price <= 0 or price >= 1:
            return 0
            
        odds = (1 - price) / price
        q = 1 - probability
        
        kelly_fraction_calc = (probability * odds - q) / odds
        
        # Apply fractional Kelly
        kelly_fraction_calc = kelly_fraction_calc * self.kelly_fraction
        
        # Ensure non-negative
        kelly_fraction_calc = max(0, kelly_fraction_calc)
        
        # Calculate dollar amount
        kelly_dollars = bankroll * kelly_fraction_calc
        
        # Convert to contracts (each contract costs price * 1)
        if price > 0:
            kelly_contracts = int(kelly_dollars / price)
        else:
            kelly_contracts = 0
            
        return kelly_contracts
        
    def get_current_exposure(self) -> float:
        """Calculate current exposure across all positions."""
        exposure = sum(
            pos.quantity * pos.current_price
            for pos in self.positions.values()
        )
        return exposure
        
    def calculate_risk_metrics(
        self,
        signal: TradingSignal,
        bankroll: float
    ) -> RiskMetrics:
        """Calculate risk metrics for a signal."""
        self.reset_daily_metrics()
        
        # Calculate Kelly size
        kelly_size = self.calculate_kelly_size(
            signal.edge,
            signal.probability,
            signal.kalshi_price,
            bankroll
        )
        
        # Apply position limits
        max_position = min(kelly_size, self.max_position_size)
        
        # Check daily loss limit
        remaining_loss_capacity = self.max_daily_loss + self.daily_pnl
        
        # Calculate current exposure
        current_exposure = self.get_current_exposure()
        
        # Remaining capacity considering daily limits
        remaining_capacity = min(
            remaining_loss_capacity,
            bankroll - current_exposure
        )
        
        # Adjust position size based on remaining capacity
        if signal.kalshi_price > 0:
            capacity_size = int(remaining_capacity / signal.kalshi_price)
            max_position = min(max_position, capacity_size)
        
        max_position = max(0, max_position)
        
        return RiskMetrics(
            kelly_size=kelly_size,
            max_position=max_position,
            current_exposure=current_exposure,
            daily_pnl=self.daily_pnl,
            remaining_capacity=remaining_capacity
        )
        
    def can_trade(self, signal: TradingSignal, risk_metrics: RiskMetrics) -> bool:
        """Check if we can trade based on risk limits."""
        self.reset_daily_metrics()
        
        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            logger.warning(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
            return False
            
        # Check position size
        if risk_metrics.max_position <= 0:
            logger.warning("Position size is zero or negative")
            return False
            
        # Check remaining capacity
        if risk_metrics.remaining_capacity <= 0:
            logger.warning("No remaining capacity")
            return False
            
        return True
        
    def update_position(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        current_price: float
    ):
        """Update position information."""
        pnl = quantity * (current_price - entry_price)
        
        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            pnl=pnl,
            timestamp=datetime.now()
        )
        
    def record_trade(self, order: Order, pnl: float = 0):
        """Record a completed trade."""
        self.reset_daily_metrics()
        
        self.daily_pnl += pnl
        self.daily_trades += 1
        self.orders_today.append(order)
        
        logger.info(
            f"Trade recorded: {order.symbol} {order.side.value} "
            f"PnL: ${pnl:.2f}, Daily PnL: ${self.daily_pnl:.2f}"
        )
        
    def get_position_summary(self) -> dict:
        """Get summary of all positions."""
        return {
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'pnl': pos.pnl
                }
                for symbol, pos in self.positions.items()
            },
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'current_exposure': self.get_current_exposure()
        }
