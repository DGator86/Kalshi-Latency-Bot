"""Black-Scholes probability engine for short-term volatility."""

import logging
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
from scipy.stats import norm

from ..models import VolatilityData

logger = logging.getLogger(__name__)


class ProbabilityEngine:
    """Calculate probabilities using Black-Scholes model for short-term moves."""
    
    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        self.price_history = {}
        
    def update_price_history(self, symbol: str, price: float, timestamp: datetime):
        """Update price history for volatility calculations."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            
        self.price_history[symbol].append((timestamp, price))
        
        # Keep only recent history
        if len(self.price_history[symbol]) > self.lookback_periods * 2:
            self.price_history[symbol] = self.price_history[symbol][-self.lookback_periods * 2:]
            
    def calculate_historical_volatility(self, symbol: str) -> Optional[float]:
        """Calculate historical volatility from price history."""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
            return None
            
        prices = [p[1] for p in self.price_history[symbol][-self.lookback_periods:]]
        if len(prices) < 2:
            return None
            
        # Calculate log returns
        log_returns = np.diff(np.log(prices))
        
        if len(log_returns) == 0:
            return None
            
        # Annualized volatility (assuming 1-minute intervals)
        volatility = np.std(log_returns) * np.sqrt(525600)  # minutes in a year
        
        return volatility
        
    def get_volatility_data(self, symbol: str) -> Optional[VolatilityData]:
        """Get volatility data for a symbol."""
        hist_vol = self.calculate_historical_volatility(symbol)
        if hist_vol is None:
            return None
            
        prices = [p[1] for p in self.price_history[symbol][-self.lookback_periods:]]
        std_dev = np.std(prices)
        
        return VolatilityData(
            symbol=symbol,
            historical_vol=hist_vol,
            implied_vol=None,  # Could be calculated from options if available
            std_dev=std_dev,
            timestamp=datetime.now()
        )
        
    def calculate_probability(
        self,
        current_price: float,
        strike_price: float,
        time_to_expiry_minutes: float,
        volatility: float,
        direction: str = "above"
    ) -> float:
        """
        Calculate probability of price being above/below strike at expiry.
        
        Args:
            current_price: Current spot price
            strike_price: Target price level
            time_to_expiry_minutes: Time to expiry in minutes
            volatility: Annualized volatility
            direction: "above" or "below"
            
        Returns:
            Probability between 0 and 1
        """
        if time_to_expiry_minutes <= 0:
            return 1.0 if (
                (direction == "above" and current_price >= strike_price) or
                (direction == "below" and current_price <= strike_price)
            ) else 0.0
            
        # Convert time to years
        T = time_to_expiry_minutes / 525600
        
        # Handle edge cases
        if T <= 0 or volatility <= 0:
            return 0.5
            
        # Black-Scholes probability calculation
        # d2 in Black-Scholes formula
        d2 = (np.log(current_price / strike_price) - 0.5 * volatility**2 * T) / (volatility * np.sqrt(T))
        
        if direction == "above":
            # Probability of price being above strike
            probability = norm.cdf(d2)
        else:
            # Probability of price being below strike
            probability = 1 - norm.cdf(d2)
            
        return probability
        
    def calculate_edge(
        self,
        fair_probability: float,
        market_price: float,
        side: str = "yes"
    ) -> float:
        """
        Calculate edge between fair probability and market price.
        
        Args:
            fair_probability: Our calculated fair probability
            market_price: Market price (0-1 or 0-100 scale)
            side: "yes" or "no"
            
        Returns:
            Edge as a decimal (e.g., 0.05 for 5% edge)
        """
        # Convert market price to probability if needed
        if market_price > 1:
            market_price = market_price / 100
            
        if side == "yes":
            # Buying yes: edge when our probability is higher than market
            edge = fair_probability - market_price
        else:
            # Buying no: edge when our probability is lower than market
            edge = (1 - fair_probability) - market_price
            
        return edge
        
    def calculate_fair_value(
        self,
        symbol: str,
        current_price: float,
        strike_price: float,
        time_to_expiry_minutes: float,
        direction: str = "above"
    ) -> Optional[float]:
        """Calculate fair value for a binary option."""
        volatility_data = self.get_volatility_data(symbol)
        if volatility_data is None or volatility_data.historical_vol is None:
            logger.warning(f"No volatility data for {symbol}")
            return None
            
        probability = self.calculate_probability(
            current_price,
            strike_price,
            time_to_expiry_minutes,
            volatility_data.historical_vol,
            direction
        )
        
        # Fair value is the probability (for a $1 payout binary option)
        return probability
