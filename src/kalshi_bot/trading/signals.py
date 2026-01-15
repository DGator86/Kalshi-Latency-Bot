"""Signal generator with edge threshold."""

import logging
from datetime import datetime
from typing import Optional, List

from ..models import TradingSignal, OrderSide, AggregatedPrice, MarketData
from .probability import ProbabilityEngine

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generate trading signals based on edge calculations."""
    
    def __init__(self, edge_threshold: float = 0.03):
        self.edge_threshold = edge_threshold
        self.probability_engine = ProbabilityEngine()
        
    def update_price_history(self, symbol: str, price: float, timestamp: datetime):
        """Update price history in probability engine."""
        self.probability_engine.update_price_history(symbol, price, timestamp)
        
    def generate_signal(
        self,
        aggregated_price: AggregatedPrice,
        market_data: MarketData,
        time_to_expiry_minutes: float
    ) -> Optional[TradingSignal]:
        """
        Generate a trading signal if edge exceeds threshold.
        
        Args:
            aggregated_price: Current aggregated price from exchanges
            market_data: Kalshi market data
            time_to_expiry_minutes: Time until market expiry
            
        Returns:
            TradingSignal if edge exceeds threshold, None otherwise
        """
        symbol = aggregated_price.symbol
        current_price = aggregated_price.vwap
        
        # Update price history
        self.update_price_history(symbol, current_price, aggregated_price.timestamp)
        
        # Extract strike from market_id (assumes format like "BTC-15MIN-50000")
        # This is a simplified example - adjust based on actual market_id format
        try:
            parts = market_data.market_id.split('-')
            if len(parts) >= 3:
                strike_price = float(parts[2])
            else:
                logger.warning(f"Cannot parse strike from market_id: {market_data.market_id}")
                return None
        except (ValueError, IndexError):
            logger.warning(f"Cannot parse strike from market_id: {market_data.market_id}")
            return None
            
        # Calculate fair value for "yes" (price above strike)
        fair_value = self.probability_engine.calculate_fair_value(
            symbol,
            current_price,
            strike_price,
            time_to_expiry_minutes,
            direction="above"
        )
        
        if fair_value is None:
            return None
            
        # Check both yes and no sides for best edge
        yes_ask = market_data.yes_ask / 100 if market_data.yes_ask > 1 else market_data.yes_ask
        no_ask = market_data.no_ask / 100 if market_data.no_ask > 1 else market_data.no_ask
        
        # Edge for buying yes
        yes_edge = fair_value - yes_ask
        
        # Edge for buying no
        no_edge = (1 - fair_value) - no_ask
        
        # Find best edge
        best_edge = max(yes_edge, no_edge)
        
        if best_edge < self.edge_threshold:
            logger.debug(f"Edge {best_edge:.3f} below threshold {self.edge_threshold}")
            return None
            
        # Determine side
        if yes_edge > no_edge:
            side = OrderSide.BUY  # Buy yes
            market_price = yes_ask
            probability = fair_value
        else:
            side = OrderSide.SELL  # Buy no (sell yes)
            market_price = no_ask
            probability = 1 - fair_value
            
        # Calculate confidence based on edge magnitude and volatility
        volatility_data = self.probability_engine.get_volatility_data(symbol)
        if volatility_data and volatility_data.historical_vol:
            # Higher volatility = lower confidence
            confidence = min(1.0, best_edge / volatility_data.historical_vol * 10)
        else:
            confidence = 0.5
            
        signal = TradingSignal(
            symbol=symbol,
            side=side,
            edge=best_edge,
            probability=probability,
            kalshi_price=market_price,
            fair_value=fair_value,
            timestamp=datetime.now(),
            confidence=confidence
        )
        
        logger.info(
            f"Signal generated: {symbol} {side.value} "
            f"edge={best_edge:.3%} prob={probability:.3f} conf={confidence:.2f}"
        )
        
        return signal
        
    def get_signals(
        self,
        aggregated_prices: List[AggregatedPrice],
        market_data_list: List[MarketData],
        time_to_expiry_minutes: float
    ) -> List[TradingSignal]:
        """Generate signals for multiple markets."""
        signals = []
        
        # Match aggregated prices with market data
        for agg_price in aggregated_prices:
            matching_markets = [
                m for m in market_data_list
                if m.symbol == agg_price.symbol
            ]
            
            for market_data in matching_markets:
                signal = self.generate_signal(
                    agg_price,
                    market_data,
                    time_to_expiry_minutes
                )
                if signal:
                    signals.append(signal)
                    
        return signals
