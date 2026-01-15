"""Price aggregator with VWAP calculation."""

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

from ..models import PriceData, AggregatedPrice
from .exchanges import BinanceWebSocket, CoinbaseWebSocket, KrakenWebSocket

logger = logging.getLogger(__name__)


class PriceAggregator:
    """Aggregates prices from multiple exchanges and calculates VWAP."""
    
    def __init__(self, symbols: List[str], window_seconds: int = 60):
        self.symbols = symbols
        self.window_seconds = window_seconds
        self.price_history: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=1000))
        )
        self.latest_prices: Dict[str, Dict[str, PriceData]] = defaultdict(dict)
        self.ws_clients = []
        self.running = False
        
    async def on_price_update(self, price_data: PriceData):
        """Handle incoming price updates."""
        symbol = price_data.symbol
        exchange = price_data.exchange
        
        # Store latest price
        self.latest_prices[symbol][exchange] = price_data
        
        # Store in history
        self.price_history[symbol][exchange].append(price_data)
        
        logger.debug(f"Price update: {exchange} {symbol} @ ${price_data.price:.2f}")
        
    def calculate_vwap(self, symbol: str, exchange: str) -> Optional[float]:
        """Calculate VWAP for a symbol on an exchange."""
        history = self.price_history[symbol][exchange]
        if not history:
            return None
            
        cutoff_time = datetime.now() - timedelta(seconds=self.window_seconds)
        
        # Filter to window
        recent_prices = [p for p in history if p.timestamp >= cutoff_time]
        if not recent_prices:
            return None
            
        # Calculate VWAP
        total_pv = sum(p.price * p.volume for p in recent_prices)
        total_volume = sum(p.volume for p in recent_prices)
        
        if total_volume == 0:
            return None
            
        return total_pv / total_volume
        
    def get_aggregated_price(self, symbol: str) -> Optional[AggregatedPrice]:
        """Get aggregated price across all exchanges."""
        prices = {}
        volumes = {}
        vwaps = []
        weights = []
        
        for exchange, price_data in self.latest_prices[symbol].items():
            vwap = self.calculate_vwap(symbol, exchange)
            if vwap is not None:
                prices[exchange] = vwap
                
                # Calculate total volume in window
                cutoff_time = datetime.now() - timedelta(seconds=self.window_seconds)
                history = self.price_history[symbol][exchange]
                recent_volume = sum(
                    p.volume for p in history if p.timestamp >= cutoff_time
                )
                volumes[exchange] = recent_volume
                
                vwaps.append(vwap)
                weights.append(recent_volume)
        
        if not vwaps:
            return None
            
        # Calculate weighted average across exchanges
        total_weight = sum(weights)
        if total_weight == 0:
            overall_vwap = np.mean(vwaps)
        else:
            overall_vwap = sum(v * w for v, w in zip(vwaps, weights)) / total_weight
            
        return AggregatedPrice(
            symbol=symbol,
            vwap=overall_vwap,
            prices=prices,
            volumes=volumes,
            timestamp=datetime.now()
        )
        
    async def start(self, config: dict):
        """Start all WebSocket connections."""
        self.running = True
        
        for symbol in self.symbols:
            # Create WebSocket clients for each exchange
            if config.get('exchanges', {}).get('binance', {}).get('enabled', True):
                binance_ws = BinanceWebSocket(symbol, self.on_price_update)
                self.ws_clients.append(binance_ws)
                asyncio.create_task(binance_ws.run())
                
            if config.get('exchanges', {}).get('coinbase', {}).get('enabled', True):
                coinbase_ws = CoinbaseWebSocket(symbol, self.on_price_update)
                self.ws_clients.append(coinbase_ws)
                asyncio.create_task(coinbase_ws.run())
                
            if config.get('exchanges', {}).get('kraken', {}).get('enabled', True):
                kraken_ws = KrakenWebSocket(symbol, self.on_price_update)
                self.ws_clients.append(kraken_ws)
                asyncio.create_task(kraken_ws.run())
                
        logger.info(f"Started price aggregator for {self.symbols}")
        
    async def stop(self):
        """Stop all WebSocket connections."""
        self.running = False
        for client in self.ws_clients:
            await client.stop()
        logger.info("Stopped price aggregator")
