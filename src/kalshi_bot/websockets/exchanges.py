"""WebSocket clients for various exchanges."""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Optional
import websockets

from ..models import PriceData

logger = logging.getLogger(__name__)


class ExchangeWebSocket(ABC):
    """Base class for exchange WebSocket clients."""
    
    def __init__(self, symbol: str, callback: Callable[[PriceData], None]):
        self.symbol = symbol
        self.callback = callback
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.running = False
        
    @abstractmethod
    async def connect(self):
        """Connect to the WebSocket."""
        pass
    
    @abstractmethod
    async def subscribe(self):
        """Subscribe to price updates."""
        pass
    
    @abstractmethod
    def parse_message(self, message: dict) -> Optional[PriceData]:
        """Parse incoming message."""
        pass
    
    async def run(self):
        """Run the WebSocket client."""
        self.running = True
        while self.running:
            try:
                await self.connect()
                await self.subscribe()
                
                async for message in self.ws:
                    if not self.running:
                        break
                    
                    try:
                        data = json.loads(message)
                        price_data = self.parse_message(data)
                        if price_data:
                            await self.callback(price_data)
                    except Exception as e:
                        logger.error(f"Error parsing message: {e}")
                        
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self.running:
                    await asyncio.sleep(5)  # Reconnect delay
                    
    async def stop(self):
        """Stop the WebSocket client."""
        self.running = False
        if self.ws:
            await self.ws.close()


class BinanceWebSocket(ExchangeWebSocket):
    """Binance WebSocket client."""
    
    def __init__(self, symbol: str, callback: Callable[[PriceData], None]):
        super().__init__(symbol, callback)
        self.ws_url = "wss://stream.binance.com:9443/ws"
        
    async def connect(self):
        """Connect to Binance WebSocket."""
        # Convert symbol format (e.g., BTC -> btcusdt)
        stream_symbol = f"{self.symbol.lower()}usdt"
        url = f"{self.ws_url}/{stream_symbol}@trade"
        self.ws = await websockets.connect(url)
        logger.info(f"Connected to Binance WebSocket for {self.symbol}")
        
    async def subscribe(self):
        """Subscribe to price updates."""
        # Binance doesn't require explicit subscription for trade streams
        pass
    
    def parse_message(self, message: dict) -> Optional[PriceData]:
        """Parse Binance message."""
        try:
            if 'e' in message and message['e'] == 'trade':
                return PriceData(
                    exchange="binance",
                    symbol=self.symbol,
                    price=float(message['p']),
                    volume=float(message['q']),
                    timestamp=datetime.fromtimestamp(message['T'] / 1000)
                )
        except Exception as e:
            logger.error(f"Error parsing Binance message: {e}")
        return None


class CoinbaseWebSocket(ExchangeWebSocket):
    """Coinbase WebSocket client."""
    
    def __init__(self, symbol: str, callback: Callable[[PriceData], None]):
        super().__init__(symbol, callback)
        self.ws_url = "wss://ws-feed.exchange.coinbase.com"
        
    async def connect(self):
        """Connect to Coinbase WebSocket."""
        self.ws = await websockets.connect(self.ws_url)
        logger.info(f"Connected to Coinbase WebSocket for {self.symbol}")
        
    async def subscribe(self):
        """Subscribe to price updates."""
        product_id = f"{self.symbol}-USD"
        subscribe_message = {
            "type": "subscribe",
            "product_ids": [product_id],
            "channels": ["matches"]
        }
        await self.ws.send(json.dumps(subscribe_message))
        
    def parse_message(self, message: dict) -> Optional[PriceData]:
        """Parse Coinbase message."""
        try:
            if message.get('type') == 'match':
                return PriceData(
                    exchange="coinbase",
                    symbol=self.symbol,
                    price=float(message['price']),
                    volume=float(message['size']),
                    timestamp=datetime.fromisoformat(message['time'].replace('Z', '+00:00'))
                )
        except Exception as e:
            logger.error(f"Error parsing Coinbase message: {e}")
        return None


class KrakenWebSocket(ExchangeWebSocket):
    """Kraken WebSocket client."""
    
    def __init__(self, symbol: str, callback: Callable[[PriceData], None]):
        super().__init__(symbol, callback)
        self.ws_url = "wss://ws.kraken.com"
        
    async def connect(self):
        """Connect to Kraken WebSocket."""
        self.ws = await websockets.connect(self.ws_url)
        logger.info(f"Connected to Kraken WebSocket for {self.symbol}")
        
    async def subscribe(self):
        """Subscribe to price updates."""
        pair = f"{self.symbol}/USD"
        subscribe_message = {
            "event": "subscribe",
            "pair": [pair],
            "subscription": {"name": "trade"}
        }
        await self.ws.send(json.dumps(subscribe_message))
        
    def parse_message(self, message: dict) -> Optional[PriceData]:
        """Parse Kraken message."""
        try:
            if isinstance(message, list) and len(message) > 1:
                if isinstance(message[1], list) and len(message[1]) > 0:
                    trade = message[1][0]
                    return PriceData(
                        exchange="kraken",
                        symbol=self.symbol,
                        price=float(trade[0]),
                        volume=float(trade[1]),
                        timestamp=datetime.fromtimestamp(float(trade[2]))
                    )
        except Exception as e:
            logger.error(f"Error parsing Kraken message: {e}")
        return None
