"""Kalshi API client with HMAC authentication."""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from typing import Dict, List, Optional
import aiohttp

from ..models import MarketData, Order, OrderSide, OrderStatus

logger = logging.getLogger(__name__)


class KalshiClient:
    """Async client for Kalshi API with HMAC authentication."""
    
    def __init__(self, api_base: str, api_key: str, api_secret: str):
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.api_secret = api_secret
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            
    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Generate HMAC signature for request."""
        # Create message to sign
        message = f"{timestamp}{method}{path}{body}"
        
        # Create HMAC signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
        
    def _get_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Get headers with HMAC signature."""
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, method, path, body)
        
        headers = {
            'Content-Type': 'application/json',
            'KALSHI-ACCESS-KEY': self.api_key,
            'KALSHI-ACCESS-SIGNATURE': signature,
            'KALSHI-ACCESS-TIMESTAMP': timestamp
        }
        
        return headers
        
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        data: Optional[dict] = None
    ) -> dict:
        """Make an authenticated request to Kalshi API."""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        url = f"{self.api_base}{endpoint}"
        body = json.dumps(data) if data else ""
        headers = self._get_headers(method, endpoint, body)
        
        try:
            async with self.session.request(
                method,
                url,
                headers=headers,
                params=params,
                data=body if body else None
            ) as response:
                response_data = await response.json()
                
                if response.status >= 400:
                    logger.error(f"API error: {response.status} - {response_data}")
                    raise Exception(f"API error: {response_data}")
                    
                return response_data
                
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
            
    async def get_markets(self, symbols: Optional[List[str]] = None) -> List[MarketData]:
        """Get market data for symbols."""
        try:
            # Get all markets or filter by symbols
            response = await self._request('GET', '/v1/markets')
            
            markets = []
            for market in response.get('markets', []):
                # Filter by symbols if provided
                if symbols:
                    # Extract symbol from market ticker
                    market_symbol = None
                    for symbol in symbols:
                        if symbol in market.get('ticker', ''):
                            market_symbol = symbol
                            break
                    if not market_symbol:
                        continue
                else:
                    market_symbol = market.get('ticker', '').split('-')[0]
                    
                # Extract market data
                market_data = MarketData(
                    market_id=market.get('id', ''),
                    symbol=market_symbol,
                    yes_bid=float(market.get('yes_bid', 0)),
                    yes_ask=float(market.get('yes_ask', 0)),
                    no_bid=float(market.get('no_bid', 0)),
                    no_ask=float(market.get('no_ask', 0)),
                    volume=int(market.get('volume', 0)),
                    timestamp=time.time()
                )
                markets.append(market_data)
                
            return markets
            
        except Exception as e:
            logger.error(f"Failed to get markets: {e}")
            return []
            
    async def place_order(
        self,
        market_id: str,
        side: OrderSide,
        quantity: int,
        price: float
    ) -> Optional[Order]:
        """Place an order."""
        try:
            order_data = {
                'market_id': market_id,
                'side': 'yes' if side == OrderSide.BUY else 'no',
                'quantity': quantity,
                'price': int(price * 100),  # Price in cents
                'type': 'limit'
            }
            
            response = await self._request('POST', '/v1/orders', data=order_data)
            
            order_info = response.get('order', {})
            
            order = Order(
                order_id=order_info.get('id', ''),
                symbol=market_id,
                side=side,
                quantity=quantity,
                price=price,
                status=OrderStatus.PENDING,
                timestamp=time.time(),
                filled_quantity=0
            )
            
            logger.info(f"Order placed: {order.order_id} {side.value} {quantity} @ ${price:.2f}")
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None
            
    async def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status."""
        try:
            response = await self._request('GET', f'/v1/orders/{order_id}')
            
            status_str = response.get('order', {}).get('status', 'pending')
            status_map = {
                'pending': OrderStatus.PENDING,
                'filled': OrderStatus.FILLED,
                'cancelled': OrderStatus.CANCELLED,
                'rejected': OrderStatus.REJECTED
            }
            
            return status_map.get(status_str, OrderStatus.PENDING)
            
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return None
            
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            await self._request('DELETE', f'/v1/orders/{order_id}')
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
            
    async def get_positions(self) -> List[dict]:
        """Get current positions."""
        try:
            response = await self._request('GET', '/v1/portfolio/positions')
            return response.get('positions', [])
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
            
    async def get_balance(self) -> float:
        """Get account balance."""
        try:
            response = await self._request('GET', '/v1/portfolio/balance')
            balance = float(response.get('balance', 0)) / 100  # Convert from cents
            return balance
            
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0
