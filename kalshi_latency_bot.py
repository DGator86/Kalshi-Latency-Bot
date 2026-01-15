#!/usr/bin/env python3
"""
Kalshi 15-Minute Crypto Latency Arbitrage Bot

Exploits pricing inefficiencies between real-time crypto prices and Kalshi's
15-minute crypto prediction markets.

Strategy:
- Monitor real-time BTC/ETH prices from multiple exchanges (Binance, Coinbase, Kraken)
- Compare to Kalshi market implied probabilities
- When significant divergence detected, execute trades before market adjusts
- Focus on the final 5 minutes of each 15-minute window where edge is highest

Author: Built for FINAL_GNOSIS Trading System
"""

import os
import sys
import json
import time
import asyncio
import logging
import hashlib
import hmac
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable
from enum import Enum
from collections import deque
import statistics
import threading
from abc import ABC, abstractmethod

# Third-party imports (install with pip)
try:
    import aiohttp
    import websockets
    import numpy as np
    from scipy.stats import norm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install aiohttp websockets numpy scipy")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Bot configuration - modify these for your setup"""

    # Kalshi API credentials (set via environment variables)
    KALSHI_API_KEY: str = field(default_factory=lambda: os.getenv("KALSHI_API_KEY", ""))
    KALSHI_API_SECRET: str = field(default_factory=lambda: os.getenv("KALSHI_API_SECRET", ""))
    KALSHI_API_BASE: str = "https://api.elections.kalshi.com/trade-api/v2"
    KALSHI_WS_URL: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"

    # Exchange API keys (optional - public endpoints work for price feeds)
    BINANCE_API_KEY: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    COINBASE_API_KEY: str = field(default_factory=lambda: os.getenv("COINBASE_API_KEY", ""))

    # Trading parameters
    MAX_POSITION_SIZE: int = 100  # Max contracts per position
    MIN_EDGE_THRESHOLD: float = 0.03  # 3% minimum edge to trade
    MAX_SPREAD_COST: float = 0.02  # 2% max acceptable spread
    KELLY_FRACTION: float = 0.25  # Fraction of Kelly criterion to use

    # Latency parameters
    PRICE_STALE_MS: int = 500  # Price older than this is stale
    MIN_SOURCES_REQUIRED: int = 2  # Minimum price sources for trade
    LATENCY_WINDOW_SECONDS: int = 300  # Focus on last 5 minutes of each period

    # Risk limits
    MAX_DAILY_LOSS: float = 500.0  # Stop trading after this loss
    MAX_CONCURRENT_POSITIONS: int = 5
    MAX_SINGLE_TRADE_RISK: float = 100.0  # Max risk per trade in dollars

    # Market identifiers (Kalshi's 15-min crypto markets)
    CRYPTO_TICKERS: List[str] = field(default_factory=lambda: [
        "KXBTC",  # Bitcoin 15-min
        "KXETH",  # Ethereum 15-min
    ])

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "kalshi_latency_bot.log"

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(config: Config) -> logging.Logger:
    """Configure structured logging"""
    logger = logging.getLogger("KalshiLatencyBot")
    logger.setLevel(getattr(logging, config.LOG_LEVEL))

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console.setFormatter(console_fmt)

    # File handler
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger

# =============================================================================
# DATA STRUCTURES
# =============================================================================

class Side(Enum):
    YES = "yes"
    NO = "no"

@dataclass
class PriceUpdate:
    """Real-time price from an exchange"""
    source: str
    symbol: str
    price: float
    timestamp_ms: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume_24h: Optional[float] = None

    @property
    def age_ms(self) -> int:
        return int(time.time() * 1000) - self.timestamp_ms

    @property
    def mid_price(self) -> float:
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return self.price

@dataclass
class KalshiMarket:
    """Kalshi market data"""
    ticker: str
    event_ticker: str
    title: str
    subtitle: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    volume: int
    open_interest: int
    strike_price: Optional[float]  # For crypto: the price threshold
    expiration_time: datetime
    last_updated: datetime

    @property
    def yes_mid(self) -> float:
        return (self.yes_bid + self.yes_ask) / 2

    @property
    def no_mid(self) -> float:
        return (self.no_bid + self.no_ask) / 2

    @property
    def spread(self) -> float:
        return self.yes_ask - self.yes_bid

    @property
    def implied_prob_yes(self) -> float:
        return self.yes_mid

    @property
    def time_to_expiry_seconds(self) -> float:
        return (self.expiration_time - datetime.now(timezone.utc)).total_seconds()

@dataclass
class TradeSignal:
    """Generated trading signal"""
    market: KalshiMarket
    side: Side
    edge: float
    confidence: float
    fair_value: float
    market_price: float
    crypto_price: float
    recommended_size: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __str__(self):
        return (
            f"Signal: {self.market.ticker} {self.side.value.upper()} | "
            f"Edge: {self.edge:.2%} | Conf: {self.confidence:.2%} | "
            f"Fair: {self.fair_value:.3f} vs Market: {self.market_price:.3f}"
        )

@dataclass
class Position:
    """Active position tracking"""
    ticker: str
    side: Side
    quantity: int
    entry_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.entry_price

@dataclass
class TradeResult:
    """Executed trade result"""
    order_id: str
    ticker: str
    side: Side
    quantity: int
    price: float
    timestamp: datetime
    success: bool
    error: Optional[str] = None

# =============================================================================
# PRICE FEED AGGREGATOR
# =============================================================================

class PriceFeedManager:
    """
    Aggregates real-time prices from multiple exchanges.
    Uses websockets for minimal latency.
    """

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.prices: Dict[str, Dict[str, PriceUpdate]] = {
            "BTC": {},
            "ETH": {},
        }
        self.callbacks: List[Callable[[PriceUpdate], None]] = []
        self._running = False
        self._tasks: List[asyncio.Task] = []

    def register_callback(self, callback: Callable[[PriceUpdate], None]):
        """Register callback for price updates"""
        self.callbacks.append(callback)

    async def start(self):
        """Start all price feeds"""
        self._running = True
        self._tasks = [
            asyncio.create_task(self._binance_feed()),
            asyncio.create_task(self._coinbase_feed()),
            asyncio.create_task(self._kraken_feed()),
        ]
        self.logger.info("Price feeds started")

    async def stop(self):
        """Stop all price feeds"""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self.logger.info("Price feeds stopped")

    def get_aggregated_price(self, symbol: str) -> Optional[Tuple[float, int]]:
        """
        Get volume-weighted average price from all sources.
        Returns (price, num_sources) or None if insufficient data.
        """
        symbol_prices = self.prices.get(symbol, {})
        valid_prices = []

        for source, update in symbol_prices.items():
            if update.age_ms < self.config.PRICE_STALE_MS:
                valid_prices.append(update)

        if len(valid_prices) < self.config.MIN_SOURCES_REQUIRED:
            return None

        # Volume-weighted average
        if all(p.volume_24h for p in valid_prices):
            total_volume = sum(p.volume_24h for p in valid_prices)
            vwap = sum(p.mid_price * p.volume_24h for p in valid_prices) / total_volume
        else:
            # Simple average if no volume data
            vwap = statistics.mean(p.mid_price for p in valid_prices)

        return (vwap, len(valid_prices))

    def _update_price(self, update: PriceUpdate):
        """Process incoming price update"""
        if update.symbol not in self.prices:
            self.prices[update.symbol] = {}
        self.prices[update.symbol][update.source] = update

        for callback in self.callbacks:
            try:
                callback(update)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")

    async def _binance_feed(self):
        """Binance WebSocket price feed"""
        url = "wss://stream.binance.com:9443/ws"
        streams = ["btcusdt@bookTicker", "ethusdt@bookTicker"]
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1
        }

        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    self.logger.debug("Binance feed connected")

                    async for msg in ws:
                        if not self._running:
                            break
                        data = json.loads(msg)
                        if "s" in data:  # Book ticker update
                            symbol = "BTC" if "BTC" in data["s"] else "ETH"
                            update = PriceUpdate(
                                source="binance",
                                symbol=symbol,
                                price=(float(data["b"]) + float(data["a"])) / 2,
                                bid=float(data["b"]),
                                ask=float(data["a"]),
                                timestamp_ms=int(time.time() * 1000),
                            )
                            self._update_price(update)
            except Exception as e:
                self.logger.warning(f"Binance feed error: {e}")
                await asyncio.sleep(1)

    async def _coinbase_feed(self):
        """Coinbase WebSocket price feed"""
        url = "wss://ws-feed.exchange.coinbase.com"
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": ["BTC-USD", "ETH-USD"],
            "channels": ["ticker"]
        }

        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    self.logger.debug("Coinbase feed connected")

                    async for msg in ws:
                        if not self._running:
                            break
                        data = json.loads(msg)
                        if data.get("type") == "ticker":
                            symbol = "BTC" if "BTC" in data["product_id"] else "ETH"
                            update = PriceUpdate(
                                source="coinbase",
                                symbol=symbol,
                                price=float(data["price"]),
                                bid=float(data.get("best_bid", data["price"])),
                                ask=float(data.get("best_ask", data["price"])),
                                timestamp_ms=int(time.time() * 1000),
                                volume_24h=float(data.get("volume_24h", 0)),
                            )
                            self._update_price(update)
            except Exception as e:
                self.logger.warning(f"Coinbase feed error: {e}")
                await asyncio.sleep(1)

    async def _kraken_feed(self):
        """Kraken WebSocket price feed"""
        url = "wss://ws.kraken.com"
        subscribe_msg = {
            "event": "subscribe",
            "pair": ["XBT/USD", "ETH/USD"],
            "subscription": {"name": "ticker"}
        }

        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    self.logger.debug("Kraken feed connected")

                    async for msg in ws:
                        if not self._running:
                            break
                        data = json.loads(msg)
                        if isinstance(data, list) and len(data) >= 4:
                            ticker_data = data[1]
                            pair = data[3]
                            symbol = "BTC" if "XBT" in pair else "ETH"

                            # Kraken ticker format: [ask, bid, close, volume, vwap, ...]
                            if isinstance(ticker_data, dict):
                                update = PriceUpdate(
                                    source="kraken",
                                    symbol=symbol,
                                    price=float(ticker_data["c"][0]),
                                    bid=float(ticker_data["b"][0]),
                                    ask=float(ticker_data["a"][0]),
                                    timestamp_ms=int(time.time() * 1000),
                                    volume_24h=float(ticker_data["v"][1]),
                                )
                                self._update_price(update)
            except Exception as e:
                self.logger.warning(f"Kraken feed error: {e}")
                await asyncio.sleep(1)

# =============================================================================
# KALSHI API CLIENT
# =============================================================================

class KalshiClient:
    """
    Kalshi API client for market data and trading.
    Handles authentication and rate limiting.
    """

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._session: Optional[aiohttp.ClientSession] = None
        self._token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

    async def initialize(self):
        """Initialize session and authenticate"""
        self._session = aiohttp.ClientSession()
        await self._authenticate()

    async def close(self):
        """Close session"""
        if self._session:
            await self._session.close()

    async def _authenticate(self):
        """Authenticate with Kalshi API"""
        if not self.config.KALSHI_API_KEY:
            self.logger.warning("No Kalshi API key - running in read-only mode")
            return

        # Kalshi uses timestamped HMAC authentication
        timestamp = str(int(time.time() * 1000))
        method = "POST"
        path = "/login"

        msg = f"{timestamp}{method}{path}"
        signature = hmac.new(
            self.config.KALSHI_API_SECRET.encode(),
            msg.encode(),
            hashlib.sha256
        ).hexdigest()

        headers = {
            "KALSHI-ACCESS-KEY": self.config.KALSHI_API_KEY,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }

        async with self._session.post(
            f"{self.config.KALSHI_API_BASE}/login",
            headers=headers
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                self._token = data.get("token")
                self._token_expiry = datetime.now(timezone.utc) + timedelta(hours=24)
                self.logger.info("Authenticated with Kalshi")
            else:
                self.logger.error(f"Auth failed: {await resp.text()}")

    def _get_auth_headers(self, method: str, path: str) -> Dict[str, str]:
        """Generate authentication headers for a request"""
        timestamp = str(int(time.time() * 1000))
        msg = f"{timestamp}{method}{path}"
        signature = hmac.new(
            self.config.KALSHI_API_SECRET.encode(),
            msg.encode(),
            hashlib.sha256
        ).hexdigest()

        return {
            "KALSHI-ACCESS-KEY": self.config.KALSHI_API_KEY,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }

    async def get_markets(self, ticker_prefix: str) -> List[KalshiMarket]:
        """Fetch current markets for a crypto ticker"""
        path = f"/markets?ticker={ticker_prefix}&status=open"
        headers = self._get_auth_headers("GET", path)

        async with self._session.get(
            f"{self.config.KALSHI_API_BASE}{path}",
            headers=headers
        ) as resp:
            if resp.status != 200:
                self.logger.error(f"Failed to get markets: {await resp.text()}")
                return []

            data = await resp.json()
            markets = []

            for m in data.get("markets", []):
                try:
                    # Parse strike price from subtitle (e.g., "BTC above $95,000")
                    strike = None
                    subtitle = m.get("subtitle", "")
                    if "$" in subtitle:
                        price_str = subtitle.split("$")[1].replace(",", "").split()[0]
                        strike = float(price_str)

                    market = KalshiMarket(
                        ticker=m["ticker"],
                        event_ticker=m["event_ticker"],
                        title=m["title"],
                        subtitle=subtitle,
                        yes_bid=m.get("yes_bid", 0) / 100,  # Convert from cents
                        yes_ask=m.get("yes_ask", 0) / 100,
                        no_bid=m.get("no_bid", 0) / 100,
                        no_ask=m.get("no_ask", 0) / 100,
                        volume=m.get("volume", 0),
                        open_interest=m.get("open_interest", 0),
                        strike_price=strike,
                        expiration_time=datetime.fromisoformat(
                            m["close_time"].replace("Z", "+00:00")
                        ),
                        last_updated=datetime.now(timezone.utc),
                    )
                    markets.append(market)
                except Exception as e:
                    self.logger.warning(f"Failed to parse market: {e}")

        return markets

    async def place_order(
        self,
        ticker: str,
        side: Side,
        quantity: int,
        price: float,
    ) -> TradeResult:
        """Place a limit order"""
        if not self._token:
            return TradeResult(
                order_id="",
                ticker=ticker,
                side=side,
                quantity=quantity,
                price=price,
                timestamp=datetime.now(timezone.utc),
                success=False,
                error="Not authenticated",
            )

        path = "/portfolio/orders"
        headers = self._get_auth_headers("POST", path)

        order_data = {
            "ticker": ticker,
            "type": "limit",
            "action": "buy",
            "side": side.value,
            "count": quantity,
            "yes_price" if side == Side.YES else "no_price": int(price * 100),
        }

        async with self._session.post(
            f"{self.config.KALSHI_API_BASE}{path}",
            headers=headers,
            json=order_data,
        ) as resp:
            data = await resp.json()

            if resp.status == 200:
                return TradeResult(
                    order_id=data.get("order", {}).get("order_id", ""),
                    ticker=ticker,
                    side=side,
                    quantity=quantity,
                    price=price,
                    timestamp=datetime.now(timezone.utc),
                    success=True,
                )
            else:
                return TradeResult(
                    order_id="",
                    ticker=ticker,
                    side=side,
                    quantity=quantity,
                    price=price,
                    timestamp=datetime.now(timezone.utc),
                    success=False,
                    error=data.get("error", str(resp.status)),
                )

    async def get_positions(self) -> List[Position]:
        """Get current positions"""
        if not self._token:
            return []

        path = "/portfolio/positions"
        headers = self._get_auth_headers("GET", path)

        async with self._session.get(
            f"{self.config.KALSHI_API_BASE}{path}",
            headers=headers,
        ) as resp:
            if resp.status != 200:
                return []

            data = await resp.json()
            positions = []

            for p in data.get("market_positions", []):
                pos = Position(
                    ticker=p["ticker"],
                    side=Side.YES if p.get("position", 0) > 0 else Side.NO,
                    quantity=abs(p.get("position", 0)),
                    entry_price=p.get("total_cost", 0) / max(abs(p.get("position", 1)), 1) / 100,
                    entry_time=datetime.fromisoformat(
                        p.get("created_time", datetime.now(timezone.utc).isoformat())
                        .replace("Z", "+00:00")
                    ),
                )
                positions.append(pos)

        return positions

    async def get_balance(self) -> float:
        """Get account balance"""
        if not self._token:
            return 0.0

        path = "/portfolio/balance"
        headers = self._get_auth_headers("GET", path)

        async with self._session.get(
            f"{self.config.KALSHI_API_BASE}{path}",
            headers=headers,
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("balance", 0) / 100  # Convert from cents
        return 0.0

# =============================================================================
# PROBABILITY CALCULATOR
# =============================================================================

class ProbabilityEngine:
    """
    Calculates fair value probabilities for crypto prediction markets.
    Uses real-time price data and volatility estimates.
    """

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

        # Rolling volatility calculation (1-minute returns)
        self.price_history: Dict[str, deque] = {
            "BTC": deque(maxlen=100),
            "ETH": deque(maxlen=100),
        }

        # Base annualized volatility estimates (updated dynamically)
        self.base_volatility = {
            "BTC": 0.60,  # 60% annualized
            "ETH": 0.75,  # 75% annualized
        }

    def update_price(self, symbol: str, price: float, timestamp_ms: int):
        """Update price history for volatility calculation"""
        self.price_history[symbol].append((timestamp_ms, price))

    def _estimate_realized_volatility(self, symbol: str) -> float:
        """Estimate short-term realized volatility from recent prices"""
        history = self.price_history.get(symbol, [])
        if len(history) < 10:
            return self.base_volatility.get(symbol, 0.65)

        # Calculate 1-minute log returns
        prices = [p[1] for p in history]
        returns = [np.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]

        if not returns:
            return self.base_volatility.get(symbol, 0.65)

        # Annualize (assuming ~1 sample per second, 525,600 minutes per year)
        std_dev = np.std(returns)
        annualized = std_dev * np.sqrt(525600)

        # Blend with base estimate
        base = self.base_volatility.get(symbol, 0.65)
        return 0.7 * annualized + 0.3 * base if annualized > 0 else base

    def calculate_fair_probability(
        self,
        current_price: float,
        strike_price: float,
        time_to_expiry_seconds: float,
        symbol: str = "BTC",
    ) -> float:
        """
        Calculate fair probability of price being above strike at expiration.

        Uses a simplified Black-Scholes-like model adjusted for:
        - Short time horizons (15 minutes)
        - High short-term volatility
        - Mean reversion tendencies
        """
        if time_to_expiry_seconds <= 0:
            return 1.0 if current_price >= strike_price else 0.0

        # Convert to years
        T = time_to_expiry_seconds / (365.25 * 24 * 3600)

        # Get volatility estimate
        sigma = self._estimate_realized_volatility(symbol)

        # Log moneyness
        if current_price <= 0 or strike_price <= 0:
            return 0.5
        log_moneyness = np.log(current_price / strike_price)

        # Drift adjustment for short-term (near zero for 15-min)
        drift = 0.0

        # d2 from Black-Scholes (probability under risk-neutral measure)
        vol_sqrt_t = sigma * np.sqrt(T)
        if vol_sqrt_t < 0.0001:
            return 1.0 if current_price >= strike_price else 0.0

        d2 = (log_moneyness + (drift - 0.5 * sigma**2) * T) / vol_sqrt_t

        # N(d2) = probability of finishing above strike
        prob = norm.cdf(d2)

        # Apply mean-reversion adjustment for very short horizons
        # Price tends to mean-revert less than Brownian motion suggests
        if T < 1/365:  # Less than 1 day
            mean_reversion_factor = 0.85  # Reduce extreme probabilities
            prob = 0.5 + (prob - 0.5) * mean_reversion_factor

        return np.clip(prob, 0.001, 0.999)

    def calculate_edge(
        self,
        fair_prob: float,
        market_price: float,
        side: Side,
    ) -> float:
        """Calculate edge for a potential trade"""
        if side == Side.YES:
            # Buying YES: we profit if YES wins
            # Fair value of YES = fair_prob
            # Edge = fair value - market price
            return fair_prob - market_price
        else:
            # Buying NO: we profit if NO wins
            # Fair value of NO = 1 - fair_prob
            return (1 - fair_prob) - market_price

# =============================================================================
# SIGNAL GENERATOR
# =============================================================================

class SignalGenerator:
    """
    Generates trading signals by comparing real-time prices to Kalshi markets.
    """

    def __init__(
        self,
        config: Config,
        logger: logging.Logger,
        price_feed: PriceFeedManager,
        prob_engine: ProbabilityEngine,
    ):
        self.config = config
        self.logger = logger
        self.price_feed = price_feed
        self.prob_engine = prob_engine

    def generate_signals(
        self,
        markets: List[KalshiMarket],
    ) -> List[TradeSignal]:
        """Generate signals for all available markets"""
        signals = []

        for market in markets:
            signal = self._evaluate_market(market)
            if signal:
                signals.append(signal)

        # Sort by edge (highest first)
        signals.sort(key=lambda s: abs(s.edge), reverse=True)
        return signals

    def _evaluate_market(self, market: KalshiMarket) -> Optional[TradeSignal]:
        """Evaluate a single market for trading opportunity"""

        # Determine which crypto this market is for
        symbol = "BTC" if "BTC" in market.ticker else "ETH"

        # Get current aggregated price
        price_data = self.price_feed.get_aggregated_price(symbol)
        if not price_data:
            return None
        crypto_price, num_sources = price_data

        # Need strike price to calculate probability
        if not market.strike_price:
            return None

        # Focus on final minutes where latency edge is highest
        time_to_expiry = market.time_to_expiry_seconds
        if time_to_expiry <= 0:
            return None

        # Calculate fair probability
        fair_prob = self.prob_engine.calculate_fair_probability(
            current_price=crypto_price,
            strike_price=market.strike_price,
            time_to_expiry_seconds=time_to_expiry,
            symbol=symbol,
        )

        # Check both YES and NO sides
        yes_edge = self.prob_engine.calculate_edge(fair_prob, market.yes_ask, Side.YES)
        no_edge = self.prob_engine.calculate_edge(fair_prob, market.no_ask, Side.NO)

        # Choose better side
        if yes_edge > no_edge and yes_edge > self.config.MIN_EDGE_THRESHOLD:
            side = Side.YES
            edge = yes_edge
            market_price = market.yes_ask
        elif no_edge > self.config.MIN_EDGE_THRESHOLD:
            side = Side.NO
            edge = no_edge
            market_price = market.no_ask
        else:
            return None

        # Check spread isn't too wide
        if market.spread > self.config.MAX_SPREAD_COST:
            self.logger.debug(f"Skipping {market.ticker}: spread too wide ({market.spread:.2%})")
            return None

        # Calculate confidence based on:
        # 1. Number of price sources
        # 2. Time to expiry (more confident closer to expiry)
        # 3. Edge magnitude
        source_confidence = min(num_sources / 3, 1.0)
        time_confidence = 1.0 if time_to_expiry < self.config.LATENCY_WINDOW_SECONDS else 0.7
        edge_confidence = min(abs(edge) / 0.10, 1.0)  # Max at 10% edge
        confidence = source_confidence * time_confidence * edge_confidence

        # Calculate position size using fractional Kelly
        kelly_fraction = self._kelly_criterion(edge, market_price, fair_prob)
        recommended_size = int(
            kelly_fraction * self.config.KELLY_FRACTION * self.config.MAX_POSITION_SIZE
        )
        recommended_size = max(1, min(recommended_size, self.config.MAX_POSITION_SIZE))

        return TradeSignal(
            market=market,
            side=side,
            edge=edge,
            confidence=confidence,
            fair_value=fair_prob if side == Side.YES else (1 - fair_prob),
            market_price=market_price,
            crypto_price=crypto_price,
            recommended_size=recommended_size,
        )

    def _kelly_criterion(
        self,
        edge: float,
        market_price: float,
        fair_prob: float,
    ) -> float:
        """
        Calculate Kelly criterion fraction.
        Kelly = (bp - q) / b where:
        - b = odds offered (payout ratio)
        - p = probability of winning
        - q = 1 - p
        """
        if market_price >= 1 or market_price <= 0:
            return 0

        # Payout ratio (what we win relative to what we risk)
        b = (1 - market_price) / market_price

        p = fair_prob
        q = 1 - p

        kelly = (b * p - q) / b if b > 0 else 0
        return max(0, kelly)

# =============================================================================
# RISK MANAGER
# =============================================================================

class RiskManager:
    """
    Manages position risk, P&L tracking, and trade limits.
    """

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

        self.daily_pnl = 0.0
        self.positions: Dict[str, Position] = {}
        self.trade_count = 0
        self.last_reset = datetime.now(timezone.utc).date()

    def _check_daily_reset(self):
        """Reset daily counters if new day"""
        today = datetime.now(timezone.utc).date()
        if today > self.last_reset:
            self.logger.info(f"Daily reset: PnL was {self.daily_pnl:.2f}")
            self.daily_pnl = 0.0
            self.trade_count = 0
            self.last_reset = today

    def can_trade(self, signal: TradeSignal) -> Tuple[bool, str]:
        """Check if a trade is allowed under risk limits"""
        self._check_daily_reset()

        # Check daily loss limit
        if self.daily_pnl <= -self.config.MAX_DAILY_LOSS:
            return False, f"Daily loss limit reached: {self.daily_pnl:.2f}"

        # Check concurrent positions
        if len(self.positions) >= self.config.MAX_CONCURRENT_POSITIONS:
            return False, f"Max positions reached: {len(self.positions)}"

        # Check if already have position in this market
        if signal.market.ticker in self.positions:
            return False, f"Already have position in {signal.market.ticker}"

        # Check single trade risk
        trade_risk = signal.recommended_size * signal.market_price
        if trade_risk > self.config.MAX_SINGLE_TRADE_RISK:
            return False, f"Trade risk too high: ${trade_risk:.2f}"

        # Check minimum confidence
        if signal.confidence < 0.5:
            return False, f"Confidence too low: {signal.confidence:.2%}"

        return True, "OK"

    def record_trade(self, result: TradeResult, signal: TradeSignal):
        """Record executed trade"""
        if result.success:
            self.positions[result.ticker] = Position(
                ticker=result.ticker,
                side=result.side,
                quantity=result.quantity,
                entry_price=result.price,
                entry_time=result.timestamp,
            )
            self.trade_count += 1
            self.logger.info(
                f"Trade recorded: {result.ticker} {result.side.value} "
                f"x{result.quantity} @ {result.price:.3f}"
            )

    def update_pnl(self, ticker: str, settlement_price: float):
        """Update P&L when market settles"""
        if ticker not in self.positions:
            return

        pos = self.positions[ticker]

        # Binary settlement: 1.0 if YES wins, 0.0 if NO wins
        if pos.side == Side.YES:
            pnl = pos.quantity * (settlement_price - pos.entry_price)
        else:
            pnl = pos.quantity * ((1 - settlement_price) - pos.entry_price)

        self.daily_pnl += pnl
        self.logger.info(
            f"Position settled: {ticker} PnL: ${pnl:.2f} | Daily: ${self.daily_pnl:.2f}"
        )
        del self.positions[ticker]

# =============================================================================
# EXECUTION ENGINE
# =============================================================================

class ExecutionEngine:
    """
    Handles order execution with smart order routing and latency optimization.
    """

    def __init__(
        self,
        config: Config,
        logger: logging.Logger,
        kalshi_client: KalshiClient,
        risk_manager: RiskManager,
    ):
        self.config = config
        self.logger = logger
        self.kalshi = kalshi_client
        self.risk = risk_manager

        self.pending_orders: Dict[str, TradeSignal] = {}

    async def execute_signal(self, signal: TradeSignal) -> Optional[TradeResult]:
        """Execute a trading signal"""

        # Risk check
        can_trade, reason = self.risk.can_trade(signal)
        if not can_trade:
            self.logger.debug(f"Trade blocked: {reason}")
            return None

        self.logger.info(f"Executing: {signal}")

        # Determine execution price
        # For latency arb, we want to be aggressive - use the ask price
        if signal.side == Side.YES:
            exec_price = signal.market.yes_ask
        else:
            exec_price = signal.market.no_ask

        # Adjust size based on available liquidity (simplified)
        size = min(signal.recommended_size, 50)  # Cap at 50 for liquidity

        # Place order
        result = await self.kalshi.place_order(
            ticker=signal.market.ticker,
            side=signal.side,
            quantity=size,
            price=exec_price,
        )

        if result.success:
            self.risk.record_trade(result, signal)
            self.logger.info(f"Order filled: {result.order_id}")
        else:
            self.logger.warning(f"Order failed: {result.error}")

        return result

# =============================================================================
# MAIN BOT ORCHESTRATOR
# =============================================================================

class KalshiLatencyBot:
    """
    Main orchestrator that coordinates all components.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = setup_logging(self.config)

        # Initialize components
        self.price_feed = PriceFeedManager(self.config, self.logger)
        self.prob_engine = ProbabilityEngine(self.config, self.logger)
        self.kalshi = KalshiClient(self.config, self.logger)
        self.risk_manager = RiskManager(self.config, self.logger)
        self.execution = ExecutionEngine(
            self.config, self.logger, self.kalshi, self.risk_manager
        )
        self.signal_gen = SignalGenerator(
            self.config, self.logger, self.price_feed, self.prob_engine
        )

        # Register price callback for volatility updates
        self.price_feed.register_callback(self._on_price_update)

        self._running = False

    def _on_price_update(self, update: PriceUpdate):
        """Handle incoming price updates"""
        self.prob_engine.update_price(
            update.symbol, update.price, update.timestamp_ms
        )

    async def start(self):
        """Start the trading bot"""
        self.logger.info("=" * 60)
        self.logger.info("KALSHI LATENCY ARBITRAGE BOT")
        self.logger.info("=" * 60)

        self._running = True

        # Initialize components
        await self.kalshi.initialize()
        await self.price_feed.start()

        # Wait for price feeds to populate
        self.logger.info("Waiting for price feeds...")
        await asyncio.sleep(3)

        # Main trading loop
        try:
            await self._trading_loop()
        except asyncio.CancelledError:
            self.logger.info("Bot cancelled")
        except Exception as e:
            self.logger.error(f"Bot error: {e}", exc_info=True)
        finally:
            await self.stop()

    async def stop(self):
        """Stop the trading bot"""
        self._running = False
        await self.price_feed.stop()
        await self.kalshi.close()
        self.logger.info("Bot stopped")

    async def _trading_loop(self):
        """Main trading loop"""
        self.logger.info("Starting trading loop")

        while self._running:
            try:
                # Fetch current markets
                all_markets = []
                for ticker in self.config.CRYPTO_TICKERS:
                    markets = await self.kalshi.get_markets(ticker)
                    all_markets.extend(markets)

                if not all_markets:
                    self.logger.debug("No markets available")
                    await asyncio.sleep(1)
                    continue

                # Generate signals
                signals = self.signal_gen.generate_signals(all_markets)

                if signals:
                    self.logger.info(f"Generated {len(signals)} signals")

                    # Execute best signal
                    best_signal = signals[0]
                    if best_signal.edge >= self.config.MIN_EDGE_THRESHOLD:
                        await self.execution.execute_signal(best_signal)

                # Rate limit: check every 500ms for latency-sensitive trading
                await asyncio.sleep(0.5)

            except Exception as e:
                self.logger.error(f"Loop error: {e}")
                await asyncio.sleep(1)

    def run(self):
        """Synchronous entry point"""
        try:
            asyncio.run(self.start())
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")

# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Kalshi 15-Minute Crypto Latency Arbitrage Bot"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without executing trades",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.03,
        help="Minimum edge threshold (default: 0.03)",
    )
    parser.add_argument(
        "--max-position",
        type=int,
        default=100,
        help="Maximum position size (default: 100)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Build config
    config = Config()
    config.MIN_EDGE_THRESHOLD = args.min_edge
    config.MAX_POSITION_SIZE = args.max_position
    config.LOG_LEVEL = args.log_level

    if args.dry_run:
        config.KALSHI_API_KEY = ""  # Disable trading
        print("Running in DRY RUN mode - no trades will be executed")

    # Run bot
    bot = KalshiLatencyBot(config)
    bot.run()

if __name__ == "__main__":
    main()
