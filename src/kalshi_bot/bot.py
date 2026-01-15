"""Main trading bot application."""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Optional

from .api import KalshiClient
from .config import Config
from .monitoring import MonitoringDashboard
from .trading import RiskManager, SignalGenerator
from .websockets import PriceAggregator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('kalshi_bot.log')
    ]
)

logger = logging.getLogger(__name__)


class TradingBot:
    """Main trading bot orchestrator."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = Config(config_path)
        self.running = False
        
        # Initialize components
        self.price_aggregator: Optional[PriceAggregator] = None
        self.signal_generator: Optional[SignalGenerator] = None
        self.risk_manager: Optional[RiskManager] = None
        self.kalshi_client: Optional[KalshiClient] = None
        self.dashboard: Optional[MonitoringDashboard] = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing trading bot...")
        
        # Get configuration
        symbols = self.config.get('trading.target_markets', ['BTC', 'ETH'])
        edge_threshold = self.config.get('trading.edge_threshold', 0.03)
        kelly_fraction = self.config.get('trading.kelly_fraction', 0.25)
        max_daily_loss = self.config.get('trading.max_daily_loss', 1000)
        max_position_size = self.config.get('trading.max_position_size', 100)
        dashboard_port = self.config.get('monitoring.dashboard_port', 8080)
        
        # Initialize price aggregator
        self.price_aggregator = PriceAggregator(symbols)
        
        # Initialize signal generator
        self.signal_generator = SignalGenerator(edge_threshold=edge_threshold)
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            max_daily_loss=max_daily_loss,
            max_position_size=max_position_size,
            kelly_fraction=kelly_fraction
        )
        
        # Initialize Kalshi client
        api_base = self.config.get('kalshi.api_base')
        api_key = self.config.get('kalshi.api_key')
        api_secret = self.config.get('kalshi.api_secret')
        
        if not api_key or not api_secret:
            logger.warning("Kalshi API credentials not configured - running in simulation mode")
            self.kalshi_client = None
        else:
            self.kalshi_client = KalshiClient(api_base, api_key, api_secret)
            await self.kalshi_client.__aenter__()
            
        # Initialize dashboard
        self.dashboard = MonitoringDashboard(port=dashboard_port)
        
        logger.info("Initialization complete")
        
    async def trading_loop(self):
        """Main trading loop."""
        logger.info("Starting trading loop...")
        
        while self.running:
            try:
                # Get current prices
                aggregated_prices = []
                for symbol in self.config.get('trading.target_markets', ['BTC', 'ETH']):
                    agg_price = self.price_aggregator.get_aggregated_price(symbol)
                    if agg_price:
                        aggregated_prices.append(agg_price)
                        
                        # Update dashboard
                        if self.dashboard:
                            latest_prices = self.dashboard.stats.get('latest_prices', {})
                            latest_prices[symbol] = {
                                'vwap': agg_price.vwap,
                                'prices': agg_price.prices,
                                'timestamp': agg_price.timestamp.isoformat()
                            }
                            self.dashboard.update_stats(latest_prices=latest_prices)
                            
                # Get Kalshi market data
                if self.kalshi_client:
                    market_data_list = await self.kalshi_client.get_markets(
                        self.config.get('trading.target_markets', ['BTC', 'ETH'])
                    )
                else:
                    market_data_list = []
                    
                # Generate signals
                if aggregated_prices and market_data_list:
                    # Assume 15-minute markets
                    time_to_expiry = 15.0
                    
                    signals = self.signal_generator.get_signals(
                        aggregated_prices,
                        market_data_list,
                        time_to_expiry
                    )
                    
                    # Update dashboard
                    if self.dashboard:
                        current_signals = self.dashboard.stats.get('signals_generated', 0)
                        self.dashboard.update_stats(
                            signals_generated=current_signals + len(signals)
                        )
                        
                    # Execute signals
                    for signal in signals:
                        await self.execute_signal(signal)
                        
                # Update dashboard with risk metrics
                if self.dashboard and self.risk_manager:
                    position_summary = self.risk_manager.get_position_summary()
                    self.dashboard.update_stats(
                        daily_pnl=position_summary['daily_pnl'],
                        daily_trades=position_summary['daily_trades'],
                        positions=position_summary['positions']
                    )
                    
                # Wait before next iteration
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                await asyncio.sleep(5)
                
    async def execute_signal(self, signal):
        """Execute a trading signal."""
        try:
            logger.info(f"Processing signal: {signal.symbol} {signal.side.value} edge={signal.edge:.3%}")
            
            # Get current balance
            if self.kalshi_client:
                balance = await self.kalshi_client.get_balance()
            else:
                balance = 10000  # Simulation mode
                
            # Calculate risk metrics
            risk_metrics = self.risk_manager.calculate_risk_metrics(signal, balance)
            
            # Check if we can trade
            if not self.risk_manager.can_trade(signal, risk_metrics):
                logger.info("Signal rejected by risk manager")
                return
                
            # Place order
            if self.kalshi_client:
                # Real trading
                # Find matching market (simplified - would need better matching logic)
                markets = await self.kalshi_client.get_markets([signal.symbol])
                if not markets:
                    logger.warning(f"No markets found for {signal.symbol}")
                    return
                    
                market = markets[0]
                order = await self.kalshi_client.place_order(
                    market.market_id,
                    signal.side,
                    risk_metrics.max_position,
                    signal.kalshi_price
                )
                
                if order:
                    logger.info(f"Order placed: {order.order_id}")
                    
                    # Update dashboard
                    if self.dashboard:
                        recent_trades = self.dashboard.stats.get('recent_trades', [])
                        recent_trades.insert(0, {
                            'timestamp': datetime.now().isoformat(),
                            'symbol': signal.symbol,
                            'side': signal.side.value,
                            'quantity': risk_metrics.max_position,
                            'price': signal.kalshi_price,
                            'pnl': 0.0  # Will be updated when position closes
                        })
                        # Keep only last 10 trades
                        recent_trades = recent_trades[:10]
                        self.dashboard.update_stats(recent_trades=recent_trades)
                        
                        signals_executed = self.dashboard.stats.get('signals_executed', 0)
                        self.dashboard.update_stats(signals_executed=signals_executed + 1)
            else:
                # Simulation mode
                logger.info(
                    f"SIMULATION: Would place order for {risk_metrics.max_position} "
                    f"contracts @ ${signal.kalshi_price:.2f}"
                )
                
        except Exception as e:
            logger.error(f"Error executing signal: {e}", exc_info=True)
            
    async def run(self):
        """Run the trading bot."""
        try:
            await self.initialize()
            
            self.running = True
            
            # Start components
            await self.price_aggregator.start(self.config.config)
            
            if self.dashboard:
                await self.dashboard.start()
                
            # Start trading loop
            await self.trading_loop()
            
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            await self.shutdown()
            
    async def shutdown(self):
        """Shutdown the bot gracefully."""
        logger.info("Shutting down trading bot...")
        
        self.running = False
        
        if self.price_aggregator:
            await self.price_aggregator.stop()
            
        if self.kalshi_client:
            await self.kalshi_client.__aexit__(None, None, None)
            
        if self.dashboard:
            await self.dashboard.stop()
            
        logger.info("Shutdown complete")


def main():
    """Main entry point."""
    import sys
    
    config_path = "config.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        
    bot = TradingBot(config_path)
    
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
