# Kalshi Latency Bot

Python async trading bot for Kalshi 15-minute crypto markets. Exploits latency between real-time BTC/ETH WebSocket feeds (Binance, Coinbase, Kraken) and Kalshi market maker quotes.

## Features

- **Price Aggregator with VWAP**: Real-time price aggregation from multiple exchanges (Binance, Coinbase, Kraken) with Volume Weighted Average Price calculation
- **Black-Scholes Probability Engine**: Short-term volatility modeling and probability calculations for binary options
- **Signal Generator**: Identifies trading opportunities with 3%+ edge threshold
- **Risk Manager**: Kelly criterion position sizing with daily loss limits
- **Kalshi API Client**: Full integration with HMAC authentication
- **Backtester**: Historical strategy validation and performance metrics
- **Live Monitoring Dashboard**: Real-time web dashboard for monitoring bot performance

## Technology Stack

- `asyncio` - Asynchronous event loop
- `aiohttp` - Async HTTP client/server
- `websockets` - WebSocket client library
- `numpy` - Numerical computing
- `scipy` - Scientific computing and statistics

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/DGator86/Kalshi-Latency-Bot.git
cd Kalshi-Latency-Bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

4. Configure the bot:
```bash
cp config.example.json config.json
# Edit config.json with your API credentials
```

## Configuration

Edit `config.json` with your settings:

```json
{
  "kalshi": {
    "api_base": "https://trading-api.kalshi.com",
    "api_key": "your_api_key_here",
    "api_secret": "your_api_secret_here"
  },
  "trading": {
    "edge_threshold": 0.03,
    "kelly_fraction": 0.25,
    "max_daily_loss": 1000,
    "max_position_size": 100,
    "target_markets": ["BTC", "ETH"]
  },
  "exchanges": {
    "binance": {"enabled": true},
    "coinbase": {"enabled": true},
    "kraken": {"enabled": true}
  },
  "monitoring": {
    "dashboard_port": 8080,
    "log_level": "INFO"
  }
}
```

### Configuration Parameters

- **kalshi.api_key / api_secret**: Your Kalshi API credentials
- **trading.edge_threshold**: Minimum edge required to generate a signal (default: 3%)
- **trading.kelly_fraction**: Fraction of Kelly criterion to use (default: 0.25 for quarter-Kelly)
- **trading.max_daily_loss**: Maximum daily loss limit in dollars
- **trading.max_position_size**: Maximum contracts per position
- **trading.target_markets**: List of symbols to trade (BTC, ETH)

## Usage

### Live Trading

Run the bot in live trading mode:

```bash
kalshi-bot live --config config.json
```

Or using Python directly:

```bash
python -m kalshi_bot live --config config.json
```

### Backtest Mode

Run backtests on historical data:

```bash
kalshi-bot backtest --capital 10000 --max-loss 1000
```

### Monitoring Dashboard

When the bot is running, access the live monitoring dashboard at:

```
http://localhost:8080
```

The dashboard displays:
- Current bot status and uptime
- Daily P&L and trade count
- Real-time price feeds from all exchanges
- Recent trades and positions
- Risk metrics

## Architecture

### Components

1. **WebSocket Clients** (`src/kalshi_bot/websockets/`)
   - Exchange-specific WebSocket clients (Binance, Coinbase, Kraken)
   - Async price feed aggregation
   - VWAP calculation across exchanges

2. **Probability Engine** (`src/kalshi_bot/trading/probability.py`)
   - Black-Scholes probability calculations
   - Historical volatility estimation
   - Fair value computation for binary options

3. **Signal Generator** (`src/kalshi_bot/trading/signals.py`)
   - Edge calculation (fair value vs market price)
   - Signal filtering by edge threshold
   - Confidence scoring

4. **Risk Manager** (`src/kalshi_bot/trading/risk.py`)
   - Kelly criterion position sizing
   - Daily loss limits
   - Position tracking and P&L calculation

5. **Kalshi API Client** (`src/kalshi_bot/api/`)
   - HMAC authentication
   - Market data retrieval
   - Order placement and management

6. **Backtester** (`src/kalshi_bot/backtester/`)
   - Historical strategy validation
   - Performance metrics (Sharpe ratio, max drawdown, win rate)
   - Trade simulation

7. **Monitoring Dashboard** (`src/kalshi_bot/monitoring/`)
   - Real-time web interface
   - Live statistics and charts
   - Health monitoring

### Trading Logic

1. **Price Aggregation**: WebSocket clients stream real-time prices from Binance, Coinbase, and Kraken
2. **VWAP Calculation**: Compute volume-weighted average price across exchanges
3. **Volatility Estimation**: Calculate historical volatility from price history
4. **Probability Calculation**: Use Black-Scholes to estimate probability of price movements
5. **Edge Detection**: Compare fair value to Kalshi market prices
6. **Signal Generation**: Generate signals when edge exceeds threshold (default 3%)
7. **Risk Management**: Calculate position size using Kelly criterion with safety factors
8. **Order Execution**: Place orders on Kalshi when all risk checks pass

## Risk Management

The bot implements multiple layers of risk management:

- **Kelly Criterion**: Optimal position sizing based on edge and probability
- **Fractional Kelly**: Uses 25% of full Kelly by default for safety
- **Position Limits**: Maximum position size per trade
- **Daily Loss Limits**: Automatic shutdown if daily loss exceeds threshold
- **Diversification**: Trades multiple symbols independently

## Development

### Project Structure

```
Kalshi-Latency-Bot/
├── src/
│   └── kalshi_bot/
│       ├── __init__.py
│       ├── bot.py              # Main bot orchestrator
│       ├── config.py           # Configuration management
│       ├── __main__.py         # CLI entry point
│       ├── api/                # Kalshi API client
│       ├── backtester/         # Backtesting engine
│       ├── models/             # Data models
│       ├── monitoring/         # Dashboard and monitoring
│       ├── trading/            # Trading logic
│       └── websockets/         # Exchange WebSocket clients
├── config.example.json         # Example configuration
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
└── README.md
```

### Testing

The bot includes simulation mode for testing without real API credentials:

```bash
# Run without API credentials for testing
kalshi-bot live --config config.json
```

When API credentials are not configured, the bot runs in simulation mode and logs what trades it would make without executing them.

## Disclaimer

**IMPORTANT**: This software is for educational purposes only. Trading cryptocurrencies and derivatives involves substantial risk of loss. The bot makes automated trading decisions based on algorithms that may not be suitable for your financial situation.

- No guarantees of profitability
- Past performance does not guarantee future results
- Use at your own risk
- Start with small position sizes
- Always test in simulation mode first
- Never risk more than you can afford to lose

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please use the GitHub issue tracker.