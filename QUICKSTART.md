# Quick Start Guide

## Installation

```bash
# Clone repository
git clone https://github.com/DGator86/Kalshi-Latency-Bot.git
cd Kalshi-Latency-Bot

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Configuration

```bash
# Copy example config
cp config.example.json config.json

# Edit with your API credentials
nano config.json
```

Minimum required configuration:
```json
{
  "kalshi": {
    "api_key": "your_api_key_here",
    "api_secret": "your_api_secret_here"
  }
}
```

## Running the Bot

### Simulation Mode (No API Credentials)
```bash
# Test without real trading
kalshi-bot live --config config.json
```

### Live Trading Mode
```bash
# After configuring API credentials
kalshi-bot live --config config.json
```

### Backtest Mode
```bash
# Run backtests
kalshi-bot backtest --capital 10000 --max-loss 1000
```

## Monitoring

### Dashboard
Open browser to http://localhost:8080 to view:
- Live prices from all exchanges
- Current positions and P&L
- Recent trades
- Bot status and uptime

### Logs
Check `kalshi_bot.log` for detailed logs

## Testing

```bash
# Run integration tests
python tests/integration_test.py

# Run example backtest
python examples/backtest_example.py
```

## Key Parameters

### Trading Settings (config.json)
- `edge_threshold`: Minimum edge to trade (default: 0.03 = 3%)
- `kelly_fraction`: Kelly sizing fraction (default: 0.25)
- `max_daily_loss`: Maximum daily loss in $ (default: 1000)
- `max_position_size`: Maximum contracts per trade (default: 100)

### Risk Management
The bot will automatically:
- Size positions using Kelly criterion
- Stop trading if daily loss exceeds limit
- Respect maximum position sizes
- Track and report all trades

## Troubleshooting

### WebSocket Connection Issues
- Check internet connection
- Verify exchange URLs in config
- Check firewall settings

### API Authentication Errors
- Verify API credentials
- Check API key permissions
- Ensure correct API endpoint

### No Signals Generated
- Check edge threshold (may be too high)
- Verify price feeds are working
- Confirm markets are active

## Safety Tips

1. **Start Small**: Begin with small position sizes
2. **Test First**: Always run in simulation mode first
3. **Monitor**: Watch the dashboard regularly
4. **Set Limits**: Use conservative daily loss limits
5. **Understand Risk**: Only trade with money you can afford to lose

## Support

For issues and questions, use the GitHub issue tracker:
https://github.com/DGator86/Kalshi-Latency-Bot/issues

## Documentation

- [README.md](README.md) - Full documentation
- [IMPLEMENTATION.md](IMPLEMENTATION.md) - Technical details
- [examples/](examples/) - Usage examples
