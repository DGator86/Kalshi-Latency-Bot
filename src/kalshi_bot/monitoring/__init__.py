"""Live monitoring dashboard using aiohttp web server."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Optional
from aiohttp import web

logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """Live monitoring dashboard for the trading bot."""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        
        # Monitoring data
        self.stats = {
            'status': 'stopped',
            'uptime_seconds': 0,
            'start_time': None,
            'positions': {},
            'daily_pnl': 0.0,
            'daily_trades': 0,
            'signals_generated': 0,
            'signals_executed': 0,
            'latest_prices': {},
            'risk_metrics': {},
            'recent_trades': [],
            'errors': []
        }
        
        # Setup routes
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/api/stats', self.handle_stats)
        self.app.router.add_get('/api/health', self.handle_health)
        
    async def handle_index(self, request):
        """Serve main dashboard page."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Kalshi Latency Bot - Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        .status {
            font-size: 18px;
            font-weight: bold;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .status.running { background-color: #4CAF50; color: white; }
        .status.stopped { background-color: #f44336; color: white; }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #4CAF50;
        }
        .metric-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .section {
            margin: 20px 0;
        }
        .section h2 {
            color: #555;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f0f0f0;
            font-weight: bold;
        }
        .positive { color: #4CAF50; }
        .negative { color: #f44336; }
        #refresh {
            float: right;
            padding: 10px 20px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        #refresh:hover {
            background: #45a049;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Kalshi Latency Bot Dashboard</h1>
        <button id="refresh" onclick="loadStats()">Refresh</button>
        
        <div id="status" class="status">Loading...</div>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Daily PnL</div>
                <div class="metric-value" id="daily-pnl">$0.00</div>
            </div>
            <div class="metric">
                <div class="metric-label">Daily Trades</div>
                <div class="metric-value" id="daily-trades">0</div>
            </div>
            <div class="metric">
                <div class="metric-label">Signals Generated</div>
                <div class="metric-value" id="signals-generated">0</div>
            </div>
            <div class="metric">
                <div class="metric-label">Uptime</div>
                <div class="metric-value" id="uptime">0h 0m</div>
            </div>
        </div>
        
        <div class="section">
            <h2>Latest Prices</h2>
            <table id="prices-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>VWAP</th>
                        <th>Binance</th>
                        <th>Coinbase</th>
                        <th>Kraken</th>
                    </tr>
                </thead>
                <tbody id="prices-body">
                    <tr><td colspan="5">No data</td></tr>
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>Recent Trades</h2>
            <table id="trades-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Quantity</th>
                        <th>Price</th>
                        <th>PnL</th>
                    </tr>
                </thead>
                <tbody id="trades-body">
                    <tr><td colspan="6">No trades</td></tr>
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                
                // Update status
                const statusDiv = document.getElementById('status');
                statusDiv.textContent = `Status: ${data.status.toUpperCase()}`;
                statusDiv.className = `status ${data.status}`;
                
                // Update metrics
                document.getElementById('daily-pnl').textContent = `$${data.daily_pnl.toFixed(2)}`;
                document.getElementById('daily-pnl').className = `metric-value ${data.daily_pnl >= 0 ? 'positive' : 'negative'}`;
                document.getElementById('daily-trades').textContent = data.daily_trades;
                document.getElementById('signals-generated').textContent = data.signals_generated;
                
                // Update uptime
                const hours = Math.floor(data.uptime_seconds / 3600);
                const minutes = Math.floor((data.uptime_seconds % 3600) / 60);
                document.getElementById('uptime').textContent = `${hours}h ${minutes}m`;
                
                // Update prices
                const pricesBody = document.getElementById('prices-body');
                if (Object.keys(data.latest_prices).length > 0) {
                    pricesBody.innerHTML = Object.entries(data.latest_prices).map(([symbol, prices]) => `
                        <tr>
                            <td>${symbol}</td>
                            <td>$${prices.vwap.toFixed(2)}</td>
                            <td>$${(prices.prices.binance || 0).toFixed(2)}</td>
                            <td>$${(prices.prices.coinbase || 0).toFixed(2)}</td>
                            <td>$${(prices.prices.kraken || 0).toFixed(2)}</td>
                        </tr>
                    `).join('');
                } else {
                    pricesBody.innerHTML = '<tr><td colspan="5">No data</td></tr>';
                }
                
                // Update trades
                const tradesBody = document.getElementById('trades-body');
                if (data.recent_trades.length > 0) {
                    tradesBody.innerHTML = data.recent_trades.map(trade => `
                        <tr>
                            <td>${new Date(trade.timestamp).toLocaleTimeString()}</td>
                            <td>${trade.symbol}</td>
                            <td>${trade.side}</td>
                            <td>${trade.quantity}</td>
                            <td>$${trade.price.toFixed(2)}</td>
                            <td class="${trade.pnl >= 0 ? 'positive' : 'negative'}">$${trade.pnl.toFixed(2)}</td>
                        </tr>
                    `).join('');
                } else {
                    tradesBody.innerHTML = '<tr><td colspan="6">No trades</td></tr>';
                }
                
            } catch (error) {
                console.error('Failed to load stats:', error);
            }
        }
        
        // Load stats on page load
        loadStats();
        
        // Auto-refresh every 5 seconds
        setInterval(loadStats, 5000);
    </script>
</body>
</html>
        """
        return web.Response(text=html, content_type='text/html')
        
    async def handle_stats(self, request):
        """Serve statistics JSON."""
        return web.json_response(self.stats)
        
    async def handle_health(self, request):
        """Health check endpoint."""
        return web.json_response({'status': 'healthy', 'timestamp': datetime.now().isoformat()})
        
    def update_stats(self, **kwargs):
        """Update dashboard statistics."""
        self.stats.update(kwargs)
        
        # Update uptime
        if self.stats['start_time']:
            self.stats['uptime_seconds'] = (
                datetime.now() - self.stats['start_time']
            ).total_seconds()
            
    async def start(self):
        """Start the dashboard server."""
        self.stats['status'] = 'running'
        self.stats['start_time'] = datetime.now()
        
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, 'localhost', self.port)
        await self.site.start()
        
        logger.info(f"Dashboard started at http://localhost:{self.port}")
        
    async def stop(self):
        """Stop the dashboard server."""
        self.stats['status'] = 'stopped'
        
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
            
        logger.info("Dashboard stopped")
