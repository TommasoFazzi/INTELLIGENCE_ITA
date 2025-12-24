"""
External integrations module for INTELLIGENCE_ITA.

Currently includes:
- market_data: Yahoo Finance integration for stock market data
"""

from .market_data import MarketDataService

__all__ = ['MarketDataService']
