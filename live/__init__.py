"""Live trading package for the Bank Nifty Supertrend + EMA strategy on Kotak Neo.

Reuses the backtest SignalEngine/StateManager unchanged and executes signals as
2-leg synthetic-future option orders. Paper mode is the default; live mode is a
single config flag (`live.mode: live` in config/strategy.yaml).
"""
