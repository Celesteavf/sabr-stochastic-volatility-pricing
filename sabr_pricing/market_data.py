"""Market data extraction and implied volatility computation via Newton-Raphson."""

import numpy as np
from scipy.stats import norm
import yfinance as yf
import pandas as pd


class MarketData:
    """Handles market data extraction from yfinance and implied volatility computation.

    Attributes:
        ticker: Equity ticker symbol (e.g. 'AAPL').
        rate: Risk-free rate used for implied vol inversion.
        S0: Spot price, set after calling fetch_spot().
        T: Time-to-maturity in years, set after calling load_chain().
        expiry: Selected expiry date string, set after calling load_chain().
        df_calls: Filtered call option chain with implied vols.
    """

    def __init__(self, ticker: str, rate: float = 0.045) -> None:
        self.ticker = ticker
        self.rate = rate
        self._yf_ticker = yf.Ticker(ticker)
        self.S0: float | None = None
        self.T: float | None = None
        self.expiry: str | None = None
        self.df_calls: pd.DataFrame | None = None

    def fetch_spot(self) -> float:
        """Retrieve the latest closing price for the ticker.

        Returns:
            Spot price S0.
        """
        history = self._yf_ticker.history(period="1d")
        self.S0 = float(history["Close"].iloc[-1])
        return self.S0

    def get_expiries(self) -> tuple:
        """Return the available option expiry dates from the exchange.

        Returns:
            Tuple of date strings in 'YYYY-MM-DD' format.
        """
        return self._yf_ticker.options

    def load_chain(
        self,
        expiry_index: int = 2,
        spot_range: float = 0.20,
        min_price: float = 0.1,
    ) -> pd.DataFrame:
        """Load, filter, and enrich the call option chain with implied volatilities.

        Args:
            expiry_index: Index into the available expiry list.
            spot_range: Fraction around spot to keep (e.g. 0.20 → ±20%).
            min_price: Minimum last price to exclude illiquid strikes.

        Returns:
            DataFrame with columns ['strike', 'lastPrice', 'vol_implicite'].
        """
        if self.S0 is None:
            self.fetch_spot()

        expiries = self.get_expiries()
        self.expiry = expiries[expiry_index]

        expiry_dt = pd.to_datetime(self.expiry)
        self.T = (expiry_dt - pd.Timestamp.now()).days / 365.25

        chain = self._yf_ticker.option_chain(self.expiry)
        df = chain.calls[["strike", "lastPrice"]].copy()

        lo = (1.0 - spot_range) * self.S0
        hi = (1.0 + spot_range) * self.S0
        df = df[(df["strike"] >= lo) & (df["strike"] <= hi) & (df["lastPrice"] > min_price)]

        df["vol_implicite"] = df.apply(
            lambda row: self._implied_vol(row["lastPrice"], self.S0, row["strike"], self.T, self.rate),
            axis=1,
        )
        df = df.dropna(subset=["vol_implicite"])
        self.df_calls = df.reset_index(drop=True)
        return self.df_calls

    @staticmethod
    def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes European call price."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def _bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes vega: ∂C/∂σ = S √T N'(d1)."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.sqrt(T) * norm.pdf(d1)

    @classmethod
    def _implied_vol(
        cls,
        C_market: float,
        S: float,
        K: float,
        T: float,
        r: float,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> float:
        """Newton-Raphson inversion of Black-Scholes to recover implied volatility.

        Returns nan if the algorithm does not converge or vega collapses.
        """
        sigma = 0.2
        for _ in range(max_iter):
            diff = cls._bs_call(S, K, T, r, sigma) - C_market
            if abs(diff) < tol:
                return sigma
            vega = cls._bs_vega(S, K, T, r, sigma)
            if abs(vega) < 1e-8:
                return np.nan
            sigma -= diff / vega
        return np.nan
