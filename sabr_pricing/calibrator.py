"""SABR model calibration using Hagan's closed-form approximation and L-BFGS-B."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize, OptimizeResult


@dataclass
class SABRParams:
    """Calibrated SABR parameters.

    Attributes:
        alpha: Initial volatility α₀ (ATM forward vol).
        rho: Spot-vol correlation ρ ∈ (-1, 1).
        nu: Vol-of-vol ν > 0.
    """

    alpha: float
    rho: float
    nu: float

    def __repr__(self) -> str:
        return f"SABRParams(alpha={self.alpha:.4f}, rho={self.rho:.4f}, nu={self.nu:.4f})"


class SABRCalibrator:
    """Calibrates SABR parameters by fitting Hagan's approximation to market implied vols.

    Uses the log-normal (β=1) variant of the Hagan (2002) formula and minimises the
    sum of squared errors between model and market vols via L-BFGS-B.

    Attributes:
        params: Calibrated SABRParams, available after calling calibrate().
        result: Raw scipy OptimizeResult from the last calibration run.
    """

    def __init__(self) -> None:
        self.params: Optional[SABRParams] = None
        self.result: Optional[OptimizeResult] = None

    @staticmethod
    def sabr_vol(
        alpha: float,
        rho: float,
        nu: float,
        S0: float,
        r: float,
        T: float,
        K: float,
    ) -> float:
        """Hagan (2002) approximation for the Black implied vol of a log-normal SABR model.

        Handles the ATM case separately to avoid the 0/0 indeterminate form.

        Args:
            alpha: Initial vol α₀.
            rho: Correlation ρ.
            nu: Vol-of-vol ν.
            S0: Current spot price.
            r: Risk-free rate.
            T: Time-to-maturity in years.
            K: Strike price.

        Returns:
            Implied Black volatility.
        """
        F0 = S0 * np.exp(r * T)
        time_correction = 1.0 + ((2.0 - 3.0 * rho**2) * nu**2 / 24.0 + rho * alpha * nu / 4.0) * T

        if abs(K - F0) < 1e-3:
            return alpha * time_correction

        eta = (nu / alpha) * np.log(F0 / K)
        x = np.log((np.sqrt(1.0 - 2.0 * rho * eta + eta**2) + eta - rho) / (1.0 - rho))
        return alpha * (eta / x) * time_correction

    def _objective(
        self,
        params: np.ndarray,
        strikes: np.ndarray,
        market_vols: np.ndarray,
        S0: float,
        r: float,
        T: float,
    ) -> float:
        """Sum of squared errors between SABR model vols and market implied vols."""
        alpha, rho, nu = params
        model_vols = np.array([self.sabr_vol(alpha, rho, nu, S0, r, T, K) for K in strikes])
        return float(np.sum((model_vols - market_vols) ** 2))

    def calibrate(
        self,
        strikes: np.ndarray,
        market_vols: np.ndarray,
        S0: float,
        r: float,
        T: float,
        initial_guess: Optional[list] = None,
    ) -> SABRParams:
        """Calibrate SABR parameters to market implied vols.

        Args:
            strikes: Array of strike prices.
            market_vols: Array of market implied volatilities (same length as strikes).
            S0: Spot price.
            r: Risk-free rate.
            T: Time-to-maturity in years.
            initial_guess: Starting point [α₀, ρ, ν] for the optimiser.

        Returns:
            Calibrated SABRParams.
        """
        if initial_guess is None:
            initial_guess = [0.2, 0.0, 0.3]

        bounds = [(1e-3, None), (-0.999, 0.999), (1e-3, None)]

        self.result = minimize(
            self._objective,
            x0=initial_guess,
            args=(strikes, market_vols, S0, r, T),
            bounds=bounds,
            method="L-BFGS-B",
        )

        alpha, rho, nu = self.result.x
        self.params = SABRParams(alpha=alpha, rho=rho, nu=nu)
        return self.params

    def smile(self, strikes: np.ndarray, S0: float, r: float, T: float) -> np.ndarray:
        """Compute the SABR implied vol smile over an array of strikes.

        Requires calibrate() to have been called first.

        Args:
            strikes: Array of strike prices.
            S0: Spot price.
            r: Risk-free rate.
            T: Time-to-maturity in years.

        Returns:
            Array of SABR implied volatilities.
        """
        if self.params is None:
            raise RuntimeError("Model not calibrated. Call calibrate() first.")
        p = self.params
        return np.array([self.sabr_vol(p.alpha, p.rho, p.nu, S0, r, T, K) for K in strikes])
