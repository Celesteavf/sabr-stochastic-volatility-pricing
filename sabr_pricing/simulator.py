"""Vectorised Monte Carlo simulation of SABR paths via Euler-Maruyama."""

import numpy as np


class SABRSimulator:
    """Generates Monte Carlo price paths under the log-normal SABR model (β=1).

    The discretisation uses exact log-normal increments (Itô's lemma applied to
    ln S_t and ln α_t) to eliminate the discretisation bias of the naive Euler scheme.

    Attributes:
        alpha: Initial volatility α₀.
        rho: Spot-vol correlation ρ.
        nu: Vol-of-vol ν.
    """

    def __init__(self, alpha: float, rho: float, nu: float) -> None:
        self.alpha = alpha
        self.rho = rho
        self.nu = nu

    def simulate(
        self,
        S0: float,
        r: float,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """Simulate SABR price trajectories.

        Correlated Brownian increments are constructed via Cholesky decomposition:
            Z₂ = ρ Z₁ + √(1−ρ²) Z̃₂,  (Z₁, Z̃₂) i.i.d. N(0,1)

        The volatility process is simulated exactly:
            α_{t+Δt} = α_t · exp(−½ν²Δt + ν√Δt · Z₂)

        The price process is simulated as:
            S_{t+Δt} = S_t · exp((r − ½α_t²)Δt + α_t√Δt · Z₁)

        Args:
            S0: Initial spot price.
            r: Risk-free rate (annualised).
            T: Horizon in years.
            n_paths: Number of Monte Carlo scenarios.
            n_steps: Number of time steps.
            seed: Optional random seed for reproducibility.

        Returns:
            Array of shape (n_paths, n_steps + 1) containing spot price paths.
            Column 0 is S0 for all paths.
        """
        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps

        Z1 = np.random.normal(0.0, 1.0, (n_paths, n_steps))
        Z2 = self.rho * Z1 + np.sqrt(1.0 - self.rho**2) * np.random.normal(0.0, 1.0, (n_paths, n_steps))

        # Volatility paths: α_t (shape: n_paths × n_steps+1)
        log_alpha_inc = -0.5 * self.nu**2 * dt + self.nu * np.sqrt(dt) * Z2
        at = self.alpha * np.exp(np.cumsum(log_alpha_inc, axis=1))
        at_paths = np.hstack([np.full((n_paths, 1), self.alpha), at])

        # Price paths: S_t using α at the left endpoint of each interval
        log_S_inc = (r - 0.5 * at_paths[:, :-1] ** 2) * dt + at_paths[:, :-1] * np.sqrt(dt) * Z1
        St = S0 * np.exp(np.cumsum(log_S_inc, axis=1))
        St_paths = np.hstack([np.full((n_paths, 1), S0), St])

        return St_paths
