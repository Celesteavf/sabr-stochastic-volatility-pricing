"""Option pricing and Greek computation via Monte Carlo simulation."""

import numpy as np
from .simulator import SABRSimulator


class OptionPricer:
    """Abstract base class for Monte Carlo option pricers under the SABR model.

    Subclasses implement price() to define the option payoff. Greeks are computed
    here via central finite differences (bump-and-reprice), using the same random
    seed for each bump to suppress Monte Carlo noise.

    Attributes:
        simulator: SABRSimulator used to generate price trajectories.
    """

    def __init__(self, simulator: SABRSimulator) -> None:
        self.simulator = simulator

    def price(self, St_paths: np.ndarray, K: float, r: float, T: float) -> float:
        """Compute the discounted Monte Carlo price. Must be overridden by subclasses."""
        raise NotImplementedError

    def greeks(
        self,
        S0: float,
        r: float,
        T: float,
        K: float,
        n_paths: int,
        n_steps: int,
        eps_S_pct: float = 0.01,
        eps_a_pct: float = 0.01,
        seed: int = 42,
    ) -> dict:
        """Compute Delta, Gamma and Vega via central finite differences.

        Each simulation uses the same seed so that the random noise cancels in the
        difference quotients (variance-reduction by common random numbers).

        Delta:  ΔC / ΔS   (central, order 1)
        Gamma:  Δ²C / ΔS² (central, order 2)
        Vega:   ΔC / Δα   (central with respect to α₀)

        Args:
            S0: Current spot price.
            r: Risk-free rate.
            T: Time-to-maturity in years.
            K: Strike price.
            n_paths: Number of Monte Carlo paths per bump.
            n_steps: Number of time steps per path.
            eps_S_pct: Spot bump size as a fraction of S0.
            eps_a_pct: Alpha bump size as a fraction of α₀.
            seed: Random seed shared across all bump simulations.

        Returns:
            Dict with keys 'delta', 'gamma', 'vega'.
        """
        eps_S = eps_S_pct * S0
        eps_a = eps_a_pct * self.simulator.alpha
        sim = self.simulator

        paths_base = sim.simulate(S0, r, T, n_paths, n_steps, seed=seed)
        C_base = self.price(paths_base, K, r, T)

        paths_S_up = sim.simulate(S0 + eps_S, r, T, n_paths, n_steps, seed=seed)
        C_S_up = self.price(paths_S_up, K, r, T)

        paths_S_down = sim.simulate(S0 - eps_S, r, T, n_paths, n_steps, seed=seed)
        C_S_down = self.price(paths_S_down, K, r, T)

        sim_a_up = SABRSimulator(sim.alpha + eps_a, sim.rho, sim.nu)
        paths_a_up = sim_a_up.simulate(S0, r, T, n_paths, n_steps, seed=seed)
        C_a_up = self.price(paths_a_up, K, r, T)

        sim_a_down = SABRSimulator(sim.alpha - eps_a, sim.rho, sim.nu)
        paths_a_down = sim_a_down.simulate(S0, r, T, n_paths, n_steps, seed=seed)
        C_a_down = self.price(paths_a_down, K, r, T)

        delta = (C_S_up - C_S_down) / (2.0 * eps_S)
        gamma = (C_S_up - 2.0 * C_base + C_S_down) / eps_S**2
        vega = (C_a_up - C_a_down) / (2.0 * eps_a)

        return {"delta": delta, "gamma": gamma, "vega": vega}


class AsianCallPricer(OptionPricer):
    """Prices arithmetic-average Asian call options: payoff = max(S̄ − K, 0).

    The average S̄ is computed over all simulated time steps (including t=0).
    """

    def price(self, St_paths: np.ndarray, K: float, r: float, T: float) -> float:
        """Discounted Monte Carlo price of the arithmetic Asian call.

        Args:
            St_paths: Simulated paths array of shape (n_paths, n_steps+1).
            K: Strike price.
            r: Risk-free rate.
            T: Time-to-maturity in years.

        Returns:
            Estimated option price.
        """
        S_mean = np.mean(St_paths, axis=1)
        payoff = np.maximum(S_mean - K, 0.0)
        return float(np.exp(-r * T) * np.mean(payoff))


class VanillaCallPricer(OptionPricer):
    """Prices European vanilla call options: payoff = max(S_T − K, 0)."""

    def price(self, St_paths: np.ndarray, K: float, r: float, T: float) -> float:
        """Discounted Monte Carlo price of the European vanilla call.

        Args:
            St_paths: Simulated paths array of shape (n_paths, n_steps+1).
            K: Strike price.
            r: Risk-free rate.
            T: Time-to-maturity in years.

        Returns:
            Estimated option price.
        """
        ST = St_paths[:, -1]
        payoff = np.maximum(ST - K, 0.0)
        return float(np.exp(-r * T) * np.mean(payoff))
