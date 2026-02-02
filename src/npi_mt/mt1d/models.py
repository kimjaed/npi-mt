# src/npi_mt/mt1d/models.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LayeredEarthModel:
    """
    1D layered Earth model parameterized by layer resistivities and depth interfaces.

    Conventions
    -----------
    - rho_ohm_m: resistivity per layer, shape (Z,)
    - depth_interfaces_m: layer boundaries, shape (Z+1,)
      where thicknesses are diff(depth_interfaces_m) for the top Z-1 layers,
      and the last layer is the bottom half-space.

    Notes
    -----
      thicknesses = np.diff(depth_interfaces_m)  # shape (Z,)
    thickness for the last layer is unused in the recursion.
    """
    rho_ohm_m: np.ndarray
    depth_interfaces_m: np.ndarray

    def __post_init__(self) -> None:
        rho = np.asarray(self.rho_ohm_m, dtype=float)
        zif = np.asarray(self.depth_interfaces_m, dtype=float)

        if rho.ndim != 1:
            raise ValueError(f"rho_ohm_m must be 1D, got shape {rho.shape}")
        if zif.ndim != 1:
            raise ValueError(f"depth_interfaces_m must be 1D, got shape {zif.shape}")
        if zif.size != rho.size + 1:
            raise ValueError(
                f"depth_interfaces_m must have length Z+1. "
                f"Got len(zif)={zif.size} and len(rho)={rho.size}."
            )
        if not np.all(np.diff(zif) > 0):
            raise ValueError("depth_interfaces_m must be strictly increasing.")
        if np.any(rho <= 0):
            raise ValueError("All resistivities must be > 0.")

        object.__setattr__(self, "rho_ohm_m", rho)
        object.__setattr__(self, "depth_interfaces_m", zif)

    @property
    def n_layers(self) -> int:
        return int(self.rho_ohm_m.size)

    @property
    def thicknesses_m(self) -> np.ndarray:
        """
        Thicknesses for each layer interval between interfaces, shape (Z,).

        The last entry corresponds to the interval between the last two interfaces.
        If you treat the last layer as half-space, only thicknesses[:Z-1] are used.
        """
        return np.diff(self.depth_interfaces_m)

    @property
    def depth_midpoints_m(self) -> np.ndarray:
        z = self.depth_interfaces_m
        return 0.5 * (z[:-1] + z[1:])
