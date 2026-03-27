"""
Velocity field computation utilities for flow-based cartography.

This module provides functions and classes for computing velocity fields from
density distributions using FFT-based methods with anisotropic diffusion. These
velocity fields drive the deformation process in flow-based cartogram algorithms.

Functions
---------
compute_velocity_anisotropic
    Basic FFT-based velocity computation.
compute_velocity_anisotropic_rfft
    Optimized real FFT version with caching.

Classes
-------
VelocityComputerFFTW
    High-performance FFTW-based velocity computer for repeated use.

Examples
--------
>>> from carto_flow.flow_cartogram.velocity import VelocityComputerFFTW
>>> from carto_flow.flow_cartogram.grid import Grid
>>> grid = Grid.from_bounds((0, 0, 100, 80), size=100)
>>> computer = VelocityComputerFFTW(grid, Dx=1.0, Dy=1.0)
>>> vx, vy = computer.compute(density)
"""

from functools import lru_cache

import numpy as np

# Import grid utilities
from .grid import Grid

# Module-level exports - Public API
__all__ = [
    "VelocityComputerFFTW",
    "compute_velocity_anisotropic",
    "compute_velocity_anisotropic_rfft",
]


def compute_velocity_anisotropic(
    rho: np.ndarray, grid: Grid, Dx: float = 1.0, Dy: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute anisotropic velocity field from density field using FFT-based method.

    This function implements a fast Fourier transform (FFT) based approach to compute
    the velocity field for flow-based cartography. The method solves a Poisson-like
    equation using spectral methods with anisotropic diffusion coefficients.

    The algorithm:
    1. Computes FFT of normalized density field
    2. Applies anisotropic diffusion operator in frequency domain
    3. Computes velocity components using inverse FFT of frequency derivatives

    Parameters
    ----------
    rho : np.ndarray
        2D density field array with shape (ny, nx). Should represent
        population or mass density distribution.
    grid : Grid
        Grid information containing spatial discretization details.
        Must provide dx and dy attributes for frequency computation.
    Dx : float, default=1.0
        Flow amplification factor in x-direction. Higher values produce
        stronger horizontal flow; lower values suppress it.
    Dy : float, default=1.0
        Flow amplification factor in y-direction. Higher values produce
        stronger vertical flow; lower values suppress it.

    Returns
    -------
    vx : np.ndarray
        X-component of velocity field with same shape as rho.
        Represents horizontal flow velocities.
    vy : np.ndarray
        Y-component of velocity field with same shape as rho.
        Represents vertical flow velocities.

    Notes
    -----
    The function uses spectral methods for computational efficiency:

    - **FFT-based computation**: O(n log n) complexity vs O(n²) for finite differences
    - **Anisotropic diffusion**: Different diffusion rates in x and y directions
    - **Zero-mean normalization**: Removes global mean to ensure convergence
    - **Frequency domain filtering**: Avoids division by zero at k=0

    The velocity field satisfies a relationship derived from mass conservation
    and anisotropic diffusion principles commonly used in flow-based cartography.

    Examples
    --------
    >>> import numpy as np
    >>> from carto_flow.velocity import compute_velocity_anisotropic
    >>> from carto_flow.grid import Grid
    >>>
    >>> # Create sample density field
    >>> density = np.random.rand(50, 50)
    >>> bounds = (0, 0, 10, 10)
    >>> grid = Grid.from_bounds(bounds, size=50)
    >>>
    >>> # Compute isotropic velocity field
    >>> vx, vy = compute_velocity_anisotropic(density, grid, Dx=1.0, Dy=1.0)
    >>> print(f"Velocity field shape: {vx.shape}")
    >>>
    >>> # Compute anisotropic velocity field (stronger in x-direction)
    >>> vx_aniso, vy_aniso = compute_velocity_anisotropic(density, grid, Dx=2.0, Dy=1.0)
    """
    ny, nx = rho.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, grid.dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, grid.dy)
    kx, ky = np.meshgrid(kx, ky)

    rho_hat = np.fft.fft2((rho - np.mean(rho)) / np.mean(rho))
    denom = kx**2 / Dx + ky**2 / Dy
    denom[0, 0] = 1.0

    phi_hat = -rho_hat / denom
    phi_hat[0, 0] = 0.0

    vx = np.real(np.fft.ifft2(1j * kx * phi_hat))
    vy = np.real(np.fft.ifft2(1j * ky * phi_hat))

    return vx, vy


@lru_cache(maxsize=16)
def _cached_kspace_grad(
    shape: tuple[int, int],
    dx: float,
    dy: float,
    Dx: float,
    Dy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cache k-space arrays for the given grid and anisotropic diffusion parameters.
    Compatible with rFFT (real input → half-spectrum output).
    """
    ny, nx = shape
    kx = 2 * np.pi * np.fft.rfftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    Kx, Ky = np.meshgrid(kx, ky)
    denom = Kx**2 / Dx + Ky**2 / Dy
    denom[0, 0] = 1e-9  # avoid division by zero
    return Kx, Ky, denom


def compute_velocity_anisotropic_rfft(
    rho: np.ndarray,
    grid: Grid,
    Dx: float = 1.0,
    Dy: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute anisotropic velocity field using real FFTs and cached k-space arrays.

    Parameters
    ----------
    rho : 2D array
        Density field.
    grid : Grid
        Grid information containing spatial discretization details.
        Must provide dx and dy attributes for frequency computation.
    Dx, Dy : float
        Flow amplification factors along x and y axes. Higher values
        produce stronger flow in the corresponding direction.

    Returns
    -------
    vx, vy : 2D arrays
        Velocity components in x and y directions.
    """
    ny, nx = rho.shape
    rho_dev = (rho - np.mean(rho)) / np.mean(rho)

    # Get cached k-space arrays
    Kx, Ky, denom = _cached_kspace_grad((ny, nx), grid.dx, grid.dy, Dx, Dy)

    # rFFT forward
    rho_hat = np.fft.rfft2(rho_dev)

    # Solve for potential in k-space
    phi_hat = -rho_hat / denom
    phi_hat[0, 0] = 0.0

    # Compute velocity components directly in k-space
    vx_hat = 1j * Kx * phi_hat
    vy_hat = 1j * Ky * phi_hat

    # Inverse rFFT (real result)
    vx = np.fft.irfft2(vx_hat, s=(ny, nx))
    vy = np.fft.irfft2(vy_hat, s=(ny, nx))

    return vx, vy


# Option 2: Pre-allocated FFTW with wisdom (best for repeated calls)
try:
    import pyfftw

    HAS_PYFFTW = True
except ImportError:
    HAS_PYFFTW = False
    pyfftw = None


class VelocityComputerFFTW:
    """
    Pre-allocate all arrays and FFTW plans for maximum performance.
    Use this when calling the function many times with same grid size.

    Example:
        computer = VelocityComputerFFTW(grid, Dx=1.0, Dy=1.0)
        for i in range(n_steps):
            vx, vy = computer.compute(rho)
    """

    def __init__(self, grid: Grid, Dx: float = 1.0, Dy: float = 1.0, threads: int | None = None) -> None:
        """
        Initialize with grid parameters.

        Parameters
        ----------
        grid : Grid
            Grid information containing spatial discretization details
        Dx, Dy : float
            Flow amplification factors along x and y axes. Higher values
            produce stronger flow in the corresponding direction.
        threads : int, None
            Number of threads for FFT computation
        """
        if not HAS_PYFFTW:
            raise ImportError("pyfftw is required for VelocityComputerFFTW. Install with: pip install pyfftw")

        if threads is None:
            import multiprocessing

            threads = max(multiprocessing.cpu_count() - 1, 1)

        self.ny, self.nx = grid.sy, grid.sx
        self.grid = grid

        # Cache k-space arrays
        self.Kx, self.Ky, self.denom = _cached_kspace_grad((grid.sy, grid.sx), grid.dx, grid.dy, Dx, Dy)

        # Pre-allocate arrays (aligned for SIMD)
        self.rho_dev = pyfftw.empty_aligned((self.ny, self.nx), dtype="float64")
        self.rho_hat = pyfftw.empty_aligned((self.ny, self.nx // 2 + 1), dtype="complex128")
        self.vx_hat = pyfftw.empty_aligned((self.ny, self.nx // 2 + 1), dtype="complex128")
        self.vy_hat = pyfftw.empty_aligned((self.ny, self.nx // 2 + 1), dtype="complex128")
        self.vx = pyfftw.empty_aligned((self.ny, self.nx), dtype="float64")
        self.vy = pyfftw.empty_aligned((self.ny, self.nx), dtype="float64")

        # Create FFTW plans (these are reused for maximum performance)
        self.fft_forward = pyfftw.FFTW(
            self.rho_dev, self.rho_hat, axes=(0, 1), direction="FFTW_FORWARD", threads=threads
        )
        self.fft_backward_x = pyfftw.FFTW(self.vx_hat, self.vx, axes=(0, 1), direction="FFTW_BACKWARD", threads=threads)
        self.fft_backward_y = pyfftw.FFTW(self.vy_hat, self.vy, axes=(0, 1), direction="FFTW_BACKWARD", threads=threads)

    def compute(self, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute velocity field from density.

        Parameters
        ----------
        rho : 2D array
            Density field

        Returns
        -------
        vx, vy : 2D arrays
            Velocity components (views of internal buffers)
        """
        # Compute deviation
        rho_mean = np.mean(rho)
        np.subtract(rho, rho_mean, out=self.rho_dev)
        self.rho_dev /= rho_mean

        # Forward FFT
        self.fft_forward()

        # Solve for potential and compute velocities in k-space
        np.negative(self.rho_hat, out=self.vx_hat)  # Reuse vx_hat buffer
        self.vx_hat /= self.denom
        self.vx_hat[0, 0] = 0.0

        # Compute vx in k-space
        np.multiply(1j * self.Kx, self.vx_hat, out=self.vx_hat)

        # Compute vy in k-space (reuse rho_hat as temporary)
        np.negative(self.rho_hat, out=self.vy_hat)
        self.vy_hat /= self.denom
        self.vy_hat[0, 0] = 0.0
        np.multiply(1j * self.Ky, self.vy_hat, out=self.vy_hat)

        # Inverse FFTs
        self.fft_backward_x()
        self.fft_backward_y()

        return self.vx, self.vy
