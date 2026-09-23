# ruff: noqa: RUF002, RUF003
"""
Velocity modulation system for cartogram algorithms.

This module provides a comprehensive system for velocity field modulation,
supporting boundary decay, smoothing, and anisotropy transformations.
Distance and sigma parameters are specified in world/data-coordinate units
(same CRS as the input geometries) and converted to pixel space automatically
at call time, so modulators work consistently across grid resolutions.

Classes
-------
VelocityModulator
    Base class for velocity field modulators.
BoundaryDecay
    Distance-based multiplicative falloff applied to all velocity components near
    geometry boundaries.
BoundaryNormalDecay
    Damps only the boundary-normal velocity component while preserving tangential
    flow near geometry boundaries.
DirectionalTensor
    Anisotropy aligned with a direction field (uniform angle, callable, or raster).
    Convenience constructors: ``radial``, ``tangential``, ``from_seeds``.
LocalizedTensor
    Anisotropy from seed points with Gaussian spatial influence; blends toward
    identity where seeds don't reach.
Smooth
    Gaussian smoothing of velocity field.
Multiplicative
    Multiplicative velocity modulation.
Tensor
    Low-level 2x2 tensor-based velocity modulation.
Pipeline
    Sequence of modulators applied in order.

Functions
---------
apply_anisotropy_tensor
    Apply anisotropy tensor to velocity field components.
build_axis_aligned_tensor
    Construct axis-aligned anisotropy field with optional rotation.

Examples
--------
>>> from carto_flow.flow_cartogram.anisotropy import DirectionalTensor, BoundaryDecay, BoundaryNormalDecay, Smooth
>>> import numpy as np
>>>
>>> # Uniform 30° tilt
>>> DirectionalTensor(theta=np.pi / 6, Dpar=2.0, Dperp=0.5)
>>>
>>> # Radially outward from a fixed center
>>> DirectionalTensor.radial(center=(500_000, 200_000), Dpar=2.0)
>>>
>>> # Tangential (counter-clockwise vortex)
>>> DirectionalTensor.tangential(center=(500_000, 200_000), Dpar=2.0)
>>>
>>> # Direction field from control points, blended with boundary decay
>>> seeds = [(1e5, 2e5, 0), (4e5, 5e5, np.pi / 2)]
>>> mod = DirectionalTensor.from_seeds(seeds, Dpar=3.0) + BoundaryDecay(decay_length=5)
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from .grid import Grid

# Module-level exports - Public API
__all__ = [
    "BoundaryDecay",
    "BoundaryNormalDecay",
    "DirectionalTensor",
    "LocalizedTensor",
    "Multiplicative",
    "Pipeline",
    "Smooth",
    "Tensor",
    "VelocityModulator",
    "apply_anisotropy_tensor",
    "build_axis_aligned_tensor",
    "preview_modulator",
]


class VelocityModulator:
    """Abstract base class for velocity field modulators.

    Subclasses implement ``__call__(vx, vy, grid, mask)`` and return
    ``(vx_new, vy_new)``.  Modulators can be chained with ``+`` into a
    :class:`Pipeline` that applies them left-to-right::

        mod = BoundaryDecay(decay_length=5) + Smooth(sigma=2)
    """

    def __call__(self, vx: np.ndarray, vy: np.ndarray, grid: Grid, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply modulation to velocity field."""
        raise NotImplementedError

    def __add__(self, other: "VelocityModulator") -> "Pipeline":
        """Chain modulators in sequence (self followed by other)."""
        if isinstance(self, Pipeline):
            return Pipeline([*self.modulators, other])
        else:
            return Pipeline([self, other])


class Pipeline(VelocityModulator):
    """Sequence of velocity modulators applied left-to-right.

    Constructed automatically when two modulators are combined with ``+``.
    Further ``+`` calls append to the same pipeline rather than nesting
    pipelines::

        pipe = BoundaryDecay() + Smooth(sigma=2)
    """

    def __init__(self, modulators: list):
        self.modulators = modulators

    def __call__(self, vx: np.ndarray, vy: np.ndarray, grid: Grid, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply all modulators in sequence."""
        for mod in self.modulators:
            vx, vy = mod(vx, vy, grid, mask)
        return vx, vy

    def __add__(self, other: "VelocityModulator") -> "Pipeline":
        """Chain another modulator after this pipeline."""
        return Pipeline([*self.modulators, other])

    def __repr__(self) -> str:
        return f"Pipeline({self.modulators})"


class BoundaryDecay(VelocityModulator):
    """Distance-based multiplicative velocity falloff near geometry boundaries.

    Multiplies the entire velocity field by a smooth spatial mask that
    transitions from ``damping_floor`` at the outer boundary to 1 deep in
    the interior over ``decay_length`` world-coordinate units.  Outside
    cells decay from ``damping_floor`` to 0 over ``outside_decay`` units.

    This is the simpler of the two boundary-decay modulators: it treats
    all velocity components identically.  For a geometrically aware variant
    that preserves tangential flow, use :class:`BoundaryNormalDecay`.

    Parameters
    ----------
    decay_length : float, default 3.0
        Width of the inside transition zone in world/data-coordinate units.
        Controls how quickly the velocity recovers from ``damping_floor``
        to 1 moving inward from the boundary.
    damping_floor : float, default 0.0
        Velocity factor applied exactly at the outer boundary; in [0, 1).
        A small positive value (e.g. 0.1–0.3) prevents complete freezing
        of boundary vertices and avoids jagged edges.
    smooth : float or None, default None
        Optional Gaussian smoothing sigma applied to the signed distance
        field before computing the falloff, in world-coordinate units.
        Smoothing blurs the boundary mask, creating a softer transition.
    outside_decay : float or None, default None
        Width of the outside decay zone in world-coordinate units.  If
        ``None``, uses the same value as ``decay_length``.  Smaller values
        suppress outside-cell drift more aggressively while leaving the
        interior unaffected.

    Examples
    --------
    >>> BoundaryDecay(decay_length=5, damping_floor=0.1)
    >>> BoundaryDecay(decay_length=10) + Smooth(sigma=2)
    """

    def __init__(
        self,
        decay_length: float = 3.0,
        damping_floor: float = 0.0,
        smooth: float | None = None,
        outside_decay: float | None = None,
    ):
        self.decay_length = decay_length
        self.damping_floor = damping_floor
        self.smooth = smooth
        self.outside_decay = outside_decay

    def __call__(self, vx: np.ndarray, vy: np.ndarray, grid: Grid, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply distance-based multiplicative falloff to velocity field."""
        cell_size = (grid.dx + grid.dy) / 2
        decay_px = self.decay_length / cell_size
        smooth_px = self.smooth / cell_size if self.smooth is not None else None

        outside = mask == -1

        dist_inside = distance_transform_edt(~outside)  # 0 outside, positive inside
        dist_outside = distance_transform_edt(outside)  # 0 inside, positive outside
        signed_dist = dist_inside - dist_outside  # positive inside, negative outside

        if smooth_px is not None and smooth_px > 0:
            signed_dist = gaussian_filter(signed_dist, smooth_px, mode="reflect")

        falloff = self.damping_floor + (1.0 - self.damping_floor) * np.clip(signed_dist / decay_px, 0.0, 1.0)
        return vx * falloff, vy * falloff

    def __repr__(self) -> str:
        return f"BoundaryDecay(decay_length={self.decay_length}, damping_floor={self.damping_floor})"


class BoundaryNormalDecay(VelocityModulator):
    """Damp the boundary-normal velocity component while preserving tangential flow.

    Near the outer boundary the velocity field is decomposed into a
    component **normal** to the boundary and a **tangential** component.
    The normal component is exponentially damped over ``decay_length``
    world-coordinate units; the tangential component is left unchanged.
    This suppresses the outward drift responsible for boundary distortion
    without affecting flow parallel to the boundary.

    Parameters
    ----------
    decay_length : float, default 3.0
        Length scale over which the normal velocity component is damped,
        in world/data-coordinate units.
    damping_floor : float, default 0.0
        Minimum damping factor at the boundary; in [0, 1].  At distance
        ``d`` from the boundary, the normal factor is
        ``floor + (1 - floor) * (1 - exp(-d / decay_length))``.
    smooth : float or None, default None
        If not ``None``, applies Gaussian smoothing to the signed distance
        field before computing boundary normals, in world-coordinate units.
        Useful to reduce noise in the normal direction for complex geometries.
    renormalize : bool, default False
        If ``True``, the output velocity is rescaled to preserve the
        original magnitude at each grid cell.  This keeps the speed of
        flow constant while only rotating the direction.

    Examples
    --------
    >>> BoundaryNormalDecay(decay_length=5)
    >>> BoundaryNormalDecay(decay_length=3, damping_floor=0.1, smooth=1.0)
    """

    def __init__(
        self,
        decay_length: float = 3.0,
        damping_floor: float = 0.0,
        smooth: float | None = None,
        renormalize: bool = False,
    ):
        self.decay_length = decay_length
        self.damping_floor = damping_floor
        self.smooth = smooth
        self.renormalize = renormalize

    def __call__(self, vx: np.ndarray, vy: np.ndarray, grid: Grid, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Damp the normal velocity component near geometry boundaries."""
        cell_size = (grid.dx + grid.dy) / 2
        decay_px = self.decay_length / cell_size
        smooth_px = self.smooth / cell_size if self.smooth is not None else None
        eps = 1e-12

        vx = vx.astype(float)
        vy = vy.astype(float)

        outside = mask == -1

        dist_in = distance_transform_edt(~outside)
        dist_out = distance_transform_edt(outside)
        signed_dist = dist_in - dist_out
        abs_dist = np.abs(signed_dist)

        if smooth_px is not None and smooth_px > 0:
            signed_dist = gaussian_filter(signed_dist, smooth_px, mode="reflect")

        # Boundary normal from the signed distance gradient
        gy, gx = np.gradient(signed_dist)
        norm = np.sqrt(gx**2 + gy**2) + eps
        nx, ny = gx / norm, gy / norm

        # Tangent vector (90° counter-clockwise from normal)
        tx, ty = -ny, nx

        # Decompose velocity into normal and tangential components
        v_n = vx * nx + vy * ny
        v_t = vx * tx + vy * ty

        # Exponential damping of the normal component with floor
        damping_floor = np.clip(self.damping_floor, 0.0, 1.0)
        normal_factor = damping_floor + (1.0 - damping_floor) * (1.0 - np.exp(-abs_dist / decay_px))
        v_n_new = v_n * normal_factor

        # Reconstruct velocity
        vx_new = v_n_new * nx + v_t * tx
        vy_new = v_n_new * ny + v_t * ty

        if self.renormalize:
            vmag = np.sqrt(vx**2 + vy**2) + eps
            new_mag = np.sqrt(vx_new**2 + vy_new**2) + eps
            vx_new *= vmag / new_mag
            vy_new *= vmag / new_mag

        return vx_new, vy_new

    def __repr__(self) -> str:
        return f"BoundaryNormalDecay(decay_length={self.decay_length}, damping_floor={self.damping_floor})"


class Smooth(VelocityModulator):
    """Gaussian smoothing of the velocity field.

    Convolves both velocity components with an isotropic Gaussian kernel.
    Smoothing reduces high-frequency oscillations and can improve
    convergence, at the cost of some sharpness in the final cartogram.

    Parameters
    ----------
    sigma : float, default 3.0
        Standard deviation of the Gaussian kernel in world/data-coordinate
        units.  Converted to pixels automatically at call time.

    Examples
    --------
    >>> Smooth(sigma=2)
    >>> BoundaryDecay(decay_length=5) + Smooth(sigma=1)
    """

    def __init__(self, sigma: float = 3.0):
        self.sigma = sigma

    def __call__(self, vx: np.ndarray, vy: np.ndarray, grid: Grid, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply Gaussian smoothing to velocity field."""
        sigma_yx = (self.sigma / grid.dy, self.sigma / grid.dx)
        vx_smooth = gaussian_filter(vx, sigma=sigma_yx, mode="reflect")
        vy_smooth = gaussian_filter(vy, sigma=sigma_yx, mode="reflect")
        return vx_smooth, vy_smooth

    def __repr__(self) -> str:
        return f"Smooth(sigma={self.sigma})"


class Multiplicative(VelocityModulator):
    """Element-wise scaling of velocity components.

    Multiplies ``vx`` by ``fx`` and ``vy`` by ``fy`` independently.
    Each factor can be a scalar, a pre-computed ``(ny, nx)`` array, or a
    callable ``(grid) -> (ny, nx) array`` evaluated lazily at call time.

    Parameters
    ----------
    fx : float or np.ndarray or callable
        Scaling factor for the x-component of velocity.
    fy : float or np.ndarray or callable
        Scaling factor for the y-component of velocity.

    Examples
    --------
    Suppress horizontal flow everywhere:

    >>> Multiplicative(fx=0.0, fy=1.0)

    Spatially varying x-suppression based on grid position:

    >>> Multiplicative(fx=lambda g: np.clip(g.X / g.X.max(), 0, 1), fy=1.0)
    """

    def __init__(self, fx, fy):
        self.fx = fx
        self.fy = fy

    def __call__(self, vx: np.ndarray, vy: np.ndarray, grid: Grid, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply multiplicative modulation to velocity field."""
        fx = self._resolve_field(self.fx, grid)
        fy = self._resolve_field(self.fy, grid)
        return vx * fx, vy * fy

    def _resolve_field(self, field, grid):
        """Resolve field to array."""
        if callable(field):
            return field(grid)
        elif np.isscalar(field):
            return np.full(grid.shape, field)
        else:
            return field

    def __repr__(self) -> str:
        return f"Multiplicative(fx={self.fx}, fy={self.fy})"


class Tensor(VelocityModulator):
    """Low-level 2×2 matrix transform of the velocity field.

    Applies ``[vx', vy'] = A @ [vx, vy]`` pointwise, where ``A`` has
    components ``Axx``, ``Axy``, ``Ayx``, ``Ayy``.  Each component can be
    a scalar, a pre-computed ``(ny, nx)`` array, or a callable
    ``(grid) -> (ny, nx) array``.

    This is the most general velocity modulator; higher-level classes such
    as :class:`DirectionalTensor` and :class:`LocalizedTensor` construct
    their tensors automatically from geometric parameters.

    Parameters
    ----------
    Axx, Axy, Ayx, Ayy : float or np.ndarray or callable
        Components of the 2×2 transform matrix.  The identity is
        ``Axx=Ayy=1, Axy=Ayx=0``.

    Examples
    --------
    Uniform 45° rotation:

    >>> import numpy as np
    >>> c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    >>> Tensor(Axx=c, Axy=-s, Ayx=s, Ayy=c)
    """

    def __init__(self, Axx, Axy, Ayx, Ayy):
        self.Axx = Axx
        self.Axy = Axy
        self.Ayx = Ayx
        self.Ayy = Ayy

    def __call__(self, vx: np.ndarray, vy: np.ndarray, grid: Grid, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply tensor modulation to velocity field."""
        Axx = self._resolve_field(self.Axx, grid)
        Axy = self._resolve_field(self.Axy, grid)
        Ayx = self._resolve_field(self.Ayx, grid)
        Ayy = self._resolve_field(self.Ayy, grid)
        return apply_anisotropy_tensor(vx, vy, Axx, Axy, Ayx, Ayy)

    def _resolve_field(self, field, grid):
        """Resolve field to array."""
        if callable(field):
            return field(grid)
        elif np.isscalar(field):
            return np.full(grid.shape, field)
        else:
            return field

    def __repr__(self) -> str:
        return "Tensor(Axx, Axy, Ayx, Ayy)"


class DirectionalTensor(VelocityModulator):
    """Anisotropic velocity modulation aligned with a direction field.

    Amplifies velocity along a preferred direction (`Dpar`) and optionally
    suppresses the perpendicular component (`Dperp`). Works correctly at any
    grid resolution: the rotation tensor is built lazily on the first call at
    each grid shape and cached for all subsequent iterations at that resolution.

    Parameters
    ----------
    theta : float | callable | np.ndarray
        Preferred flow direction in radians (0 = positive x-axis, π/2 = positive y-axis).

        - **float**: uniform angle applied everywhere.
        - **callable** ``(grid) -> (ny, nx) array``: evaluated once per grid shape.
          The callable receives the current :class:`Grid` object, giving access to
          ``grid.X``, ``grid.Y``, ``grid.shape``, etc.
        - **ndarray**: pre-computed angle field; automatically interpolated to the
          current grid shape with bilinear interpolation if shapes differ.
    Dpar : float, default 2.0
        Amplification factor along the preferred direction.
    Dperp : float, default 1.0
        Amplification factor perpendicular to the preferred direction.

    Examples
    --------
    Uniform 45° tilt with double amplification along that axis:

    >>> DirectionalTensor(theta=np.pi / 4, Dpar=2.0, Dperp=0.5)

    Flow aligned radially outward from the domain center:

    >>> DirectionalTensor(
    ...     theta=lambda g: np.arctan2(g.Y - g.Y.mean(), g.X - g.X.mean()),
    ...     Dpar=2.0,
    ... )

    Direction field from an external raster (e.g. slope aspect) — auto-resized:

    >>> DirectionalTensor(theta=aspect_array, Dpar=3.0, Dperp=0.5)
    """

    def __init__(self, theta, Dpar: float = 2.0, Dperp: float = 1.0):
        self.theta = theta
        self.Dpar = Dpar
        self.Dperp = Dperp
        self._tensor_cache: dict = {}  # keyed by grid.shape

    def _resolve_theta(self, grid) -> np.ndarray:
        """Return a (ny, nx) angle array for the given grid."""
        if callable(self.theta):
            return self.theta(grid)
        elif np.isscalar(self.theta):
            return np.full(grid.shape, float(self.theta))  # type: ignore[arg-type]
        else:
            arr = np.asarray(self.theta, dtype=float)
            if arr.shape == grid.shape:
                return arr
            from scipy.ndimage import zoom

            zy = grid.shape[0] / arr.shape[0]
            zx = grid.shape[1] / arr.shape[1]
            return zoom(arr, (zy, zx), order=1)

    def __call__(self, vx: np.ndarray, vy: np.ndarray, grid, mask: np.ndarray) -> tuple:
        """Apply directional tensor modulation, using a cached tensor when possible."""
        shape = grid.shape
        if shape not in self._tensor_cache:
            theta = self._resolve_theta(grid)
            self._tensor_cache[shape] = build_axis_aligned_tensor(
                grid.sx, grid.sy, Dx=self.Dpar, Dy=self.Dperp, theta_field=theta
            )
        Axx, Axy, Ayx, Ayy = self._tensor_cache[shape]
        return apply_anisotropy_tensor(vx, vy, Axx, Axy, Ayx, Ayy)

    @classmethod
    def radial(cls, center=None, Dpar: float = 2.0, Dperp: float = 1.0, inward: bool = False) -> "DirectionalTensor":
        """Flow aligned radially outward from (or inward toward) a point.

        Parameters
        ----------
        center : (float, float) or None
            ``(x, y)`` of the origin in data coordinates.  If ``None``, the
            centroid of the grid bounding box is used (evaluated lazily).
        Dpar : float, default 2.0
            Amplification along the radial direction.
        Dperp : float, default 1.0
            Amplification tangential to the radial direction.
        inward : bool, default False
            If ``True``, preferred direction points toward *center* instead of
            away from it.

        Examples
        --------
        >>> DirectionalTensor.radial(center=(500_000, 200_000), Dpar=3.0, Dperp=0.5)
        >>> DirectionalTensor.radial(inward=True)          # toward domain centroid
        """
        sign = -1.0 if inward else 1.0
        if center is not None:
            cx, cy = float(center[0]), float(center[1])

            def theta(g, cx=cx, cy=cy, sign=sign):
                return np.arctan2(sign * (g.Y - cy), sign * (g.X - cx))

        else:

            def theta(g, sign=sign):  # type: ignore[misc]
                return np.arctan2(sign * (g.Y - g.Y.mean()), sign * (g.X - g.X.mean()))

        obj = cls(theta=theta, Dpar=Dpar, Dperp=Dperp)
        obj._description = f"DirectionalTensor.radial(center={center!r}, inward={inward})"  # type: ignore[attr-defined]
        return obj

    @classmethod
    def tangential(
        cls, center=None, Dpar: float = 2.0, Dperp: float = 1.0, clockwise: bool = False
    ) -> "DirectionalTensor":
        """Flow aligned tangentially around a point (counter-clockwise by default).

        Parameters
        ----------
        center : (float, float) or None
            ``(x, y)`` of the rotation origin in data coordinates.  If ``None``,
            the centroid of the grid bounding box is used.
        Dpar : float, default 2.0
            Amplification along the tangential direction.
        Dperp : float, default 1.0
            Amplification along the radial direction.
        clockwise : bool, default False
            If ``True``, the preferred rotation direction is clockwise.

        Examples
        --------
        >>> DirectionalTensor.tangential(center=(500_000, 200_000), Dpar=2.0)
        """
        # Counter-clockwise tangent of radial angle θ is θ + π/2;
        # clockwise tangent is θ - π/2.
        offset = -np.pi / 2 if clockwise else np.pi / 2
        if center is not None:
            cx, cy = float(center[0]), float(center[1])

            def theta(g, cx=cx, cy=cy, offset=offset):
                return np.arctan2(g.Y - cy, g.X - cx) + offset

        else:

            def theta(g, offset=offset):  # type: ignore[misc]
                return np.arctan2(g.Y - g.Y.mean(), g.X - g.X.mean()) + offset

        obj = cls(theta=theta, Dpar=Dpar, Dperp=Dperp)
        obj._description = f"DirectionalTensor.tangential(center={center!r}, clockwise={clockwise})"  # type: ignore[attr-defined]
        return obj

    @classmethod
    def from_seeds(cls, seeds, Dpar: float = 2.0, Dperp: float = 1.0, power: float = 2.0) -> "DirectionalTensor":
        """Build a direction field from a sparse set of (x, y, θ) control points.

        The direction at each grid cell is computed as the inverse-distance-weighted
        (IDW) average of the seed angles.  Angle averaging is circular: each seed
        contributes ``(cos θ, sin θ)`` proportional to its weight, and the
        resultant angle is recovered with ``arctan2``.

        Parameters
        ----------
        seeds : array-like of shape (n, 3)
            Each row is ``(x, y, theta)`` where ``x, y`` are data coordinates
            and ``theta`` is the preferred flow direction in radians.
        Dpar : float, default 2.0
            Amplification along the preferred direction.
        Dperp : float, default 1.0
            Amplification perpendicular to the preferred direction.
        power : float, default 2.0
            IDW distance exponent.  Higher values give more localised influence
            (1 = linear falloff, 2 = classic IDW, ≥3 = near-Voronoi).

        Examples
        --------
        >>> seeds = [
        ...     (100_000, 200_000, 0),          # flow east at this point
        ...     (400_000, 500_000, np.pi / 2),  # flow north at this point
        ...     (700_000, 150_000, np.pi / 4),  # flow north-east at this point
        ... ]
        >>> DirectionalTensor.from_seeds(seeds, Dpar=3.0, Dperp=0.5)
        """
        pts = np.asarray(seeds, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("seeds must have shape (n, 3): each row is (x, y, theta)")
        xs, ys = pts[:, 0], pts[:, 1]
        sin_t = np.sin(pts[:, 2])
        cos_t = np.cos(pts[:, 2])

        def theta(g, xs=xs, ys=ys, sin_t=sin_t, cos_t=cos_t, power=power):
            # Broadcast: (ny, nx, n)
            dx = g.X[..., np.newaxis] - xs
            dy = g.Y[..., np.newaxis] - ys
            d2 = dx**2 + dy**2

            exact = d2 < 1e-30  # grid point coincides with a seed
            # IDW weight = 1 / d^power = (d²)^(-power/2); zero at exact matches
            with np.errstate(divide="ignore", invalid="ignore"):
                w = np.where(exact, 0.0, d2 ** (-power / 2))

            w_sum = w.sum(axis=-1, keepdims=True)
            w_norm = np.where(w_sum > 0, w / w_sum, 1.0 / w.shape[-1])

            sin_avg = (w_norm * sin_t).sum(axis=-1)
            cos_avg = (w_norm * cos_t).sum(axis=-1)
            result = np.arctan2(sin_avg, cos_avg)

            # At exact seed locations use the seed angle directly
            any_exact = exact.any(axis=-1)
            if any_exact.any():
                seed_idx = np.argmax(exact, axis=-1)
                thetas = np.arctan2(sin_t, cos_t)
                result = np.where(any_exact, thetas[seed_idx], result)

            return result

        obj = cls(theta=theta, Dpar=Dpar, Dperp=Dperp)
        obj._description = f"DirectionalTensor.from_seeds(n={len(pts)}, power={power})"  # type: ignore[attr-defined]
        return obj

    def __repr__(self) -> str:
        if hasattr(self, "_description"):
            return self._description
        return f"DirectionalTensor(theta={self.theta!r}, Dpar={self.Dpar}, Dperp={self.Dperp})"


class LocalizedTensor(VelocityModulator):
    """Velocity modulation from localized seed points with Gaussian influence.

    Each seed defines a preferred flow direction and amplification at a
    specific location.  Influence decays as a Gaussian whose shape is
    controlled by ``sigma``.  Where seeds do not reach (total Gaussian
    weight < 1) the field blends toward the identity — isotropic, velocity
    unchanged.

    Parameters
    ----------
    seeds : list of dict or list of tuple
        Each element describes one seed.  Dict keys:

        - ``x``, ``y``: location in data coordinates (required)
        - ``theta``: preferred flow direction in radians, 0 = +x axis.
          Required unless ``sigma`` is a (2, 2) covariance matrix, in which
          case it is derived from the matrix's major eigenvector.
        - ``sigma``: influence zone — three forms accepted:

          - **float** — isotropic circle of radius σ
          - **array-like (σ_par, σ_perp)** — ellipse aligned with ``theta``;
            σ_par along the flow direction, σ_perp across it.  ``theta`` is
            required.
          - **(2, 2) array Σ** — full covariance matrix; ``theta`` may be
            omitted (derived from the major eigenvector of Σ) or specified
            independently

        - ``Dpar``, ``Dperp``: optional per-seed amplification overrides

        Tuple form ``(x, y, theta, sigma)`` or ``(x, y, theta, sigma, Dpar,
        Dperp)`` is also accepted; the tuple form always requires ``theta``
        and ``sigma`` must be a scalar.

    default_Dpar : float, default 2.0
        Amplification along preferred direction for seeds that omit ``Dpar``.
    default_Dperp : float, default 1.0
        Amplification perpendicular to preferred direction for seeds that omit
        ``Dperp``.

    Examples
    --------
    Isotropic influence zone:

    >>> LocalizedTensor([dict(x=500_000, y=200_000, theta=np.pi/4, sigma=80_000)])

    Elliptical zone aligned with flow (3× wider along flow than across):

    >>> LocalizedTensor([
    ...     dict(x=500_000, y=200_000, theta=np.pi/4,
    ...          sigma=(120_000, 40_000), Dpar=3.0, Dperp=0.5),
    ... ])

    Full covariance; theta derived from the major eigenvector:

    >>> theta = np.pi / 4
    >>> c, s = np.cos(theta), np.sin(theta)
    >>> Sigma = (120_000**2 * np.outer([c, s], [c, s])
    ...        + 40_000**2 * np.outer([-s, c], [-s, c]))
    >>> LocalizedTensor([dict(x=500_000, y=200_000, sigma=Sigma, Dpar=3.0)])
    """

    def __init__(self, seeds, default_Dpar: float = 2.0, default_Dperp: float = 1.0):
        self._parsed = self._parse_seeds(seeds, default_Dpar, default_Dperp)
        self._tensor_cache: dict = {}  # keyed by grid.shape

    @staticmethod
    def _sigma_to_precision(sigma, theta):
        """Convert a sigma specification to a (2, 2) precision matrix.

        Returns ``(P, theta)`` where ``theta`` may have been derived from
        ``sigma`` if it was ``None`` and ``sigma`` is a covariance matrix.
        """
        sigma_arr = np.asarray(sigma, dtype=float)

        if sigma_arr.shape == ():
            s_val = float(sigma_arr)
            P = np.array([[1.0 / s_val**2, 0.0], [0.0, 1.0 / s_val**2]])

        elif sigma_arr.shape == (2,):
            if theta is None:
                raise ValueError("theta is required when sigma is a (sigma_par, sigma_perp) pair.")
            sp, sn = float(sigma_arr[0]), float(sigma_arr[1])
            c, s_ = np.cos(theta), np.sin(theta)
            P = 1.0 / sp**2 * np.outer([c, s_], [c, s_]) + 1.0 / sn**2 * np.outer([-s_, c], [-s_, c])

        elif sigma_arr.shape == (2, 2):
            eigvals = np.linalg.eigvalsh(sigma_arr)
            if eigvals.min() <= 0:
                raise ValueError(
                    f"sigma covariance matrix must be positive definite (smallest eigenvalue = {eigvals.min():.3g})."
                )
            if theta is None:
                _, vecs = np.linalg.eigh(sigma_arr)  # ascending eigenvalues
                major = vecs[:, -1]
                theta = float(np.arctan2(major[1], major[0]))
            P = np.linalg.inv(sigma_arr)

        else:
            raise ValueError(
                "sigma must be a scalar, a (sigma_par, sigma_perp) 2-tuple, or a (2, 2) covariance matrix."
            )

        return P, theta

    @staticmethod
    def _parse_seeds(seeds, default_Dpar, default_Dperp):
        """Return list of (cx, cy, P, Axx, Ayy, Axy) per seed.

        P is the (2, 2) precision matrix used in the Gaussian weight.
        """
        result = []
        for s in seeds:
            if isinstance(s, dict):
                x, y = float(s["x"]), float(s["y"])
                theta = s.get("theta", None)
                if theta is not None:
                    theta = float(theta)
                sigma = s["sigma"]
                dp = float(s.get("Dpar", default_Dpar))
                dpe = float(s.get("Dperp", default_Dperp))
            else:
                s = tuple(s)
                if len(s) == 4:
                    x, y, theta, sigma = s
                    x, y, theta = float(x), float(y), float(theta)
                    dp, dpe = default_Dpar, default_Dperp
                elif len(s) == 6:
                    x, y, theta, sigma, dp, dpe = s
                    x, y, theta = float(x), float(y), float(theta)
                    dp, dpe = float(dp), float(dpe)
                else:
                    raise ValueError(
                        "Each seed must be (x, y, theta, sigma), "
                        "(x, y, theta, sigma, Dpar, Dperp), or a dict with those keys."
                    )

            P, theta = LocalizedTensor._sigma_to_precision(sigma, theta)

            if theta is None:
                theta = 0.0  # isotropic velocity (dp == dpe); direction irrelevant

            c, s_ = np.cos(theta), np.sin(theta)
            result.append((
                x,
                y,
                P,
                dp * c * c + dpe * s_ * s_,  # Axx
                dp * s_ * s_ + dpe * c * c,  # Ayy
                (dp - dpe) * c * s_,
            ))  # Axy = Ayx (symmetric)
        if not result:
            raise ValueError("seeds must be non-empty")
        return result

    def _build_tensor(self, grid):
        """Build blended (Axx, Axy, Ayx, Ayy) arrays for the given grid."""
        xs = np.array([p[0] for p in self._parsed])
        ys = np.array([p[1] for p in self._parsed])
        Ps = np.stack([p[2] for p in self._parsed])  # (n_seeds, 2, 2)
        Axxs = np.array([p[3] for p in self._parsed])
        Ayys = np.array([p[4] for p in self._parsed])
        Axys = np.array([p[5] for p in self._parsed])

        # (ny, nx, n_seeds) displacement arrays
        dx = grid.X[..., np.newaxis] - xs
        dy = grid.Y[..., np.newaxis] - ys

        # Quadratic form xᵀPx using the three unique elements of symmetric P
        P00 = Ps[:, 0, 0]  # (n_seeds,)
        P01 = Ps[:, 0, 1]  # (n_seeds,) — equals P10
        P11 = Ps[:, 1, 1]  # (n_seeds,)
        quad = P00 * dx**2 + 2.0 * P01 * dx * dy + P11 * dy**2  # (ny, nx, n_seeds)
        w = np.exp(-0.5 * quad)
        W = w.sum(axis=-1)  # (ny, nx)

        # Normalize so total seed weight never exceeds 1;
        # remainder blends toward identity (isotropic background).
        norm = np.maximum(1.0, W)
        w_n = w / norm[..., np.newaxis]  # (ny, nx, n_seeds)
        w_bg = np.maximum(0.0, 1.0 - W / norm)  # (ny, nx)

        Axx = (w_n * Axxs).sum(axis=-1) + w_bg  # identity contributes 1
        Ayy = (w_n * Ayys).sum(axis=-1) + w_bg  # identity contributes 1
        Axy = (w_n * Axys).sum(axis=-1)  # identity contributes 0
        return Axx, Axy, Axy, Ayy  # Ayx = Axy (symmetric tensor)

    def __call__(self, vx: np.ndarray, vy: np.ndarray, grid, mask: np.ndarray) -> tuple:
        """Apply localised tensor modulation, using cached arrays when possible."""
        shape = grid.shape
        if shape not in self._tensor_cache:
            self._tensor_cache[shape] = self._build_tensor(grid)
        Axx, Axy, Ayx, Ayy = self._tensor_cache[shape]
        return apply_anisotropy_tensor(vx, vy, Axx, Axy, Ayx, Ayy)

    def __repr__(self) -> str:
        return f"LocalizedTensor(n_seeds={len(self._parsed)})"


def apply_anisotropy_tensor(vx, vy, Axx, Axy, Ayx, Ayy):
    """
    Apply anisotropy tensor to velocity field components.

    Axx etc are (ny,nx) arrays. Compute:
      vx' = Axx*vx + Axy*vy
      vy' = Ayx*vx + Ayy*vy

    Parameters
    ----------
    vx, vy : np.ndarray
        Velocity field components with shape (ny, nx)
    Axx, Axy, Ayx, Ayy : np.ndarray
        Anisotropy tensor components with shape (ny, nx)

    Returns
    -------
    vx_new, vy_new : np.ndarray
        Transformed velocity field components
    """
    vx_new = Axx * vx + Axy * vy
    vy_new = Ayx * vx + Ayy * vy
    return vx_new, vy_new


def build_axis_aligned_tensor(nx, ny, Dx=1.0, Dy=1.0, theta_field=None):
    """
    Build axis-aligned anisotropy field (scalar Dx, Dy) optionally rotated by theta(x,y).

    Returns Axx, Axy, Ayx, Ayy arrays. If theta_field is None, axis-aligned (no rotation).
    theta_field shape (ny,nx) in radians if provided.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions (x, y)
    Dx, Dy : float, default=1.0
        Anisotropy scaling factors in x and y directions
    theta_field : np.ndarray, optional
        Rotation field with shape (ny, nx) in radians

    Returns
    -------
    Axx, Axy, Ayx, Ayy : np.ndarray
        Anisotropy tensor components
    """
    if theta_field is None:
        Axx = np.full((ny, nx), Dx)
        Ayy = np.full((ny, nx), Dy)
        Axy = Ayx = np.zeros((ny, nx))
        return Axx, Axy, Ayx, Ayy
    else:
        # rotated tensors
        c = np.cos(theta_field)
        s = np.sin(theta_field)
        Axx = Dx * c * c + Dy * s * s
        Ayy = Dx * s * s + Dy * c * c
        Axy = (Dx - Dy) * c * s
        Ayx = Axy
        return Axx, Axy, Ayx, Ayy


def preview_modulator(
    modulator: VelocityModulator | None = None,
    gdf=None,
    grid_size: int = 64,
    skip: int = 4,
    input_angle: float = 0.0,
    margin: float = 0.1,
    show_geometry: bool = True,
    ax=None,
    cmap="viridis",
    show_colorbar: bool = True,
    arrows_kwargs: dict | None = None,
    colorbar_kwargs: dict | None = None,
    values=None,
    column: str | None = None,
    Dx: float = 1.0,
    Dy: float = 1.0,
    show_vectors: bool | str | tuple = "output",
    input_arrows_kwargs: dict | None = None,
    diff_cmap: str = "Reds",
    diff_arrows_kwargs: dict | None = None,
    diff_colorbar_kwargs: dict | None = None,
    heatmap: str | None = None,
    heatmap_type: str = "magnitude",
    heatmap_cmap: str | None = None,
    heatmap_kwargs: dict | None = None,
    heatmap_colorbar_kwargs: dict | None = None,
    heatmap_alpha_from_magnitude: bool = False,
    arrow_scale: float = 1.0,
):
    """Preview a velocity modulator on a uniform probe field.

    Applies the modulator to a spatially uniform input field and plots the
    result as a quiver diagram.  Arrow color encodes the local amplification
    factor (output magnitude ÷ input magnitude = 1.0 for identity).  For
    :class:`LocalizedTensor`, an optional background heatmap shows the total
    Gaussian seed weight (how much each location is dominated by seeds vs.
    the isotropic background).

    The geometry mask passed to the modulator is rasterized from *gdf*, so
    modulators that use boundary proximity (e.g. :class:`BoundaryDecay`) work
    correctly in the preview.

    Parameters
    ----------
    modulator : VelocityModulator
        The modulator to preview.
    gdf : GeoDataFrame
        Geometries used to derive the spatial extent and the geometry mask
        passed to the modulator.
    grid_size : int, default 64
        Number of grid cells along the longer axis.  Kept small for a fast,
        clutter-free preview.
    skip : int, default 4
        Plot every *skip*-th arrow to reduce clutter.
    input_angle : float, default 0.0
        Direction of the uniform probe field in radians (0 = +x = east).
        Change this to see how the modulator responds to a different incoming
        flow direction.
    margin : float, default 0.05
        Fractional margin added around the bounds before building the grid,
        matching the convention used by the real algorithm.
    show_geometry : bool, default True
        When ``True``, draw the geometry outlines as a light background layer.
    show_colorbar : bool, default True
        When ``False``, the colorbar for the arrows is omitted.  Useful when
        embedding the preview in a figure that provides its own colorbar.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created if ``None``.
    values : array-like, optional
        Per-geometry values (same length as *gdf*) used to compute a realistic
        density field and velocity field via the FFT Poisson solver — identical
        to what :func:`morph_geometries` computes internally.  When provided,
        the probe field is the normalized velocity field derived from those
        values rather than a uniform field.  Useful for modulators that depend
        on the spatial structure of the field.
    column : str, optional
        Column name in *gdf* to use as values.  Equivalent to passing
        ``values=gdf[column].to_numpy()``.  Takes precedence over *values* if
        both are given.
    Dx : float, default 1.0
        Anisotropic diffusion factor in x passed to :class:`VelocityComputerFFTW`
        when computing the density-based velocity field.
    Dy : float, default 1.0
        Anisotropic diffusion factor in y passed to :class:`VelocityComputerFFTW`.
    show_vectors : bool or str or tuple, default 'output'
        Controls which velocity quiver layers are drawn:

        - ``'output'`` — only the modulated output field (default).
        - ``True`` — input + output (comparison view).
        - ``False`` or ``None`` — no quivers (useful when only a heatmap is wanted).
        - ``'input'`` or ``'diff'`` — only that single layer.
        - tuple, e.g. ``('output', 'diff')`` — any combination.
    input_arrows_kwargs : dict, optional
        Extra keyword arguments for the input quiver ``ax.quiver`` call.
    diff_cmap : str or Colormap, default 'Reds'
        Colormap for the difference quiver.
    diff_arrows_kwargs : dict, optional
        Extra keyword arguments for the difference quiver ``ax.quiver`` call.
    arrow_scale : float, default 1.0
        Length of the longest output arrow as a fraction of the skipped-cell
        diagonal (``sqrt((dx*skip)² + (dy*skip)²)``).  All quivers share this
        scale so arrows are directly comparable.  Values < 1 add a gap between
        adjacent arrows; values > 1 allow overlap.
    cmap : str or Colormap, default 'viridis'
        Colormap for the quiver arrows, encoding amplification factor.
    arrows_kwargs : dict, optional
        Extra keyword arguments merged into the ``ax.quiver`` call for the
        modulated-velocity arrows.  Override any of the defaults (``cmap``,
        ``pivot``, ``zorder``, …).
    colorbar_kwargs : dict, optional
        Extra keyword arguments merged into the ``plt.colorbar`` call (e.g.
        ``label``, ``shrink``, ``pad``).
    heatmap : {'weight', 'input', 'output', 'diff'} or None, default None
        Optional background heatmap overlay:

        - ``'weight'`` — Gaussian seed-weight field for :class:`LocalizedTensor`
          (replaces the old *show_weight* parameter).
        - ``'input'``, ``'output'``, ``'diff'`` — scalar reduction of the
          corresponding velocity field controlled by *heatmap_type*.
    heatmap_type : {'magnitude', 'angle', 'magnitude_diff', 'angle_diff'}, default 'magnitude'
        How to reduce the 2-D vector field to a scalar for the heatmap:

        - ``'magnitude'`` — ``‖field‖``; valid for all heatmap modes.
        - ``'angle'`` — direction of the *field* vector in [−π, π]; valid for all
          heatmap modes.
        - ``'magnitude_diff'`` — ``‖v_out‖ − ‖v_in‖``; signed speed change.
          Positive where the modulator amplified speed, negative where it
          suppressed it.  Uses a divergent colormap.  Only valid with
          ``heatmap='diff'``.
        - ``'angle_diff'`` — ``angle(v_out) − angle(v_in)`` wrapped to [−π, π];
          how much the modulator rotated the flow direction.  Uses a cyclic
          colormap (``'twilight_shifted'``) so that −π and +π (both a 180° flip)
          have the same color.  Only valid with ``heatmap='diff'``.
    heatmap_alpha_from_magnitude : bool, default False
        When ``True`` and ``heatmap_type='angle'``, the imshow alpha channel is
        set to the normalized velocity magnitude so regions with near-zero
        velocity become transparent.  Has no effect for ``heatmap='weight'``.

    Returns
    -------
    ModulatorPreviewResult
        Named container with all produced artists:

        - ``ax`` — the axes
        - ``arrows`` — the :class:`~matplotlib.quiver.Quiver` artist
        - ``colorbar`` — :class:`~matplotlib.colorbar.Colorbar` for arrows
        - ``geometry_collections`` — list of collections from ``gdf.plot()``;
          empty when *show_geometry* is ``False``
        - ``weight_image`` — :class:`~matplotlib.image.AxesImage` for the
          seed-weight heatmap, or ``None``
        - ``input_arrows`` — :class:`~matplotlib.quiver.Quiver` for the
          pre-modulation arrows, or ``None``
        - ``diff_arrows`` — :class:`~matplotlib.quiver.Quiver` for the
          difference field, or ``None``

    Examples
    --------
    >>> from carto_flow.flow_cartogram import preview_modulator, DirectionalTensor
    >>> mod = DirectionalTensor.radial(center=(500_000, 200_000), Dpar=3.0)
    >>> result = preview_modulator(mod, gdf)
    >>> result.arrows.set_alpha(0.8)
    >>> result.colorbar.set_label("strength")

    >>> # Density-based velocity:
    >>> result = preview_modulator(mod, gdf, column="population", show_vectors=True)
    """
    import matplotlib.pyplot as plt
    import shapely

    from .grid import Grid
    from .plot_results import ModulatorPreviewResult

    bounds = gdf.total_bounds  # (xmin, ymin, xmax, ymax)
    grid = Grid.from_bounds(bounds, size=grid_size, margin=margin)

    # Rasterize geometries to produce a proper mask (-1 = outside, k = inside geometry k)
    geom_mask = np.full(grid.shape, -1, dtype=int)
    for idx, geom in enumerate(gdf.geometry):
        if not geom.is_empty:
            inside = shapely.contains_xy(geom, grid.X, grid.Y)
            geom_mask[inside] = idx

    # Resolve values from column if given
    if column is not None:
        values = gdf[column].to_numpy()

    if values is not None:
        from .density import compute_density_field_from_geometries
        from .velocity import VelocityComputerFFTW

        rho = compute_density_field_from_geometries(gdf.geometry, values, grid)

        # Input: isotropic baseline (Dx=Dy=1)
        vx_in, vy_in = VelocityComputerFFTW(grid, Dx=1, Dy=1).compute(rho)
        vmax_in = np.nanmax(np.sqrt(vx_in**2 + vy_in**2))
        if vmax_in > 1e-12:
            vx_in /= vmax_in
            vy_in /= vmax_in

        # Output: anisotropic velocity (Dx, Dy); skip second FFT when identity
        if Dx != 1.0 or Dy != 1.0:
            vx_out, vy_out = VelocityComputerFFTW(grid, Dx=Dx, Dy=Dy).compute(rho)
            if vmax_in > 1e-12:
                vx_out /= vmax_in  # same scale as input for comparability
                vy_out /= vmax_in
        else:
            vx_out, vy_out = vx_in.copy(), vy_in.copy()
    else:
        # Input: unit probe field (no Dx/Dy)
        vx_in = np.full(grid.shape, np.cos(input_angle))
        vy_in = np.full(grid.shape, np.sin(input_angle))

        # Output: Dx/Dy-scaled probe field
        vx_out = np.full(grid.shape, np.cos(input_angle) * Dx)
        vy_out = np.full(grid.shape, np.sin(input_angle) * Dy)

    # Apply modulator to the output field
    if modulator is not None:
        vx_out, vy_out = modulator(vx_out, vy_out, grid, geom_mask)

    # Post-modulation normalization (mirrors algorithm.py:392-395)
    # if values is not None:
    vmax2 = np.nanmax(np.sqrt(vx_out**2 + vy_out**2))
    if vmax2 > 1e-12:
        vx_out /= vmax2
        vy_out /= vmax2

    magnitude = np.sqrt(vx_out**2 + vy_out**2)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    # Optional geometry outlines
    geom_collections = []
    if show_geometry:
        before = len(ax.collections)
        gdf.plot(ax=ax, facecolor="none", edgecolor="0.6", linewidth=0.6, zorder=1)
        geom_collections = list(ax.collections[before:])

    # Normalize show_vectors into a set of layer names
    def _sv_set(sv):
        if sv is True:
            return {"input", "output"}
        if not sv:
            return set()
        if isinstance(sv, str):
            return {sv}
        return set(sv)

    _show_v = _sv_set(show_vectors)

    # Unified heatmap block: 'weight' or velocity field (input/output/diff)
    heatmap_image = None
    heatmap_cb = None
    if heatmap is not None:
        gxmin, gymin, gxmax, gymax = grid.bounds
        if heatmap == "weight":
            if isinstance(modulator, LocalizedTensor):
                xs = np.array([p[0] for p in modulator._parsed])
                ys = np.array([p[1] for p in modulator._parsed])
                Ps = np.stack([p[2] for p in modulator._parsed])
                _dx = grid.X[..., np.newaxis] - xs
                _dy = grid.Y[..., np.newaxis] - ys
                P00, P01, P11 = Ps[:, 0, 0], Ps[:, 0, 1], Ps[:, 1, 1]
                quad = P00 * _dx**2 + 2.0 * P01 * _dx * _dy + P11 * _dy**2
                W = np.clip(np.exp(-0.5 * quad).sum(axis=-1), 0.0, 1.0)
                _hkw = {
                    "origin": "lower",
                    "extent": [gxmin, gxmax, gymin, gymax],
                    "cmap": heatmap_cmap or "Blues",
                    "alpha": 0.35,
                    "vmin": 0.0,
                    "vmax": 1.0,
                    "zorder": 2,
                    **(heatmap_kwargs or {}),
                }
                heatmap_image = ax.imshow(W, **_hkw)
                if show_colorbar:
                    _hcb_kw = {"label": "Weight", **(heatmap_colorbar_kwargs or {})}
                    heatmap_cb = plt.colorbar(heatmap_image, ax=ax, **_hcb_kw)
        elif heatmap in ("input", "output", "diff"):
            if heatmap == "input":
                _hx, _hy = vx_in, vy_in
            elif heatmap == "output":
                _hx, _hy = vx_out, vy_out
            else:
                _hx, _hy = vx_out - vx_in, vy_out - vy_in

            if heatmap_type == "magnitude":
                _hdata = np.sqrt(_hx**2 + _hy**2)
                _hcmap = heatmap_cmap or cmap
                _hclim = {"vmin": 0.0, "vmax": 1.0} if heatmap != "diff" else {"vmin": 0.0}
            elif heatmap_type == "angle":
                _hdata = np.arctan2(_hy, _hx)
                _hcmap = heatmap_cmap or "twilight"
                _hclim = {"vmin": -np.pi, "vmax": np.pi}
            elif heatmap_type == "magnitude_diff":
                if heatmap != "diff":
                    raise ValueError("heatmap_type='magnitude_diff' is only valid with heatmap='diff'")
                _hdata = np.sqrt(vx_out**2 + vy_out**2) - np.sqrt(vx_in**2 + vy_in**2)
                _hcmap = heatmap_cmap or "PiYG"
                _abs_max = float(np.nanmax(np.abs(_hdata))) or 1.0
                _hclim = {"vmin": -_abs_max, "vmax": _abs_max}
            elif heatmap_type == "angle_diff":
                if heatmap != "diff":
                    raise ValueError("heatmap_type='angle_diff' is only valid with heatmap='diff'")
                _raw = np.arctan2(vy_out, vx_out) - np.arctan2(vy_in, vx_in)
                _hdata = np.arctan2(np.sin(_raw), np.cos(_raw))  # wrap to [-π, π]
                _hcmap = heatmap_cmap or "twilight_shifted"
                _hclim = {"vmin": -np.pi, "vmax": np.pi}
            else:
                raise ValueError(
                    "heatmap_type must be 'magnitude', 'angle', 'magnitude_diff', or "
                    f"'angle_diff', got {heatmap_type!r}"
                )

            _hkw = {
                "origin": "lower",
                "extent": [gxmin, gxmax, gymin, gymax],
                "cmap": _hcmap,
                "zorder": 1,
                **_hclim,
                **(heatmap_kwargs or {}),
            }
            if heatmap_alpha_from_magnitude and heatmap_type == "angle":
                _hmag = np.sqrt(_hx**2 + _hy**2)
                _hmax = float(_hmag.max())
                _hkw["alpha"] = (_hmag / _hmax) if _hmax > 1e-12 else np.ones_like(_hmag)
            heatmap_image = ax.imshow(_hdata, **_hkw)
            if show_colorbar:
                _hlabel = {
                    "magnitude": f"{heatmap} magnitude",
                    "angle": f"{heatmap} direction",
                    "magnitude_diff": "speed change (\u2016out\u2016 \u2212 \u2016in\u2016)",
                    "angle_diff": "rotation (angle out \u2212 angle in)",
                }.get(heatmap_type, heatmap_type)
                _hcb_kw = {"label": _hlabel, **(heatmap_colorbar_kwargs or {})}
                heatmap_cb = plt.colorbar(heatmap_image, ax=ax, **_hcb_kw)
                if heatmap_type == "angle":
                    heatmap_cb.set_ticks(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) * np.pi)  # type: ignore[arg-type]
                    heatmap_cb.set_ticklabels(["W", "S", "E", "N", "W"])
                if heatmap_type == "angle_diff":
                    heatmap_cb.set_ticks(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) * np.pi)  # type: ignore[arg-type]
                    heatmap_cb.set_ticklabels(["-π", "-π/2", "0", "+π/2", "+π"])
        else:
            raise ValueError(f"heatmap must be 'weight', 'input', 'output', or 'diff', got {heatmap!r}")

    # Shared quiver scale: longest output arrow spans arrow_scale × skipped-cell diagonal.
    # scale_units='xy' puts arrows in data coordinates so the same scale is directly
    # comparable across output, diff, and input quivers.
    # zorder controls rendering order: input(2) < diff(3) < output(4).
    _diag = np.sqrt((grid.dx * skip) ** 2 + (grid.dy * skip) ** 2)
    _out_max = float(np.nanmax(magnitude))
    _ref_scale = _out_max / (arrow_scale * _diag) if _diag > 0 and _out_max > 1e-12 else None
    _scale_kw = {"scale": _ref_scale, "scale_units": "xy"} if _ref_scale is not None else {}

    q = None
    cb = None
    if "output" in _show_v:
        _arrows_kw = {"cmap": cmap, "zorder": 4, "clim": (0.0, 1.0), **_scale_kw, **(arrows_kwargs or {})}
        q = ax.quiver(
            grid.X[::skip, ::skip],
            grid.Y[::skip, ::skip],
            vx_out[::skip, ::skip],
            vy_out[::skip, ::skip],
            magnitude[::skip, ::skip],
            **_arrows_kw,
        )
        if show_colorbar:
            _colorbar_kw = {
                "label": (
                    "Output velocity magnitude" if values is not None else "Amplification (output / input magnitude)"
                ),
                **(colorbar_kwargs or {}),
            }
            cb = plt.colorbar(q, ax=ax, **_colorbar_kw)

    # Difference quiver
    q_diff = None
    cb_diff = None
    if "diff" in _show_v:
        vx_diff = vx_out - vx_in
        vy_diff = vy_out - vy_in
        diff_magnitude = np.sqrt(vx_diff**2 + vy_diff**2)
        _diff_kw = {"cmap": diff_cmap, "zorder": 3, "clim": (0.0, None), **_scale_kw, **(diff_arrows_kwargs or {})}
        q_diff = ax.quiver(
            grid.X[::skip, ::skip],
            grid.Y[::skip, ::skip],
            vx_diff[::skip, ::skip],
            vy_diff[::skip, ::skip],
            diff_magnitude[::skip, ::skip],
            **_diff_kw,
        )
        cb_diff = None
        if show_colorbar:
            _diff_cb_kw = {"label": "Difference magnitude", **(diff_colorbar_kwargs or {})}
            cb_diff = plt.colorbar(q_diff, ax=ax, **_diff_cb_kw)

    # Input quiver
    q_in = None
    if "input" in _show_v:
        _in_kw = {"color": "0.6", "alpha": 0.4, "zorder": 2, **_scale_kw, **(input_arrows_kwargs or {})}
        q_in = ax.quiver(
            grid.X[::skip, ::skip],
            grid.Y[::skip, ::skip],
            vx_in[::skip, ::skip],
            vy_in[::skip, ::skip],
            **_in_kw,
        )

    ax.set_aspect("equal")
    _dx_dy = f"Dx/Dy={Dx / Dy:.3g}" if (Dx != 1.0 or Dy != 1.0) else None
    _mod_str = repr(modulator) if modulator is not None else None
    _title_parts = [p for p in [_dx_dy, _mod_str] if p is not None]
    ax.set_title(" + ".join(_title_parts) if _title_parts else "Identity (no modulator)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return ModulatorPreviewResult(
        ax=ax,
        arrows=q,  # type: ignore[arg-type]
        colorbar=cb,  # type: ignore[arg-type]
        geometry_collections=geom_collections,
        input_arrows=q_in,
        diff_arrows=q_diff,
        diff_colorbar=cb_diff,
        heatmap_image=heatmap_image,
        heatmap_colorbar=heatmap_cb,
    )
