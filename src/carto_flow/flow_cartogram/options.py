"""
Configuration options and status enums for morphing algorithms.

Configuration classes for controlling cartogram generation behavior.

Classes
-------
MorphStatus
    Enum for morphing operation status values.
MorphOptions
    Dataclass with comprehensive validation.
MorphOptionsError
    Base exception for validation errors.

Examples
--------
>>> from carto_flow.flow_cartogram import MorphOptions
>>>
>>> # Use preset configurations
>>> options = MorphOptions.preset_fast()
>>>
>>> # Or customize parameters
>>> options = MorphOptions(grid=256, n_iter=100, mean_tol=0.05)
"""

import warnings
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Literal, Optional

from .grid import Grid, GridSpec, LadderSpec

if TYPE_CHECKING:
    from .anisotropy import VelocityModulator
    from .density import DensityModulator

__all__ = [
    "MorphOptions",
    "MorphOptionsConsistencyError",
    "MorphOptionsError",
    "MorphOptionsValidationError",
    "MorphStatus",
]


class MorphStatus(str, Enum):
    """Status values for morphing operations.

    Inherits from str for backward compatibility with string comparisons.

    Values
    ------
    ORIGINAL : str
        Original unmorphed state (no morphing performed)
    CONVERGED : str
        Algorithm converged within tolerance thresholds
    STALLED : str
        Algorithm stopped because ``stall_patience`` consecutive cycles
        (windows of ``max(recompute_every, 10)`` iterations) made no progress
    COMPLETED : str
        Algorithm ran all ``n_iter`` iterations without converging or stalling
    RUNNING : str
        Algorithm is still running (intermediate status)
    FAILED : str
        Algorithm failed due to an error
    """

    ORIGINAL = "original"
    CONVERGED = "converged"
    STALLED = "stalled"
    COMPLETED = "completed"
    RUNNING = "running"
    FAILED = "failed"

    def __str__(self) -> str:
        return self.value


# ============================================================================
# Custom Exceptions
# ============================================================================


class MorphOptionsError(ValueError):
    """Base exception for MorphOptions validation errors."""

    pass


class MorphOptionsValidationError(MorphOptionsError):
    """Field validation errors (type, range, format)."""

    pass


class MorphOptionsConsistencyError(MorphOptionsError):
    """Logical consistency errors between related fields."""

    pass


# ============================================================================
# Configuration Data Class
# ============================================================================


@dataclass
class MorphOptions:
    """Configuration options for morphing algorithms with comprehensive validation.

    This class validates all options to ensure numerical stability and logical
    consistency for cartogram generation algorithms. Validation occurs during
    construction, direct attribute assignment, and when creating modified copies.

    Grid specification (``grid``):
        - an int or ``(nx, ny)``: one grid of that many points, built from the bounds of the data
          with a margin of 0.5 and rectangular cells
        - ``GridSpec(size, margin, square)``: one grid with its margin and squareness
        - ``LadderSpec(min_resolution, levels, margin, square)`` or ``"auto"`` (``LadderSpec()``): a ladder of
          resolutions that ``create_flow_cartogram`` and ``CartogramWorkflow.morph`` run from coarse to fine
        - a ``Grid``: a prebuilt grid in data coordinates
        - ``None``: a 100 x 100 grid

    Note
    ----
    ``grid_size`` is a deprecated alias of ``grid``. ``grid_margin`` and ``grid_square`` are deprecated:
    they apply to an int or tuple ``grid`` (defaults 0.5 and False); put them in a ``GridSpec`` or
    ``LadderSpec`` instead.

    Validation
    ----------
    - All numeric parameters are validated for type and range constraints
    - Logical consistency between related parameters is enforced
    - Invalid values raise MorphOptionsValidationError with detailed messages
    - Validation occurs on construction, direct assignment, and copy operations
    """

    # Grid construction parameters
    grid: int | tuple[int | None, int | None] | GridSpec | LadderSpec | Grid | Literal["auto"] | None = 256
    grid_size: int | tuple[int | None, int | None] | GridSpec | LadderSpec | Literal["auto"] | None = None
    grid_margin: float | None = None
    grid_square: bool | None = None

    # Computation options
    dt: float = 0.2
    n_iter: int = 500
    """Maximum number of iterations. For a ladder of resolutions (``LadderSpec`` or ``"auto"`` in ``grid``) it is
    the total over all levels: each level can use what the previous levels left."""
    recompute_every: int | None = 10
    """Maximum number of iterations a computed velocity field is reused; ``recompute_on_rise`` can recompute it earlier
    (None: not recomputed on the clock)."""
    recompute_on_rise: float | None = 0.01
    """Relative rise of the score that triggers an early recompute of the velocity field.

    This adds to ``recompute_every`` and does not replace it. If None, the field is recomputed only every
    ``recompute_every`` iterations. Otherwise (default 0.01) it is also
    recomputed before the next iteration whenever the score ``max(mean_error / mean_tol,
    max_error / max_tol)`` (log2 errors) rose by more than this fraction (0.0 for any rise, 0.01 for
    1%) compared with the previous iteration, provided the field has been used for at least one
    iteration. ``recompute_every`` stays the maximum interval between recomputes. The window used for
    stall detection does not depend on when recomputes happen.
    """
    adaptive_dt: bool = True
    """Shrink the step when the area errors alternate in sign.

    The step moves the fastest point ``dt`` grid cells whatever the error, so close to the target
    each step overshoots: the areas alternate around their targets with an amplitude proportional to
    ``dt``, and a tolerance below that amplitude is met only when the oscillation happens to dip
    below it. With ``adaptive_dt`` a region counts as inverted if the sign of its area error
    (current area against target area) changed in any of the last 3 iterations, between two
    iterations in which its log2 error exceeds 10 % of ``log2(1 + max_tol)`` (a tight tolerance
    sees the sign of smaller errors). Far from the target almost no region inverts. When more than
    30 % of the regions are inverted the step is multiplied by ``1 - 0.6 * share`` (share of
    inverted regions), down to ``dt / 8``; the next 3 iterations cannot shrink it again. The step
    never grows back. Apart from that dead band the rule does not depend on the tolerances, and it
    does not depend on ``recompute_every`` or ``recompute_on_rise``.

    During the first ``max(recompute_every, 10)`` iterations the step is also limited to the score
    of the current state divided by 10 (not below ``dt / 8``), so a run that continues from a
    geometry near the target (a multiresolution level, a refinement, a second phase) does not start
    with a full step that undoes it. A run that starts far from the target (score above 10) is not
    affected.

    False: constant ``dt``.
    """
    snapshot_every: int | None = None
    mean_tol: float = 0.05  # Percentage tolerance, e.g., 0.05 = 5%
    max_tol: float = 0.10  # Percentage tolerance, e.g., 0.10 = 10%

    # Anisotropy options
    Dx: float = 1.0  # Flow amplification in x: higher = stronger horizontal flow
    Dy: float = 1.0  # Flow amplification in y: higher = stronger vertical flow
    anisotropy: Optional["VelocityModulator"] = None
    density_mod: Optional["DensityModulator"] = None

    # Unit scaling options
    area_scale: float = 1.0  # Multiplier for area values (e.g., 1e6 to convert m² to km²)

    prescale_components: bool | None = None
    """Pre-scale each connected component to its target total area before morphing.

    When True, the algorithm detects groups of adjacent geometries (connected
    components) and uniformly scales each group so its total area equals the
    area implied by the global target density. This reduces distortion of the outer boundary
    during morphing, especially for maps with multiple disconnected regions (e.g.,
    archipelagos, separate land masses). By pre-scaling components to their target areas,
    the algorithm can focus on redistributing density within each component rather than
    applying large global transformations that can excessively distort the outer boundary.
    The landmarks are scaled with the component that contains them; the displacement coordinates are only with
    ``prescale_coords``.

    None (default) means False, and ``create_voronoi_cartogram(premorph=...)`` then follows
    ``VoronoiOptions.prescale_components``; an explicit True or False is used as given.
    """
    prescale_coords: bool = False
    """Also scale the displacement coordinates with the component that contains them when ``prescale_components`` is on.

    The landmarks follow the geometries they refer to, so they are always scaled with them. Displacement coordinates
    are not tied to the geometries: a grid of points that is carried along to show the deformation of the plane may
    be wanted without the pre-scaling, points that mark positions on the map with it. With False (default) the
    displacement coordinates start where they are given. To mix both kinds, pass the points that follow the
    geometries as ``landmarks`` (point geometries are accepted) and the others as ``displacement_coords``.
    """

    # Output options
    landmark_probes: int = 0
    """Number of probe points per region for ``get_landmarks(uniform=True)`` and ``get_coords(uniform=True)``.

    Points that are uniform in a region before the morph crowd the border of an expanding region, because the flow
    scales the area less near the border. The correction needs the crowding as a function of the distance to the border,
    which probe points, uniform in every region before the morph and carried with the morph by ``create_flow_cartogram``
    and ``CartogramWorkflow``, measure independently of the landmarks. With 0 (default) no probes are carried, and
    ``get_landmarks(uniform=True)`` uses the point landmarks themselves, which works when they were uniform in their
    region and numerous (dot density, about 30 per region). Lines, polygons, other point sets and displacement
    coordinates need probes; 50 per region are enough. Each saved snapshot stores the position and area scaling of the probes.
    """
    save_internals: bool = False
    show_progress: bool = True
    progress_message: str | None = None
    benchmark: bool = False
    """Record per-phase wall-clock time inside the morphing loop.

    When True, a Benchmark object is attached to the returned Cartogram
    as ``cartogram.benchmark``.  Useful for diagnosing which phase
    dominates runtime and whether parallelism overhead outweighs its benefit.
    """

    error_floor: float = 1e-3
    """Smallest area the error metrics resolve, as a fraction of the mean target area.

    Before taking the log2 ratio, both the current and the target area of a region are raised to
    ``error_floor * mean(target areas)``, so a collapsed or zero-valued region has a finite error and
    a region below the floor counts as on target while it stays below it. The floor does not depend
    on the grid, so the reported errors (``mean_error_pct``, ``max_error_pct``, the convergence history
    and the snapshot errors) are comparable between grid sizes and between the levels of a
    multiresolution morph. ``0`` disables the floor (a zero area then gives an infinite error).
    """

    density_rasterization: str = "exact"
    """How the density is rasterized onto the grid.

    - ``"exact"`` (default): each cell takes the area-weighted mean density over the cell, so every
      region takes part in proportion to its area and the integral of the density over the grid
      equals the total value. Costs about as much as ``"center"`` (less from grid 256 up).
      ``parallel_density`` is ignored.
    - ``"center"``: each cell takes the density of the geometry that contains its center. Every
      region is quantized to whole cells, and a region smaller than a cell that contains no cell
      center is invisible to the flow.
    """

    # Parallel computation options
    parallel_fft: bool | int = False
    """Control FFTW thread count for the velocity (Poisson solve) phase.

    FFT parallelism benefits emerge at grid ≥ 1024 (~1.2x) and become
    substantial at ≥ 2048 (~1.8-2x). At smaller grids, thread-launch overhead
    exceeds the benefit. Displacement and area phases always run serially.

    - False: 1 FFT thread (serial)
    - True: max(cpu_count()-1, 1) threads
    - N (int > 1): exactly N threads (capped at cpu_count()-1)
    """

    parallel_density: bool | int = False
    """Control thread count for the density rasterisation phase (ThreadPoolExecutor).

    Whether threading helps depends on geometry count and grid size:

    - Few large geometries (e.g. ~50 states, grid ≥ 1024): ~1.4x speedup
    - Many small geometries (e.g. ~430 districts): threading is slower because
      Future-submission overhead exceeds per-geometry work

    - False: serial density loop
    - True: all available cores (os.cpu_count())
    - N (int > 1): exactly N worker threads
    """

    # Stall detection
    stall_patience: int | None = 4
    """Number of consecutive cycles without progress that ends the run as stalled.

    The velocity field is reused between recomputes, so the error follows a sawtooth. Stall
    detection therefore judges windows of ``max(recompute_every, 10)`` iterations ("cycles",
    10 iterations if ``recompute_every`` is below 10 or None), counted from the first iteration
    and independent of when recomputes actually happen (also with ``recompute_on_rise``). For each
    completed window the minimum of the mean-error ratio ``mean_error / mean_tol`` and the minimum
    of the max-error ratio ``max_error / max_tol`` (log2 errors, as in the convergence test) are
    recorded; a ratio below 1 means that component is satisfied. A component counts toward
    progress only while it is still violated (its window minimum is above 1) and its window
    minimum is lower than that component's best over the earlier windows by at least the improvement
    required by ``stall_min_efficiency``. The window is progress if any component counts, so a max error that
    stays fixed while a still violated mean error falls is progress, whereas improvements of a
    satisfied component, or of a component that regressed above 1 without beating its earlier
    best, are not. A window in which neither component is violated does not count either.
    The run stops with status ``STALLED`` after ``stall_patience``
    consecutive windows without progress, so a stall is detected after at least ``stall_patience``
    windows, and an incomplete last window is not judged.

    When the run does not converge, the returned state is the best iterate (see
    ``Cartogram.best_iteration``).

    If None, stall detection is disabled and the algorithm runs until convergence or ``n_iter``.
    """

    stall_min_efficiency: float = 0.05
    """Improvement of a cycle minimum per unit of distance moved that counts as progress (see ``stall_patience``).

    The expected improvement of an iteration follows from the step: the fastest point moves ``dt`` times the step
    fraction of the adaptive step cells, and a region of linear size ``L`` changes its area, and with it its error, by a
    fraction of order ``distance / L``. A cycle therefore has to lower a violated component's minimum below its best
    earlier minimum by at least ``stall_min_efficiency`` times the distance moved in the cycle divided by ``L``, the
    square root of the area-weighted mean target area. Smaller decreases do not reset the stall count. The required
    improvement thus falls with the cell size (a finer grid moves less per iteration), with a smaller ``dt`` or
    step fraction, and with larger regions, so that a run that makes steady progress is not ended as stalled. At 0.05
    a cycle of 10 iterations at the default ``dt`` of 0.2 on regions of 8 cells has to improve by 1%.
    """

    def get_grid(self, bounds: tuple[float, float, float, float]) -> "Grid":
        """Get the Grid object for computation with clear precedence.

        Parameters
        ----------
        bounds : Tuple[float, float, float, float]
            Bounding box as (min_x, min_y, max_x, max_y)

        Returns
        -------
        Grid
            Grid object for spatial discretization
        """
        from .grid import Grid

        if isinstance(self.grid, Grid):
            # Grid object provided - use it, ignore other grid params
            return self.grid
        spec = self.grid_spec
        if isinstance(spec, LadderSpec):
            raise ValueError(  # noqa: TRY004
                "grid describes a ladder of resolutions; run it with create_flow_cartogram or CartogramWorkflow.morph"
            )
        if spec is not None:
            return Grid.from_bounds(bounds, spec.size, spec.margin, spec.square)
        # Default grid
        return Grid.from_bounds(bounds, size=100)

    @property
    def grid_spec(self) -> "GridSpec | LadderSpec | None":
        """``grid`` as a ``GridSpec`` or ``LadderSpec`` (None when ``grid`` is None or a ``Grid``)."""
        size = self.grid
        if size is None or isinstance(size, Grid):
            return None
        if isinstance(size, (GridSpec, LadderSpec)):
            return size
        if isinstance(size, str):
            return LadderSpec()
        return GridSpec(
            tuple(size) if isinstance(size, list) else size,
            0.5 if self.grid_margin is None else self.grid_margin,
            False if self.grid_square is None else self.grid_square,
        )

    def copy_with(self, **kwargs) -> "MorphOptions":
        """Create a copy of this MorphOptions instance with specified options overridden.

        Parameters
        ----------
        **kwargs
            Keyword arguments corresponding to MorphOptions fields to override

        Returns
        -------
        MorphOptions
            New MorphOptions instance with the specified options overridden

        Examples
        --------
        Create new options with different iteration count and time step:

        >>> new_opts = opts.copy_with(n_iter=200, dt=0.15)

        Create options with different tolerance values:

        >>> strict_opts = opts.copy_with(mean_tol=0.01, max_tol=0.02)

        Chain multiple modifications:

        >>> final_opts = opts.copy_with(n_iter=150).with_options(dt=0.1)
        """
        # Validate that all provided kwargs are valid MorphOptions fields
        valid_fields = {field.name for field in self.__dataclass_fields__.values()}
        invalid_fields = set(kwargs.keys()) - valid_fields
        if invalid_fields:
            raise ValueError(f"Invalid field(s): {invalid_fields}. Valid fields are: {valid_fields}")

        # Use dataclasses.replace to create a new instance with overridden fields
        # This will automatically trigger validation via __post_init__
        try:
            new_options = replace(self, **kwargs)
        except Exception as e:
            # Re-raise validation errors with additional context about which fields were being modified
            field_list = ", ".join(kwargs.keys())
            if isinstance(e, MorphOptionsValidationError):
                raise MorphOptionsValidationError(f"Failed to create options with {field_list}: {e!s}") from e
            else:
                # Re-raise other exceptions (like dataclass field errors) as-is
                raise
        else:
            return new_options

    def __post_init__(self):
        """Validate options after dataclass initialization."""
        if self.grid_size is not None:
            self._warn_grid_size()
            self.grid = self.grid_size
            self.grid_size = None
        self._validate()
        for name in ("grid_margin", "grid_square"):
            if getattr(self, name) is not None:
                warnings.warn(
                    f"MorphOptions.{name} is deprecated and will be removed in the next major release; "
                    "pass a GridSpec or LadderSpec as grid instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )
        # Mark that validation has been initialized
        super().__setattr__("_validation_initialized", True)

    @staticmethod
    def _warn_grid_size() -> None:
        warnings.warn(
            "MorphOptions.grid_size is deprecated and will be removed in the next major release; use grid.",
            DeprecationWarning,
            stacklevel=4,
        )

    def __setattr__(self, name: str, value) -> None:
        """Validate individual fields when assigned directly."""
        if name == "grid_size" and hasattr(self, "_validation_initialized"):
            # Deprecated alias: assign to grid
            if value is None:
                return
            self._warn_grid_size()
            name = "grid"
        # Only validate after full initialization to avoid recursion during __init__
        # Check if we're fully initialized by looking for a marker attribute
        if (
            hasattr(self, "_validation_initialized") and name != "_validation_initialized" and not name.startswith("_")
        ):  # Skip private attributes
            # Validate the new value before setting it
            self._validate_single_field(name, value)

        # Set the attribute (only if validation passed or during initialization)
        super().__setattr__(name, value)

    def _validate(self) -> None:
        """Comprehensive validation of all fields and their consistency."""
        errors = []

        # Field-level validation
        field_errors = self._validate_all_fields()
        errors.extend(field_errors)

        # Logical consistency validation
        consistency_errors = self._validate_consistency()
        errors.extend(consistency_errors)

        if errors:
            raise MorphOptionsValidationError("; ".join(errors))

    def _validate_all_fields(self) -> list[str]:
        """Validate all individual fields and return list of error messages."""
        errors = []

        # Validate each field using the single validation method
        field_names = [
            "grid",
            "grid_margin",
            "grid_square",
            "dt",
            "n_iter",
            "recompute_every",
            "snapshot_every",
            "mean_tol",
            "max_tol",
            "Dx",
            "Dy",
            "anisotropy",
            "density_mod",
            "area_scale",
            "save_internals",
            "landmark_probes",
            "show_progress",
            "progress_message",
            "benchmark",
            "prescale_components",
            "prescale_coords",
            "adaptive_dt",
            "stall_patience",
            "stall_min_efficiency",
            "recompute_on_rise",
            "error_floor",
            "density_rasterization",
            "parallel_fft",
            "parallel_density",
        ]

        for field_name in field_names:
            field_error = self._validate_field_value(field_name, getattr(self, field_name))
            if field_error:
                errors.append(field_error)

        return errors

    def _validate_field_value(self, field_name: str, value) -> str | None:
        """Single source of truth for field validation logic."""
        # Grid parameters
        if field_name == "grid":
            if value is not None:
                return self._validate_grid(value)
        elif field_name == "grid_margin":
            if value is not None and (not isinstance(value, (int, float)) or value < 0):
                return "grid_margin must be a non-negative number"
        elif field_name == "grid_square":
            if value is not None and not isinstance(value, bool):
                return "grid_square must be a boolean"
        elif field_name == "density_rasterization":
            if value not in ("center", "exact"):
                return 'density_rasterization must be "center" or "exact"'
        elif field_name in ("parallel_fft", "parallel_density"):
            if not isinstance(value, (bool, int)) or (
                isinstance(value, int) and not isinstance(value, bool) and value <= 0
            ):
                return f"{field_name} must be True, False, or a positive integer"

        # Computation parameters
        elif field_name == "dt":
            if not isinstance(value, (int, float)) or value <= 0 or value > 1.0:
                return "dt must be > 0 and <= 1.0"
        elif field_name == "n_iter":
            if not isinstance(value, int) or value <= 0:
                return "n_iter must be a positive integer"
        elif field_name == "recompute_every":
            if value is not None and (not isinstance(value, int) or value <= 0):
                return "recompute_every must be a positive integer"
        elif field_name == "snapshot_every":
            if value is not None and (not isinstance(value, int) or value <= 0):
                return "snapshot_every must be a positive integer"
        elif field_name == "mean_tol":
            if not isinstance(value, (int, float)) or value <= 0 or value > 1.0:
                return "mean_tol must be > 0 and <= 1.0"
        elif field_name == "max_tol":
            if not isinstance(value, (int, float)) or value <= 0 or value > 1.0:
                return "max_tol must be > 0 and <= 1.0"

        # Anisotropy parameters
        elif field_name == "Dx":
            if not isinstance(value, (int, float)) or value <= 0:
                return "Dx must be a positive number"
        elif field_name == "Dy":
            if not isinstance(value, (int, float)) or value <= 0:
                return "Dy must be a positive number"
        elif field_name == "anisotropy":
            if value is not None:
                from .anisotropy import VelocityModulator

                if not isinstance(value, VelocityModulator):
                    return "anisotropy must be a VelocityModulator instance or None"

        elif field_name == "density_mod":
            if value is not None:
                from .density import DensityModulator

                if not isinstance(value, DensityModulator):
                    return "density_mod must be a DensityModulator instance or None"

        # Unit scaling parameters
        elif field_name == "area_scale":
            if not isinstance(value, (int, float)) or value <= 0:
                return "area_scale must be a positive number"

        # Output parameters
        elif field_name == "landmark_probes":
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                return "landmark_probes must be a non-negative integer"
        elif field_name == "save_internals":
            if not isinstance(value, bool):
                return "save_internals must be a boolean"
        elif field_name == "show_progress":
            if not isinstance(value, bool):
                return "show_progress must be a boolean"
        elif field_name == "progress_message" and value is not None and not isinstance(value, str):
            return "progress_message must be a string or None"
        elif field_name == "benchmark" and not isinstance(value, bool):
            return "benchmark must be a boolean"

        # Stall detection options
        elif field_name == "stall_patience":
            if value is not None:
                if not isinstance(value, int) or isinstance(value, bool):
                    return "stall_patience must be an integer or None"
                elif value < 0:
                    return "stall_patience must be a non-negative integer or None"

        elif field_name == "error_floor":
            if not isinstance(value, int | float) or isinstance(value, bool) or not 0 <= value < 1:
                return "error_floor must be a number in [0, 1)"
        elif field_name == "recompute_on_rise":
            if value is not None and (not isinstance(value, int | float) or isinstance(value, bool) or value < 0):
                return "recompute_on_rise must be a non-negative number or None"
        elif field_name == "stall_min_efficiency":
            if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
                return "stall_min_efficiency must be a non-negative number"

        # Outer-boundary distortion reduction options
        elif field_name == "adaptive_dt" and not isinstance(value, bool):
            return "adaptive_dt must be a boolean"
        elif field_name == "prescale_components" and value is not None and not isinstance(value, bool):
            return "prescale_components must be a boolean or None"
        elif field_name == "prescale_coords" and not isinstance(value, bool):
            return "prescale_coords must be a boolean"

        return ""  # No errors

    def _validate_consistency(self) -> list[str]:
        """Validate logical consistency between related fields."""
        errors = []

        # Tolerance ordering
        if self.mean_tol > self.max_tol:
            errors.append(f"mean_tol ({self.mean_tol}) must be <= max_tol ({self.max_tol})")

        # Anisotropy consistency
        if self.anisotropy is not None and (self.Dx != 1.0 or self.Dy != 1.0):
            errors.append("cannot combine anisotropy modulator with Dx/Dy != 1.0")

        return errors

    def _validate_grid(self, grid) -> str | None:
        """Validate the grid specification."""
        if isinstance(grid, (Grid, GridSpec, LadderSpec)):
            return None
        if isinstance(grid, str):
            return None if grid == "auto" else "grid as a string must be 'auto'"
        if isinstance(grid, int):
            if grid <= 0:
                return "grid as int must be positive"
        elif isinstance(grid, (tuple, list)) and len(grid) == 2:
            for i, size in enumerate(grid):
                if size is not None and (not isinstance(size, int) or size <= 0):
                    return f"grid[{i}] must be a positive integer or None"
        else:
            return "grid must be a positive integer, a tuple of (int, int), a GridSpec, a LadderSpec, 'auto' or a Grid"

        return None

    def _validate_single_field(self, field_name: str, value) -> None:
        """Validate a single field and its consistency with other fields."""
        errors = []

        # Validate the specific field using single source of truth
        field_errors = self._validate_field_value(field_name, value)
        if field_errors:
            errors.append(field_errors)

        # Check logical consistency with current object state
        # Create a temporary copy to check consistency without modifying self
        temp_kwargs = {field_name: value}
        temp_obj = replace(self, **temp_kwargs)
        consistency_errors = temp_obj._validate_consistency()
        if consistency_errors:
            errors.extend(consistency_errors)

        if errors:
            raise MorphOptionsValidationError(f"Invalid {field_name}={value}: {', '.join(errors)}")

    @staticmethod
    def _calculate_bounds_from_geometries(geometries) -> tuple[float, float, float, float]:
        """
        Calculate bounding box from a list of geometries.

        This helper method computes the overall bounds (min_x, min_y, max_x, max_y)
        that encompass all provided geometries. Used internally for automatic grid
        creation when no explicit bounds are provided.

        Parameters
        ----------
        geometries : iterable
            Collection of shapely geometries with bounds attribute

        Returns
        -------
        Tuple[float, float, float, float]
            Bounding box as (min_x, min_y, max_x, max_y)

        Raises
        ------
        AttributeError
            If geometries don't have bounds attribute
        """

        # Calculate bounds from all geometries
        bounds = None
        for geom in geometries:
            if bounds is None:
                bounds = geom.bounds
            else:
                bounds = (
                    min(bounds[0], geom.bounds[0]),
                    min(bounds[1], geom.bounds[1]),
                    max(bounds[2], geom.bounds[2]),
                    max(bounds[3], geom.bounds[3]),
                )
        return bounds  # type: ignore[return-value]

    @classmethod
    def preset_fast(cls) -> "MorphOptions":
        """Create options for a quick preview.

        A coarse grid (128), a larger step (``dt=0.3``) and loose tolerances (15 % mean, 25 % maximum
        area error) for exploring a map. The regions are only approximately on target, and a coarse
        grid does not resolve inputs with many small regions, such as the congressional districts.

        Returns
        -------
        MorphOptions
            Configuration for fast preview morphing

        Examples
        --------
        >>> options = MorphOptions.preset_fast()
        >>> result = create_flow_cartogram(gdf, 'population', options=options)
        """
        return cls(
            grid=128,
            n_iter=150,
            dt=0.3,
            mean_tol=0.15,
            max_tol=0.25,
            show_progress=True,
        )

    @classmethod
    def preset_balanced(cls) -> "MorphOptions":
        """Create options with balanced speed/quality trade-off.

        The same values as ``MorphOptions()``: grid 256, ``dt=0.2`` and tolerances of 5 % mean and 10 %
        maximum area error. This is the setting for most uses, and the one ``premorph`` of
        :func:`~carto_flow.voronoi_cartogram.create_voronoi_cartogram` builds on.

        Returns
        -------
        MorphOptions
            Configuration for balanced morphing

        Examples
        --------
        >>> options = MorphOptions.preset_balanced()
        >>> result = create_flow_cartogram(gdf, 'population', options=options)
        """
        return cls(
            grid=256,
            n_iter=500,
            dt=0.2,
            mean_tol=0.05,
            max_tol=0.1,
            show_progress=True,
        )

    @classmethod
    def preset_high_quality(cls) -> "MorphOptions":
        """Create options for publication-quality results.

        A fine grid (512), a small step (``dt=0.1``) and strict tolerances (1 % mean, 3 % maximum area
        error). The small step needs about twice the iterations of ``dt=0.2``; the maximum error
        at the end is lower and fewer regions end self-intersecting. Inputs with many regions smaller than
        a grid cell, such as the 3,108 counties, do not converge with any preset.

        Returns
        -------
        MorphOptions
            Configuration for high-quality morphing

        Examples
        --------
        >>> options = MorphOptions.preset_high_quality()
        >>> result = create_flow_cartogram(gdf, 'population', options=options)
        """
        return cls(
            grid=512,
            n_iter=1000,
            dt=0.1,
            mean_tol=0.01,
            max_tol=0.03,
            show_progress=True,
        )
