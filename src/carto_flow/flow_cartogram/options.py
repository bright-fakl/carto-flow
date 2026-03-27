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
>>> options = MorphOptions(grid_size=256, n_iter=100, mean_tol=0.05)
"""

from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .anisotropy import VelocityModulator
    from .density import DensityModulator
    from .grid import Grid

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
        Algorithm stopped improving (error increasing)
    COMPLETED : str
        Algorithm completed all iterations without converging
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

    Grid specification (in order of precedence):
        1. grid: Pre-constructed Grid object (highest precedence)
        2. grid_size + grid_margin + grid_square: Construction parameters
        3. Default: 100x100 grid if nothing specified

    Note
    ----
    When 'grid' is provided, 'grid_size', 'grid_margin', and 'grid_square' are ignored.

    Validation
    ----------
    - All numeric parameters are validated for type and range constraints
    - Logical consistency between related parameters is enforced
    - Invalid values raise MorphOptionsValidationError with detailed messages
    - Validation occurs on construction, direct assignment, and copy operations
    """

    # Grid construction parameters
    grid: Optional["Grid"] = None  # For advanced users
    grid_size: int | tuple[int | None, int | None] | None = 256
    grid_margin: float = 0.5
    grid_square: bool = False

    # Computation options
    dt: float = 0.2
    n_iter: int = 500
    recompute_every: int | None = 10
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

    prescale_components: bool = False
    """Pre-scale each connected component to its target total area before morphing.

    When True, the algorithm detects groups of adjacent geometries (connected
    components) and uniformly scales each group so its total area equals the
    area implied by the global target density. This reduces distortion of the outer boundary
    during morphing, especially for maps with multiple disconnected regions (e.g.,
    archipelagos, separate land masses). By pre-scaling components to their target areas,
    the algorithm can focus on redistributing density within each component rather than
    applying large global transformations that can excessively distort the outer boundary.
    """

    # Output options
    save_internals: bool = False
    show_progress: bool = True
    progress_message: str | None = None

    # Rasterization optimization options
    use_bounding_box_filter: bool = True
    """Use bounding box pre-filtering before point-in-polygon checks.

    This can significantly speed up density field computation by first filtering
    grid cells to only those within each geometry's bounding box before
    performing the more expensive point-in-polygon checks.
    """

    # Parallel computation options
    parallel: bool | int = True
    """Control parallel computation across all subsystems.

    - True (default): use all available cores (max(cpu_count()-1, 1) for FFT, all cores for density/areas)
    - False: serial everywhere (1 FFT thread, no parallel density or area computation)
    - N (int > 1): use exactly N threads/jobs for FFT, density, and area computation
    """

    # Stall detection
    stall_patience: int | None = 5
    """Maximum number of iterations to allow error to increase before considering algorithm stalled.

    If None, stall detection is disabled and the algorithm will run until convergence or
    maximum iterations. If 0, algorithm will stall immediately if error increases.
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
        elif self.grid_size is not None:
            # Use grid construction parameters
            return Grid.from_bounds(bounds, self.grid_size, self.grid_margin, self.grid_square)
        else:
            # Default grid
            return Grid.from_bounds(bounds, size=100)

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
        self._validate()
        # Mark that validation has been initialized
        super().__setattr__("_validation_initialized", True)

    def __setattr__(self, name: str, value) -> None:
        """Validate individual fields when assigned directly."""
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
            "grid_size",
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
            "show_progress",
            "progress_message",
            "prescale_components",
            "stall_patience",
            "parallel",
        ]

        for field_name in field_names:
            field_error = self._validate_field_value(field_name, getattr(self, field_name))
            if field_error:
                errors.append(field_error)

        return errors

    def _validate_field_value(self, field_name: str, value) -> str | None:
        """Single source of truth for field validation logic."""
        # Grid parameters
        if field_name == "grid_size":
            if value is not None:
                return self._validate_grid_size(value)
        elif field_name == "grid_margin":
            if not isinstance(value, (int, float)) or value < 0:
                return "grid_margin must be a non-negative number"
        elif field_name == "grid_square":
            if not isinstance(value, bool):
                return "grid_square must be a boolean"
        elif field_name == "use_bounding_box_filter":
            if not isinstance(value, bool):
                return "use_bounding_box_filter must be a boolean"
        elif field_name == "parallel":
            if not isinstance(value, (bool, int)) or (
                isinstance(value, int) and not isinstance(value, bool) and value <= 0
            ):
                return "parallel must be True, False, or a positive integer"

        # Computation parameters
        elif field_name == "dt":
            if not isinstance(value, (int, float)) or value <= 0 or value > 1.0:
                return "dt must be > 0 and <= 1.0"
        elif field_name == "n_iter":
            if not isinstance(value, int) or value <= 0:
                return "n_iter must be a positive integer"
        elif field_name == "recompute_every":
            if value is not None:
                if not isinstance(value, int) or value <= 0:
                    return "recompute_every must be a positive integer"
                elif hasattr(self, "n_iter") and value > self.n_iter:
                    return "recompute_every cannot be greater than n_iter"
        elif field_name == "snapshot_every":
            if value is not None:
                if not isinstance(value, int) or value <= 0:
                    return "snapshot_every must be a positive integer"
                elif hasattr(self, "n_iter") and value > self.n_iter:
                    return "snapshot_every cannot be greater than n_iter"
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
        elif field_name == "save_internals":
            if not isinstance(value, bool):
                return "save_internals must be a boolean"
        elif field_name == "show_progress":
            if not isinstance(value, bool):
                return "show_progress must be a boolean"
        elif field_name == "progress_message" and value is not None and not isinstance(value, str):
            return "progress_message must be a string or None"

        # Stall detection options
        elif field_name == "stall_patience":
            if value is not None:
                if not isinstance(value, int):
                    return "stall_patience must be an integer or None"
                elif value < 0:
                    return "stall_patience must be a non-negative integer or None"

        # Outer-boundary distortion reduction options
        elif field_name == "prescale_components" and not isinstance(value, bool):
            return "prescale_components must be a boolean"

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

    def _validate_grid_size(self, grid_size) -> str | None:
        """Validate grid_size parameter format and values."""
        if isinstance(grid_size, int):
            if grid_size <= 0:
                return "grid_size as int must be positive"
        elif isinstance(grid_size, (tuple, list)) and len(grid_size) == 2:
            for i, size in enumerate(grid_size):
                if size is not None and (not isinstance(size, int) or size <= 0):
                    return f"grid_size[{i}] must be a positive integer or None"
        else:
            return "grid_size must be a positive integer or tuple of (int, int)"

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
        """Create options optimized for quick preview.

        Uses lower grid resolution, fewer iterations, and looser tolerances
        for fast interactive exploration.

        Returns
        -------
        MorphOptions
            Configuration for fast preview morphing

        Examples
        --------
        >>> options = MorphOptions.preset_fast()
        >>> result = morph_gdf(gdf, 'population', options=options)
        """
        return cls(
            grid_size=128,
            n_iter=30,
            dt=0.3,
            mean_tol=0.15,
            max_tol=0.25,
            show_progress=True,
        )

    @classmethod
    def preset_balanced(cls) -> "MorphOptions":
        """Create options with balanced speed/quality trade-off.

        Default production setting suitable for most use cases.

        Returns
        -------
        MorphOptions
            Configuration for balanced morphing

        Examples
        --------
        >>> options = MorphOptions.preset_balanced()
        >>> result = morph_gdf(gdf, 'population', options=options)
        """
        return cls(
            grid_size=256,
            n_iter=100,
            dt=0.2,
            mean_tol=0.05,
            max_tol=0.1,
            show_progress=True,
        )

    @classmethod
    def preset_high_quality(cls) -> "MorphOptions":
        """Create options optimized for publication-quality results.

        Uses higher grid resolution, more iterations, and stricter tolerances
        for maximum accuracy.

        Returns
        -------
        MorphOptions
            Configuration for high-quality morphing

        Examples
        --------
        >>> options = MorphOptions.preset_high_quality()
        >>> result = morph_gdf(gdf, 'population', options=options)
        """
        return cls(
            grid_size=512,
            n_iter=300,
            dt=0.1,
            mean_tol=0.01,
            max_tol=0.03,
            show_progress=True,
        )
