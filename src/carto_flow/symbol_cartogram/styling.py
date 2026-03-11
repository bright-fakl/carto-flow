"""Styling class for symbol cartograms.

This module defines the Styling class that collects styling decisions
and applies them to LayoutResult to produce SymbolCartogram instances.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .layout_result import Transform

if TYPE_CHECKING:
    from .layout_result import LayoutResult
    from .result import SymbolCartogram
    from .symbols import Symbol


class FitMode(Enum):
    """How to fit a styled symbol inside the canonical tile.

    Attributes
    ----------
    INSIDE : str
        Conservative fit: symbol is guaranteed to fit inside tile (centered).
        Uses inscribed_radius / bounding_radius ratio.
    AREA : str
        Area-preserving fit: symbol area equals tile area.
        Symbol may extend outside tile boundaries.
    FILL : str
        Optimal fit with center shift: maximizes symbol size while staying
        inside tile. May shift symbol center from tile center.

    """

    INSIDE = "inside"
    AREA = "area"
    FILL = "fill"


class Styling:
    """Collects styling decisions, then applies to LayoutResult.

    Supports incremental styling with fluent API.

    Parameters
    ----------
    symbol : Symbol or str or None
        Symbol instance or string shorthand ("circle", "hexagon", "square").
        If None, uses the canonical symbol from LayoutResult.
    scale : float
        Global scale multiplier. Default: 1.0
    rotation : float
        Additional rotation in degrees (counter-clockwise). Default: 0.0
    reflection : bool
        Whether to reflect symbols. Default: False
    fit_mode : FitMode or str
        How to fit styled symbol inside canonical tile:
        - "inside": Conservative fit, symbol guaranteed inside tile (default)
        - "area": Area-preserving, symbol area equals tile area
        - "fill": Optimal fit with center shift, maximizes symbol size

    Examples
    --------
    >>> # Simple styling
    >>> styling = Styling(symbol="hexagon", scale=0.9)
    >>> cartogram = styling.apply(layout_result)

    >>> # With fit mode
    >>> styling = Styling(symbol="circle", fit_mode="fill")
    >>> cartogram = styling.apply(layout_result)

    >>> # Fluent API
    >>> styling = (Styling()
    ...     .set_symbol("hexagon")
    ...     .transform(scale=0.9)
    ...     .set_symbol("star", indices=[5, 10, 15]))
    >>> cartogram = styling.apply(layout_result)

    >>> # Per-geometry overrides
    >>> styling = Styling().set_symbol("hexagon").transform(scale=0.8, indices=[0, 1, 2])
    >>> cartogram = styling.apply(layout_result)

    """

    def _is_array_like(self, value: Any) -> bool:
        """Check if value is array-like (not string, not Symbol, not dict)."""
        if isinstance(value, str):
            return False
        if isinstance(value, dict):
            return False
        if hasattr(self, "_resolve_symbol") and hasattr(value, "__class__"):
            from .symbols import Symbol

            if isinstance(value, Symbol):
                return False
        return hasattr(value, "__len__") and not isinstance(value, (str, bytes))

    def _mask_to_indices(self, mask: list[bool] | np.ndarray) -> list[int]:
        """Convert boolean mask to list of True indices."""
        if isinstance(mask, np.ndarray):
            return mask.nonzero()[0].tolist()
        return [i for i, val in enumerate(mask) if val]

    def __init__(
        self,
        symbol: Symbol | str | None = None,
        scale: float = 1.0,
        rotation: float = 0.0,
        reflection: bool = False,
        fit_mode: FitMode | Literal["inside", "area", "fill"] = FitMode.INSIDE,
    ):
        self._global_symbol: Symbol | str | None = symbol
        self._global_params: dict[str, Any] = {}
        # Convert rotation from degrees to radians for internal use
        self._global_transform = Transform(
            scale=scale,
            rotation=np.radians(rotation) if rotation != 0.0 else 0.0,
            reflection=reflection,
        )
        self._per_geometry: dict[int, dict[str, Any]] = {}

        # Normalize fit_mode to enum
        if isinstance(fit_mode, str):
            fit_mode = FitMode(fit_mode)
        self._fit_mode = fit_mode

    def set_symbol(
        self,
        symbol: Symbol | str | list | np.ndarray,
        indices: list[int] | None = None,
        mask: list[bool] | np.ndarray | None = None,
    ) -> Styling:
        """Set symbol for all or specific geometries.

        Parameters
        ----------
        symbol : Symbol or str or list or np.ndarray
            Symbol instance, string shorthand ("circle", "hexagon", "square"), or array of values.
        indices : list of int or None
            Geometry indices to apply to. None = all geometries.
        mask : list of bool or np.ndarray or None
            Boolean mask indicating which geometries to apply the symbol to.

        Returns
        -------
        Styling
            Self for method chaining.

        Examples
        --------
        >>> styling = Styling().set_symbol("hexagon")
        >>> styling = Styling().set_symbol("circle", indices=[0, 1, 2])
        >>> styling = Styling().set_symbol(["circle", "hexagon", "square"])  # Positional
        >>> styling = Styling().set_symbol("star", mask=[True, False, True])  # Boolean mask

        """
        if indices is not None and mask is not None:
            raise ValueError("Cannot specify both 'indices' and 'mask'")

        # Case 1: Array of values (positional)
        if self._is_array_like(symbol):
            if indices is not None or mask is not None:
                raise ValueError("Cannot use array values with 'indices' or 'mask'")
            for i, sym in enumerate(symbol):
                self._per_geometry.setdefault(i, {})["symbol"] = sym
            return self

        # Case 2: Boolean mask
        if mask is not None:
            for i in self._mask_to_indices(mask):
                self._per_geometry.setdefault(i, {})["symbol"] = symbol
            return self

        # Case 3: Explicit indices (existing behavior)
        if indices is not None:
            for i in indices:
                self._per_geometry.setdefault(i, {})["symbol"] = symbol
            return self

        # Case 4: Global (existing behavior)
        self._global_symbol = symbol
        return self

    def set_params(
        self,
        params: dict[str, Any] | list | np.ndarray,
        indices: list[int] | None = None,
        mask: list[bool] | np.ndarray | None = None,
    ) -> Styling:
        """Set symbol parameters for all or specific geometries.

        Parameters are passed to symbol.modify() to create modified symbols.
        Useful for symbols with configurable parameters like IsohedralTileSymbol.

        Parameters
        ----------
        params : dict or list or np.ndarray
            Parameters to pass to symbol.modify(), or array of parameter dicts.
        indices : list of int or None
            Geometry indices to apply to. None = all geometries.
        mask : list of bool or np.ndarray or None
            Boolean mask indicating which geometries to apply the parameters to.

        Returns
        -------
        Styling
            Self for method chaining.

        Examples
        --------
        >>> styling = Styling().set_params({"pointy_top": False})
        >>> styling = Styling().set_params({"prototile_params": [0.5, 0.3]})
        >>> styling = Styling().set_params([{"a": 1}, {"b": 2}])  # Positional
        >>> styling = Styling().set_params({"pointy_top": True}, mask=[True, False, True])  # Masked

        """
        if indices is not None and mask is not None:
            raise ValueError("Cannot specify both 'indices' and 'mask'")

        # Case 1: Array of values (positional)
        if self._is_array_like(params):
            if indices is not None or mask is not None:
                raise ValueError("Cannot use array values with 'indices' or 'mask'")
            for i, p in enumerate(params):
                self._per_geometry.setdefault(i, {}).setdefault("params", {}).update(p)
            return self

        # Case 2: Boolean mask
        if mask is not None:
            for i in self._mask_to_indices(mask):
                self._per_geometry.setdefault(i, {}).setdefault("params", {}).update(params)
            return self

        # Case 3: Explicit indices (existing behavior)
        if indices is not None:
            for i in indices:
                self._per_geometry.setdefault(i, {}).setdefault("params", {}).update(params)
            return self

        # Case 4: Global (existing behavior)
        self._global_params.update(params)
        return self

    def transform(
        self,
        scale: float | list | np.ndarray = 1.0,
        rotation: float | list | np.ndarray = 0.0,
        reflection: bool | list | np.ndarray = False,
        indices: list[int] | None = None,
        mask: list[bool] | np.ndarray | None = None,
    ) -> Styling:
        """Apply additional transform for all or specific geometries.

        Parameters
        ----------
        scale : float or list or np.ndarray
            Scale multiplier (default 1.0)
        rotation : float or list or np.ndarray
            Additional rotation in degrees (counter-clockwise, default 0.0)
        reflection : bool or list or np.ndarray
            Whether to reflect (default False)
        indices : list of int or None
            Geometry indices to apply to. None = all geometries.
        mask : list of bool or np.ndarray or None
            Boolean mask indicating which geometries to apply the transform to.

        Returns
        -------
        Styling
            Self for method chaining.

        Examples
        --------
        >>> styling = Styling().transform(scale=0.9)
        >>> styling = Styling().transform(rotation=45, reflection=True)
        >>> styling = Styling().transform(scale=0.8, indices=[0, 1, 2])
        >>> styling = Styling().transform(scale=[0.9, 1.0, 0.8])  # Positional
        >>> styling = Styling().transform(rotation=45, mask=[True, False, True])  # Masked

        """
        import numpy as np

        if indices is not None and mask is not None:
            raise ValueError("Cannot specify both 'indices' and 'mask'")

        # Check if any parameter is array-like
        scale_is_array = self._is_array_like(scale)
        rotation_is_array = self._is_array_like(rotation)
        reflection_is_array = self._is_array_like(reflection)
        any_array = scale_is_array or rotation_is_array or reflection_is_array

        if any_array:
            if indices is not None or mask is not None:
                raise ValueError("Cannot use array values with 'indices' or 'mask'")

            # Determine maximum length to handle
            lengths = []
            if scale_is_array:
                lengths.append(len(scale))
            if rotation_is_array:
                lengths.append(len(rotation))
            if reflection_is_array:
                lengths.append(len(reflection))

            max_len = max(lengths)

            for i in range(max_len):
                s = scale[i] if scale_is_array else scale
                r = rotation[i] if rotation_is_array else rotation
                rf = reflection[i] if reflection_is_array else reflection

                t = Transform(
                    scale=s,
                    rotation=np.radians(r) if r != 0.0 else 0.0,
                    reflection=rf,
                )

                existing = self._per_geometry.get(i, {}).get("transform", Transform())
                self._per_geometry.setdefault(i, {})["transform"] = existing.compose(t)
            return self

        # Case 2: Boolean mask
        if mask is not None:
            t = Transform(
                scale=scale,
                rotation=np.radians(rotation) if rotation != 0.0 else 0.0,
                reflection=reflection,
            )
            for i in self._mask_to_indices(mask):
                existing = self._per_geometry.get(i, {}).get("transform", Transform())
                self._per_geometry.setdefault(i, {})["transform"] = existing.compose(t)
            return self

        # Case 3: Explicit indices (existing behavior)
        if indices is not None:
            t = Transform(
                scale=scale,
                rotation=np.radians(rotation) if rotation != 0.0 else 0.0,
                reflection=reflection,
            )
            for i in indices:
                existing = self._per_geometry.get(i, {}).get("transform", Transform())
                self._per_geometry.setdefault(i, {})["transform"] = existing.compose(t)
            return self

        # Case 4: Global (existing behavior)
        t = Transform(
            scale=scale,
            rotation=np.radians(rotation) if rotation != 0.0 else 0.0,
            reflection=reflection,
        )
        self._global_transform = self._global_transform.compose(t)
        return self

    def apply(self, layout_result: LayoutResult) -> SymbolCartogram:
        """Apply styling to layout and produce cartogram.

        Parameters
        ----------
        layout_result : LayoutResult
            Immutable layout output from compute().

        Returns
        -------
        SymbolCartogram
            Rendered cartogram with styled symbols.

        """
        import geopandas as gpd
        import numpy as np

        from .result import SymbolCartogram

        # Validate indices
        n_geoms = len(layout_result.transforms)
        invalid_indices = [i for i in self._per_geometry if i >= n_geoms]
        if invalid_indices:
            raise IndexError(f"Invalid geometry indices: {invalid_indices}. Layout has {n_geoms} geometries.")

        geometries: list[Any] = []
        symbol_x: list[float] = []
        symbol_y: list[float] = []
        symbol_sizes: list[float] = []

        # Cache for symbol fit factors (to avoid recomputing)
        symbol_fit_cache: dict[int, tuple[float, tuple[float, float]]] = {}

        for i, base_transform in enumerate(layout_result.transforms):
            # Get effective symbol
            symbol = self._per_geometry.get(i, {}).get("symbol", self._global_symbol)
            if symbol is None:
                symbol = layout_result.canonical_symbol
            if isinstance(symbol, str):
                symbol = self._resolve_symbol(symbol)

            # Get effective params
            params = {
                **self._global_params,
                **self._per_geometry.get(i, {}).get("params", {}),
            }
            if params:
                symbol = symbol.modify(**params)

            # Get effective transform
            global_t = self._global_transform
            per_t = self._per_geometry.get(i, {}).get("transform", Transform())
            effective_t = base_transform.compose(global_t).compose(per_t)

            # If styling has rotation/reflection, wrap symbol for proper fitting
            # This ensures fit is computed on the transformed shape
            fit_symbol = symbol
            styling_rotation = global_t.rotation
            styling_reflection = global_t.reflection
            if styling_rotation != 0.0 or styling_reflection:
                from .symbols import TransformedSymbol

                # Convert radians back to degrees for TransformedSymbol
                fit_symbol = TransformedSymbol(
                    symbol,
                    rotation=np.degrees(styling_rotation),
                    reflection=styling_reflection,
                )

            # Compute fit factor and offset based on fit_mode
            symbol_id = id(fit_symbol)
            if symbol_id not in symbol_fit_cache:
                symbol_fit_cache[symbol_id] = self._compute_fit(layout_result.canonical_symbol, fit_symbol)

            fit_factor, unit_offset = symbol_fit_cache[symbol_id]
            effective_size = layout_result.base_size * fit_factor

            # The offset is in unit coordinates and will be applied inside
            # get_geometry() after scaling but before rotation/reflection.
            # This ensures the offset transforms naturally with the symbol.

            # Generate geometry
            geom = symbol.get_geometry(
                position=effective_t.position,
                size=effective_size,
                transform=effective_t,
                offset=unit_offset,
            )
            geometries.append(geom)

            # Store symbol metadata (use tile center position)
            symbol_x.append(effective_t.position[0])
            symbol_y.append(effective_t.position[1])
            # Symbol size is effective_size * scale (effective scale from transform)
            symbol_sizes.append(effective_size * effective_t.scale)

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=geometries, crs=layout_result.crs)  # type: ignore[arg-type]
        gdf["original_index"] = range(len(geometries))
        gdf["_symbol_x"] = symbol_x
        gdf["_symbol_y"] = symbol_y
        gdf["_symbol_size"] = symbol_sizes

        # Compute displacement from original positions
        displacements = np.zeros(len(geometries))
        if layout_result.positions is not None:
            displacements = np.sqrt(
                (np.array(symbol_x) - layout_result.positions[:, 0]) ** 2
                + (np.array(symbol_y) - layout_result.positions[:, 1]) ** 2,
            )
        gdf["_displacement"] = displacements

        # Compute metrics
        metrics = {
            "displacement_mean": float(np.mean(displacements)),
            "displacement_max": float(np.max(displacements)),
            "displacement_std": float(np.std(displacements)),
        }

        # Add algorithm-specific metrics
        if layout_result.algorithm_info and "info" in layout_result.algorithm_info:
            info = layout_result.algorithm_info["info"]
            if isinstance(info, dict):
                if "iterations" in info:
                    metrics["iterations"] = info["iterations"]
                if "converged" in info:
                    metrics["converged"] = info["converged"]

        # Extract tiling_result and assignments for grid layouts
        tiling_result = layout_result.algorithm_info.get("tiling_result") if layout_result.algorithm_info else None
        assignments = layout_result.algorithm_info.get("assignments") if layout_result.algorithm_info else None

        return SymbolCartogram(
            symbols=gdf,
            layout_result=layout_result,
            styling=self,
            metrics=metrics,
            _tiling_result=tiling_result,
            _assignments=assignments,
            simulation_history=layout_result.simulation_history,
        )

    def _compute_fit(self, canonical: Symbol, styled: Symbol) -> tuple[float, tuple[float, float]]:
        """Compute fit factor and offset for styled symbol inside canonical tile.

        Parameters
        ----------
        canonical : Symbol
            The canonical tile symbol.
        styled : Symbol
            The styled symbol to fit inside.

        Returns
        -------
        tuple[float, tuple[float, float]]
            (fit_factor, (offset_x, offset_y)) where:
            - fit_factor: scale multiplier for base_size
            - offset: position shift in unit coordinates (multiplied by effective_size)

        """
        if styled is canonical:
            return 1.0, (0.0, 0.0)

        if self._fit_mode == FitMode.INSIDE:
            # Conservative fit: symbol guaranteed inside tile (centered)
            # Use the actual polygon containment check for better fit than
            # the simple inscribed_radius / bounding_radius ratio.
            import numpy as np
            from shapely.affinity import scale as shapely_scale

            container = canonical.unit_polygon()
            contained = styled.unit_polygon()

            # Binary search for the largest scale that fits
            lo, hi = 0.01, 2.0
            for _ in range(20):  # 20 iterations gives ~1e-6 precision
                mid = (lo + hi) / 2
                scaled = shapely_scale(contained, xfact=mid, yfact=mid, origin=(0, 0))
                if scaled.within(container):
                    lo = mid  # Can go bigger
                else:
                    hi = mid  # Too big
            fit_factor = lo
            return fit_factor, (0.0, 0.0)

        if self._fit_mode == FitMode.AREA:
            # Area-preserving fit: symbol area equals tile area
            # Both symbols define size such that area = base_size² * unit_area * 4
            # For equal area: styled.size² * styled.unit_area = canonical.size² * canonical.unit_area
            # fit_factor = sqrt(canonical.unit_area / styled.unit_area)
            import numpy as np

            canonical_area = canonical.unit_polygon().area
            styled_area = styled.unit_polygon().area
            fit_factor = np.sqrt(canonical_area / styled_area)
            return fit_factor, (0.0, 0.0)

        if self._fit_mode == FitMode.FILL:
            # Optimal fit with center shift: maximize symbol size
            # This requires optimization to find best scale and position
            return self._compute_optimal_fit(canonical, styled)

        # Default to INSIDE
        fit_factor = canonical.inscribed_radius / styled.bounding_radius
        return fit_factor, (0.0, 0.0)

    def _compute_optimal_fit(self, canonical: Symbol, styled: Symbol) -> tuple[float, tuple[float, float]]:
        """Compute optimal fit with center shift using optimization.

        Uses grid search followed by local optimization to find the maximum
        scale and optimal position for the styled symbol to fit inside the
        canonical tile.

        Parameters
        ----------
        canonical : Symbol
            The canonical tile symbol.
        styled : Symbol
            The styled symbol to fit inside.

        Returns
        -------
        tuple[float, tuple[float, float]]
            (fit_factor, (offset_x, offset_y))

        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            # Fall back to INSIDE mode if scipy not available
            import warnings

            warnings.warn(
                "scipy not available, falling back to 'inside' fit mode",
                RuntimeWarning,
                stacklevel=2,
            )
            fit_factor = canonical.inscribed_radius / styled.bounding_radius
            return fit_factor, (0.0, 0.0)

        import numpy as np
        from shapely.affinity import scale, translate

        container = canonical.unit_polygon()
        contained = styled.unit_polygon()

        # Initial scale from INSIDE mode (use the improved binary search result)
        canonical.inscribed_radius / styled.bounding_radius

        # First, find the centered fit using binary search
        lo, hi = 0.01, 2.0
        for _ in range(20):
            mid = (lo + hi) / 2
            scaled = scale(contained, xfact=mid, yfact=mid, origin=(0, 0))
            if scaled.within(container):
                lo = mid
            else:
                hi = mid
        centered_scale = lo

        # Grid search to find good starting point (with offset)
        best_scale = centered_scale
        best_offset = (0.0, 0.0)

        # Test scales from centered fit to 1.5x the centered fit
        for scale_factor in np.linspace(centered_scale, centered_scale * 1.5, 20):
            for dx in np.linspace(-0.3, 0.3, 15):
                for dy in np.linspace(-0.3, 0.3, 15):
                    scaled = scale(contained, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
                    shifted = translate(scaled, dx, dy)
                    if shifted.within(container) and scale_factor > best_scale:
                        best_scale = scale_factor
                        best_offset = (dx, dy)

        # Local optimization starting from best grid point
        def objective(x):
            scale_factor, dx, dy = x
            scaled = scale(contained, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
            shifted = translate(scaled, dx, dy)

            if shifted.within(container):
                return -scale_factor

            outside_area = shifted.area - shifted.intersection(container).area
            return -scale_factor + 100 * outside_area

        x0 = [best_scale, best_offset[0], best_offset[1]]
        bounds = [(0.01, 10.0), (-0.5, 0.5), (-0.5, 0.5)]
        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

        # Verify the result actually fits
        scale_factor, dx, dy = result.x
        scaled = scale(contained, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
        shifted = translate(scaled, dx, dy)

        if shifted.within(container) and scale_factor >= best_scale:
            # The optimization finds scale_factor and dx, dy in unit coordinates.
            # - scale_factor: how much to scale the styled unit polygon
            # - dx, dy: offset in the scaled coordinate system
            #
            # In get_geometry(), we scale by effective_size * 2, where
            # effective_size = base_size * fit_factor.
            #
            # For the relative sizes to match:
            # - Optimization: square fits in triangle with scale_factor
            # - World: square fits in triangle with effective_size / base_size ratio
            #
            # So fit_factor = scale_factor (direct ratio)
            #
            # For the offset:
            # - Optimization offset is in scaled coords (multiplied by scale_factor)
            # - World offset = dx * (base_size / 0.5) = dx * 2 * base_size
            # - get_geometry applies offset * effective_size = offset * base_size * scale_factor
            # - So: offset * base_size * scale_factor = dx * 2 * base_size
            # - unit_offset = dx * 2 / scale_factor
            fit_factor = scale_factor
            unit_offset = (dx * 2.0 / scale_factor, dy * 2.0 / scale_factor)
            return fit_factor, unit_offset

        # Fall back to best grid search result (which is guaranteed to fit)
        fit_factor = best_scale
        unit_offset = (best_offset[0] * 2.0 / best_scale, best_offset[1] * 2.0 / best_scale)
        return fit_factor, unit_offset

    def _resolve_symbol(self, symbol: str) -> Symbol:
        """Resolve string shorthand to Symbol instance.

        Parameters
        ----------
        symbol : str
            Built-in symbol name: ``"circle"``, ``"square"``, ``"hexagon"``,
            ``"triangle"``, ``"diamond"``, ``"pentagon"``, ``"star"``.

        Returns
        -------
        Symbol
            Symbol instance.

        Raises
        ------
        ValueError
            If symbol string is not recognized.

        """
        from .symbols import resolve_symbol

        return resolve_symbol(symbol)

    def __repr__(self) -> str:
        """Return string representation of Styling."""
        import numpy as np

        parts = []
        if self._global_symbol is not None:
            parts.append(f"symbol={self._global_symbol!r}")
        if self._global_transform.scale != 1.0:
            parts.append(f"scale={self._global_transform.scale}")
        if self._global_transform.rotation != 0.0:
            # Show rotation in degrees for user-friendliness
            parts.append(f"rotation={np.degrees(self._global_transform.rotation):.1f}")
        if self._global_transform.reflection:
            parts.append("reflection=True")
        if self._global_params:
            parts.append(f"params={self._global_params}")
        if self._per_geometry:
            parts.append(f"per_geometry={len(self._per_geometry)} overrides")

        return f"Styling({', '.join(parts) if parts else 'default'})"
