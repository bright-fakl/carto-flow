"""Core functions for the flow-density layout algorithm."""

from __future__ import annotations

import numba
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay, cKDTree


def _gabriel_pairs(pts: NDArray) -> set[tuple[int, int]]:
    """Return Gabriel graph edges as (i,j) pairs with i < j.

    The Gabriel graph is the subset of Delaunay edges where no other point
    lies strictly inside the diametral circle of the edge.
    """
    tri = Delaunay(pts)
    edge_set: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        a, b, c = simplex
        for p, q in [(a, b), (a, c), (b, c)]:
            edge_set.add((min(p, q), max(p, q)))

    edges = np.array(list(edge_set), dtype=np.intp)  # (E, 2)
    ei, ej = edges[:, 0], edges[:, 1]
    mids = (pts[ei] + pts[ej]) / 2  # (E, 2)
    diff_ij = pts[ei] - pts[ej]
    r2s = (diff_ij * diff_ij).sum(axis=1) / 4  # (E,)

    diff = pts[np.newaxis, :, :] - mids[:, np.newaxis, :]  # (E, n, 2)
    sq_dists = (diff * diff).sum(axis=2)  # (E, n)

    n = len(pts)
    idx = np.arange(n)
    exclude = (idx[np.newaxis, :] == ei[:, np.newaxis]) | (idx[np.newaxis, :] == ej[:, np.newaxis])  # (E, n)
    violates = (sq_dists < r2s[:, np.newaxis]) & ~exclude
    gabriel_mask = ~violates.any(axis=1)  # (E,)

    return {(int(i), int(j)) for i, j in edges[gabriel_mask]}


def _get_pairs(centroids: NDArray, use_gabriel: bool) -> set[tuple[int, int]]:
    """Return adjacency pairs as (i,j) with i < j (Gabriel or full Delaunay)."""
    if use_gabriel:
        return _gabriel_pairs(centroids)
    tri = Delaunay(centroids)
    pairs: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        a, b, c = simplex
        for p, q in [(a, b), (a, c), (b, c)]:
            pairs.add((min(p, q), max(p, q)))
    return pairs


@numba.njit(fastmath=True, cache=True)
def _density_field_numba_grouped(
    pairs_arr: NDArray,
    centroids: NDArray,
    radii: NDArray,
    spacing_abs: float,
    x1d: NDArray,
    y1d: NDArray,
    damp: bool,
    sigma_perp_factor: float,
    push_scale: float,
    pull_scale: float,
    group_ids: NDArray,
    cross_pull_scale: float,
) -> NDArray:
    NY = len(y1d)
    NX = len(x1d)
    rho = np.zeros((NY, NX))
    dx = x1d[1] - x1d[0]
    dy = y1d[1] - y1d[0]
    x0 = x1d[0]
    y0 = y1d[0]

    for e in range(len(pairs_arr)):
        ci = pairs_arr[e, 0]
        cj = pairs_arr[e, 1]
        cx_i = centroids[ci, 0]
        cy_i = centroids[ci, 1]
        cx_j = centroids[cj, 0]
        cy_j = centroids[cj, 1]
        r_i = radii[ci]
        r_j = radii[cj]
        d = np.sqrt((cx_j - cx_i) ** 2 + (cy_j - cy_i) ** 2)
        r_target = r_i + r_j + spacing_abs
        if r_target <= 0.0:
            # Both symbols are zero-sized and there is no spacing: the pair
            # has no target separation and contributes no density.
            continue

        ratio = d / r_target
        w = np.exp(1.0 - ratio) if (damp and ratio > 1.0) else 1.0
        amplitude = ((r_target / d) ** 2 - 1.0) * w
        if amplitude > 0.0:
            scale = push_scale
        else:
            eff_pull = pull_scale if group_ids[ci] == group_ids[cj] else pull_scale * cross_pull_scale
            scale = eff_pull

        ux = (cx_j - cx_i) / d
        uy = (cy_j - cy_i) / d
        half_spacing = 0.5 * spacing_abs
        claim_i = d * (r_i + half_spacing) / r_target
        claim_j = d * (r_j + half_spacing) / r_target
        px = cx_i + claim_i * ux
        py = cy_i + claim_i * uy

        max_claim = max(claim_i, claim_j)
        max_sigma_perp = sigma_perp_factor * max_claim
        hw_x = 4.0 * np.sqrt((ux * max_claim) ** 2 + (uy * max_sigma_perp) ** 2)
        hw_y = 4.0 * np.sqrt((uy * max_claim) ** 2 + (ux * max_sigma_perp) ** 2)

        ix0 = max(0, int((px - hw_x - x0) / dx))
        ix1 = min(NX, int((px + hw_x - x0) / dx) + 2)
        iy0 = max(0, int((py - hw_y - y0) / dy))
        iy1 = min(NY, int((py + hw_y - y0) / dy) + 2)

        if ix0 >= ix1 or iy0 >= iy1:
            continue

        coeff = scale * amplitude
        inv2_ci2 = 1.0 / (2.0 * claim_i * claim_i)
        inv2_cj2 = 1.0 / (2.0 * claim_j * claim_j)
        inv2_pi2 = 1.0 / (2.0 * (sigma_perp_factor * claim_i) ** 2)
        inv2_pj2 = 1.0 / (2.0 * (sigma_perp_factor * claim_j) ** 2)

        for iy in range(iy0, iy1):
            dY = y1d[iy] - py
            for ix in range(ix0, ix1):
                dX = x1d[ix] - px
                s_par = dX * ux + dY * uy
                s_perp = -dX * uy + dY * ux
                if s_par <= 0.0:
                    val = coeff * np.exp(-(s_par * s_par * inv2_ci2 + s_perp * s_perp * inv2_pi2))
                else:
                    val = coeff * np.exp(-(s_par * s_par * inv2_cj2 + s_perp * s_perp * inv2_pj2))
                rho[iy, ix] += val

    return rho


@numba.njit(fastmath=True, cache=True)
def _density_field_numba(
    pairs_arr: NDArray,
    centroids: NDArray,
    radii: NDArray,
    spacing_abs: float,
    x1d: NDArray,
    y1d: NDArray,
    damp: bool,
    sigma_perp_factor: float,
    push_scale: float,
    pull_scale: float,
) -> NDArray:
    NY = len(y1d)
    NX = len(x1d)
    rho = np.zeros((NY, NX))
    dx = x1d[1] - x1d[0]
    dy = y1d[1] - y1d[0]
    x0 = x1d[0]
    y0 = y1d[0]

    for e in range(len(pairs_arr)):
        ci = pairs_arr[e, 0]
        cj = pairs_arr[e, 1]
        cx_i = centroids[ci, 0]
        cy_i = centroids[ci, 1]
        cx_j = centroids[cj, 0]
        cy_j = centroids[cj, 1]
        r_i = radii[ci]
        r_j = radii[cj]
        d = np.sqrt((cx_j - cx_i) ** 2 + (cy_j - cy_i) ** 2)
        r_target = r_i + r_j + spacing_abs
        if r_target <= 0.0:
            # Both symbols are zero-sized and there is no spacing: the pair
            # has no target separation and contributes no density.
            continue

        ratio = d / r_target
        w = np.exp(1.0 - ratio) if (damp and ratio > 1.0) else 1.0
        amplitude = ((r_target / d) ** 2 - 1.0) * w
        scale = push_scale if amplitude > 0.0 else pull_scale

        ux = (cx_j - cx_i) / d
        uy = (cy_j - cy_i) / d
        half_spacing = 0.5 * spacing_abs
        claim_i = d * (r_i + half_spacing) / r_target
        claim_j = d * (r_j + half_spacing) / r_target
        px = cx_i + claim_i * ux
        py = cy_i + claim_i * uy

        max_claim = max(claim_i, claim_j)
        max_sigma_perp = sigma_perp_factor * max_claim
        hw_x = 4.0 * np.sqrt((ux * max_claim) ** 2 + (uy * max_sigma_perp) ** 2)
        hw_y = 4.0 * np.sqrt((uy * max_claim) ** 2 + (ux * max_sigma_perp) ** 2)

        ix0 = max(0, int((px - hw_x - x0) / dx))
        ix1 = min(NX, int((px + hw_x - x0) / dx) + 2)
        iy0 = max(0, int((py - hw_y - y0) / dy))
        iy1 = min(NY, int((py + hw_y - y0) / dy) + 2)

        if ix0 >= ix1 or iy0 >= iy1:
            continue

        coeff = scale * amplitude
        inv2_ci2 = 1.0 / (2.0 * claim_i * claim_i)
        inv2_cj2 = 1.0 / (2.0 * claim_j * claim_j)
        inv2_pi2 = 1.0 / (2.0 * (sigma_perp_factor * claim_i) ** 2)
        inv2_pj2 = 1.0 / (2.0 * (sigma_perp_factor * claim_j) ** 2)

        for iy in range(iy0, iy1):
            dY = y1d[iy] - py
            for ix in range(ix0, ix1):
                dX = x1d[ix] - px
                s_par = dX * ux + dY * uy
                s_perp = -dX * uy + dY * ux
                if s_par <= 0.0:
                    val = coeff * np.exp(-(s_par * s_par * inv2_ci2 + s_perp * s_perp * inv2_pi2))
                else:
                    val = coeff * np.exp(-(s_par * s_par * inv2_cj2 + s_perp * s_perp * inv2_pj2))
                rho[iy, ix] += val

    return rho


def _relative_to_target(deviation: NDArray, target: NDArray) -> NDArray:
    """Express a deviation as a fraction of its target separation.

    A target separation of zero means both symbols are zero-sized and there
    is no separation to achieve, so the deviation counts as no error.
    """
    return np.divide(deviation, target, out=np.zeros(np.shape(deviation)), where=target > 0)


def _build_density_field_midpoint(
    centroids: NDArray,
    radii: NDArray,
    spacing_abs: float,
    grid,
    damp: bool,
    sigma_perp_factor: float,
    pairs: set[tuple[int, int]],
    push_scale: float = 1.0,
    pull_scale: float = 1.0,
    group_ids: NDArray | None = None,
    cross_group_pull_scale: float = 1.0,
) -> NDArray:
    """Build density field using a split Gaussian at the proportional midpoint M."""
    pairs_arr = np.array(list(pairs), dtype=np.int32)
    if group_ids is not None:
        return _density_field_numba_grouped(
            pairs_arr,
            centroids,
            radii,
            spacing_abs,
            grid.x_coords,
            grid.y_coords,
            damp,
            sigma_perp_factor,
            push_scale,
            pull_scale,
            group_ids,
            cross_group_pull_scale,
        )
    return _density_field_numba(
        pairs_arr,
        centroids,
        radii,
        spacing_abs,
        grid.x_coords,
        grid.y_coords,
        damp,
        sigma_perp_factor,
        push_scale,
        pull_scale,
    )


def build_density_field(
    centroids: NDArray,
    radii: NDArray,
    spacing_abs: float,
    grid,
    smooth: float,
    damp: bool,
    sigma_perp_factor: float,
    use_gabriel: bool,
    force_balance: float | str = 1.0,
    group_ids: NDArray | None = None,
    cross_group_pull_scale: float = 1.0,
) -> NDArray:
    """Build the Gaussian midpoint density field, with optional smoothing.

    Parameters
    ----------
    centroids : NDArray, shape (N, 2)
        Current centroid positions.
    radii : NDArray, shape (N,)
        Circle radii in the same coordinate units as centroids.
    spacing_abs : float
        Absolute gap between circle boundaries (added to r_i + r_j).
    grid : Grid
        Discretisation grid.
    smooth : float
        Gaussian filter sigma in real-world coordinate units (same as
        positions/radii). Converted to grid cells via grid.dx / grid.dy.
        0 = no smoothing.
    damp : bool
        Apply exponential dampening when circles are far from their targets.
    sigma_perp_factor : float
        Perpendicular Gaussian width = factor * max(claim_i, claim_j),
        where claim_i = d * (r_i + spacing_abs/2) / (r_i + r_j + spacing_abs).
    use_gabriel : bool
        Use Gabriel graph for pairs; False = full Delaunay.
    force_balance : float or {"count", "rms", "repulse"}, default 1.0
        Relative weighting of push vs pull pair contributions. Only the
        ratio push_scale/pull_scale affects the velocity field shape.
        ``float``: push_scale = force_balance, pull_scale = 1.0.
        ``"count"``: push_scale = 1/N_push, pull_scale = 1/N_pull —
          each pair contributes equally regardless of push/pull count imbalance.
        ``"rms"``: push_scale = 1/rms(push amplitudes),
          pull_scale = 1/rms(pull amplitudes) — equalises field energy per type.
        ``"repulse"``: pull_scale = 0 — only overlapping pairs contribute.

    Returns
    -------
    NDArray, shape (NY, NX)
        Density field, renormalized to preserve mean after smoothing.
    """
    pairs = _get_pairs(centroids, use_gabriel)

    push_scale = pull_scale = 1.0
    if force_balance != 1.0:
        if force_balance == "repulse":
            pull_scale = 0.0
        else:
            push_amps: list[float] = []
            pull_amps: list[float] = []
            for i, j in pairs:
                d = np.hypot(centroids[j, 0] - centroids[i, 0], centroids[j, 1] - centroids[i, 1])
                r_target = radii[i] + radii[j] + spacing_abs
                ratio = d / r_target
                w = np.exp(1.0 - ratio) if (damp and ratio > 1.0) else 1.0
                a = ((r_target / d) ** 2 - 1.0) * w
                (push_amps if a > 0 else pull_amps).append(a)
            if isinstance(force_balance, str):
                if force_balance == "count":
                    push_scale = 1.0 / float(len(push_amps)) if push_amps else 1.0
                    pull_scale = 1.0 / float(len(pull_amps)) if pull_amps else 1.0
                else:  # "rms"
                    rms_push = float(np.sqrt(np.mean(np.array(push_amps) ** 2))) if push_amps else 1.0
                    rms_pull = float(np.sqrt(np.mean(np.array(pull_amps) ** 2))) if pull_amps else 1.0
                    push_scale = 1.0 / rms_push if rms_push > 0 else 1.0
                    pull_scale = 1.0 / rms_pull if rms_pull > 0 else 1.0
            else:
                push_scale = float(force_balance)

    rho = _build_density_field_midpoint(
        centroids,
        radii,
        spacing_abs,
        grid,
        damp,
        sigma_perp_factor,
        pairs,
        push_scale=push_scale,
        pull_scale=pull_scale,
        group_ids=group_ids,
        cross_group_pull_scale=cross_group_pull_scale,
    )

    if smooth > 0:
        mu = np.mean(rho)
        sigma_yx = (smooth / grid.dy, smooth / grid.dx)
        rho = gaussian_filter(rho, sigma=sigma_yx, mode="reflect")
        rho *= mu / np.mean(rho)

    return rho


def run_flow_density(
    centroids: NDArray,
    radii: NDArray,
    spacing_abs: float,
    grid_size: int,
    smooth: float,
    sigma_perp_factor: float,
    damp: bool,
    use_gabriel: bool,
    max_iterations: int,
    recompute_every: int,
    dt_factor: float,
    convergence_tolerance: float,
    show_progress: bool,
    save_history: bool,
    save_density_fields: bool = False,
    force_balance: float | str = 1.0,
    group_ids: NDArray | None = None,
    cross_group_pull_scale: float = 1.0,
) -> tuple[NDArray, dict, list[NDArray] | None]:
    """Run the flow-density layout algorithm.

    Builds a Gaussian density field from per-pair contact-point blobs and
    advects centroid positions through the resulting velocity field until
    convergence or max_iterations is reached.

    Parameters
    ----------
    centroids : NDArray, shape (N, 2)
        Initial centroid positions (projected coordinates).
    radii : NDArray, shape (N,)
        Circle radii in the same coordinate units as centroids.
    spacing_abs : float
        Absolute gap added to r_i + r_j for the target distance.
    grid_size : int
        Grid resolution (square); larger values increase accuracy and runtime.
    smooth : float
        Gaussian filter sigma in real-world coordinate units (same as
        positions/radii). Converted internally to grid cells via
        grid.dx / grid.dy. 0 = no smoothing.
    sigma_perp_factor : float
        Perpendicular Gaussian width = factor * max(claim_i, claim_j),
        where claim_i = d * (r_i + spacing_abs/2) / (r_i + r_j + spacing_abs).
    damp : bool
        Apply exponential dampening when circles are far from their targets.
    use_gabriel : bool
        Use Gabriel graph for adjacency; False = full Delaunay.
    max_iterations : int
        Maximum number of advection steps.
    recompute_every : int
        Rebuild density and velocity fields every N steps.
    dt_factor : float
        Timestep = factor * min(dx, dy) / max_velocity.
    convergence_tolerance : float
        Stop when mean |d_nn - target| / target falls below this threshold,
        where target = r_i + r_nn + spacing_abs.
    show_progress : bool
        Print progress every 20 steps.
    save_history : bool
        Record centroid positions at every step.
    save_density_fields : bool
        Record rho, vx, vy at every density recompute step. Default: False.
        Results stored in ``info["density_fields"]`` (list of dicts with keys
        ``iteration``, ``rho``, ``vx``, ``vy``) and grid metadata in
        ``info["grid_bounds"]``, ``info["grid_shape"]``,
        ``info["grid_x_coords"]``, ``info["grid_y_coords"]``.

    Returns
    -------
    positions : NDArray, shape (N, 2)
        Final centroid positions.
    info : dict
        ``iterations``, ``final_error``, ``final_max_error``, ``n_overlaps``,
        ``converged``, ``errors`` (mean NN error per step),
        ``max_errors`` (max NN error per step), ``n_overlaps_history``,
        and optionally ``density_fields`` + grid metadata.
    history : list[NDArray] or None
        Per-step position snapshots when save_history=True, else None.
    """
    from carto_flow.flow_cartogram.displacement import displace_coords_numba
    from carto_flow.flow_cartogram.grid import Grid
    from carto_flow.flow_cartogram.velocity import VelocityComputerFFTW

    bounds = (
        centroids[:, 0].min(),
        centroids[:, 1].min(),
        centroids[:, 0].max(),
        centroids[:, 1].max(),
    )
    grid = Grid.from_bounds(bounds, size=grid_size, margin=0.5, square=True)
    velocity_computer = VelocityComputerFFTW(grid)

    pts = centroids.copy().astype(np.float64)

    # Arrange coincident tiles on a small circle so every pair has d ≥ one grid cell.
    cell = min(grid.dx, grid.dy)
    _, inverse = np.unique(pts, axis=0, return_inverse=True)
    for g in range(inverse.max() + 1):
        idx = np.where(inverse == g)[0]
        K = len(idx)
        if K > 1:
            r = K * cell / (2.0 * np.pi)
            angles = 2.0 * np.pi * np.arange(K) / K
            pts[idx, 0] += r * np.cos(angles)
            pts[idx, 1] += r * np.sin(angles)

    history: list[NDArray] | None = [] if save_history else None
    density_snapshots: list[dict] = [] if save_density_fields else None  # type: ignore[assignment]
    errors: list[float] = []
    max_errors: list[float] = []
    n_overlaps_list: list[int] = []

    rho: NDArray[np.floating] = np.zeros((1, 1))
    vx: NDArray[np.floating] = np.zeros((1, 1))
    vy: NDArray[np.floating] = np.zeros((1, 1))

    for step in range(max_iterations):
        if step % recompute_every == 0:
            rho = build_density_field(
                pts,
                radii,
                spacing_abs,
                grid,
                smooth=smooth,
                damp=damp,
                sigma_perp_factor=sigma_perp_factor,
                use_gabriel=use_gabriel,
                force_balance=force_balance,
                group_ids=group_ids,
                cross_group_pull_scale=cross_group_pull_scale,
            )
            vx, vy = velocity_computer.compute(rho)
            if save_density_fields:
                density_snapshots.append({
                    "iteration": step,
                    "rho": rho.copy(),
                    "vx": vx.copy(),
                    "vy": vy.copy(),
                })

        vmax = float(np.hypot(vx, vy).max())
        if vmax < 1e-10:
            if show_progress:
                print(f"Step {step}: velocity vanished, stopping.")
            break

        dt = dt_factor * min(grid.dx, grid.dy) / vmax
        pts = displace_coords_numba(pts, grid.x_coords, grid.y_coords, vx, vy, dt, grid.dx, grid.dy)

        if history is not None:
            history.append(pts.copy())

        # Convergence: nearest-neighbor spacing error (one NN per circle)
        tree = cKDTree(pts)
        nn_dists, nn_idx = tree.query(pts, k=2)  # k=2: [self (0), nearest other (1)]
        nn_dists = nn_dists[:, 1]
        nn_idx = nn_idx[:, 1]
        nn_targets = radii + radii[nn_idx] + spacing_abs
        if force_balance == "repulse":
            nn_errs = np.maximum(0.0, _relative_to_target(nn_targets - nn_dists, nn_targets))
        else:
            nn_errs = np.abs(_relative_to_target(nn_dists - nn_targets, nn_targets))
        nn_err = float(np.mean(nn_errs))
        nn_max_err = float(np.max(nn_errs))
        overlaps = int(np.sum(nn_dists < nn_targets))

        errors.append(nn_err)
        max_errors.append(nn_max_err)
        n_overlaps_list.append(overlaps)

        if show_progress and step % 20 == 0:
            print(f"Step {step:4d}: mean NN-error = {nn_err:.3f}, max = {nn_max_err:.3f}, overlaps = {overlaps}")

        if nn_err < convergence_tolerance:
            if show_progress:
                print(f"Converged at step {step}, mean error = {nn_err:.4f}")
            break

    final_error = errors[-1] if errors else float("nan")
    final_max_error = max_errors[-1] if max_errors else float("nan")
    final_overlaps = n_overlaps_list[-1] if n_overlaps_list else 0
    converged = bool(errors and errors[-1] < convergence_tolerance)

    tree = cKDTree(pts)
    nn_dists_f, nn_idx_f = tree.query(pts, k=2)
    nn_targets_f = radii + radii[nn_idx_f[:, 1]] + spacing_abs
    final_signed_errors = _relative_to_target(nn_targets_f - nn_dists_f[:, 1], nn_targets_f)

    info: dict = {
        "iterations": len(errors),
        "final_error": final_error,
        "final_max_error": final_max_error,
        "n_overlaps": final_overlaps,
        "converged": converged,
        "errors": errors,
        "max_errors": max_errors,
        "n_overlaps_history": n_overlaps_list,
        "final_signed_errors": final_signed_errors,
    }
    if save_density_fields:
        info["density_fields"] = density_snapshots
        info["grid_bounds"] = bounds
        info["grid_shape"] = grid.shape
        info["grid_x_coords"] = grid.x_coords.copy()
        info["grid_y_coords"] = grid.y_coords.copy()

    return pts, info, history
