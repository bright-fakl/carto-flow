"""Geodesic (topology-respecting) Voronoi labeling via multi-source BFS.

Replaces Euclidean nearest-neighbor assignment when ``geodesic_voronoi=True``.
The BFS wavefront propagates only through active (land) pixels, so it cannot
cross water bodies or other inactive regions — eliminating cross-bay assignments
that arise from straight-line distance.

Time complexity: O(n_active_pixels) — comparable to cKDTree labeling.

When the active region has multiple disconnected components (e.g. from elastic
boundary deformation creating isolated pockets), each component is seeded
dynamically via the unseeded-component fix: the nearest centroid to each
component's center-of-mass is used as the seed label.
"""

from __future__ import annotations

import numba
import numpy as np


@numba.njit(cache=True)
def _bfs(
    active_2d: np.ndarray,
    seed_rows: np.ndarray,
    seed_cols: np.ndarray,
    seed_labels: np.ndarray,
) -> np.ndarray:
    """Multi-source BFS on a boolean raster.

    Parameters
    ----------
    active_2d : (ny, nx) bool
        True for land/active pixels.
    seed_rows, seed_cols : (S,) int32
        2-D grid coordinates of each seed.  Duplicate positions are silently
        skipped (first seed to claim a cell wins).
    seed_labels : (S,) int32
        Label assigned to each seed.

    Returns
    -------
    labels : (ny, nx) int32
        Label for each pixel; -1 for unreachable pixels.
    """
    ny, nx = active_2d.shape
    S = len(seed_rows)
    labels = np.full((ny, nx), -1, dtype=np.int32)
    # Pre-allocate flat queue arrays (each pixel enqueued at most once).
    qr = np.empty(ny * nx, dtype=np.int32)
    qc = np.empty(ny * nx, dtype=np.int32)
    ql = np.empty(ny * nx, dtype=np.int32)
    head = tail = np.int32(0)
    for i in range(S):
        r = seed_rows[i]
        c = seed_cols[i]
        if 0 <= r < ny and 0 <= c < nx and active_2d[r, c] and labels[r, c] < 0:
            labels[r, c] = seed_labels[i]
            qr[tail] = r
            qc[tail] = c
            ql[tail] = seed_labels[i]
            tail += np.int32(1)
    DR = (-1, 1, 0, 0)
    DC = (0, 0, -1, 1)
    while head < tail:
        r = qr[head]
        c = qc[head]
        lbl = ql[head]
        head += np.int32(1)
        for d in range(4):
            nr = r + DR[d]
            nc = c + DC[d]
            if 0 <= nr < ny and 0 <= nc < nx and active_2d[nr, nc] and labels[nr, nc] < 0:
                labels[nr, nc] = lbl
                qr[tail] = nr
                qc[tail] = nc
                ql[tail] = lbl
                tail += np.int32(1)
    return labels


def geodesic_label_active(
    points: np.ndarray,
    gx_active: np.ndarray,
    gy_active: np.ndarray,
    active_mask_flat: np.ndarray,
    grid_nx: int,
    grid_ny: int,
    return_debug: bool = False,
    grid_x_coords: np.ndarray | None = None,
    grid_y_coords: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Snap centroids to nearest active pixel, run BFS, return flat labels.

    Parameters
    ----------
    points : (G, 2) float
        Centroid world coordinates.
    gx_active, gy_active : (m,) float
        World coordinates of active pixels (same order as the active mask).
    active_mask_flat : (grid_nx * grid_ny,) bool
        Flat active-pixel mask for the full grid.
    grid_nx, grid_ny : int
        Grid dimensions.
    return_debug : bool
        When ``True``, also return a diagnostic dict that reports any centroid
        whose BFS seed was placed in a different connected component of the
        active grid than the component the centroid is closest to.  This
        detects the "cross-water labeling" failure mode where Euclidean seed
        snapping puts a seed in a topologically disconnected region.

        The dict contains:

        ``seed_rows``, ``seed_cols`` : (n_seeds,) int32
            Grid coordinates of the placed seeds.
        ``seed_components`` : (n_seeds,) int32
            Connected-component id for each seed.
        ``centroid_components`` : (G,) int32
            Component id of the nearest active pixel to each original centroid.
        ``misplaced_mask`` : (G,) bool
            ``True`` for centroids whose seed is in a different component.
        ``n_misplaced`` : int
            Number of misplaced seeds.
        ``comp_2d`` : (ny, nx) int32
            Full component label grid (useful for visualization).
        ``unseeded_comp_ids`` : (k,) int32
            Component IDs (1-based, scipy convention) that had no seed before
            the fix was applied.  Non-empty means the fix fired this call.
        ``n_unseeded`` : int
            Number of unseeded components.
        ``unseeded_pixel_mask`` : (ny, nx) bool
            Active pixels that belonged to unseeded components (for overlay).
        ``seed_centroid_distances`` : (G,) float
            Euclidean distance from each centroid to its seed world position.
            Values >> ``grid_dx`` indicate possible within-component
            misplacements (seed placed on the far side of a bay).
    grid_x_coords, grid_y_coords : (nx,) and (ny,) float or None
        World coordinates of the grid columns/rows.  When provided, they are
        used to compute ``seed_centroid_distances`` in the debug dict.

    Returns
    -------
    labels : (m,) intp
        Centroid index for each active pixel.  If ``return_debug=True``,
        returns ``(labels, debug_dict)`` instead.
    """
    from scipy.spatial import cKDTree

    G = len(points)
    active_2d = active_mask_flat.reshape(grid_ny, grid_nx)
    active_flat_idx = np.where(active_mask_flat)[0]  # flat indices of active pixels

    # Snap each centroid to the nearest active grid point.
    # Use k-NN + greedy unique assignment so that two centroids that land on the
    # same pixel both get a unique slot (first one wins per pixel).
    active_pts = np.column_stack([gx_active, gy_active])
    k = min(5, len(active_pts))
    _, snap_all = cKDTree(active_pts).query(points, k=k)  # (G, k)
    if snap_all.ndim == 1:
        snap_all = snap_all[:, np.newaxis]

    used: set = set()
    seed_flat: np.ndarray = np.empty(G, dtype=np.intp)
    seed_labels: np.ndarray = np.empty(G, dtype=np.int32)
    n_seeds = 0
    for i in range(G):
        for j in range(k):
            idx = int(snap_all[i, j])
            if idx not in used:
                seed_flat[n_seeds] = active_flat_idx[idx]
                seed_labels[n_seeds] = i
                used.add(idx)
                n_seeds += 1
                break
        else:
            # All k candidates already claimed — fall back to nearest
            seed_flat[n_seeds] = active_flat_idx[int(snap_all[i, 0])]
            seed_labels[n_seeds] = i
            n_seeds += 1

    seed_flat = seed_flat[:n_seeds]
    seed_labels = seed_labels[:n_seeds]

    # Convert flat pixel indices to 2-D (row, col).
    seed_rows: np.ndarray = (seed_flat // grid_nx).astype(np.int32)
    seed_cols: np.ndarray = (seed_flat % grid_nx).astype(np.int32)

    labels_2d = _bfs(active_2d, seed_rows, seed_cols, seed_labels)

    # Extract labels for active pixels only.
    flat_labels = labels_2d.ravel()[active_flat_idx]

    # ------------------------------------------------------------------
    # Fix: if any active pixels are unreachable (label == -1), they belong
    # to components that received no seed.  Add one seed per unseeded
    # component (nearest centroid to component's center of mass) and re-run
    # BFS so that every component has a starting point.
    # ------------------------------------------------------------------
    bad = flat_labels < 0
    comp_grid = None  # computed lazily below; shared with debug path
    n_comp = 0
    flat_comp = None
    unseeded_ids: list[int] = []

    if bad.any():
        from scipy.ndimage import label as scipy_label

        comp_grid, n_comp = scipy_label(active_2d)  # 0=inactive, 1..n_comp=components
        flat_comp = comp_grid.ravel()[active_flat_idx]

        labeled_comp = flat_comp[~bad]
        seeded_set = {int(c) for c in labeled_comp}
        unseeded_ids = [c for c in range(1, n_comp + 1) if c not in seeded_set]

        if unseeded_ids:
            centroid_tree = cKDTree(points)
            extra_r: list[int] = []
            extra_c: list[int] = []
            extra_lbl: list[int] = []
            for cid in unseeded_ids:
                pix = np.where(flat_comp == cid)[0]
                xs = gx_active[pix]
                ys = gy_active[pix]
                cx, cy = float(xs.mean()), float(ys.mean())
                best = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
                fi = int(active_flat_idx[pix[best]])
                r_, c_ = int(fi // grid_nx), int(fi % grid_nx)
                _, lbl = centroid_tree.query([[xs[best], ys[best]]])
                extra_r.append(r_)
                extra_c.append(c_)
                extra_lbl.append(int(lbl))

            seed_rows = np.concatenate([seed_rows, np.array(extra_r, dtype=np.int32)])
            seed_cols = np.concatenate([seed_cols, np.array(extra_c, dtype=np.int32)])
            seed_labels = np.concatenate([seed_labels, np.array(extra_lbl, dtype=np.int32)])
            labels_2d = _bfs(active_2d, seed_rows, seed_cols, seed_labels)
            flat_labels = labels_2d.ravel()[active_flat_idx]
            bad = flat_labels < 0

    # Remaining fallback: truly isolated pixels (e.g. a component whose
    # representative pixel was inactive — should be extremely rare).
    if bad.any():
        good_pts = active_pts[~bad]
        bad_pts = active_pts[bad]
        _, nn = cKDTree(good_pts).query(bad_pts)
        flat_labels[bad] = flat_labels[~bad][nn]

    if not return_debug:
        return flat_labels.astype(np.intp)

    # ------------------------------------------------------------------
    # Debug path
    # ------------------------------------------------------------------
    # Ensure component grid is available (may already be computed above).
    if comp_grid is None:
        from scipy.ndimage import label as scipy_label

        comp_grid, n_comp = scipy_label(active_2d)
        flat_comp = comp_grid.ravel()[active_flat_idx]

    # Component of each placed seed (use the final seed arrays, including
    # any extras added for unseeded components).
    sr = seed_rows
    sc = seed_cols
    seed_comps_all = comp_grid[sr, sc]

    # For each centroid (0..G-1): component of its nearest active pixel.
    _, nn_idx = cKDTree(active_pts).query(points)
    centroid_comps = comp_grid.ravel()[active_flat_idx[nn_idx]]

    # Misplacement: first G seeds' component != centroid's nearest-pixel component.
    seed_comps_g = seed_comps_all[:G]
    misplaced = seed_comps_g != centroid_comps

    # Unseeded component info (what the fix handled).
    unseeded_arr = np.array(unseeded_ids, dtype=np.int32)
    unseeded_pixel_mask = (
        np.isin(comp_grid, unseeded_arr) if unseeded_arr.size else (np.zeros((grid_ny, grid_nx), dtype=bool))
    )

    # Seed-to-centroid distances (requires grid coordinate arrays).
    if grid_x_coords is not None and grid_y_coords is not None:
        seed_wx = grid_x_coords[sc[:G].astype(np.intp)]
        seed_wy = grid_y_coords[sr[:G].astype(np.intp)]
        seed_centroid_distances = np.hypot(seed_wx - points[:, 0], seed_wy - points[:, 1])
    else:
        seed_centroid_distances = None

    debug_info = {
        "seed_rows": sr.copy(),
        "seed_cols": sc.copy(),
        "seed_labels": seed_labels.copy(),
        "seed_components": seed_comps_all.copy(),
        "centroid_components": centroid_comps.copy(),
        "misplaced_mask": misplaced,
        "n_misplaced": int(misplaced.sum()),
        "comp_2d": comp_grid,
        "unseeded_comp_ids": unseeded_arr,
        "n_unseeded": len(unseeded_ids),
        "unseeded_pixel_mask": unseeded_pixel_mask,
        "seed_centroid_distances": seed_centroid_distances,
    }
    return flat_labels.astype(np.intp), debug_info
