"""Figures of the device: error map, chosen placement and patch partitions (Fig. 1).

The error map and the placement figure need live calibration data of the backend and are
therefore produced by ``scripts/run_experiment.py``; the partition figure needs only
``data/layout.json`` (``scripts/plot_partitions.py``).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

Edge = Tuple[int, int]
PatchQubits = Dict[str, List[int]]
Matchings = Dict[str, Iterable[Edge]]


def _coords_from_phys(log_to_phys: np.ndarray) -> Dict[int, Tuple[float, float]]:
    """Derive (x, y) display coords from IBM physical qubit indices.

    IBM physical qubit p sits at grid position (row=p//10, col=p%10).
    Y is inverted so qubit 0 appears at the top.
    """
    coords: Dict[int, Tuple[float, float]] = {}
    for logical, physical in enumerate(log_to_phys):
        col = float(int(physical) % 10)
        row = -float(int(physical) // 10)
        coords[logical] = (col, row)
    return coords


def plot_patch_coupling_map(
    patch_qubits: PatchQubits,
    boundary_edges: Set[Edge],
    all_edges: Set[Edge],
    log_to_phys: Optional[np.ndarray] = None,
    qubit_coords: Optional[Dict[int, Tuple[float, float]]] = None,
    figsize: Tuple[float, float] = (12, 10),
    node_size: float = 280,
    label_qubits: bool = True,
    title: str = "Patched coupling map",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Visualize a patched coupling map.

    Qubits in the same patch share a color; boundary edges are drawn as red
    dashed lines and internal edges are solid gray lines.

    Parameters
    ----------
    patch_qubits:
        Mapping from patch name to list of qubit indices in that patch.
    boundary_edges:
        Set of (a, b) edges that cross patch boundaries (drawn dashed red).
    all_edges:
        Complete set of coupling-map edges to draw (logical qubit pairs).
    log_to_phys:
        Logical-to-physical index array for IBM devices; used to derive
        qubit grid coordinates when ``qubit_coords`` is None.
    qubit_coords:
        Explicit ``{qubit: (x, y)}`` coordinate mapping. Overrides
        ``log_to_phys`` when provided.
    figsize:
        Matplotlib figure size in inches.
    node_size:
        Scatter marker size for qubit nodes.
    label_qubits:
        Whether to draw the qubit index inside each node.
    title:
        Figure title.
    ax:
        Axes to draw into; if None a fresh figure and axes are created.

    Returns
    -------
    plt.Figure
    """
    if qubit_coords is None:
        if log_to_phys is None:
            raise ValueError("Provide either qubit_coords or log_to_phys.")
        qubit_coords = _coords_from_phys(log_to_phys)

    patch_names = list(patch_qubits.keys())
    n_patches = max(len(patch_names), 1)
    cmap = plt.colormaps.get_cmap("tab10")

    qubit_color: Dict[int, tuple] = {}
    for idx, name in enumerate(patch_names):
        color = cmap(idx % 10)
        for q in patch_qubits[name]:
            qubit_color[q] = color

    boundary_norm: Set[Edge] = {(min(a, b), max(a, b)) for a, b in boundary_edges}

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # --- edges ---
    for a, b in all_edges:
        if a not in qubit_coords or b not in qubit_coords:
            continue
        xa, ya = qubit_coords[a]
        xb, yb = qubit_coords[b]
        key: Edge = (min(a, b), max(a, b))
        if key in boundary_norm:
            ax.plot(
                [xa, xb], [ya, yb],
                color="crimson", linewidth=2.0, linestyle=(0, (4, 3)),
                zorder=1, alpha=0.9,
            )
        else:
            ax.plot(
                [xa, xb], [ya, yb],
                color="#888888", linewidth=1.5, linestyle="-",
                zorder=1, alpha=0.55,
            )

    # --- qubit nodes ---
    qubits_in_map = sorted(qubit_coords.keys())
    xs = [qubit_coords[q][0] for q in qubits_in_map]
    ys = [qubit_coords[q][1] for q in qubits_in_map]
    colors = [qubit_color.get(q, (0.75, 0.75, 0.75, 1.0)) for q in qubits_in_map]

    ax.scatter(
        xs, ys,
        c=colors, s=node_size,
        zorder=2, edgecolors="black", linewidths=0.6,
    )

    if label_qubits:
        font_size = max(4, min(8, int(node_size ** 0.5) - 4))
        for q in qubits_in_map:
            x, y = qubit_coords[q]
            ax.text(x, y, str(q), ha="center", va="center",
                    fontsize=font_size, zorder=3, fontweight="bold",
                    color="black")

    # --- legend ---
    handles = [
        mpatches.Patch(facecolor=cmap(i % 10), edgecolor="black",
                       linewidth=0.5, label=f"Patch {name}")
        for i, name in enumerate(patch_names)
    ]
    handles += [
        plt.Line2D([0], [0], color="crimson", linestyle="--",
                   linewidth=2, label="Boundary edge"),
        plt.Line2D([0], [0], color="#888888", linestyle="-",
                   linewidth=1.5, alpha=0.7, label="Internal edge"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9,
              framealpha=0.9, edgecolor="gray")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=10)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Error diagnostics: bar charts + connectivity heatmap
# -----------------------------------------------------------------------------


def _collect_cz_errors(backend, props, max_qubit: Optional[int] = None) -> Dict[Edge, float]:
    """Map each undirected CZ coupler to its calibrated error rate."""
    out: Dict[Edge, float] = {}
    for (a, b), _ in backend.target["cz"].items():
        if max_qubit is not None and (a >= max_qubit or b >= max_qubit):
            continue
        e = props.gate_error("cz", [a, b])
        if e is None:
            continue
        out[(min(a, b), max(a, b))] = float(e)
    return out


def _collect_readout_errors(backend, props, max_qubit: Optional[int] = None) -> Dict[int, float]:
    """Map each qubit to its readout error."""
    out: Dict[int, float] = {}
    n = max_qubit if max_qubit is not None else backend.num_qubits
    for q in range(n):
        try:
            e = props.readout_error(q)
        except Exception:
            continue
        if e is None:
            continue
        out[q] = float(e)
    return out


def plot_top_backend_errors(
    backend,
    props,
    top_n: int = 100,
    cz_thresh: Optional[float] = None,
    ro_thresh: Optional[float] = None,
    figsize: Tuple[float, float] = (18, 9),
) -> plt.Figure:
    """Stacked top-N CZ + readout error bar charts, sorted high to low.

    Parameters
    ----------
    top_n:
        Number of worst items shown per panel.
    cz_thresh, ro_thresh:
        Optional horizontal reference lines (e.g. the pruning thresholds used
        by placement search) on the CZ and readout panels respectively.
    """
    cz_errors = _collect_cz_errors(backend, props)
    cz_filtered = {k: v for k, v in cz_errors.items() if v < 1}

    cz_sorted = sorted(cz_filtered.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    cz_labels = [f"{a}-{b}" for (a, b), _ in cz_sorted]
    cz_vals = [e for _, e in cz_sorted]

    ro_errors = _collect_readout_errors(backend, props)
    ro_sorted = sorted(ro_errors.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    ro_labels = [str(q) for q, _ in ro_sorted]
    ro_vals = [e for _, e in ro_sorted]

    fig, (ax_cz, ax_ro) = plt.subplots(2, 1, figsize=figsize)

    ax_cz.bar(range(len(cz_vals)), cz_vals, color="steelblue", edgecolor="none")
    if cz_thresh is not None:
        ax_cz.axhline(cz_thresh, color="crimson", linestyle="--", lw=1,
                      label=f"thresh = {cz_thresh}")
        ax_cz.legend()
    ax_cz.set_xticks(range(len(cz_vals)))
    ax_cz.set_xticklabels(cz_labels, rotation=90, fontsize=7)
    ax_cz.set_xlabel("coupler (qubit1-qubit2)")
    ax_cz.set_ylabel("CZ error rate")
    ax_cz.set_title(f"{backend.name} top-{len(cz_vals)} CZ error rates")
    ax_cz.set_xlim(-0.5, len(cz_vals) - 0.5)

    ax_ro.bar(range(len(ro_vals)), ro_vals, color="darkorange", edgecolor="none")
    if ro_thresh is not None:
        ax_ro.axhline(ro_thresh, color="crimson", linestyle="--", lw=1,
                      label=f"thresh = {ro_thresh}")
        ax_ro.legend()
    ax_ro.set_xticks(range(len(ro_vals)))
    ax_ro.set_xticklabels(ro_labels, rotation=90, fontsize=7)
    ax_ro.set_xlabel("qubit")
    ax_ro.set_ylabel("readout error")
    ax_ro.set_title(f"{backend.name} top-{len(ro_vals)} readout errors")
    ax_ro.set_xlim(-0.5, len(ro_vals) - 0.5)

    fig.tight_layout()
    return fig


def plot_backend_placement(
    backend,
    props,
    placement,
    log_to_phys,
    filtered_matchings: Matchings,
    phys_rows: int,
    phys_cols: int,
    thresh_cz: float,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Backend connectivity + chosen placement + matching schedule overlay.

    Layer order (bottom → top): pruned edges, good edges outside the chosen
    subset, chosen-but-unscheduled, then the four scheduled matchings.
    """
    n_phys = phys_rows * phys_cols

    all_device_edges: Set[Edge] = set()
    good_edges: Set[Edge] = set()
    for (a, b), _ in backend.target["cz"].items():
        if a >= n_phys or b >= n_phys:
            continue
        all_device_edges.add((min(a, b), max(a, b)))
        e = props.gate_error("cz", [a, b])
        if e is not None and float(e) < thresh_cz:
            good_edges.add((min(a, b), max(a, b)))
    pruned_edges = all_device_edges - good_edges

    chosen_q = {int(q) for q in log_to_phys}
    chosen_phys_edges = {(a, b) for (a, b) in good_edges
                         if a in chosen_q and b in chosen_q}

    log_to_phys_map = {i: int(p) for i, p in enumerate(log_to_phys)}
    matching_colors = {"A": "tab:blue",  "B": "tab:green",
                       "C": "tab:red",   "D": "tab:purple"}
    phys_edge_color: Dict[Edge, Tuple[str, str]] = {}
    for k, edges in filtered_matchings.items():
        for (la, lb) in edges:
            pa, pb = log_to_phys_map[la], log_to_phys_map[lb]
            phys_edge_color[(min(pa, pb), max(pa, pb))] = (k, matching_colors[k])

    scheduled = set(phys_edge_color.keys())
    chosen_unscheduled = chosen_phys_edges - scheduled
    good_outside_subset = good_edges - chosen_phys_edges

    def _xy(p):
        return (p % phys_cols, -(p // phys_cols))

    if ax is None:
        fig, ax = plt.subplots(figsize=(phys_cols * 0.95 + 1, phys_rows * 0.95 + 1))
    else:
        fig = ax.figure

    for (a, b) in pruned_edges:
        xa, ya = _xy(a); xb, yb = _xy(b)
        ax.plot([xa, xb], [ya, yb], color="crimson", lw=1.0,
                linestyle="--", alpha=0.5, zorder=1)
    for (a, b) in good_outside_subset:
        xa, ya = _xy(a); xb, yb = _xy(b)
        ax.plot([xa, xb], [ya, yb], color="#cccccc", lw=1.2,
                alpha=0.6, zorder=2)
    for (a, b) in chosen_unscheduled:
        xa, ya = _xy(a); xb, yb = _xy(b)
        ax.plot([xa, xb], [ya, yb], color="#666666", lw=1.6,
                alpha=0.8, zorder=3)

    for (a, b), (_, color) in phys_edge_color.items():
        xa, ya = _xy(a); xb, yb = _xy(b)
        ax.plot([xa, xb], [ya, yb], color=color, lw=2.6, alpha=0.95, zorder=4)

    for q in range(n_phys):
        x, y = _xy(q)
        if q in chosen_q:
            ax.scatter([x], [y], s=330, c="white",
                       edgecolors="black", linewidths=0.9, zorder=5)
            ax.text(x, y, str(q), ha="center", va="center",
                    fontsize=6, fontweight="bold", color="black", zorder=6)
        else:
            ax.scatter([x], [y], s=180, c="#eeeeee",
                       edgecolors="#888888", linewidths=0.5, zorder=5)
            ax.text(x, y, str(q), ha="center", va="center",
                    fontsize=5, color="#888888", zorder=6)

    handles = []
    for k in ("A", "B", "C", "D"):
        n = len(filtered_matchings.get(k, []))
        handles.append(mlines.Line2D([], [], color=matching_colors[k], lw=2.6,
                                     label=f"Matching {k} ({n} edges)"))
    if chosen_unscheduled:
        handles.append(mlines.Line2D(
            [], [], color="#666666", lw=1.6,
            label=f"Chosen, unscheduled — {len(chosen_unscheduled)} edge(s)"))
    handles += [
        mlines.Line2D([], [], color="#cccccc", lw=1.2,
                      label=f"Good (outside subset) — {len(good_outside_subset)} edges"),
        mlines.Line2D([], [], color="crimson", lw=1.0, linestyle="--",
                      label=f"Pruned (CZ err ≥ {thresh_cz}) — {len(pruned_edges)} edges"),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(4, len(handles)),
        fontsize=8,
        framealpha=0.95,
        borderaxespad=0.3,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    ax.set_aspect("equal")
    ax.axis("off")
    dr = getattr(placement, "dr", "?")
    dc = getattr(placement, "dc", "?")
    ax.set_title(
        f"Backend + chosen placement (dr={dr}, dc={dc}) + matching schedule",
        pad=42,
    )
    return fig


def plot_backend_error_map(
    backend,
    props,
    phys_rows: int,
    phys_cols: int,
    chosen_qubits: Optional[Set[int]] = None,
    excluded_qubits: Optional[Set[int]] = None,
    ax: Optional[plt.Axes] = None,
    cz_cmap: str = "Reds",
    ro_cmap: str = "Blues",
    cz_broken_threshold: float = 1.0,
) -> plt.Figure:
    """Connectivity map with qubits colored by readout error, edges by CZ error.

    Two horizontal colorbars on top (readout) and bottom (CZ) of the axes.

    ``excluded_qubits`` (e.g. qubits dropped by placement due to bad readout
    or full edge isolation) are rendered grey instead of colormap-colored,
    and edges touching them are drawn faint grey too. ``chosen_qubits`` is
    accepted but no longer affects styling.

    Edges with CZ error ≥ ``cz_broken_threshold`` (default 1.0 — i.e. fully
    broken/uncalibrated gates) are drawn as faint grey lines and excluded
    from the CZ colormap normalisation so the remaining gradient is useful.
    """
    n_phys = phys_rows * phys_cols
    excluded = set(excluded_qubits or ())
    ro_vals = _collect_readout_errors(backend, props, max_qubit=n_phys)
    cz_vals = _collect_cz_errors(backend, props, max_qubit=n_phys)

    cz_normal = {e: v for e, v in cz_vals.items()
                 if v < cz_broken_threshold
                 and e[0] not in excluded and e[1] not in excluded}
    cz_broken = {e: v for e, v in cz_vals.items() if v >= cz_broken_threshold}
    cz_touching_excluded = {
        e for e in cz_vals
        if e not in cz_broken and (e[0] in excluded or e[1] in excluded)
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(phys_cols * 1.0 + 2, phys_rows * 1.0 + 3))
    else:
        fig = ax.figure

    # Log-scale norms require strictly positive bounds — skip zero/negative
    # calibration readings when computing vmin.
    ro_pos = [v for v in ro_vals.values() if v > 0]
    cz_pos = [v for v in cz_normal.values() if v > 0]
    ro_norm = mpl.colors.LogNorm(
        vmin=min(ro_pos) if ro_pos else 1e-4,
        vmax=max(ro_pos) if ro_pos else 1.0,
    )
    cz_norm = mpl.colors.LogNorm(
        vmin=min(cz_pos) if cz_pos else 1e-4,
        vmax=max(cz_pos) if cz_pos else 1.0,
    )
    ro_cmap_obj = plt.colormaps.get_cmap(ro_cmap)
    cz_cmap_obj = plt.colormaps.get_cmap(cz_cmap)

    def _xy(p):
        return (p % phys_cols, -(p // phys_cols))

    for (a, b), _ in cz_broken.items():
        xa, ya = _xy(a); xb, yb = _xy(b)
        ax.plot([xa, xb], [ya, yb], color="#bbbbbb",
                lw=1.2, linestyle="--", alpha=0.6, zorder=0)

    for (a, b) in cz_touching_excluded:
        xa, ya = _xy(a); xb, yb = _xy(b)
        ax.plot([xa, xb], [ya, yb], color="#cccccc",
                lw=1.5, alpha=0.7, zorder=0)

    for (a, b), v in cz_normal.items():
        xa, ya = _xy(a); xb, yb = _xy(b)
        ax.plot([xa, xb], [ya, yb], color=cz_cmap_obj(cz_norm(v)),
                lw=2.8, alpha=0.95, zorder=1)

    for q in range(n_phys):
        x, y = _xy(q)
        if q in excluded:
            color = (0.85, 0.85, 0.85, 1.0)
            edge_color = "#888888"
            text_color = "#666666"
        elif q in ro_vals:
            color = ro_cmap_obj(ro_norm(max(ro_vals[q], ro_norm.vmin)))
            edge_color = "black"
            text_color = "black"
        else:
            color = (0.93, 0.93, 0.93, 1.0)
            edge_color = "black"
            text_color = "black"
        ax.scatter([x], [y], s=260, c=[color],
                   edgecolors=edge_color, linewidths=0.7, zorder=2)
        ax.text(x, y, str(q), ha="center", va="center",
                fontsize=6, fontweight="bold", color=text_color, zorder=3)

    ax.set_aspect("equal")
    ax.axis("off")
    title = f"{backend.name} CZ + readout error map (log scale)"
    if cz_broken:
        title += f"  ({len(cz_broken)} broken CZ shown in grey)"
    ax.set_title(title)

    sm_ro = ScalarMappable(norm=ro_norm, cmap=ro_cmap_obj); sm_ro.set_array([])
    sm_cz = ScalarMappable(norm=cz_norm, cmap=cz_cmap_obj); sm_cz.set_array([])
    # Use make_axes_locatable so the colorbars steal from ax's own region.
    # Combined with matching invisible spacers on any sibling axes (see
    # plot_placement_and_errors), this keeps connectivity grids the same size
    # across subplots.
    divider = make_axes_locatable(ax)
    cax_top = divider.append_axes("top", size="3%", pad=0.45)
    cax_bot = divider.append_axes("bottom", size="3%", pad=0.45)
    cb_ro = fig.colorbar(sm_ro, cax=cax_top, orientation="horizontal")
    cax_top.xaxis.set_ticks_position("top")
    cax_top.xaxis.set_label_position("top")
    cb_ro.set_label("readout error (qubits)")
    cb_cz = fig.colorbar(sm_cz, cax=cax_bot, orientation="horizontal")
    cb_cz.set_label(f"CZ error (edges, <{cz_broken_threshold})")

    return fig


def plot_placement_and_errors(
    backend,
    props,
    placement,
    log_to_phys,
    filtered_matchings: Matchings,
    phys_rows: int,
    phys_cols: int,
    thresh_cz: float,
    logical_rows: Optional[int] = None,
    logical_cols: Optional[int] = None,
    figsize: Tuple[float, float] = (22, 13),
) -> plt.Figure:
    """Side-by-side error heatmap (left) + placement schedule (right).

    ``logical_rows`` / ``logical_cols`` describe the original target grid the
    placement was searching for; they are needed to reconstruct which
    physical qubits the placement dropped (so the error map can grey them
    out). If omitted, the dropped-qubit overlay is skipped.
    """
    fig, (ax_e, ax_p) = plt.subplots(1, 2, figsize=figsize)

    plot_backend_error_map(
        backend=backend, props=props,
        phys_rows=phys_rows, phys_cols=phys_cols,
        chosen_qubits={int(q) for q in log_to_phys},
        ax=ax_e,
    )
    plot_backend_placement(
        backend=backend, props=props, placement=placement,
        log_to_phys=log_to_phys, filtered_matchings=filtered_matchings,
        phys_rows=phys_rows, phys_cols=phys_cols, thresh_cz=thresh_cz,
        ax=ax_p,
    )
    # Reserve invisible spacers on the placement ax that mirror the error-map's
    # top/bottom colorbars; this makes both connectivity grids render at the
    # same physical size under aspect="equal".
    div_p = make_axes_locatable(ax_p)
    for loc in ("top", "bottom"):
        spacer = div_p.append_axes(loc, size="3%", pad=0.45)
        spacer.axis("off")
    fig.tight_layout()
    return fig


