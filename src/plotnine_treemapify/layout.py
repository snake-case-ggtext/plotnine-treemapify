"""Clean-room treemap layout algorithms.

Implements:
- Bruls, Huizing, & van Wijk (1999) squarified treemaps (`layout="squarified"`)
- Column-first (`layout="scol"`) and row-first (`layout="srow"`) squarified variants
- Order-preserving column layout (`layout="fixed"`)
- All 4 starting corners (`bottomleft`, `topleft`, `topright`, `bottomright`)
- Up to 3 nested subgroup hierarchy levels (`subgroup`, `subgroup2`, `subgroup3`)
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

VALID_LAYOUTS = ("squarified", "scol", "srow", "fixed")
VALID_STARTS = ("bottomleft", "topleft", "topright", "bottomright")


def _worst_aspect(areas: list[float], side: float) -> float:
    """Worst aspect ratio max(w/h, h/w) for a strip of `areas` laid along `side`."""
    if not areas or side <= 0.0:
        return float("inf")
    s = sum(areas)
    if s <= 0.0:
        return float("inf")
    s2 = s * s
    w2 = side * side
    a_max = max(areas)
    a_min = min(areas)
    if a_min <= 0.0:
        return float("inf")
    return max((w2 * a_max) / s2, s2 / (w2 * a_min))


def _layout_fixed(
    items: list[tuple[int, float]],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> dict[int, tuple[float, float, float, float]]:
    """Order-preserving vertical column layout with ceil(sqrt(n)) tiles per column."""
    n = len(items)
    if n == 0:
        return {}
    total = sum(a for _, a in items)
    if total <= 0.0:
        return {}

    per_col = max(1, math.ceil(math.sqrt(n)))
    cols = [items[i : i + per_col] for i in range(0, n, per_col)]

    w = x1 - x0
    h = y1 - y0
    out: dict[int, tuple[float, float, float, float]] = {}
    cx = x0
    for col in cols:
        col_sum = sum(a for _, a in col)
        cw = w * (col_sum / total)
        cy = y0
        for idx, a in col:
            th = h * (a / col_sum) if col_sum > 0.0 else 0.0
            out[idx] = (cx, cy, cx + cw, cy + th)
            cy += th
        cx += cw
    return out


def _layout_squarified(
    items: list[tuple[int, float]],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    layout: str = "squarified",
) -> dict[int, tuple[float, float, float, float]]:
    """Bruls-Huizing-van Wijk (1999) squarified treemap with optional scol/srow first strip."""
    if not items:
        return {}
    total_raw = sum(a for _, a in items)
    w_tot = x1 - x0
    h_tot = y1 - y0
    rect_area = w_tot * h_tot
    if total_raw <= 0.0 or rect_area <= 0.0:
        return {}

    # Sort descending by area (stable on original index)
    scale = rect_area / total_raw
    remaining = [(idx, a * scale) for idx, a in sorted(items, key=lambda kv: (-kv[1], kv[0]))]

    rx0, ry0, rx1, ry1 = x0, y0, x1, y1
    out: dict[int, tuple[float, float, float, float]] = {}
    first_strip = True

    while remaining:
        w = rx1 - rx0
        h = ry1 - ry0
        if w <= 1e-12 or h <= 1e-12:
            for idx, _ in remaining:
                out[idx] = (rx0, ry0, rx1, ry1)
            break

        if first_strip and layout == "scol":
            is_col = True
        elif first_strip and layout == "srow":
            is_col = False
        else:
            is_col = w >= h
        first_strip = False

        side = h if is_col else w
        strip = [remaining[0]]
        strip_areas = [remaining[0][1]]
        pos = 1
        while pos < len(remaining):
            cand = remaining[pos][1]
            if _worst_aspect(strip_areas + [cand], side) <= _worst_aspect(strip_areas, side) + 1e-12:
                strip.append(remaining[pos])
                strip_areas.append(cand)
                pos += 1
            else:
                break

        strip_sum = sum(strip_areas)
        if is_col:
            sw = strip_sum / h if h > 0.0 else 0.0
            sw = min(sw, w)
            cy = ry0
            for idx, a in strip:
                th = (a / sw) if sw > 1e-12 else (h / len(strip))
                ty1 = min(cy + th, ry1)
                out[idx] = (rx0, cy, rx0 + sw, ty1)
                cy = ty1
            rx0 += sw
        else:
            sh = strip_sum / w if w > 0.0 else 0.0
            sh = min(sh, h)
            cx = rx0
            for idx, a in strip:
                tw = (a / sh) if sh > 1e-12 else (w / len(strip))
                tx1 = min(cx + tw, rx1)
                out[idx] = (cx, ry0, tx1, ry0 + sh)
                cx = tx1
            ry0 += sh

        remaining = remaining[pos:]

    return out


def _partition_rect(
    items: list[tuple[int, float]],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    layout: str,
) -> dict[int, tuple[float, float, float, float]]:
    if layout == "fixed":
        return _layout_fixed(items, x0, y0, x1, y1)
    return _layout_squarified(items, x0, y0, x1, y1, layout=layout)


def _mirror_box(
    box: tuple[float, float, float, float],
    start: str,
    xlim: tuple[float, float] = (0.0, 1.0),
    ylim: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    lx0, lx1 = xlim
    ly0, ly1 = ylim
    if start == "bottomleft":
        return (x0, y0, x1, y1)
    if start == "topleft":
        return (x0, ly0 + ly1 - y1, x1, ly0 + ly1 - y0)
    if start == "bottomright":
        return (lx0 + lx1 - x1, y0, lx0 + lx1 - x0, y1)
    if start == "topright":
        return (lx0 + lx1 - x1, ly0 + ly1 - y1, lx0 + lx1 - x0, ly0 + ly1 - y0)
    raise ValueError(f"start must be one of {VALID_STARTS}, got {start!r}")


def _recursive_layout(
    df: pd.DataFrame,
    row_indices: list[int],
    areas: np.ndarray,
    group_cols: Sequence[str],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    layout: str,
    tile_boxes: dict[int, tuple[float, float, float, float]],
    group_boxes: dict[str, list[dict]],
) -> None:
    if not row_indices:
        return
    if not group_cols:
        items = [(idx, float(areas[idx])) for idx in row_indices if float(areas[idx]) > 0.0]
        rects = _partition_rect(items, x0, y0, x1, y1, layout)
        tile_boxes.update(rects)
        return

    col = group_cols[0]
    rest = group_cols[1:]
    # Preserve order of appearance for fixed layout; unique groups with total positive area
    groups: dict[object, list[int]] = {}
    for idx in row_indices:
        val = df.iloc[idx][col]
        groups.setdefault(val, []).append(idx)

    group_keys = list(groups.keys())
    group_items = []
    for gi, key in enumerate(group_keys):
        tot = float(sum(max(float(areas[i]), 0.0) for i in groups[key]))
        if tot > 0.0:
            group_items.append((gi, tot))

    g_rects = _partition_rect(group_items, x0, y0, x1, y1, layout)
    for gi, rect in g_rects.items():
        key = group_keys[gi]
        member_indices = groups[key]
        group_boxes.setdefault(col, []).append({
            "level": col,
            "label": str(key),
            "key": key,
            "first_row": member_indices[0],
            "box": rect,
        })
        _recursive_layout(
            df, member_indices, areas, rest,
            rect[0], rect[1], rect[2], rect[3],
            layout, tile_boxes, group_boxes,
        )


def compute_treemap_layout(
    data: pd.DataFrame,
    *,
    area_col: str = "area",
    layout: str = "squarified",
    start: str = "bottomleft",
    xlim: tuple[float, float] = (0.0, 1.0),
    ylim: tuple[float, float] = (0.0, 1.0),
) -> tuple[pd.DataFrame, dict[str, list[dict]]]:
    """Compute tile boxes and nested subgroup boxes for one panel DataFrame.

    Returns `(tiles_df, subgroup_boxes)` where `tiles_df` has `xmin, ymin, xmax, ymax`
    columns added, and `subgroup_boxes[level]` is a list of dicts with mirrored
    `xmin, ymin, xmax, ymax`, `label`, and `first_row`.
    """
    if layout not in VALID_LAYOUTS:
        raise ValueError(f"layout must be one of {VALID_LAYOUTS}, got {layout!r}")
    if start not in VALID_STARTS:
        raise ValueError(f"start must be one of {VALID_STARTS}, got {start!r}")

    df = data.reset_index(drop=True).copy()
    if df.empty:
        df["xmin"] = []
        df["ymin"] = []
        df["xmax"] = []
        df["ymax"] = []
        return df, {}

    areas = pd.to_numeric(df[area_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    valid_indices = [i for i, a in enumerate(areas) if a > 0.0]

    # Collect present subgroup hierarchy columns in order: subgroup -> subgroup2 -> subgroup3
    group_cols = [c for c in ("subgroup", "subgroup2", "subgroup3") if c in df.columns and df[c].notna().any()]

    tile_boxes: dict[int, tuple[float, float, float, float]] = {}
    raw_group_boxes: dict[str, list[dict]] = {}
    _recursive_layout(
        df, valid_indices, areas, group_cols,
        xlim[0], ylim[0], xlim[1], ylim[1],
        layout, tile_boxes, raw_group_boxes,
    )

    xmins, ymins, xmaxs, ymaxs = [], [], [], []
    keep_mask = []
    for i in range(len(df)):
        if i in tile_boxes:
            bx0, by0, bx1, by1 = _mirror_box(tile_boxes[i], start, xlim=xlim, ylim=ylim)
            xmins.append(bx0)
            ymins.append(by0)
            xmaxs.append(bx1)
            ymaxs.append(by1)
            keep_mask.append(True)
        else:
            keep_mask.append(False)

    out_df = df.loc[keep_mask].copy().reset_index(drop=True)
    out_df["xmin"] = xmins
    out_df["ymin"] = ymins
    out_df["xmax"] = xmaxs
    out_df["ymax"] = ymaxs

    mirrored_groups: dict[str, list[dict]] = {}
    for lvl, entries in raw_group_boxes.items():
        mlist = []
        for entry in entries:
            bx0, by0, bx1, by1 = _mirror_box(entry["box"], start, xlim=xlim, ylim=ylim)
            mlist.append({
                "level": lvl,
                "label": entry["label"],
                "key": entry["key"],
                "first_row": entry["first_row"],
                "xmin": bx0,
                "ymin": by0,
                "xmax": bx1,
                "ymax": by1,
            })
        mirrored_groups[lvl] = mlist

    return out_df, mirrored_groups


def treemapify(
    data: pd.DataFrame,
    area: str,
    subgroup: str | None = None,
    subgroup2: str | None = None,
    subgroup3: str | None = None,
    layout: str = "squarified",
    start: str = "bottomleft",
    xlim: tuple[float, float] = (0.0, 1.0),
    ylim: tuple[float, float] = (0.0, 1.0),
) -> pd.DataFrame:
    """Compute treemap tile coordinates `(xmin, ymin, xmax, ymax)` for a DataFrame."""
    work = data.copy()
    if subgroup is not None:
        work["subgroup"] = work[subgroup]
    if subgroup2 is not None:
        work["subgroup2"] = work[subgroup2]
    if subgroup3 is not None:
        work["subgroup3"] = work[subgroup3]
    out, _ = compute_treemap_layout(
        work, area_col=area, layout=layout, start=start, xlim=xlim, ylim=ylim
    )
    return out
