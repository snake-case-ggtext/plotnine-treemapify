"""Draw-time Matplotlib text-fitting artist for treemap tile and subgroup labels.

Measures actual renderer glyph metrics during `draw(renderer)`, wraps words when
`reflow=True`, binary-searches optimal font size when `grow=True` or shrinking to
fit, omits labels below `min_size`, and aligns inside each tile/subgroup box
according to `place`, `padding_x`, and `padding_y`.
"""

from __future__ import annotations

import math
from typing import Any

import matplotlib.colors as mcolors
from matplotlib.artist import Artist
from matplotlib.font_manager import FontProperties


_VALID_PLACES = {
    "centre", "center", "middle",
    "topleft", "top", "topright",
    "left", "right",
    "bottomleft", "bottom", "bottomright",
}


def _unit_to_pt(val: Any) -> float:
    """Convert a padding value (mm float/int, or (value, unit) tuple) to points."""
    if isinstance(val, (tuple, list)) and len(val) == 2:
        amount, unit = float(val[0]), str(val[1]).lower().strip()
        if unit in ("mm", "millimeter", "millimeters"):
            return amount * 72.0 / 25.4
        if unit in ("cm", "centimeter", "centimeters"):
            return amount * 72.0 / 2.54
        if unit in ("in", "inch", "inches"):
            return amount * 72.0
        if unit in ("pt", "pts", "point", "points", "bigpts"):
            return amount
        return amount * 72.0 / 25.4
    # Default unit matches R treemapify's `grid::unit(1, "mm")`
    return float(val) * 72.0 / 25.4


def _parse_fontface(fontface: str | int | None) -> tuple[str, str]:
    """Map R/ggplot2 fontface to Matplotlib (weight, style)."""
    if fontface is None:
        return ("normal", "normal")
    if isinstance(fontface, (int, float)):
        fi = int(fontface)
        if fi == 2:
            return ("bold", "normal")
        if fi == 3:
            return ("normal", "italic")
        if fi == 4:
            return ("bold", "italic")
        return ("normal", "normal")
    s = str(fontface).strip().lower()
    if s in ("bold", "2"):
        return ("bold", "normal")
    if s in ("italic", "oblique", "3"):
        return ("normal", "italic")
    if s in ("bold.italic", "bolditalic", "bold italic", "4"):
        return ("bold", "italic")
    return ("normal", "normal")


def _place_alignments(place: str) -> tuple[str, str]:
    """Return (halign, valign) for a place string."""
    p = str(place).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if p in ("centre", "center", "middle"):
        return ("center", "center")
    if p == "topleft":
        return ("left", "top")
    if p == "top":
        return ("center", "top")
    if p == "topright":
        return ("right", "top")
    if p == "left":
        return ("left", "center")
    if p == "right":
        return ("right", "center")
    if p == "bottomleft":
        return ("left", "bottom")
    if p == "bottom":
        return ("center", "bottom")
    if p == "bottomright":
        return ("right", "bottom")
    return ("center", "center")


def _r_color_to_rgba(color: Any, alpha: float) -> tuple[float, float, float, float]:
    """Convert R-friendly color names (e.g. grey20, grey50, grey) + alpha to RGBA."""
    if color is None:
        return (0.0, 0.0, 0.0, alpha)
    if isinstance(color, str):
        s = color.strip().lower()
        if s in ("grey", "gray"):
            color = "#bebebe"
        elif (s.startswith("grey") or s.startswith("gray")) and s[4:].isdigit():
            lvl = max(0, min(100, int(s[4:])))
            v = lvl / 100.0
            return (v, v, v, float(alpha))
    r, g, b, a = mcolors.to_rgba(color)
    return (float(r), float(g), float(b), float(a) * float(alpha))


class _FitTextArtist(Artist):
    """Draw-time artist that fits, optionally reflows/grows, and aligns text inside boxes."""

    def __init__(
        self,
        ax,
        items: list[dict[str, Any]],
        *,
        padding_x: Any = 1.0,
        padding_y: Any = 1.0,
        place: str = "topleft",
        min_size: float = 4.0,
        grow: bool = False,
        reflow: bool = False,
        zorder: float = 4.0,
    ):
        super().__init__()
        self._ax = ax
        self._items = items
        self._pad_x_pt = _unit_to_pt(padding_x)
        self._pad_y_pt = _unit_to_pt(padding_y)
        self._place = place
        self._min_size = float(min_size)
        self._grow = bool(grow)
        self._reflow = bool(reflow)
        self.set_zorder(zorder)

    def _make_prop(self, item: dict[str, Any], fontsize_pt: float) -> FontProperties:
        weight, style = _parse_fontface(item.get("fontface", "plain"))
        family = item.get("family") or "sans-serif"
        return FontProperties(family=family, weight=weight, style=style, size=fontsize_pt)

    def _measure_line(self, renderer, text: str, prop: FontProperties) -> tuple[float, float, float]:
        if not text:
            _, h, d = renderer.get_text_width_height_descent("Ag", prop, ismath=False)
            return 0.0, float(h), float(d)
        w, h, d = renderer.get_text_width_height_descent(text, prop, ismath=False)
        return float(w), float(h), float(d)

    def _measure_block(
        self,
        renderer,
        lines: list[str],
        prop: FontProperties,
        lineheight: float,
    ) -> tuple[float, float, list[tuple[float, float, float]], float]:
        """Measure multi-line block returning (max_w, total_h, per_line_whd, line_step)."""
        metrics = [self._measure_line(renderer, ln, prop) for ln in lines]
        _, ref_h, _ = renderer.get_text_width_height_descent("Ag", prop, ismath=False)
        line_step = float(ref_h) * float(lineheight)
        max_w = max((m[0] for m in metrics), default=0.0)
        if len(metrics) == 1:
            total_h = metrics[0][1]
        else:
            # Top of first line to bottom of last line
            first_ascent = metrics[0][1] - metrics[0][2]
            last_descent = metrics[-1][2]
            total_h = first_ascent + (len(metrics) - 1) * line_step + last_descent
        return max_w, total_h, metrics, line_step

    def _wrap_text(
        self,
        renderer,
        text: str,
        prop: FontProperties,
        avail_w: float,
    ) -> list[str]:
        """Greedy word wrap respecting explicit newlines."""
        out_lines: list[str] = []
        for para in str(text).split("\n"):
            words = para.split()
            if not words:
                out_lines.append("")
                continue
            cur = words[0]
            for word in words[1:]:
                cand = f"{cur} {word}"
                w, _, _ = self._measure_line(renderer, cand, prop)
                if w <= avail_w:
                    cur = cand
                else:
                    out_lines.append(cur)
                    cur = word
            out_lines.append(cur)
        return out_lines

    def _try_fit(
        self,
        renderer,
        item: dict[str, Any],
        text: str,
        fs_pt: float,
        avail_w: float,
        avail_h: float,
    ) -> tuple[list[str], FontProperties, float, float, list[tuple[float, float, float]], float] | None:
        prop = self._make_prop(item, fs_pt)
        lh = float(item.get("lineheight", 1.15))
        if self._reflow:
            lines = self._wrap_text(renderer, text, prop, avail_w)
        else:
            lines = str(text).split("\n")
        max_w, total_h, metrics, line_step = self._measure_block(renderer, lines, prop, lh)
        if max_w <= avail_w + 1e-3 and total_h <= avail_h + 1e-3:
            return (lines, prop, max_w, total_h, metrics, line_step)
        return None

    def _solve_size(
        self,
        renderer,
        item: dict[str, Any],
        text: str,
        avail_w: float,
        avail_h: float,
        dpi: float,
    ):
        base_fs = float(item.get("size", 18.0))
        min_fs = max(self._min_size, 0.5)

        if not self._grow:
            # First check base size directly
            fit = self._try_fit(renderer, item, text, base_fs, avail_w, avail_h)
            if fit is not None:
                return fit
            # Check lower bound
            low_fit = self._try_fit(renderer, item, text, min_fs, avail_w, avail_h)
            if low_fit is None:
                return None
            lo, hi = min_fs, base_fs
            best = low_fit
            for _ in range(18):
                mid = 0.5 * (lo + hi)
                cand = self._try_fit(renderer, item, text, mid, avail_w, avail_h)
                if cand is not None:
                    best = cand
                    lo = mid
                else:
                    hi = mid
            return best

        # grow=True: find largest font size >= min_fs that fits inside the box
        low_fit = self._try_fit(renderer, item, text, min_fs, avail_w, avail_h)
        if low_fit is None:
            return None
        max_cap = max(base_fs, min(avail_h * 72.0 / max(dpi, 1.0) * 1.5, 450.0))
        lo, hi = min_fs, max_cap
        # Expand hi if text still fits at hi (very short text in huge box)
        best = low_fit
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            cand = self._try_fit(renderer, item, text, mid, avail_w, avail_h)
            if cand is not None:
                best = cand
                lo = mid
            else:
                hi = mid
        return best

    def draw(self, renderer) -> None:
        if not self.get_visible() or not self._items:
            return

        ax = self._ax
        fig = ax.figure
        dpi = float(fig.dpi)
        _, fig_h = fig.canvas.get_width_height()
        px_per_pt = dpi / 72.0
        pad_x_px = self._pad_x_pt * px_per_pt
        pad_y_px = self._pad_y_pt * px_per_pt
        halign, valign = _place_alignments(self._place)

        for idx, item in enumerate(self._items):
            txt = str(item.get("label", "")).strip()
            if not txt:
                continue

            p0 = ax.transAxes.transform((float(item["xmin"]), float(item["ymin"])))
            p1 = ax.transAxes.transform((float(item["xmax"]), float(item["ymax"])))
            bx0, by0 = float(min(p0[0], p1[0])), float(min(p0[1], p1[1]))
            bx1, by1 = float(max(p0[0], p1[0])), float(max(p0[1], p1[1]))

            avail_w = (bx1 - bx0) - 2.0 * pad_x_px
            avail_h = (by1 - by0) - 2.0 * pad_y_px
            if avail_w <= 1.0 or avail_h <= 1.0:
                continue

            solved = self._solve_size(renderer, item, txt, avail_w, avail_h, dpi)
            if solved is None:
                continue

            lines, prop, max_w, total_h, metrics, line_step = solved

            # Vertical block anchor (in bottom-up display pixels)
            if valign == "top":
                block_top = by1 - pad_y_px
            elif valign == "bottom":
                block_top = by0 + pad_y_px + total_h
            else:
                block_top = 0.5 * (by0 + by1) + 0.5 * total_h

            # First line baseline (bottom-up): top of block minus first line's ascent
            first_ascent = metrics[0][1] - metrics[0][2]
            first_baseline_bl = block_top - first_ascent

            rgba = _r_color_to_rgba(item.get("colour", "#000000"), float(item.get("alpha", 1.0)))
            angle = float(item.get("angle", 0.0))

            renderer.open_group("treemap_text", gid=f"tm_lbl_{idx}")
            gc = renderer.new_gc()
            try:
                gc.set_foreground(rgba, isRGBA=True)
                gc.set_alpha(rgba[3])
                gc.set_clip_rectangle(ax.bbox)

                for li, (line, (lw, lh_px, ld_px)) in enumerate(zip(lines, metrics)):
                    if halign == "left":
                        x_px = bx0 + pad_x_px
                    elif halign == "right":
                        x_px = bx1 - pad_x_px - lw
                    else:
                        x_px = 0.5 * (bx0 + bx1) - 0.5 * lw

                    baseline_bl = first_baseline_bl - li * line_step
                    # Clamp subtly inside box if rounding pushed by sub-pixel
                    y_td = float(fig_h) - baseline_bl
                    renderer.draw_text(gc, x_px, y_td, line, prop, angle, ismath=False)
            finally:
                gc.restore()
                renderer.close_group("treemap_text")
