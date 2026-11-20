"""plotnine geoms for treemap visualizations (clean-room treemapify port)."""

from __future__ import annotations

from typing import Any

import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, Rectangle
from plotnine import element_blank, theme
from plotnine.geoms.geom import geom
from plotnine.geoms.geom_polygon import geom_polygon

from .layout import compute_treemap_layout
from .text import _FitTextArtist, _r_color_to_rgba, _unit_to_pt


def _resolve_layout(params: dict[str, Any]) -> tuple[str, str]:
    layout = params.get("layout", "squarified")
    if params.get("fixed") is True:
        layout = "fixed"
    start = params.get("start", "bottomleft")
    return str(layout), str(start)


def _normalize_linetype(lt: Any) -> str:
    if lt is None:
        return "solid"
    s = str(lt).strip().lower()
    mapping = {
        "0": "none", "blank": "none",
        "1": "solid", "solid": "solid", "-": "solid",
        "2": "dashed", "dashed": "dashed", "--": "dashed",
        "3": "dotted", "dotted": "dotted", ":": "dotted",
        "4": "dashdot", "dotdash": "dashdot", "-.": "dashdot",
    }
    return mapping.get(s, "solid")


def _clean_color(col: Any, alpha: float = 1.0) -> tuple[str | tuple, float]:
    if col is None:
        return ("none", 0.0)
    if isinstance(col, float) and col != col:  # NaN
        return ("none", 0.0)
    if isinstance(col, str) and col.strip().lower() in ("", "none", "na", "transparent"):
        return ("none", 0.0)
    rgba = _r_color_to_rgba(col, alpha)
    if rgba[3] <= 0.0:
        return ("none", 0.0)
    return (rgba, rgba[3])


def _apply_treemap_theme(plot):
    """Strip x/y axes, ticks, labels, and panel gridlines like R's geom_treemap()."""
    blank_theme = theme(
        axis_text=element_blank(),
        axis_ticks=element_blank(),
        axis_title=element_blank(),
        axis_line=element_blank(),
        panel_grid=element_blank(),
    )
    plot.theme = plot.theme + blank_theme
    return plot


class geom_treemap(geom):
    """Draw a treemap of rectangular tiles proportional to `area`."""

    REQUIRED_AES = {"area"}
    NON_MISSING_AES = {"area"}
    DEFAULT_AES = {
        "fill": "#333333",
        "color": "grey",
        "alpha": 1.0,
        "size": 0.5,
        "linetype": "solid",
    }
    DEFAULT_PARAMS = {
        "stat": "identity",
        "position": "identity",
        "na_rm": False,
        "fixed": None,
        "layout": "squarified",
        "start": "bottomleft",
        "radius": 0.0,
    }

    draw_legend = staticmethod(geom_polygon.draw_legend)

    def __radd__(self, plot):
        plot = super().__radd__(plot)
        return _apply_treemap_theme(plot)

    def setup_data(self, data):
        data = data.copy()
        data["xmin"] = 0.0
        data["xmax"] = 1.0
        data["ymin"] = 0.0
        data["ymax"] = 1.0
        return data

    def draw_panel(self, data, panel_params, coord, ax):
        layout, start = _resolve_layout(self.params)
        tiles_df, _ = compute_treemap_layout(data, area_col="area", layout=layout, start=start)
        radius_pt = _unit_to_pt(self.params.get("radius", 0.0)) if self.params.get("radius", 0.0) else 0.0
        zorder = self.params.get("zorder", 2.0)

        for _, row in tiles_df.iterrows():
            x0, y0 = float(row["xmin"]), float(row["ymin"])
            w = float(row["xmax"]) - x0
            h = float(row["ymax"]) - y0
            if w <= 0.0 or h <= 0.0:
                continue

            alpha = float(row.get("alpha", 1.0))
            fc, _ = _clean_color(row.get("fill", "#333333"), alpha)
            ec, _ = _clean_color(row.get("color", "grey"), 1.0)
            size = float(row.get("size", 0.5))
            lw_pt = size * 72.0 / 96.0 if ec != "none" and size > 0.0 else 0.0
            ls = _normalize_linetype(row.get("linetype", "solid"))
            if ls == "none":
                ec = "none"
                lw_pt = 0.0

            if radius_pt > 0.0:
                patch = FancyBboxPatch(
                    (x0, y0), w, h,
                    boxstyle=f"round,pad=0,rounding_size={radius_pt / 300.0:.4f}",
                    facecolor=fc,
                    edgecolor=ec,
                    linewidth=lw_pt,
                    linestyle=ls if ls != "none" else "solid",
                    transform=ax.transAxes,
                    zorder=zorder,
                    clip_on=True,
                )
            else:
                patch = Rectangle(
                    (x0, y0), w, h,
                    facecolor=fc,
                    edgecolor=ec,
                    linewidth=lw_pt,
                    linestyle=ls if ls != "none" else "solid",
                    transform=ax.transAxes,
                    zorder=zorder,
                    clip_on=True,
                )
            ax.add_patch(patch)


class geom_treemap_subgroup_border(geom):
    """Draw enclosing border rectangles around treemap subgroups."""

    REQUIRED_AES = {"area"}
    NON_MISSING_AES = {"area"}
    DEFAULT_AES = {
        "color": "grey50",
        "size": 4.0,
        "alpha": 1.0,
        "linetype": "solid",
    }
    DEFAULT_PARAMS = {
        "stat": "identity",
        "position": "identity",
        "na_rm": False,
        "fixed": None,
        "layout": "squarified",
        "start": "bottomleft",
        "level": "subgroup",
    }

    draw_legend = staticmethod(geom_polygon.draw_legend)

    def __radd__(self, plot):
        plot = super().__radd__(plot)
        return _apply_treemap_theme(plot)

    def setup_data(self, data):
        data = data.copy()
        data["xmin"] = 0.0
        data["xmax"] = 1.0
        data["ymin"] = 0.0
        data["ymax"] = 1.0
        return data

    def draw_panel(self, data, panel_params, coord, ax):
        layout, start = _resolve_layout(self.params)
        level = str(self.params.get("level", "subgroup"))
        _, group_boxes = compute_treemap_layout(data, area_col="area", layout=layout, start=start)
        entries = group_boxes.get(level) or []
        if not entries:
            return

        df_reset = data.reset_index(drop=True)
        zorder = self.params.get("zorder", 3.0)

        for entry in entries:
            row = df_reset.iloc[entry["first_row"]]
            x0, y0 = float(entry["xmin"]), float(entry["ymin"])
            w = float(entry["xmax"]) - x0
            h = float(entry["ymax"]) - y0
            if w <= 0.0 or h <= 0.0:
                continue

            alpha = float(row.get("alpha", 1.0))
            ec, _ = _clean_color(row.get("color", "grey50"), alpha)
            size = float(row.get("size", 4.0))
            lw_pt = size * 72.0 / 96.0 if ec != "none" and size > 0.0 else 0.0
            ls = _normalize_linetype(row.get("linetype", "solid"))
            if ls == "none" or ec == "none" or lw_pt <= 0.0:
                continue

            patch = Rectangle(
                (x0, y0), w, h,
                facecolor="none",
                edgecolor=ec,
                linewidth=lw_pt,
                linestyle=ls,
                transform=ax.transAxes,
                zorder=zorder,
                clip_on=True,
            )
            ax.add_patch(patch)


class geom_treemap_subgroup2_border(geom_treemap_subgroup_border):
    """Draw enclosing border rectangles around `subgroup2` treemap groups."""

    DEFAULT_PARAMS = {
        **geom_treemap_subgroup_border.DEFAULT_PARAMS,
        "level": "subgroup2",
    }


class geom_treemap_subgroup3_border(geom_treemap_subgroup_border):
    """Draw enclosing border rectangles around `subgroup3` treemap groups."""

    DEFAULT_PARAMS = {
        **geom_treemap_subgroup_border.DEFAULT_PARAMS,
        "level": "subgroup3",
    }


class geom_treemap_text(geom):
    """Draw fitted, optionally reflowed and grown text labels inside treemap tiles."""

    REQUIRED_AES = {"area", "label"}
    NON_MISSING_AES = {"area"}
    DEFAULT_AES = {
        "color": "#000000",
        "size": 18.0,
        "alpha": 1.0,
        "family": "",
        "fontface": "plain",
        "angle": 0.0,
        "lineheight": 1.15,
    }
    DEFAULT_PARAMS = {
        "stat": "identity",
        "position": "identity",
        "na_rm": False,
        "show_legend": False,
        "padding_x": 1.0,
        "padding_y": 1.0,
        "place": "topleft",
        "min_size": 4.0,
        "grow": False,
        "reflow": False,
        "fixed": None,
        "layout": "squarified",
        "start": "bottomleft",
    }

    draw_legend = staticmethod(geom_polygon.draw_legend)

    def __init__(self, mapping=None, data=None, show_legend=False, **kwargs):
        super().__init__(mapping=mapping, data=data, show_legend=show_legend, **kwargs)

    def __radd__(self, plot):
        plot = super().__radd__(plot)
        return _apply_treemap_theme(plot)

    def setup_data(self, data):
        data = data.copy()
        data["xmin"] = 0.0
        data["xmax"] = 1.0
        data["ymin"] = 0.0
        data["ymax"] = 1.0
        return data

    def draw_panel(self, data, panel_params, coord, ax):
        layout, start = _resolve_layout(self.params)
        tiles_df, _ = compute_treemap_layout(data, area_col="area", layout=layout, start=start)
        if tiles_df.empty:
            return

        items = []
        for _, row in tiles_df.iterrows():
            lbl = row.get("label", "")
            if lbl is None or (isinstance(lbl, float) and lbl != lbl):
                continue
            items.append({
                "xmin": float(row["xmin"]),
                "ymin": float(row["ymin"]),
                "xmax": float(row["xmax"]),
                "ymax": float(row["ymax"]),
                "label": str(lbl),
                "colour": row.get("color", "#000000"),
                "size": float(row.get("size", 18.0)),
                "alpha": float(row.get("alpha", 1.0)),
                "family": row.get("family", ""),
                "fontface": row.get("fontface", "plain"),
                "angle": float(row.get("angle", 0.0)),
                "lineheight": float(row.get("lineheight", 1.15)),
            })

        artist = _FitTextArtist(
            ax,
            items,
            padding_x=self.params.get("padding_x", 1.0),
            padding_y=self.params.get("padding_y", 1.0),
            place=self.params.get("place", "topleft"),
            min_size=self.params.get("min_size", 4.0),
            grow=self.params.get("grow", False),
            reflow=self.params.get("reflow", False),
            zorder=self.params.get("zorder", 5.0),
        )
        ax.add_artist(artist)


class geom_treemap_subgroup_text(geom):
    """Draw fitted text labels for treemap subgroups."""

    REQUIRED_AES = {"area"}
    NON_MISSING_AES = {"area"}
    DEFAULT_AES = {
        "color": "grey20",
        "size": 36.0,
        "alpha": 1.0,
        "family": "",
        "fontface": "plain",
        "angle": 0.0,
        "lineheight": 1.15,
    }
    DEFAULT_PARAMS = {
        "stat": "identity",
        "position": "identity",
        "na_rm": False,
        "show_legend": False,
        "padding_x": 1.0,
        "padding_y": 1.0,
        "place": "bottom",
        "min_size": 4.0,
        "grow": False,
        "reflow": False,
        "fixed": None,
        "layout": "squarified",
        "start": "bottomleft",
        "level": "subgroup",
    }

    draw_legend = staticmethod(geom_polygon.draw_legend)

    def __init__(self, mapping=None, data=None, show_legend=False, **kwargs):
        super().__init__(mapping=mapping, data=data, show_legend=show_legend, **kwargs)

    def __radd__(self, plot):
        plot = super().__radd__(plot)
        return _apply_treemap_theme(plot)

    def setup_data(self, data):
        data = data.copy()
        data["xmin"] = 0.0
        data["xmax"] = 1.0
        data["ymin"] = 0.0
        data["ymax"] = 1.0
        return data

    def draw_panel(self, data, panel_params, coord, ax):
        layout, start = _resolve_layout(self.params)
        level = str(self.params.get("level", "subgroup"))
        _, group_boxes = compute_treemap_layout(data, area_col="area", layout=layout, start=start)
        entries = group_boxes.get(level) or []
        if not entries:
            return

        df_reset = data.reset_index(drop=True)
        items = []
        for entry in entries:
            row = df_reset.iloc[entry["first_row"]]
            items.append({
                "xmin": float(entry["xmin"]),
                "ymin": float(entry["ymin"]),
                "xmax": float(entry["xmax"]),
                "ymax": float(entry["ymax"]),
                "label": str(entry["label"]),
                "colour": row.get("color", "grey20"),
                "size": float(row.get("size", 36.0)),
                "alpha": float(row.get("alpha", 1.0)),
                "family": row.get("family", ""),
                "fontface": row.get("fontface", "plain"),
                "angle": float(row.get("angle", 0.0)),
                "lineheight": float(row.get("lineheight", 1.15)),
            })

        artist = _FitTextArtist(
            ax,
            items,
            padding_x=self.params.get("padding_x", 1.0),
            padding_y=self.params.get("padding_y", 1.0),
            place=self.params.get("place", "bottom"),
            min_size=self.params.get("min_size", 4.0),
            grow=self.params.get("grow", False),
            reflow=self.params.get("reflow", False),
            zorder=self.params.get("zorder", 4.5),
        )
        ax.add_artist(artist)


class geom_treemap_subgroup2_text(geom_treemap_subgroup_text):
    """Draw fitted text labels for `subgroup2` treemap groups."""

    DEFAULT_PARAMS = {
        **geom_treemap_subgroup_text.DEFAULT_PARAMS,
        "level": "subgroup2",
    }


class geom_treemap_subgroup3_text(geom_treemap_subgroup_text):
    """Draw fitted text labels for `subgroup3` treemap groups."""

    DEFAULT_PARAMS = {
        **geom_treemap_subgroup_text.DEFAULT_PARAMS,
        "level": "subgroup3",
    }
