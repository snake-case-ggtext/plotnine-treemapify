"""Unit tests for plotnine-treemapify layout, subgroup nesting, and fitted text rendering."""

from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import pytest
from plotnine import aes, facet_wrap, ggplot

from plotnine_treemapify import (
    geom_treemap,
    geom_treemap_subgroup2_border,
    geom_treemap_subgroup2_text,
    geom_treemap_subgroup_border,
    geom_treemap_subgroup_text,
    geom_treemap_text,
    treemapify,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "item": ["A", "B", "C", "D", "E", "F"],
        "grp": ["G1", "G1", "G1", "G2", "G2", "G2"],
        "sub2": ["S1", "S1", "S2", "S3", "S3", "S4"],
        "val": [40, 25, 15, 10, 7, 3],
    })


@pytest.mark.parametrize("layout", ["squarified", "scol", "srow", "fixed"])
@pytest.mark.parametrize("start", ["bottomleft", "topleft", "topright", "bottomright"])
def test_area_conservation_and_no_overlap(layout: str, start: str):
    df = _sample_df()
    res = treemapify(df, area="val", layout=layout, start=start)
    assert len(res) == len(df)
    total_val = df["val"].sum()
    areas = (res["xmax"] - res["xmin"]) * (res["ymax"] - res["ymin"])
    assert math.isclose(float(areas.sum()), 1.0, rel_tol=1e-6)
    for _, row in res.iterrows():
        expected = row["val"] / total_val
        observed = (row["xmax"] - row["xmin"]) * (row["ymax"] - row["ymin"])
        assert math.isclose(observed, expected, rel_tol=1e-5)

    # Pairwise zero overlap
    boxes = list(zip(res["xmin"], res["ymin"], res["xmax"], res["ymax"]))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ix = min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])
            iy = min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1])
            assert ix <= 1e-6 or iy <= 1e-6


def test_start_corner_placement():
    df = pd.DataFrame({"item": ["Big", "Small"], "val": [75, 25]})
    bl = treemapify(df, area="val", start="bottomleft").set_index("item")
    tl = treemapify(df, area="val", start="topleft").set_index("item")
    tr = treemapify(df, area="val", start="topright").set_index("item")
    br = treemapify(df, area="val", start="bottomright").set_index("item")

    assert bl.loc["Big", "xmin"] == pytest.approx(0.0)
    assert tr.loc["Big", "xmax"] == pytest.approx(1.0)
    assert br.loc["Big", "xmax"] == pytest.approx(1.0)
    assert tl.loc["Small", "ymin"] == pytest.approx(0.0)


def test_nested_subgroup_containment():
    df = _sample_df()
    res = treemapify(df, area="val", subgroup="grp", subgroup2="sub2")
    for grp, sub in res.groupby("grp"):
        gx0, gy0 = sub["xmin"].min(), sub["ymin"].min()
        gx1, gy1 = sub["xmax"].max(), sub["ymax"].max()
        expected_area = sub["val"].sum() / df["val"].sum()
        assert (gx1 - gx0) * (gy1 - gy0) == pytest.approx(expected_area, rel=1e-5)


def test_render_all_geoms_and_facets():
    df = _sample_df()
    p = (
        ggplot(df, aes(area="val", fill="grp", label="item", subgroup="grp", subgroup2="sub2"))
        + geom_treemap()
        + geom_treemap_subgroup2_border(colour="white", size=1.5)
        + geom_treemap_subgroup_border(colour="black", size=3)
        + geom_treemap_subgroup_text(place="centre", grow=True, alpha=0.5, fontface="italic")
        + geom_treemap_subgroup2_text(place="topright", min_size=4)
        + geom_treemap_text(place="topleft", reflow=True, min_size=4)
    )
    fig = p.draw()
    fig.canvas.draw()
    assert len(fig.axes) == 1

    # Faceted render
    pf = (
        ggplot(df, aes(area="val", fill="sub2", label="item"))
        + geom_treemap()
        + geom_treemap_text(place="centre")
        + facet_wrap("~grp")
    )
    fig2 = pf.draw()
    fig2.canvas.draw()
    assert len(fig2.axes) == 2
