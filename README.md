# plotnine-treemapify

Clean-room treemap geoms for `plotnine`, implementing Bruls–Huizing–van Wijk squarified treemaps (`squarified`, `scol`, `srow`, `fixed`), all 4 starting corners (`bottomleft`, `topleft`, `topright`, `bottomright`), up to 3 nested subgroup levels (`subgroup`, `subgroup2`, `subgroup3`), and draw-time auto-fitted/reflowed/grown text labels.

## Public API

- `geom_treemap`
- `geom_treemap_text`
- `geom_treemap_subgroup_border`, `geom_treemap_subgroup2_border`, `geom_treemap_subgroup3_border`
- `geom_treemap_subgroup_text`, `geom_treemap_subgroup2_text`, `geom_treemap_subgroup3_text`
- `treemapify`

## Example

```python
from plotnine import ggplot, aes
from plotnine_treemapify import (
    geom_treemap,
    geom_treemap_text,
    geom_treemap_subgroup_border,
    geom_treemap_subgroup_text,
)

(
    ggplot(df, aes(area="gdp", fill="region", label="country", subgroup="region"))
    + geom_treemap()
    + geom_treemap_subgroup_border()
    + geom_treemap_subgroup_text(place="centre", grow=True, alpha=0.5)
    + geom_treemap_text(place="topleft", reflow=True)
)
```
