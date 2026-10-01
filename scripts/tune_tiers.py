"""Search width/depth settings that land each architecture near the tier targets.
Prints the closest few candidates; the chosen ones are hard-coded in vcm/models.py."""
import itertools, sys
sys.path.insert(0, ".")
from vcm.models import build_model, count_params, TIER_TARGETS

GRIDS = {
    "ds_cnn": dict(width=range(32, 257, 8), depth=[3, 4, 5, 6]),
    "bc_resnet": dict(scale=range(4, 30)),
    "tc_resnet": dict(width=[round(0.1 * i, 1) for i in range(5, 50)]),
    "matchbox": dict(width=range(32, 192, 8), blocks=[3, 4], repeat=[1, 2, 3]),
    "crnn": dict(ch=[8, 12, 16, 20, 24, 32], hidden=range(32, 257, 16), layers=[1, 2]),
    "conformer": dict(d=range(32, 129, 8), layers=[1, 2, 3, 4], heads=[4]),
}
for arch, grid in GRIDS.items():
    keys = list(grid)
    for tier, tgt in TIER_TARGETS.items():
        res = []
        for vals in itertools.product(*[grid[k] for k in keys]):
            cfg = dict(zip(keys, vals))
            res.append((abs(count_params(build_model(arch, tier, **cfg)) - tgt), cfg))
        res.sort(key=lambda r: r[0])
        print(arch, tier, [(r[1], tgt + 0 if False else r[0]) for r in res[:3]])
