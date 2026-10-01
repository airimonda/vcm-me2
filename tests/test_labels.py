import pandas as pd

from vcm import labels as L


def test_sizes():
    assert len(L.COMMANDS) == 19 and L.N_CMD == 20 and L.CLASSES[-1] == "OUT_OF_SCOPE"
    assert L.N_SLOT_HEADS == 6 and all(len(v) == 3 for v in L.SLOT_VALUES.values())
    assert L.N_VARIATIONS == 93 and L.N_EVAL_GROUPS == 94
    assert len(L.JOINT_NAMES) == 32


def test_every_variation_row_maps_uniquely():
    v = L.load_variations()
    assert len(v) == 93
    seen = set()
    for i, r in enumerate(v.itertuples(index=False)):
        row = {"command": r.label, "variation": r.phrase,
               "slot_value": r.value if isinstance(r.value, str) else "", "out_of_scope": 0}
        cmd, head, val, var = L.row_to_labels(row)
        assert var == i and var not in seen
        seen.add(var)
        assert cmd == L.COMMANDS.index(r.label)
        if r.label in L.SLOT_VALUES:
            assert head == L.SLOT_COMMANDS.index(r.label)
            assert L.SLOT_VALUES[r.label][val].lower() == r.value.lower()
        else:
            assert head == -1 and val == -1
    assert seen == set(range(93))


def test_slot_normalisation_case_insensitive():
    assert L.slot_value_idx("COLOR", "red") == 0
    assert L.slot_value_idx("COLOR", " BLUE ") == 1
    assert L.slot_value_idx("ALARM", "9:00 pm") == 2
    assert L.slot_value_idx("TIMER", "1 MINUTE") == 2


def test_other_slot_rows_get_minus_one():
    row = {"command": "TIMER", "variation": float("nan"), "slot_value": float("nan"), "out_of_scope": 0,
           "bucket": "TIMER (other slot value)"}
    cmd, head, val, var = L.row_to_labels(row)
    assert cmd == L.COMMANDS.index("TIMER") and head == 0 and val == -1 and var == -1
    row["slot_value"] = "45 seconds"
    assert L.row_to_labels(row)[2] == -1


def test_oos_row():
    row = {"command": "OUT_OF_SCOPE", "variation": float("nan"), "slot_value": float("nan"), "out_of_scope": 1}
    assert L.row_to_labels(row) == (L.OOS_IDX, -1, -1, L.OOS_GROUP)


def test_packed_meta_matches_manifest_if_present():
    import pathlib
    p = pathlib.Path("data/packs/test_meta.parquet")
    if not p.exists():
        return
    m = pd.read_parquet(p)
    assert (m.variation_idx >= 0).all()           # test has no "other slot value" rows
    assert m.variation_idx.nunique() == 94
    assert m.cmd_idx.between(0, 19).all()
