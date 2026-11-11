import random

import pytest

from evo2_mvp.evee_short_window import select_partition
from evo2_mvp.sequence import Variant


def test_selection_uses_only_eligible_records_and_preserves_connected_groups():
    rows = [
        Variant("1", 1, "A", "G", "A", 1),
        Variant("1", 2, "A", "G", "A|B", 1),
        Variant("1", 3, "A", "G", "B", 0),
        Variant("1", 4, "A", "G", "C", 1),
        Variant("1", 5, "A", "G", "D", 0),
        Variant("1", 6, "A", "G", "E", 1),
    ]
    eligible = {v.id for v in rows} - {rows[1].id, rows[5].id}
    with pytest.raises(ValueError, match="Too few independent"):
        select_partition(rows, eligible, 4, random.Random(42))
    rows.append(Variant("1", 7, "A", "G", "F", 0))
    eligible.add(rows[-1].id)
    sample = select_partition(rows, eligible, 4, random.Random(42))
    assert sample == select_partition(rows, eligible, 4, random.Random(42))
    assert sorted(item["label"] for item in sample) == [0, 0, 1, 1]
    assert {item["group"] for item in sample} == {"A|B", "C", "D", "F"}
    assert all(item["variant_id"] in eligible for item in sample)
