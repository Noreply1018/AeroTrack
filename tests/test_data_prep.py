from aerotrack.data_prep import _bbox_from_range_angle


def test_dense_fallback_builds_bbox() -> None:
    bbox, source = _bbox_from_range_angle({"dense": [[3, 7], [1, 9], [2, 5]]})

    assert bbox == (1.0, 5.0, 3.0, 9.0)
    assert source == "range_angle.dense_bbox"
