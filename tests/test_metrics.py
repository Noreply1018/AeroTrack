from aerotrack.metrics import match_detection_counts, mean_average_precision_50


def test_detection_matching_counts_tp_fp_fn_and_duplicate_detection() -> None:
    gt = [
        {"sequence_id": "s1", "frame_id": "000001", "class_id": "0", "x1": "0", "y1": "0", "x2": "10", "y2": "10"},
        {"sequence_id": "s1", "frame_id": "000001", "class_id": "1", "x1": "20", "y1": "20", "x2": "30", "y2": "30"},
    ]
    detections = [
        {
            "sequence_id": "s1",
            "frame_id": "000001",
            "class_id": "0",
            "score": "0.9",
            "x1": "0",
            "y1": "0",
            "x2": "10",
            "y2": "10",
        },
        {
            "sequence_id": "s1",
            "frame_id": "000001",
            "class_id": "0",
            "score": "0.8",
            "x1": "0",
            "y1": "0",
            "x2": "10",
            "y2": "10",
        },
        {
            "sequence_id": "s1",
            "frame_id": "000001",
            "class_id": "2",
            "score": "0.7",
            "x1": "50",
            "y1": "50",
            "x2": "60",
            "y2": "60",
        },
    ]

    assert match_detection_counts(gt, detections, iou_threshold=0.5) == (1, 2, 1)


def test_map50_penalizes_late_false_positive() -> None:
    gt = [
        {"sequence_id": "s1", "frame_id": "000001", "class_id": "0", "x1": "0", "y1": "0", "x2": "10", "y2": "10"},
    ]
    detections = [
        {
            "sequence_id": "s1",
            "frame_id": "000001",
            "class_id": "0",
            "score": "0.9",
            "x1": "50",
            "y1": "50",
            "x2": "60",
            "y2": "60",
        },
        {
            "sequence_id": "s1",
            "frame_id": "000001",
            "class_id": "0",
            "score": "0.8",
            "x1": "0",
            "y1": "0",
            "x2": "10",
            "y2": "10",
        },
    ]

    assert mean_average_precision_50(gt, detections, iou_threshold=0.5) == 0.5
