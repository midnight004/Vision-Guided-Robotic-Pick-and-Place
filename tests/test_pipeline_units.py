"""
Lightweight unit tests for the pick-and-place project.

These avoid heavy dependencies (no MuJoCo/torch) so they run fast and keep the
core logic independently testable. Run with:
    python -m pytest tests/ -q
or without pytest:
    python tests/test_pipeline_units.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def test_color_of_mapping():
    from generate_dataset import color_of
    assert color_of("red_box") == "red"
    assert color_of("red_can") == "red"
    assert color_of("blue_capsule") == "blue"
    assert color_of("green_cylinder") == "green"
    assert color_of("yellow_sphere") == "yellow"
    assert color_of("purple_box") == "unknown"
    assert color_of("orange_sphere") == "unknown"
    assert color_of("white_cylinder") == "unknown"
    assert color_of("black_box") == "unknown"


def test_class_ids_match_detector_taxonomy():
    # Dataset class order must match detector.class_to_id
    from generate_dataset import CLASS_NAMES
    expected = {"red": 0, "blue": 1, "green": 2, "yellow": 3, "unknown": 4}
    for name, idx in expected.items():
        assert CLASS_NAMES[idx] == name


def test_expected_color():
    from run_pipeline import Pipeline
    assert Pipeline._expected_color("red_box") == "red"
    assert Pipeline._expected_color("yellow_bottle") == "yellow"
    assert Pipeline._expected_color("purple_cylinder") == "unknown"
    assert Pipeline._expected_bin("orange_box") == "trash"
    assert Pipeline._expected_bin("green_box") == "green"


def test_scene_builder_routing():
    from src.simulation.scene_builder import SceneBuilder, DESTINATIONS
    assert SceneBuilder.get_bin_name("red") == "red"
    assert SceneBuilder.get_bin_name("unknown") == "trash"
    for c in ("red", "blue", "green", "yellow", "trash"):
        assert c in DESTINATIONS


def test_eval_aggregate_math():
    from evaluate import aggregate
    records = [
        {"detected": True, "class_correct": True, "loc_error_m": 0.02,
         "inference_ms": 25.0, "pick_success": True, "landed": True,
         "routed_right": True, "failure": None},
        {"detected": True, "class_correct": False, "loc_error_m": 0.04,
         "inference_ms": 25.0, "pick_success": True, "landed": True,
         "routed_right": False, "failure": "classification"},
        {"detected": False, "class_correct": False, "loc_error_m": None,
         "inference_ms": 25.0, "pick_success": False, "landed": False,
         "routed_right": False, "failure": "detection"},
    ]
    task_summary = {"task": {"avg_cycle_time_s": 4.0, "objects_per_minute": 15.0}}
    agg = aggregate(records, task_summary)
    assert agg["counts"]["items"] == 3
    assert agg["counts"]["detected"] == 2
    assert abs(agg["vision"]["detection_rate"] - 2 / 3) < 1e-6
    assert abs(agg["vision"]["class_accuracy"] - 0.5) < 1e-6
    assert agg["end_to_end"]["successes"] == 1
    assert agg["failures"]["classification"] == 1
    assert agg["failures"]["detection"] == 1


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
