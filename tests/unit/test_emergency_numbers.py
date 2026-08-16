"""
Emergency contact numbers are India defaults for this deployment.

Upstream shipped "112 / 911 / 108" in the blocks path and "(112 / 911)" in the
sync pipeline. 911 is a US number and does nothing here, while 102 — the
ambulance line the re-skinning guide specifies alongside 112 and 108 — was
missing. Patient-facing emergency text is the last place to carry a number that
does not connect.
"""

from __future__ import annotations

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

PATIENT_FACING = [
    REPO / "graphrag/domain/messages.py",
    REPO / "graphrag/pipeline/graphrag_pipeline.py",
    REPO / "app/services/orchestration/pipeline.py",
]


@pytest.mark.parametrize("path", PATIENT_FACING, ids=lambda p: p.name)
def test_no_us_emergency_number_in_patient_facing_text(path: pathlib.Path) -> None:
    assert path.exists(), f"{path} moved — update this test"
    assert "911" not in path.read_text(), (
        f"{path.name} carries 911; this deployment is India (112 / 108 / 102)"
    )


def test_blocks_emergency_lists_india_numbers() -> None:
    from graphrag.domain.messages import canned_blocks_for

    steps: list[str] = []
    for block in canned_blocks_for("emergency_redirect"):
        data = getattr(block, "data", None)
        steps.extend(getattr(data, "steps", []) or [])
    joined = " ".join(steps)
    assert "112" in joined
    assert "108" in joined
    assert "102" in joined
    assert "911" not in joined
