import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def plan(tmp_path: Path) -> dict[str, str]:
    """Run phase2.sh in plan-only mode and return each step's command, keyed by step name."""
    result = subprocess.run(["bash", str(ROOT / "scripts/pod/phase2.sh")], env={"PLAN_ONLY": str(tmp_path),
                            "PATH": "/usr/bin:/bin"}, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    logs = tmp_path / "logs"
    order = [line.split()[0] for line in (logs / "phase2.log").read_text().splitlines() if line != "ALL_DONE"]
    return {name: (logs / f"{name}.log").read_text().strip() for name in order}


def test_each_evaluation_uses_its_own_arms_adapter(tmp_path):
    steps = plan(tmp_path)
    splits = ["heldout_test", "test", "cal_threshold", "cal_fit", "dev"]
    assert list(steps) == (["train_m4"] + [f"m4_{s}" for s in splits] + ["train_m3"] + [f"m3_{s}" for s in splits])
    for arm, kind in (("m4", "calibration"), ("m3", "correctness")):
        assert steps[f"train_{arm}"].endswith(f"artifacts/{arm} --reward {kind}")
        for s in splits:
            command = steps[f"{arm}_{s}"]
            assert f"data/processed/v1/{s}.jsonl" in command
            assert command.endswith(f"artifacts/{arm}_{s} --lora artifacts/{arm}/adapter")
