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
        assert steps[f"train_{arm}"].endswith(f"artifacts/{arm} --reward {kind} --pool artifacts/grpo_pool")
        for s in splits:
            command = steps[f"{arm}_{s}"]
            assert f"data/processed/v1/{s}.jsonl" in command
            assert command.endswith(f"artifacts/{arm}_{s} --lora artifacts/{arm}/adapter")


def plan3(tmp_path: Path, seed: int, extra: bool) -> dict[str, str]:
    env = {"PLAN_ONLY": str(tmp_path), "PATH": "/usr/bin:/bin", "SEED": str(seed)} | ({"EXTRA": "1"} if extra else {})
    result = subprocess.run(["bash", str(ROOT / "scripts/pod/phase3.sh")], env=env, cwd=ROOT, capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr
    logs = tmp_path / "logs"
    order = [line.split()[0] for line in (logs / "phase3.log").read_text().splitlines() if line != "ALL_DONE"]
    return {name: (logs / f"{name}.log").read_text().strip() for name in order}


def test_phase3_trains_each_arm_with_its_seed_and_evaluates_its_own_adapter(tmp_path):
    steps = plan3(tmp_path, seed=2, extra=False)
    splits = ["heldout_test", "test", "cal_threshold", "cal_fit", "dev"]
    arms = [("m4_v2_s2", "calibration"), ("m3_v2_s2", "correctness")]
    assert list(steps) == [step for arm, _ in arms
                           for step in [f"train_{arm}"] + [f"{arm}_{s}" for s in splits] + [f"{arm}_swap"]]
    for arm, kind in arms:
        assert steps[f"train_{arm}"].endswith(f"artifacts/{arm} --reward {kind} --seed 2")
        assert "--pool" not in steps[f"train_{arm}"]
        for s in splits:
            assert steps[f"{arm}_{s}"].endswith(f"data/processed/v1/{s}.jsonl data/images artifacts/{arm}_{s} "
                                                f"--lora artifacts/{arm}/adapter")
        assert steps[f"{arm}_swap"].endswith(f"swap_heldout.jsonl data/images artifacts/{arm}_swap "
                                             f"--lora artifacts/{arm}/adapter")


def test_phase3_extra_trains_m5_from_m1_and_swaps_the_reference_models(tmp_path):
    steps = plan3(tmp_path, seed=0, extra=True)
    assert steps["train_m5"].endswith("artifacts/m5 --lr 1e-4 --calibrated --init-adapter artifacts/m1_lr2e-4/adapter")
    assert steps["m5_heldout_test"].endswith("artifacts/m5_heldout_test --lora artifacts/m5/adapter")
    assert steps["m0_swap"].endswith("artifacts/m0_swap")
    assert steps["m1_swap"].endswith("artifacts/m1_swap --lora artifacts/m1_lr2e-4/adapter")
    assert list(steps)[-3:] == ["m5_swap", "m0_swap", "m1_swap"]
