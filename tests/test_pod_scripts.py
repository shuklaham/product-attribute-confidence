import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


SPLITS = ["heldout_test", "test", "cal_threshold", "cal_fit", "dev"]


def run_plan(tmp_path: Path, phase: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(ROOT / f"scripts/pod/{phase}.sh")],
                          env={"PLAN_ONLY": str(tmp_path), "PATH": "/usr/bin:/bin"} | (env or {}), cwd=ROOT,
                          capture_output=True, text=True)


def steps_of(tmp_path: Path, phase: str, env: dict[str, str] | None = None) -> dict[str, str]:
    """Run a phase script in plan-only mode and return each step's command, keyed by step name, in order."""
    result = run_plan(tmp_path, phase, env)
    assert result.returncode == 0, result.stderr
    logs = tmp_path / "logs"
    order = [line.split()[0] for line in (logs / f"{phase}.log").read_text().splitlines() if line != "ALL_DONE"]
    return {name: (logs / f"{name}.log").read_text().strip() for name in order}


def plan(tmp_path: Path) -> dict[str, str]:
    return steps_of(tmp_path, "phase2")


def test_each_evaluation_uses_its_own_arms_adapter(tmp_path):
    steps = plan(tmp_path)
    assert list(steps) == (["train_m4"] + [f"m4_{s}" for s in SPLITS] + ["train_m3"] + [f"m3_{s}" for s in SPLITS])
    for arm, kind in (("m4", "calibration"), ("m3", "correctness")):
        assert steps[f"train_{arm}"].endswith(f"artifacts/{arm} --reward {kind} --pool artifacts/grpo_pool")
        for s in SPLITS:
            command = steps[f"{arm}_{s}"]
            assert f"data/processed/v1/{s}.jsonl" in command
            assert command.endswith(f"artifacts/{arm}_{s} --lora artifacts/{arm}/adapter")


def plan3(tmp_path: Path, seed: int, extra: bool, rescore: int | None = None) -> dict[str, str]:
    return steps_of(tmp_path, "phase3", {"SEED": str(seed)} | ({"EXTRA": "1"} if extra else {})
                    | ({"RESCORE": str(rescore)} if rescore else {}))


def test_phase3_trains_each_arm_with_its_seed_and_evaluates_its_own_adapter(tmp_path):
    steps = plan3(tmp_path, seed=2, extra=False)
    arms = [("m4_v2_s2", "calibration"), ("m3_v2_s2", "correctness")]
    assert list(steps) == [step for arm, _ in arms
                           for step in [f"train_{arm}"] + [f"{arm}_{s}" for s in SPLITS] + [f"{arm}_swap"]]
    for arm, kind in arms:
        assert steps[f"train_{arm}"].endswith(f"artifacts/{arm} --reward {kind} --seed 2 --rescore-every 50")
        assert "--pool" not in steps[f"train_{arm}"]
        for s in SPLITS:
            assert steps[f"{arm}_{s}"].endswith(f"data/processed/v1/{s}.jsonl data/images artifacts/{arm}_{s} "
                                                f"--lora artifacts/{arm}/adapter")
        assert steps[f"{arm}_swap"].endswith(f"swap_heldout.jsonl data/images artifacts/{arm}_swap "
                                             f"--lora artifacts/{arm}/adapter")


def test_phase3_extra_trains_m5_from_m1_and_swaps_the_reference_models(tmp_path):
    steps = plan3(tmp_path, seed=0, extra=True, rescore=100)
    assert steps["train_m4_v2_s0"].endswith("--seed 0 --rescore-every 100")
    assert steps["train_m5"].endswith("artifacts/m5 --lr 1e-4 --calibrated --init-adapter artifacts/m1_lr2e-4/adapter")
    assert steps["m5_heldout_test"].endswith("artifacts/m5_heldout_test --lora artifacts/m5/adapter")
    assert steps["m0_swap"].endswith("artifacts/m0_swap")
    assert steps["m1_swap"].endswith("artifacts/m1_swap --lora artifacts/m1_lr2e-4/adapter")
    assert list(steps)[-3:] == ["m5_swap", "m0_swap", "m1_swap"]


def test_phase4_trains_m4_m5_m3_with_its_seed_from_m1_and_evaluates_each_own_adapter(tmp_path):
    steps = steps_of(tmp_path, "phase4", {"SEED": "1"})
    names = ["m4_v1_s1", "m5_s1", "m3_v1_s1"]
    assert list(steps) == [step for name in names
                           for step in [f"train_{name}"] + [f"{name}_{s}" for s in SPLITS] + [f"{name}_swap"]]
    for name, kind in (("m4_v1_s1", "calibration"), ("m3_v1_s1", "correctness")):
        assert steps[f"train_{name}"].endswith(
            f"data/images artifacts/m1_lr2e-4/adapter artifacts/{name} --reward {kind} --seed 1 "
            "--pool artifacts/grpo_pool")
    assert steps["train_m5_s1"].endswith("calibsft.jsonl data/images artifacts/m5_s1 --lr 1e-4 --calibrated "
                                         "--init-adapter artifacts/m1_lr2e-4/adapter --seed 1")
    for name in names:
        for s in SPLITS:
            assert steps[f"{name}_{s}"].endswith(f"data/processed/v1/{s}.jsonl data/images artifacts/{name}_{s} "
                                                 f"--lora artifacts/{name}/adapter")
        assert steps[f"{name}_swap"].endswith(f"swap_heldout.jsonl data/images artifacts/{name}_swap "
                                              f"--lora artifacts/{name}/adapter")


def test_phase4_runs_each_named_reference_model_with_only_the_photo_then_only_the_text(tmp_path):
    steps = steps_of(tmp_path, "phase4", {"SEED": "2", "ABLATE": "m0 m1 m4 m5"})
    lora = {"m0": "", "m1": " --lora artifacts/m1_lr2e-4/adapter", "m4": " --lora artifacts/m4/adapter",
            "m5": " --lora artifacts/m5/adapter"}
    assert list(steps)[-8:] == [f"{m}_subset_{mode}" for m in lora for mode in ("photo_only", "text_only")]
    for m, flag in lora.items():
        assert steps[f"{m}_subset_photo_only"].endswith(
            f"subset_heldout.jsonl data/images artifacts/{m}_subset_photo_only --mode photo_only "
            f"--condition redacted{flag}")
        assert steps[f"{m}_subset_text_only"].endswith(
            f"subset_heldout.jsonl data/images artifacts/{m}_subset_text_only --mode text_only{flag}")


def test_phase4_refuses_an_unknown_reference_model_before_running_anything(tmp_path):
    result = run_plan(tmp_path, "phase4", {"SEED": "1", "ABLATE": "m0 m4_v2"})
    assert result.returncode == 1 and "m4_v2" in result.stderr
    assert not (tmp_path / "logs").exists()
