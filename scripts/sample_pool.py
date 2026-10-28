"""Sort the GRPO prompts into strata by sampling M1 on every one. Runs on the GPU pod.

Every RL prompt (each training item as SFT's second pass presents it, src/pac/grpo.py) is answered
8 times at temperature 1 from the full distribution, with no top-k or top-p cut, the way GRPO
samples. A prompt whose answers are partly right and partly wrong is mixed; otherwise settled.
Prompts come from the model's chat template through its processor, as in scripts/predict.py.

Writes samples.jsonl (one line per prompt: the 8 outputs, parsed, and the stratum) and pool.json
(counts per stratum and per attribute and text condition, what the samples state as confidence,
and the number of training steps grpo.steps_for gives).

Usage:
    uv run --group gpu python -I scripts/sample_pool.py <train.jsonl> <image_dir> <adapter_dir> <out_dir>
        [--limit N]
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from PIL import Image

from pac.grpo import GROUP, prompts, steps_for, stratum
from pac.output import parse_output
from pac.prompts import messages

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
MAX_MODEL_LEN = 4096
MAX_TOKENS = 48  # the budget evaluation decodes with (scripts/predict.py)
IMAGE_TOKENS = 256
CHUNK = 256
SALT = "pool-v1"
SAMPLING = {"n": GROUP, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "max_tokens": MAX_TOKENS}


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()

    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}


def seed_for(example: dict) -> int:
    """A sampling seed per prompt, so the pass is reproducible and prompts do not share random draws."""
    return int(hashlib.sha256(f"{SALT}:{example['product_id']}:{example['attribute']}".encode()).hexdigest()[:8], 16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train", type=Path)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    with open(args.train, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
    rl = prompts(examples)[:args.limit]

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    started, clock = datetime.now(UTC), time.monotonic()
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    texts = [processor.apply_chat_template(messages(p, image=p["image"]), tokenize=False, add_generation_prompt=True)
             for p in rl]
    too_long = [p["example_id"] for p, text in zip(rl, texts)
                if len(processor.tokenizer(text).input_ids) + IMAGE_TOKENS + MAX_TOKENS > MAX_MODEL_LEN]
    if too_long:
        raise SystemExit(f"{len(too_long)} prompts do not fit in {MAX_MODEL_LEN} tokens, for example {too_long[:5]}")

    engine = {"max_model_len": MAX_MODEL_LEN, "limit_mm_per_prompt": {"image": 1}, "seed": 0,
              "enable_lora": True, "max_lora_rank": 16}
    llm = LLM(model=MODEL, revision=REVISION, **engine)
    adapter = LoRARequest("m1", 1, str(args.adapter))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    strata, by_group, confidences, errors = Counter(), Counter(), Counter(), Counter()
    settled_right = samples = right = 0
    with open(args.out_dir / "samples.jsonl", "w", encoding="utf-8") as out:
        for start in range(0, len(rl), CHUNK):
            chunk = rl[start:start + CHUNK]
            requests = []
            for p, text in zip(chunk, texts[start:start + CHUNK]):
                with Image.open(args.image_dir / p["image"]) as image:
                    requests.append({"prompt": text, "multi_modal_data": {"image": image.convert("RGB")}})
            params = [SamplingParams(**SAMPLING, seed=seed_for(p)) for p in chunk]
            for p, result in zip(chunk, llm.generate(requests, params, use_tqdm=False, lora_request=adapter)):
                letters = [o["letter"] for o in p["options"]]
                parsed = [parse_output(o.text, letters) for o in result.outputs]
                correct = [int(x.valid and x.choice == p["answer"]) for x in parsed]
                s = stratum(correct)
                strata[s] += 1
                by_group[(p["attribute"], p["condition"], s)] += 1
                settled_right += s == "settled" and all(correct)
                samples += len(correct)
                right += sum(correct)
                confidences.update(x.confidence for x in parsed if x.valid)
                errors.update(x.error for x in parsed if not x.valid)
                record = {k: p[k] for k in ("example_id", "product_id", "family_id", "category", "attribute",
                                            "condition", "answer", "reference")}
                record |= {"outputs": [o.text for o in result.outputs], "choices": [x.choice for x in parsed],
                           "confidences": [x.confidence for x in parsed], "errors": [x.error for x in parsed],
                           "correct": correct, "stratum": s}
                out.write(json.dumps(record) + "\n")
            print(f"{start + len(chunk):,} / {len(rl):,}", flush=True)

    import torch

    elapsed = time.monotonic() - clock
    valid = sum(confidences.values())
    summary = {
        "started": started.isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 1),
        "gpu_hours": round(elapsed / 3600, 4),
        "gpu": torch.cuda.get_device_name(0),
        "git": git_state(),
        "train": {"path": str(args.train), "sha256": hashlib.sha256(args.train.read_bytes()).hexdigest()},
        "model": {"name": MODEL, "revision": REVISION, "adapter": str(args.adapter),
                  "adapter_sha256": hashlib.sha256(
                      (args.adapter / "adapter_model.safetensors").read_bytes()).hexdigest()},
        "sampling": SAMPLING,
        "prompts": len(rl),
        "strata": dict(strata),
        "settled_all_right": settled_right,
        "settled_all_wrong": strata["settled"] - settled_right,
        "by_attribute_condition": {f"{a}/{c}/{s}": n for (a, c, s), n in sorted(by_group.items())},
        "sample_accuracy": right / samples,
        "invalid_rate": sum(errors.values()) / samples,
        "errors": dict(errors),
        "stated_confidence": {f"{v:g}": n / valid for v, n in confidences.most_common(12)},
        "stated_below_0_9": sum(n for v, n in confidences.items() if v < 0.9) / valid,
        "steps": steps_for(strata["mixed"]),
        "versions": {p: metadata.version(p) for p in ("vllm", "torch", "transformers")}
                    | {"python": platform.python_version()},
    }
    (args.out_dir / "pool.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("prompts", "strata", "settled_all_wrong", "sample_accuracy",
                                              "invalid_rate", "stated_below_0_9", "steps")}))


if __name__ == "__main__":
    main()
