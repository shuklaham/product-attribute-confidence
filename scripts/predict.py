"""Run a model on built examples with vLLM and write scored predictions plus a run record.

Runs on the GPU pod. Decoding is greedy. At the position where the model writes its choice,
the log-probability of every option letter is kept for the answer-letter baseline (M2-logit).
Prompts come from the model's own chat template through its Hugging Face processor, the same
path training uses, so evaluation and training see identical prompt strings.

Usage:
    uv run --group gpu python -I scripts/predict.py <examples.jsonl> <image_dir> <out_dir>
        [--mode photo_text|text_only|photo_only] [--condition full|redacted]
        [--attribute neckline|sleeve_length] [--limit N]
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from PIL import Image

from pac.examples import CONDITIONS
from pac.inference import MODES, evidence_view, letter_logprobs, letter_token_ids, score
from pac.prompts import messages

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
MAX_MODEL_LEN = 4096
MAX_TOKENS = 48
SEED = 0
CHUNK = 256


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()

    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("examples", type=Path)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--mode", choices=MODES, default="photo_text")
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--attribute")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()

    with open(args.examples, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
    examples = [e for e in examples
                if (args.condition is None or e["condition"] == args.condition)
                and (args.attribute is None or e["attribute"] == args.attribute)][:args.limit]
    if not examples:
        raise SystemExit("no examples match the filters")

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    started, clock = datetime.now(UTC), time.monotonic()
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    tokenizer = processor.tokenizer
    engine = {"max_model_len": MAX_MODEL_LEN, "limit_mm_per_prompt": {"image": 1}, "seed": SEED}
    llm = LLM(model=args.model, revision=args.revision, **engine)
    letter_ids = letter_token_ids(lambda s: tokenizer.encode(s, add_special_tokens=False))
    sampling = {"temperature": 0.0, "max_tokens": MAX_TOKENS, "seed": SEED}
    params = SamplingParams(**sampling, logprob_token_ids=list(letter_ids.values()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompt_hash, first_prompt = hashlib.sha256(), None
    with open(args.out_dir / "predictions.jsonl", "w", encoding="utf-8") as out:
        for start in range(0, len(examples), CHUNK):
            chunk = examples[start:start + CHUNK]
            requests = []
            for example in chunk:
                view, photo = evidence_view(example, args.mode)
                prompt = processor.apply_chat_template(messages(view, image=example["image"] if photo else None),
                                                       tokenize=False, add_generation_prompt=True)
                prompt_hash.update(prompt.encode())
                first_prompt = first_prompt or prompt
                request = {"prompt": prompt}
                if photo:
                    with Image.open(args.image_dir / example["image"]) as image:
                        request["multi_modal_data"] = {"image": image.convert("RGB")}
                requests.append(request)
            for example, result in zip(chunk, llm.generate(requests, params, use_tqdm=False)):
                generated = result.outputs[0]
                pieces = [tokenizer.decode([t]) for t in generated.token_ids]
                logprobs = [{t: lp.logprob for t, lp in position.items()} for position in generated.logprobs]
                letters = [o["letter"] for o in example["options"]]
                record = score(example, generated.text, letter_logprobs(pieces, logprobs, letter_ids, letters))
                record.update(mode=args.mode, finish_reason=generated.finish_reason,
                              prompt_tokens=len(result.prompt_token_ids), output_tokens=len(generated.token_ids))
                out.write(json.dumps(record) + "\n")
            print(f"{start + len(chunk):,} / {len(examples):,}", flush=True)

    import torch

    elapsed = time.monotonic() - clock
    record = {
        "started": started.isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 1),
        "gpu_hours": round(elapsed / 3600, 4),
        "gpu": torch.cuda.get_device_name(0),
        "git": git_state(),
        "examples": {"path": str(args.examples), "sha256": hashlib.sha256(args.examples.read_bytes()).hexdigest()},
        "filters": {"mode": args.mode, "condition": args.condition, "attribute": args.attribute,
                    "limit": args.limit},
        "n": len(examples),
        "model": {"name": args.model, "revision": args.revision},
        "engine": engine,
        "sampling": sampling,
        "letter_token_ids": letter_ids,
        "prompts_sha256": prompt_hash.hexdigest(),
        "first_prompt": first_prompt,
        "versions": {p: metadata.version(p) for p in ("vllm", "torch", "transformers")}
                    | {"python": platform.python_version()},
    }
    (args.out_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(examples):,} predictions in {elapsed / 60:.1f} min to {args.out_dir}")


if __name__ == "__main__":
    main()
