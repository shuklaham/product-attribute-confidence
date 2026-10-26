"""LoRA SFT of Qwen3-VL-4B-Instruct on the training split: the M1 model. Runs on the GPU pod.

The adapter (rank 16) sits on the language model's attention and MLP projections only; the vision
encoder is untouched, and the script checks that. The loss covers the answer letter, the JSON
format and the end-of-turn token, not the confidence value (src/pac/sft.py). --check builds one
batch and prints which tokens are trained, without loading the model.

Usage:
    uv run --group gpu python -I scripts/train_sft.py <train.jsonl> <image_dir> <out_dir> --lr 1e-4
        [--epochs 1] [--max-steps N] [--check]
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

import torch
from PIL import Image

from pac.prompts import messages
from pac.sft import completion_labels, presentations, target_text

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
LORA_TARGETS = r"model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
END_OF_TURN = "<|im_end|>"


class Collator:
    """Turns presentations into a padded batch: prompt tokens are ignored, the completion is learned."""

    def __init__(self, processor, image_dir: Path):
        self.processor, self.image_dir = processor, image_dir
        self.tokenizer = processor.tokenizer

    def encode(self, example: dict) -> dict:
        prompt = self.processor.apply_chat_template(messages(example, image=example["image"]), tokenize=False,
                                                    add_generation_prompt=True)
        with Image.open(self.image_dir / example["image"]) as image:
            encoded = self.processor(text=[prompt], images=[image.convert("RGB")], return_tensors="pt")
        target, span = target_text(example["answer"])
        completion = self.tokenizer(target + END_OF_TURN, add_special_tokens=False, return_offsets_mapping=True)
        prompt_ids = encoded["input_ids"][0].tolist()
        return {
            "input_ids": prompt_ids + completion["input_ids"],
            "labels": [-100] * len(prompt_ids) + completion_labels(completion["input_ids"],
                                                                  completion["offset_mapping"], span),
            "pixel_values": encoded["pixel_values"],
            "image_grid_thw": encoded["image_grid_thw"],
        }

    def __call__(self, batch: list[dict]) -> dict:
        items = [self.encode(e) for e in batch]
        width = max(len(i["input_ids"]) for i in items)
        pad = self.tokenizer.pad_token_id

        def padded(key, fill):
            return torch.tensor([i[key] + [fill] * (width - len(i[key])) for i in items])

        return {
            "input_ids": padded("input_ids", pad),
            "labels": padded("labels", -100),
            "attention_mask": torch.tensor([[1] * len(i["input_ids"]) + [0] * (width - len(i["input_ids"]))
                                            for i in items]),
            "pixel_values": torch.cat([i["pixel_values"] for i in items]),
            "image_grid_thw": torch.cat([i["image_grid_thw"] for i in items]),
        }


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()

    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train", type=Path)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

    with open(args.train, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
    shown = presentations(examples, args.epochs)
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    collate = Collator(processor, args.image_dir)

    if args.check:
        batch = collate(shown[:2])
        print({k: tuple(v.shape) for k, v in batch.items()})
        for ids, labels in zip(batch["input_ids"], batch["labels"]):
            print(repr(processor.tokenizer.decode([t for t, lab in zip(ids, labels) if lab != -100])))
        return

    started, clock = datetime.now(UTC), time.monotonic()
    model = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16,
                                                        attn_implementation="sdpa")
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    adapted = [name for name, _ in model.named_modules() if name.endswith("lora_A")]
    if not adapted or any("language_model" not in name or "visual" in name for name in adapted):
        raise SystemExit("LoRA must sit on the language model only")
    model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    training = TrainingArguments(
        output_dir=str(args.out_dir / "trainer"), per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accumulate, learning_rate=args.lr, num_train_epochs=1,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine", warmup_steps=0.03, bf16=True, logging_steps=10, save_strategy="no",
        report_to="none", dataloader_num_workers=4, remove_unused_columns=False, seed=0,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = Trainer(model=model, args=training, train_dataset=shown, data_collator=collate)
    result = trainer.train()

    adapter_dir = args.out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    elapsed = time.monotonic() - clock
    record = {
        "started": started.isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 1),
        "gpu_hours": round(elapsed / 3600, 4),
        "gpu": torch.cuda.get_device_name(0),
        "git": git_state(),
        "train": {"path": str(args.train), "sha256": hashlib.sha256(args.train.read_bytes()).hexdigest()},
        "presentations": len(shown),
        "epochs": args.epochs,
        "model": {"name": MODEL, "revision": REVISION},
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": LORA_TARGETS, "adapted_modules": len(adapted),
                 "trainable_parameters": trainable},
        "optimizer": {"lr": args.lr, "batch": args.batch, "accumulate": args.accumulate, "schedule": "cosine",
                      "warmup": 0.03, "max_steps": args.max_steps},
        "steps": result.global_step,
        "final_train_loss": round(result.training_loss, 4),
        "log_history": trainer.state.log_history,
        "adapter_sha256": hashlib.sha256((adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "versions": {p: metadata.version(p) for p in ("torch", "transformers", "peft", "accelerate")}
                    | {"python": platform.python_version()},
    }
    (args.out_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"trained {result.global_step} steps in {elapsed / 60:.1f} min; loss {result.training_loss:.4f}; "
          f"{len(adapted)} adapted modules; adapter in {adapter_dir}")


if __name__ == "__main__":
    main()
