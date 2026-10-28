"""GRPO from M1 for the two RL conditions. Runs on the GPU pod.

    M3: --reward correctness   reward = 1 if right, 0 if wrong
    M4: --reward calibration   reward = correctness − (confidence − correctness)²
    Invalid output gets −1.1 in both (src/pac/rewards.py).

Each step takes 8 prompts, 4 from the mixed stratum and 4 from the settled one (src/pac/grpo.py),
samples 8 answers to each from the current policy at temperature 1, in one left-padded batch,
scores them, and takes one optimizer step on

    loss = −(1/64) Σ_samples advantage × Σ_completion tokens log π(token)

where a sample's advantage is its reward minus its group's mean. The update uses the policy that
sampled, so no importance ratio or clipping is needed. There is no KL term, so a group whose
advantages are all zero contributes nothing and is skipped. The LoRA adapter starts as M1's, with
its rank and modules, and dropout is off so that the update scores exactly the policy that sampled.
The update scores each group on its own, without padding.

Two recipes decide the strata:
- v1 (Milestone 4, --pool DIR): the strata come from a sampling pass of M1 (scripts/sample_pool.py)
  and stay fixed. With --seed 0 it reproduces the Milestone 4 runs.
- v2 (Milestone 5, the default): every --rescore-every steps, the current policy scores every
  prompt with one forward pass up to where it writes its choice, giving p, the probability that a
  sample picks the right letter. The strata follow from p (grpo.strata_from_scores).

--seed sets the schedule and the sampling draws, so the same seed gives both arms the same
first window of prompts and the same random numbers at every step.

--check N samples N prompts without training. It prints the decoded answers with their rewards in
both arms, checks that sampling used the plain distribution and that every answer starts with the
prefix the scorer assumes, compares the training forward's and the scorer's numbers with the
sampler's, and times a sampling batch, a forward and backward pass, and a scoring pass.

Usage:
    uv run --group gpu python -I scripts/train_grpo.py <train.jsonl> <image_dir> <init_adapter> <out_dir>
        --reward correctness|calibration [--seed 0] [--pool DIR] [--rescore-every 50] [--lr 1e-5]
        [--steps N] [--max-steps N] [--check N]
"""

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import torch
from PIL import Image

from pac.grpo import (GROUP, RESCORE_EVERY, SIGNAL_THRESHOLD, TOTAL_STEPS, advantages, expected_signal, prompts,
                      schedule, steps_for, strata_from_scores, window_seed, windows)
from pac.inference import letter_token_ids
from pac.output import parse_output
from pac.prompts import messages
from pac.rewards import KINDS, reward

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
MAX_NEW_TOKENS = 48  # the budget evaluation decodes with (scripts/predict.py)
CHOICE_PREFIX = '{\n  "choice": "'  # how M1 starts every answer (src/pac/sft.py, target_text)
PER_STRATUM = 4
WARMUP = 10
CLIP = 1.0
CHECKPOINT_EVERY = 50
SCORE_BATCH = 32


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()

    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Policy:
    """M1's adapter on the base model, trainable, with sampling and scoring of completions."""

    def __init__(self, adapter: Path):
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig

        self.processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"  # batched sampling continues every prompt from its last token
        base = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16,
                                                           attn_implementation="sdpa").to("cuda")
        self.model = PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
        adapted = [name for name, _ in self.model.named_modules() if name.endswith("lora_A")]
        if not adapted or any("language_model" not in name or "visual" in name for name in adapted):
            raise SystemExit("LoRA must sit on the language model only")
        self.adapted = len(adapted)
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        self.eos = [self.tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
        # The model's own generation defaults (Qwen's top-k and top-p) must not apply, so they are replaced.
        self.generation = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                           repetition_penalty=1.0, max_new_tokens=MAX_NEW_TOKENS,
                                           eos_token_id=self.eos, pad_token_id=self.tokenizer.pad_token_id)
        base.generation_config = self.generation
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.enable_input_require_grads()
        self.keep_logits = "logits_to_keep" in inspect.signature(base.forward).parameters
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.prefix_ids = self.tokenizer.encode(CHOICE_PREFIX, add_special_tokens=False)
        self.letter_ids = letter_token_ids(lambda s: self.tokenizer.encode(s, add_special_tokens=False))

    def encode(self, examples: list[dict], image_dir: Path, copies: int = GROUP) -> dict:
        """`copies` copies of each prompt, in order, left-padded to a common length.

        For one prompt the copies are identical, so there is no padding: that is how groups are scored.
        """
        texts, images = [], []
        for example in examples:
            text = self.processor.apply_chat_template(messages(example, image=example["image"]), tokenize=False,
                                                      add_generation_prompt=True)
            with Image.open(image_dir / example["image"]) as image:
                image = image.convert("RGB")
            texts += [text] * copies
            images += [image] * copies
        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        return {k: v.to("cuda") for k, v in batch.items()}

    @torch.no_grad()
    def sample(self, inputs: dict, keep: bool = False):
        """Sampled completions, their lengths up to and including the end token, and their text."""
        self.model.eval()  # no gradient checkpointing, so generation keeps its cache
        out = self.model.generate(**inputs, generation_config=self.generation, return_dict_in_generate=True,
                                  output_logits=keep, output_scores=keep)
        self.model.train()
        completions = out.sequences[:, inputs["input_ids"].shape[1]:]
        lengths = [next((i + 1 for i, t in enumerate(row) if t in self.eos), len(row)) for row in completions.tolist()]
        texts = [self.tokenizer.decode(row[:n], skip_special_tokens=True)
                 for row, n in zip(completions.tolist(), lengths)]
        return completions, torch.tensor(lengths, device="cuda"), texts, out

    def logprobs(self, inputs: dict, completions: torch.Tensor, lengths: torch.Tensor):
        """Log-probability of every completion token under the current policy, and the mask of real tokens."""
        width = int(lengths.max())
        mask = torch.arange(width, device="cuda")[None, :] < lengths[:, None]
        tokens = completions[:, :width].masked_fill(~mask, self.tokenizer.pad_token_id)
        extra = {"logits_to_keep": width + 1} if self.keep_logits else {}
        out = self.model(input_ids=torch.cat([inputs["input_ids"], tokens], 1),
                         attention_mask=torch.cat([inputs["attention_mask"], mask.long()], 1),
                         # Marks image tokens for the multimodal position encoding; the answer is text.
                         mm_token_type_ids=torch.cat([inputs["mm_token_type_ids"], torch.zeros_like(tokens)], 1),
                         pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"],
                         use_cache=False, **extra)
        # The logit at position i predicts token i + 1, so the width logits before the last one score the completion.
        logits = out.logits[:, -(width + 1):-1].float()
        return torch.log_softmax(logits, -1).gather(-1, tokens[..., None]).squeeze(-1), mask

    @torch.no_grad()
    def score(self, examples: list[dict], image_dir: Path, batch: int = SCORE_BATCH) -> list[float]:
        """For each prompt, the probability that a sample at temperature 1 writes the right letter.

        One forward pass over the prompt and the answer's fixed opening (CHOICE_PREFIX); the next
        token is the letter, so the softmax there is the distribution the sampler draws the choice from.
        """
        self.model.eval()
        prefix = torch.tensor(self.prefix_ids, device="cuda")
        out = []
        for start in range(0, len(examples), batch):
            chunk = examples[start:start + batch]
            inputs = self.encode(chunk, image_dir, copies=1)
            n = len(chunk)
            ones = torch.ones(n, len(prefix), dtype=inputs["attention_mask"].dtype, device="cuda")
            extra = {"logits_to_keep": 1} if self.keep_logits else {}
            logits = self.model(input_ids=torch.cat([inputs["input_ids"], prefix.expand(n, -1)], 1),
                                attention_mask=torch.cat([inputs["attention_mask"], ones], 1),
                                mm_token_type_ids=torch.cat([inputs["mm_token_type_ids"], torch.zeros_like(ones)], 1),
                                pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"],
                                use_cache=False, **extra).logits[:, -1].float()
            probabilities = torch.softmax(logits, -1)
            out += [float(row[self.letter_ids[e["answer"]]]) for e, row in zip(chunk, probabilities)]
        self.model.train()
        return out


def score(texts: list[str], example: dict, kind: str) -> tuple[list, list[float]]:
    letters = [o["letter"] for o in example["options"]]
    parsed = [parse_output(t, letters) for t in texts]
    return parsed, [reward(p, example["answer"], kind) for p in parsed]


def check(policy: Policy, examples: list[dict], image_dir: Path, out_dir: Path, timing: list[dict],
          total_prompts: int) -> None:
    """Sample without training: decoded answers, the sampler's settings, and agreement of every number.

    All prompts are sampled in one left-padded batch, as in training, and each group is then scored on
    its own without padding, so a gap would also show a padding or position error.
    """
    torch.cuda.reset_peak_memory_stats()
    clock = time.monotonic()
    completions, lengths, texts, out = policy.sample(policy.encode(examples, image_dir), keep=True)
    t_sample = time.monotonic() - clock
    # With temperature 1 and no top-k or top-p cut, the processed scores are the raw logits.
    plain = all(torch.equal(s, raw) for s, raw in zip(out.scores, out.logits))
    k = len(policy.prefix_ids)
    prefix_ok = float((completions[:, :k] == torch.tensor(policy.prefix_ids, device="cuda")).all(1).float().mean())
    letter_step = torch.softmax(out.logits[k].float(), -1)  # where every answer writes its letter
    sampled = torch.log_softmax(torch.stack(out.logits, 1).float(), -1)
    sampled = sampled.gather(-1, completions[..., None]).squeeze(-1)
    del out
    scored_p = policy.score(examples, image_dir)
    gaps, p_gaps, report = [], [], []
    for i, example in enumerate(examples):
        rows = slice(i * GROUP, (i + 1) * GROUP)
        with torch.no_grad():
            scored, mask = policy.logprobs(policy.encode([example], image_dir), completions[rows], lengths[rows])
        gap = (sampled[rows, :mask.shape[1]] - scored).abs()[mask]
        gaps.append(gap)
        p_sampler = float(letter_step[i * GROUP, policy.letter_ids[example["answer"]]])
        p_gaps.append(abs(p_sampler - scored_p[i]))
        parsed, m3 = score(texts[rows], example, "correctness")
        _, m4 = score(texts[rows], example, "calibration")
        report.append({"example_id": example["example_id"], "answer": example["answer"],
                       "max_logprob_gap": float(gap.max()), "p_sampler": round(p_sampler, 4),
                       "p_scorer": round(scored_p[i], 4), "outputs": texts[rows],
                       "parsed": [[p.choice, p.confidence, p.error] for p in parsed], "reward_m3": m3,
                       "reward_m4": [round(r, 4) for r in m4]})
        print(json.dumps(report[-1]), flush=True)
    clock = time.monotonic()
    scored, mask = policy.logprobs(policy.encode(examples[:1], image_dir), completions[:GROUP], lengths[:GROUP])
    (scored * mask).sum().backward()
    torch.cuda.synchronize()
    policy.model.zero_grad(set_to_none=True)
    t_update = time.monotonic() - clock
    clock = time.monotonic()
    policy.score(timing, image_dir)
    torch.cuda.synchronize()
    t_score = time.monotonic() - clock
    gaps = torch.cat(gaps)
    summary = {"prompts": len(examples), "plain_sampling": plain, "answers_with_the_scored_prefix": prefix_ok,
               "logprob_gap_max": float(gaps.max()), "logprob_gap_mean": float(gaps.mean()),
               "p_gap_max": max(p_gaps), "seconds_sample_batch": round(t_sample, 2),
               "seconds_forward_backward_group": round(t_update, 2),
               "seconds_full_scoring_estimate": round(t_score / len(timing) * total_prompts, 1),
               "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
               "logits_to_keep": policy.keep_logits, "adapted_modules": policy.adapted,
               "trainable_parameters": sum(p.numel() for p in policy.params)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "check.json").write_text(json.dumps({"summary": summary, "rollouts": report}, indent=2) + "\n")
    print(json.dumps(summary))


def load_pool(pool_dir: Path, by_key: dict) -> dict[str, list]:
    """Recipe v1: the fixed strata from M1's sampling pass, checked against the RL prompts."""
    pool = {"mixed": [], "settled": []}
    with open(pool_dir / "samples.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["product_id"], r["attribute"])
            p = by_key[key]
            if (p["answer"], p["condition"]) != (r["answer"], r["condition"]):
                raise SystemExit(f"the pool does not describe the RL prompt {key}")
            pool[r["stratum"]].append(key)
    return pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train", type=Path)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("init_adapter", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--reward", choices=KINDS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool", type=Path, help="recipe v1: fixed strata from this sampling pass")
    parser.add_argument("--rescore-every", type=int, default=RESCORE_EVERY, help="recipe v2")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, help=f"default: {TOTAL_STEPS} for v2, the rule in grpo.steps_for for v1")
    parser.add_argument("--max-steps", type=int, help="stop early, for a smoke test")
    parser.add_argument("--check", type=int, metavar="N")
    args = parser.parse_args()

    with open(args.train, encoding="utf-8") as f:
        by_key = {(p["product_id"], p["attribute"]): p for p in prompts([json.loads(line) for line in f])}
    keys = sorted(by_key)
    recipe = "v1" if args.pool else "v2"
    if recipe == "v1":
        fixed = load_pool(args.pool, by_key)
        steps = args.steps or steps_for(len(fixed["mixed"]), PER_STRATUM)
        spans = [(0, steps)]
    else:
        steps = args.steps or TOTAL_STEPS
        spans = windows(steps, args.rescore_every)

    torch.manual_seed(args.seed)
    policy = Policy(args.init_adapter)
    if args.check:
        pool = fixed if recipe == "v1" else strata_from_scores(dict(zip(keys[:2048], policy.score(
            [by_key[k] for k in keys[:2048]], args.image_dir))))
        first = schedule(pool, 1, PER_STRATUM, seed=window_seed(args.seed, 0))[0]
        picked = first[:args.check // 2] + first[PER_STRATUM:PER_STRATUM + args.check - args.check // 2]
        check(policy, [by_key[k] for k in picked], args.image_dir, args.out_dir,
              [by_key[k] for k in keys[-256:]], len(keys))
        return

    started, clock = datetime.now(UTC), time.monotonic()
    optimizer = torch.optim.AdamW(policy.params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / WARMUP))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = open(args.out_dir / "log.jsonl", "w", encoding="utf-8")
    scores_log = open(args.out_dir / "scores.jsonl", "w", encoding="utf-8")
    last = min(steps, args.max_steps or steps)
    for window, (start, end) in enumerate(spans):
        if start >= last:
            break
        if recipe == "v1":
            pool = fixed
        else:
            t = time.monotonic()
            p = dict(zip(keys, policy.score([by_key[k] for k in keys], args.image_dir)))
            pool = strata_from_scores(p)
            signal = {k: expected_signal(v) for k, v in p.items()}
            scores_log.write(json.dumps({
                "window": window, "step": start, "seconds": round(time.monotonic() - t, 1),
                "mixed": len(pool["mixed"]), "settled": len(pool["settled"]),
                "signal_mixed": sum(signal[k] for k in pool["mixed"]) / len(pool["mixed"]),
                "signal_all": sum(signal.values()) / len(signal),
                "p": {f"{pid}:{attribute}": round(v, 5) for (pid, attribute), v in p.items()}}) + "\n")
            scores_log.flush()
        plan = schedule(pool, end - start, PER_STRATUM, seed=window_seed(args.seed, window))
        for step, batch_keys in zip(range(start, min(end, last)), plan):
            # The same seed draws the same random numbers at each step, in both arms.
            torch.manual_seed(args.seed * 1_000_003 + step)
            t0 = time.monotonic()
            examples = [by_key[key] for key in batch_keys]
            completions, lengths, texts, _ = policy.sample(policy.encode(examples, args.image_dir))
            groups = []
            for i, example in enumerate(examples):
                rows = slice(i * GROUP, (i + 1) * GROUP)
                parsed, rewards = score(texts[rows], example, args.reward)
                correct = [int(p.valid and p.choice == example["answer"]) for p in parsed]
                groups.append({"example": example, "completions": completions[rows], "lengths": lengths[rows],
                               "parsed": parsed, "correct": correct, "rewards": rewards,
                               "advantages": advantages(rewards)})
            t1 = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            trained = 0
            for g in groups:
                if max(abs(a) for a in g["advantages"]) < 1e-12:
                    continue
                scored, mask = policy.logprobs(policy.encode([g["example"]], args.image_dir), g["completions"],
                                               g["lengths"])
                weights = torch.tensor(g["advantages"], device="cuda", dtype=torch.float32)[:, None]
                loss = -(weights * scored * mask).sum() / (GROUP * len(groups))
                loss.backward()
                trained += 1
            grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.params, CLIP)) if trained else 0.0
            lr = optimizer.param_groups[0]["lr"]  # the rate this step used; the warm-up moves it after the step
            if trained:
                optimizer.step()
            warmup.step()
            t2 = time.monotonic()

            parsed = [p for g in groups for p in g["parsed"]]
            correct = [c for g in groups for c in g["correct"]]
            rewards = [r for g in groups for r in g["rewards"]]
            stated = [(p.confidence, c) for p, c in zip(parsed, correct) if p.valid]
            right, wrong = [q for q, c in stated if c], [q for q, c in stated if not c]
            record = {
                "step": step,
                "window": window,
                "lr": lr,
                "reward": sum(rewards) / len(rewards),
                "accuracy": sum(correct) / len(correct),
                "invalid": sum(not p.valid for p in parsed) / len(parsed),
                "confidence": sum(q for q, _ in stated) / len(stated) if stated else None,
                "confidence_right": sum(right) / len(right) if right else None,
                "confidence_wrong": sum(wrong) / len(wrong) if wrong else None,
                # The sampled answers' Brier score on valid outputs: comparable across arms, unlike their rewards.
                "brier": sum((q - c) ** 2 for q, c in stated) / len(stated) if stated else None,
                "stated": dict(Counter(f"{q:g}" for q, _ in stated).most_common(6)),
                # Groups whose answers are all right or all wrong: no signal for the correctness reward.
                "no_correctness_signal": sum(len(set(g["correct"])) == 1 for g in groups) / len(groups),
                "groups_trained": trained,
                "accuracy_mixed": sum(c for g in groups[:PER_STRATUM] for c in g["correct"]) / (PER_STRATUM * GROUP),
                "accuracy_settled": sum(c for g in groups[PER_STRATUM:] for c in g["correct"]) / (PER_STRATUM * GROUP),
                "grad_norm": grad_norm,
                "seconds_sample": round(t1 - t0, 2),
                "seconds_update": round(t2 - t1, 2),
                "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step % 10 == 0:
                print(f"step {step}: reward {record['reward']:.3f}, accuracy {record['accuracy']:.3f}, "
                      f"trained {trained} groups, {t2 - t0:.0f} s", flush=True)
            if (step + 1) % CHECKPOINT_EVERY == 0:
                policy.model.save_pretrained(args.out_dir / "checkpoint")
    log.close()
    scores_log.close()

    adapter_dir = args.out_dir / "adapter"
    policy.model.save_pretrained(adapter_dir)
    elapsed = time.monotonic() - clock
    history = [json.loads(line) for line in open(args.out_dir / "log.jsonl", encoding="utf-8")]
    rescored = [json.loads(line) for line in open(args.out_dir / "scores.jsonl", encoding="utf-8")]
    record = {
        "started": started.isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 1),
        "gpu_hours": round(elapsed / 3600, 4),
        "gpu": torch.cuda.get_device_name(0),
        "git": git_state(),
        "reward": args.reward,
        "recipe": recipe,
        "seed": args.seed,
        "train": {"path": str(args.train), "sha256": sha256(args.train)},
        "pool": ({"path": str(args.pool), "sha256": sha256(args.pool / "samples.jsonl"),
                  "mixed": len(fixed["mixed"]), "settled": len(fixed["settled"])} if recipe == "v1" else
                 {"rescore_every": args.rescore_every, "signal_threshold": SIGNAL_THRESHOLD,
                  "windows": [{k: w[k] for k in ("step", "seconds", "mixed", "signal_mixed", "signal_all")}
                              for w in rescored]}),
        "init_adapter": {"path": str(args.init_adapter),
                         "sha256": sha256(args.init_adapter / "adapter_model.safetensors")},
        "model": {"name": MODEL, "revision": REVISION},
        "steps": {"planned": steps, "run": len(history)},
        "settings": {"lr": args.lr, "betas": [0.9, 0.99], "weight_decay": 0.0, "warmup_steps": WARMUP, "clip": CLIP,
                     "group": GROUP, "per_stratum": PER_STRATUM, "kl": 0.0, "advantage": "reward minus group mean",
                     "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": MAX_NEW_TOKENS}},
        "no_correctness_signal": sum(h["no_correctness_signal"] for h in history) / len(history),
        "adapter_sha256": sha256(adapter_dir / "adapter_model.safetensors"),
        "versions": {p: metadata.version(p) for p in ("torch", "transformers", "peft")}
                    | {"python": platform.python_version()},
    }
    (args.out_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"trained {len(history)} steps in {elapsed / 60:.1f} min; adapter in {adapter_dir}")


if __name__ == "__main__":
    main()
