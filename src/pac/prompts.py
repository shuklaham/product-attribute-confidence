"""The shared prompt. Every model sees the same instructions and layout."""

from __future__ import annotations

INSTRUCTIONS = """Use the product photo and text to answer the question.
Choose exactly one option letter.
Return JSON with exactly two keys: "choice" and "confidence".
"confidence" is a number from 0 to 1: how likely your chosen option matches the product's catalog label.
If the evidence is incomplete, still choose the most likely option and express uncertainty through confidence.
Return only the JSON object."""


def prompt_text(example: dict) -> str:
    """Prompt text for one example. Only the evidence, the question and the options reach it."""
    lines = [
        INSTRUCTIONS,
        "",
        f"Product title: {example['title'] or '(none)'}",
        f"Product description: {example['description'] or '(none)'}",
        "",
        f"Question: {example['question']}",
        "Options:",
    ]
    lines += [f"{o['letter']}. {o['label']}: {o['definition']}" for o in example["options"]]
    return "\n".join(lines)


def messages(example: dict, image: str | None = None) -> list[dict]:
    """Chat messages in the format Qwen3-VL processors accept, with the photo first."""
    content = [{"type": "image", "image": image}] if image is not None else []
    content.append({"type": "text", "text": prompt_text(example)})
    return [{"role": "user", "content": content}]
