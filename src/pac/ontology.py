"""Label set, raw-value mapping and phrase patterns, loaded from configs/ontology.yaml."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "ontology.yaml"
MISSING_VALUES = {"", "na", "n/a", "none", "null"}

# Outcomes of mapping a raw catalog value.
OK = "ok"
UNLABELED = "unlabeled"
DROPPED = "dropped"
UNMAPPED = "unmapped"


def _compile(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


@dataclass(frozen=True)
class Label:
    name: str
    definition: str
    patterns: tuple[re.Pattern, ...]


@dataclass(frozen=True, eq=False)
class Attribute:
    key: str
    source_field: str
    question: str
    labels: tuple[Label, ...]
    raw_to_label: dict[str, str]
    dropped_raw: frozenset[str]
    dropped_patterns: tuple[re.Pattern, ...]
    attribute_words: re.Pattern
    head_modifiers: re.Pattern
    redaction_cues: tuple[re.Pattern, ...] = ()
    exclusion_reason: str = ""
    exclusion_patterns: tuple[re.Pattern, ...] = ()
    exclusion_raw: tuple[tuple[str, frozenset[str]], ...] = ()

    @property
    def label_names(self) -> list[str]:
        return [label.name for label in self.labels]

    @property
    def definitions(self) -> dict[str, str]:
        return {label.name: label.definition for label in self.labels}

    @property
    def all_patterns(self) -> tuple[re.Pattern, ...]:
        """Kept and dropped label phrases."""
        return tuple(p for label in self.labels for p in label.patterns) + self.dropped_patterns

    @property
    def redaction_patterns(self) -> tuple[re.Pattern, ...]:
        """Everything redaction removes besides the attribute words: label phrases and cues."""
        return self.all_patterns + self.redaction_cues

    def canonical(self, raw) -> tuple[str | None, str]:
        """Map a raw catalog value to (label, outcome)."""
        if raw is None or str(raw).strip().lower() in MISSING_VALUES:
            return None, UNLABELED
        value = str(raw).strip()
        if value in self.raw_to_label:
            return self.raw_to_label[value], OK
        if value in self.dropped_raw:
            return None, DROPPED
        return None, UNMAPPED

    def named_labels(self, text: str) -> set[str]:
        """Kept labels whose phrases appear in the text."""
        return {label.name for label in self.labels if any(p.search(text) for p in label.patterns)}

    def excluded_by_text(self, text: str) -> bool:
        """True when the text describes a product whose label is ambiguous by definition for this attribute."""
        return any(p.search(text) for p in self.exclusion_patterns)

    def excluded_by_catalog(self, raw_labels: dict) -> bool:
        """True when another catalog field marks the product as ambiguous for this attribute."""
        return any(str(raw_labels.get(name, "")).strip() in values for name, values in self.exclusion_raw)


@dataclass(frozen=True, eq=False)
class Ontology:
    version: str
    min_train_support: int
    attributes: dict[str, Attribute]
    sha256: str


def load(path: Path | str = DEFAULT_PATH) -> Ontology:
    raw = Path(path).read_bytes()
    spec = yaml.safe_load(raw)
    attributes = {}
    for key, a in spec["attributes"].items():
        dropped = a.get("dropped") or []
        exclude_if = a.get("exclude_if") or {}
        attributes[key] = Attribute(
            key=key,
            source_field=a["source_field"],
            question=a["question"],
            labels=tuple(
                Label(item["name"], item["definition"], tuple(_compile(p) for p in item["phrases"]))
                for item in a["labels"]
            ),
            raw_to_label={value: item["name"] for item in a["labels"] for value in item["raw_values"]},
            dropped_raw=frozenset(value for d in dropped for value in d["raw_values"]),
            dropped_patterns=tuple(_compile(p) for d in dropped for p in d.get("phrases", [])),
            attribute_words=_compile(a["attribute_words"]),
            head_modifiers=_compile(a["head_modifiers"]),
            redaction_cues=tuple(_compile(p) for p in a.get("redaction_cues") or []),
            exclusion_reason=exclude_if.get("reason", ""),
            exclusion_patterns=tuple(_compile(p) for p in exclude_if.get("phrases", [])),
            exclusion_raw=tuple((name, frozenset(values))
                                for name, values in (exclude_if.get("raw_values") or {}).items()),
        )
    return Ontology(spec["version"], spec["min_train_support"], attributes, hashlib.sha256(raw).hexdigest())
