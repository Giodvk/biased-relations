
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict


MASK_TOKEN = "[MASK]"
DEFAULT_PATTERN = "BR*_prompts.json"


def parse_relation_triplet(raw_relation: str) -> tuple[str, str, str]:
    """
    Parse a relation encoded as:

        <group, relation, stereotype>

    split(",", 2) is intentionally used so that the third field can still
    contain commas.
    """
    if not isinstance(raw_relation, str):
        raise ValueError(
            f"Relation must be a string, got {type(raw_relation).__name__}."
        )

    relation = raw_relation.strip()

    if not (relation.startswith("<") and relation.endswith(">")):
        raise ValueError(
            f"Invalid relation format: {raw_relation!r}. "
            "Expected '<group, relation, stereotype>'."
        )

    content = relation[1:-1].strip()
    parts = [part.strip() for part in content.split(",", 2)]

    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError(
            f"Could not parse relation triplet: {raw_relation!r}."
        )

    group, relation_phrase, stereotype = parts
    return group, relation_phrase, stereotype


def parse_bias_label(label: str) -> tuple[str, str]:
    """
    Parse labels such as:

        BR01(age)

    Returns:
        ("BR01", "age")

    If the textual category is absent, the second value is "".
    """
    if not isinstance(label, str):
        raise ValueError(
            f"Bias label must be a string, got {type(label).__name__}."
        )

    label = label.strip()

    match = re.fullmatch(r"(BR\d{2})(?:\((.+)\))?", label)

    if match is None:
        raise ValueError(
            f"Invalid bias label: {label!r}. "
            "Expected a format such as 'BR01(age)'."
        )

    bias_code = match.group(1)
    bias_type = (match.group(2) or "").strip()

    return bias_code, bias_type


def load_prompt_file(path: Path) -> list[dict]:
    """Load and validate one BRxx_prompts.json file."""
    print(f"[Bias] Loading {path.name}")

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {path}: {exc}"
        ) from exc

    if not isinstance(data, list):
        raise ValueError(
            f"{path.name}: top-level JSON object must be a list."
        )

    return data


def normalize_prompt_file(
    path: Path,
    strict: bool = True,
) -> list[dict]:
    """
    Flatten one prompt file into one row per generated prompt.

    The original relation and all its structured components are retained.
    """
    data = load_prompt_file(path)
    rows = []

    for relation_index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"{path.name}, relation {relation_index}: "
                "expected a JSON object."
            )

        if "input" not in item or "outputs" not in item:
            raise ValueError(
                f"{path.name}, relation {relation_index}: "
                "missing 'input' or 'outputs'."
            )

        raw_relation = item["input"]
        group, relation_phrase, stereotype = parse_relation_triplet(
            raw_relation
        )

        outputs = item["outputs"]

        if not isinstance(outputs, list) or len(outputs) == 0:
            raise ValueError(
                f"{path.name}, relation {relation_index}: "
                "'outputs' must be a non-empty list."
            )

        # Stable ID inside the complete dataset.
        relation_id = (
            f"{path.stem}__relation_{relation_index:04d}"
        )

        for prompt_index, output in enumerate(outputs):
            if (
                not isinstance(output, list)
                or len(output) < 3
            ):
                raise ValueError(
                    f"{path.name}, relation {relation_index}, "
                    f"prompt {prompt_index}: expected "
                    "[prompt, target_group, bias_label]."
                )

            prompt = str(output[0]).strip()
            target_group = str(output[1]).strip()
            bias_label = str(output[2]).strip()

            bias_code, bias_type = parse_bias_label(bias_label)

            filename_match = re.match(
                r"(BR\d{2})",
                path.stem,
            )

            filename_bias_code = (
                filename_match.group(1)
                if filename_match
                else ""
            )

            # -----------------------------
            # Integrity checks
            # -----------------------------
            if strict and MASK_TOKEN not in prompt:
                raise ValueError(
                    f"{path.name}, relation {relation_index}, "
                    f"prompt {prompt_index}: no {MASK_TOKEN} found."
                )

            if strict and prompt.count(MASK_TOKEN) != 1:
                raise ValueError(
                    f"{path.name}, relation {relation_index}, "
                    f"prompt {prompt_index}: expected exactly one "
                    f"{MASK_TOKEN}, found {prompt.count(MASK_TOKEN)}."
                )

            if strict and target_group.casefold() != group.casefold():
                raise ValueError(
                    f"{path.name}, relation {relation_index}, "
                    f"prompt {prompt_index}: target group "
                    f"{target_group!r} does not match relation group "
                    f"{group!r}."
                )

            if (
                strict
                and filename_bias_code
                and filename_bias_code != bias_code
            ):
                raise ValueError(
                    f"{path.name}, relation {relation_index}: "
                    f"file code {filename_bias_code!r} does not match "
                    f"output label {bias_code!r}."
                )

            rows.append(
                {
                    "relation_id": relation_id,
                    "raw_relation": raw_relation,
                    "group": group,
                    "relation": relation_phrase,
                    "biased_prompt": prompt,
                    "target": target_group,
                    "bias_code": bias_code,
                    "bias_type": bias_type,
                }
            )

    print(
        f"[Bias] {path.name}: "
        f"{len(data)} relations, {len(rows)} prompts"
    )

    return rows


def discover_prompt_files(
    data_dir: Path,
    pattern: str = DEFAULT_PATTERN,
) -> list[Path]:
    """Find all BR prompt files in deterministic order."""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}"
        )

    if not data_dir.is_dir():
        raise NotADirectoryError(
            f"Not a directory: {data_dir}"
        )

    files = sorted(data_dir.rglob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files matching {pattern!r} found in {data_dir}."
        )

    return files


def assign_relation_splits(
    rows: list[dict],
    seed: int,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.10,
) -> list[dict]:
    """
    Assign train/validation/test at RELATION level.

    Splitting is stratified by bias_code in the sense that each BR category
    is split independently. This prevents one category from accidentally
    disappearing from validation/test when enough relations are available.
    """
    if train_ratio <= 0.0:
        raise ValueError("train_ratio must be > 0.")

    if validation_ratio < 0.0:
        raise ValueError("validation_ratio must be >= 0.")

    if train_ratio + validation_ratio >= 1.0:
        raise ValueError(
            "train_ratio + validation_ratio must be < 1."
        )

    relation_to_bias: dict[str, str] = {}

    for row in rows:
        relation_id = row["relation_id"]
        bias_code = row["bias_code"]

        previous = relation_to_bias.get(relation_id)

        if previous is not None and previous != bias_code:
            raise ValueError(
                f"Relation {relation_id} appears under multiple "
                f"bias codes: {previous}, {bias_code}."
            )

        relation_to_bias[relation_id] = bias_code

    bias_to_relations: dict[str, list[str]] = {}

    for relation_id, bias_code in relation_to_bias.items():
        bias_to_relations.setdefault(
            bias_code,
            [],
        ).append(relation_id)

    relation_split: dict[str, str] = {}

    for bias_code, relation_ids in sorted(
        bias_to_relations.items()
    ):
        relation_ids = list(relation_ids)

        # Category-specific deterministic shuffle.
        rng = random.Random(
            f"{seed}:{bias_code}"
        )
        rng.shuffle(relation_ids)

        n = len(relation_ids)

        n_train = int(n * train_ratio)
        n_validation = int(n * validation_ratio)

        # When possible, guarantee that validation and test are represented.
        if n >= 3:
            n_train = max(1, n_train)
            n_validation = max(1, n_validation)

            if n_train + n_validation >= n:
                n_train = max(
                    1,
                    n - n_validation - 1,
                )

        train_ids = relation_ids[:n_train]
        validation_ids = relation_ids[
            n_train:n_train + n_validation
        ]
        test_ids = relation_ids[
            n_train + n_validation:
        ]

        for relation_id in train_ids:
            relation_split[relation_id] = "train"

        for relation_id in validation_ids:
            relation_split[relation_id] = "validation"

        for relation_id in test_ids:
            relation_split[relation_id] = "test"

        print(
            f"[Split] {bias_code}: "
            f"relations train={len(train_ids)}, "
            f"validation={len(validation_ids)}, "
            f"test={len(test_ids)}"
        )

    result: list[dict] = []

    for row in rows:
        copied = dict(row)
        copied["split"] = relation_split[
            row["relation_id"]
        ]
        result.append(copied)

    return result


def build_biased_dataset(
    data_dir: str,
    pattern: str = DEFAULT_PATTERN,
    seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.10,
    strict: bool = True,
) -> DatasetDict:
    """
    Build the complete raw biased-prompt DatasetDict.
    """
    data_path = Path(data_dir)

    files = discover_prompt_files(
        data_dir=data_path,
        pattern=pattern,
    )

    print(
        f"\n=== Discovered {len(files)} prompt files ==="
    )

    for file in files:
        print(f"  - {file.name}")

    all_rows: list[dict] = []

    for path in files:
        all_rows.extend(
            normalize_prompt_file(
                path,
                strict=strict,
            )
        )

    if not all_rows:
        raise ValueError(
            "No valid biased-prompt rows were produced."
        )

    all_rows = assign_relation_splits(
        rows=all_rows,
        seed=seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )

    split_rows = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for row in all_rows:
        split_rows[row["split"]].append(row)

    # Shuffle prompt order only AFTER relation-level split assignment.
    for split_name, rows in split_rows.items():
        rng = random.Random(
            f"{seed}:{split_name}"
        )
        rng.shuffle(rows)

    dataset = DatasetDict(
        {
            split_name: Dataset.from_pandas(
                pd.DataFrame(rows),
                preserve_index=False,
            )
            for split_name, rows in split_rows.items()
        }
    )

    return dataset


def validate_no_relation_leakage(
    dataset: DatasetDict,
) -> None:
    """Ensure no <G,R,S> relation occurs in multiple splits."""
    split_relations: dict[str, set[str]] = {}

    for split_name, split in dataset.items():
        split_relations[split_name] = set(
            split["relation_id"]
        )

    train = split_relations["train"]
    validation = split_relations["validation"]
    test = split_relations["test"]

    overlaps = {
        "train/validation": train & validation,
        "train/test": train & test,
        "validation/test": validation & test,
    }

    leaking = {
        name: values
        for name, values in overlaps.items()
        if values
    }

    if leaking:
        raise RuntimeError(
            f"Relation leakage detected: {leaking}"
        )

    print(
        "\n[Check] No relation leakage across splits."
    )



def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the raw biased-relation prompt dataset "
            "for surgical BERT fine-tuning."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/biased_relations",
        help=(
            "Directory containing BR01_prompts.json ... "
            "BR09_prompts.json."
        ),
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_PATTERN,
        help=(
            "Glob pattern used to discover prompt files. "
            f"Default: {DEFAULT_PATTERN}"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--no-strict",
        action="store_true",
        help=(
            "Disable strict checks for [MASK], group target, "
            "and BR code consistency."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./biased_dataset",
    )

    args = parser.parse_args()

    dataset = build_biased_dataset(
        data_dir=args.data_dir,
        pattern=args.pattern,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        strict=not args.no_strict,
    )

    validate_no_relation_leakage(dataset)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.save_to_disk(
        str(output_dir)
    )

    print(
        f"\nBiased dataset saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()
