from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


def clean_text(
    text: str,
    min_chars: int = 30,
    max_chars: int = 500,
) -> str | None:
    """
    Normalize and lightly filter neutral text.

    This function intentionally performs only raw-data cleaning.
    Tokenization and MLM masking are deferred to the preprocessing stage.
    """
    if text is None:
        return None

    text = " ".join(str(text).split()).strip()

    if len(text) < min_chars:
        return None

    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]

    return text


def load_wikitext_neutral(
    target_size: int,
    seed: int,
) -> list[dict]:
    """
    Load diverse neutral/general text from WikiText-2.
    """
    print("[Neutral] Loading WikiText-2...")

    wiki = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="train",
    )

    rows: list[dict] = []

    for text in wiki["text"]:
        cleaned = clean_text(text)

        if cleaned is None:
            continue

        rows.append(
            {
                "text": cleaned,
                "domain": "neutral",
                "source": "wikitext",
            }
        )

    rng = random.Random(seed)
    rng.shuffle(rows)

    rows = rows[:target_size]

    print(f"[Neutral] WikiText usable examples: {len(rows)}")

    return rows


def load_country_city_animals_neutral(
    target_size: int,
    seed: int,
) -> list[dict]:
    """
    Load the optional Country-city-animals corpus.

    This source is supplementary rather than the main preservation corpus.
    """
    print("[Neutral] Loading Country-city-animals...")

    try:
        data = load_dataset(
            "xiaozeroone/Country-city-animals",
            "Corpus_narrative",
            split="train",
        )

    except Exception as exc:
        print(
            "[Neutral] Country-city-animals unavailable. "
            f"Continuing without it: {exc}"
        )
        return []

    rows: list[dict] = []

    for row in data:
        text = row.get("text") or row.get("sentence")
        cleaned = clean_text(text)

        if cleaned is None:
            continue

        rows.append(
            {
                "text": cleaned,
                "domain": "neutral",
                "source": "country-city-animals",
            }
        )

    rng = random.Random(seed)
    rng.shuffle(rows)

    rows = rows[:target_size]

    print(
        "[Neutral] Country-city-animals usable examples: "
        f"{len(rows)}"
    )

    return rows


def build_neutral_pool(
    total_examples: int,
    seed: int,
    use_country_city_animals: bool = True,
    country_city_animals_ratio: float = 0.20,
) -> list[dict]:
    """
    Build the complete neutral preservation pool.

    Default source composition:
        80% WikiText-2
        20% Country-city-animals

    If the optional source cannot fill its quota, WikiText is used to fill
    the missing examples.
    """
    if total_examples <= 0:
        raise ValueError("total_examples must be > 0.")

    if not 0.0 <= country_city_animals_ratio <= 1.0:
        raise ValueError(
            "country_city_animals_ratio must be between 0 and 1."
        )

    if use_country_city_animals:
        cta_target = int(
            round(total_examples * country_city_animals_ratio)
        )
    else:
        cta_target = 0

    wiki_target = total_examples - cta_target

    wiki_rows = load_wikitext_neutral(
        target_size=wiki_target,
        seed=seed,
    )

    cta_rows: list[dict] = []

    if cta_target > 0:
        cta_rows = load_country_city_animals_neutral(
            target_size=cta_target,
            seed=seed + 1,
        )

    neutral_pool = wiki_rows + cta_rows

    # Remove exact duplicate texts across sources.
    deduplicated: list[dict] = []
    seen_texts: set[str] = set()

    for row in neutral_pool:
        if row["text"] in seen_texts:
            continue

        seen_texts.add(row["text"])
        deduplicated.append(row)

    neutral_pool = deduplicated

    # If the optional source did not fill its quota, fill the missing
    # examples from WikiText.
    if len(neutral_pool) < total_examples:
        missing = total_examples - len(neutral_pool)

        print(
            f"[Neutral] Filling {missing} missing examples "
            "from WikiText..."
        )

        # Ask for more than strictly necessary because some examples may
        # already exist in the pool and will be discarded as duplicates.
        extra_wiki = load_wikitext_neutral(
            target_size=max(missing * 3, missing),
            seed=seed + 2,
        )

        for row in extra_wiki:
            if row["text"] in seen_texts:
                continue

            neutral_pool.append(row)
            seen_texts.add(row["text"])

            if len(neutral_pool) >= total_examples:
                break

    if len(neutral_pool) < total_examples:
        print(
            "[Warning] Requested "
            f"{total_examples} neutral examples but only "
            f"{len(neutral_pool)} unique usable examples were obtained."
        )

    rng = random.Random(seed)
    rng.shuffle(neutral_pool)

    return neutral_pool[:total_examples]


def split_neutral_pool(
    neutral_pool: list[dict],
    seed: int,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.10,
) -> dict[str, list[dict]]:
    """
    Split the neutral preservation pool into train/validation/test.

    The remaining fraction after train and validation is assigned to test.
    """
    if train_ratio <= 0.0:
        raise ValueError("train_ratio must be > 0.")

    if validation_ratio < 0.0:
        raise ValueError("validation_ratio must be >= 0.")

    if train_ratio + validation_ratio >= 1.0:
        raise ValueError(
            "train_ratio + validation_ratio must be < 1."
        )

    rows = list(neutral_pool)

    rng = random.Random(seed)
    rng.shuffle(rows)

    n_total = len(rows)
    n_train = int(n_total * train_ratio)
    n_validation = int(n_total * validation_ratio)

    train_rows = rows[:n_train]
    validation_rows = rows[
        n_train:n_train + n_validation
    ]
    test_rows = rows[n_train + n_validation:]

    for row in train_rows:
        row["split"] = "train"

    for row in validation_rows:
        row["split"] = "validation"

    for row in test_rows:
        row["split"] = "test"

    return {
        "train": train_rows,
        "validation": validation_rows,
        "test": test_rows,
    }


def build_neutral_preservation_dataset(
    neutral_examples: int = 6000,
    seed: int = 42,
    use_country_city_animals: bool = True,
    country_city_animals_ratio: float = 0.20,
) -> DatasetDict:
    """
    Build the raw neutral preservation DatasetDict.

    Schema:
        text   : raw neutral sentence/passage
        domain : always "neutral"
        source : source dataset
        split  : train / validation / test

    Deliberately absent:
        input_ids
        attention_mask
        labels
        mask positions
        apply_mask

    Those fields will be produced later during BERT preprocessing.
    """
    print("\n=== Neutral preservation dataset ===")

    neutral_pool = build_neutral_pool(
        total_examples=neutral_examples,
        seed=seed,
        use_country_city_animals=use_country_city_animals,
        country_city_animals_ratio=country_city_animals_ratio,
    )

    split_rows = split_neutral_pool(
        neutral_pool=neutral_pool,
        seed=seed,
    )

    dataset = DatasetDict(
        {
            split_name: Dataset.from_pandas(
                pd.DataFrame(rows),
                preserve_index=False,
            )
            for split_name, rows in split_rows.items()
        }
    )

    print(
        "\nFinal neutral sizes:"
        f"\n  train      = {len(dataset['train'])}"
        f"\n  validation = {len(dataset['validation'])}"
        f"\n  test       = {len(dataset['test'])}"
    )

    return dataset


def print_statistics(dataset: DatasetDict) -> None:
    """
    Print basic integrity/statistical checks for the neutral dataset.
    """
    print("\n=== Neutral dataset statistics ===")

    total = 0

    for split_name, split in dataset.items():
        df = split.to_pandas()
        total += len(df)

        print(f"\n[{split_name}]")
        print(f"Examples: {len(df)}")

        if not df.empty:
            print("\nSources:")
            print(df["source"].value_counts(dropna=False))

            duplicated = int(df["text"].duplicated().sum())
            print(f"\nDuplicate texts inside split: {duplicated}")

    print(f"\nTotal examples: {total}")

    print("\nExample rows:")

    for split_name in dataset.keys():
        if len(dataset[split_name]) == 0:
            continue

        row = dataset[split_name][0]

        print(
            f"[{split_name} | {row['source']}] "
            f"{row['text'][:140]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the raw neutral preservation dataset for "
            "surgical BERT fine-tuning."
        )
    )

    parser.add_argument(
        "--neutral-examples",
        type=int,
        default=6000,
        help="Total number of neutral examples before splitting.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--no-country-city-animals",
        action="store_true",
        help=(
            "Disable the Country-city-animals supplementary source "
            "and use WikiText only."
        ),
    )

    parser.add_argument(
        "--country-city-animals-ratio",
        type=float,
        default=0.20,
        help=(
            "Fraction of the neutral pool requested from "
            "Country-city-animals. Default: 0.20."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./neutral_preservation_dataset",
    )

    args = parser.parse_args()

    dataset = build_neutral_preservation_dataset(
        neutral_examples=args.neutral_examples,
        seed=args.seed,
        use_country_city_animals=(
            not args.no_country_city_animals
        ),
        country_city_animals_ratio=(
            args.country_city_animals_ratio
        ),
    )

    print_statistics(dataset)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(str(output_dir))

    print(
        f"\nNeutral preservation dataset saved to: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
