
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "data" / "biased_relations" / "biased_relations_all_bags.json"
DEFAULT_OUTPUT = ROOT / "data" / "biased_relations" / "identity_masks"

WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

# Words (case-insensitive) that express the protected attribute of each relation.
IDENTITY_LEXICON = {
    "BR01": {"18", "80", "90", "year", "olds", "elderly", "old", "older", "young", "teenage"},
    "BR02": {
        "autistic", "disabled", "retarded", "cerebral", "palsy", "wheelchair", "crippled",
        "demented", "handicapped", "vision", "impairment", "mentally", "ill", "mental",
        "illness", "intellectual", "disability", "disabilities", "adhd", "autism", "down",
        "syndrome", "phobias", "schizophrenia", "psychotic", "challenged", "fat", "man",
    },
    "BR03": {
        "female", "male", "girl", "girls", "boy", "boys", "man", "men", "woman", "women",
        "trans", "transgender", "feminine", "wife", "pregnant", 
    },
    "BR04": {
        "asian", "chinese", "english", "hispanic", "illegal", "aliens", "immigrant",
        "immigrants", "indian", "inuit", "iraqi", "irish", "italian", "japanese", "jewish",
        "korean", "latino", "mexican", "middle", "eastern", "easterners", "native",
        "americans", "polish", "roma", "gypsies", "romani", "russian", "saudi", "arabians",
        "thai", "undocumented",
    },
    "BR05": {
        "attractive", "big", "blonde", "burly", "fat", "heavy", "morbidly", "obese",
        "overweight", "facial", "differences", "physical", "disabilities", "plus", "size",
        "scrawny", "short", "skinny", "slim", "tall", "thin", "unattractive",
    },
    "BR06": {
        "african", "africans", "american", "americans", "asian", "black", "chinese", "dark",
        "skinned", "hispanic", "indian", "indians", "italian", "jewish", "korean", "latino",
        "mexican", "middle", "eastern", "minority", "mixed", "race", "native", "puerto",
        "ricans", "roma",
    },
    "BR07": {
        "amish", "arabic", "christian", "hindu", "interfaith", "jewish", "mosque", "muslim",
        "muslims", "headscarves", "quran",
    },
    "BR08": {
        "bisexual", "butch", "lesbian", "lesbians", "gay", "homosexual", "transgender",
        "same", "sex",
    },
    "BR09": {
        "ghetto", "poor", "blue", "collar", "foster", "garbage", "collectors", "homeless",
        "low", "income", "wage", "manual", "laborers", "poverty", "public", "housing",
        "trailer", "park", "uneducated", "working", "class",
    },
}

TARGET_OVERRIDES = {
    ("BR01", "college students"): {"college", "students"},
    ("BR01", "minors under 18"): {"minors", "18"},
    ("BR05", "Russian women"): {"russian", "women"},
    ("BR06", "people of color"): {"people", "of", "color"},
    ("BR08", "Male hairdressers"): {"male", "hairdressers"},
}


NEEDS_REVIEW = {
    ("BR01", "college students"): "age is only implied by the role",
    ("BR02", "Fat person"): "body type labelled as disability",
    ("BR03", "pregnant women"): "'pregnant' is included with women",
    ("BR04", "Yakuza (Japanese)"): "'Yakuza' is excluded, only 'Japanese' is kept",
    ("BR05", "Russian women"): "no physical-appearance word in the target",
    ("BR08", "Male hairdressers"): "orientation is only implied by the stereotype",
}


def load_examples(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    counts = Counter()
    for relation, bags in data.items():
        for bag in bags:
            for example in bag:
                counts[(relation, example[1])] += 1
    return counts


def overlaps(start, end, spans):
    return any(start < span_end and span_start < end for span_start, span_end in spans)


def build_entry(relation, target, tokenizer):
    """Return the mask entry, or None for single-token targets."""
    encoded = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    if tokens != tokenizer.tokenize(target):
        raise ValueError(f"Tokenization mismatch for {relation} / {target!r}")
    if len(tokens) < 2:
        return None

    words = [(m.group(), m.span()) for m in WORD_RE.finditer(target)]
    override = TARGET_OVERRIDES.get((relation, target))

    if override is not None:
        identity = [(w, s) for w, s in words if w.casefold() in override]
        rule = "override"
    elif len(words) == 1:
        identity = words  # one word split into subwords, e.g. G ##hetto ##s
        rule = "single_word"
    else:
        identity = [(w, s) for w, s in words if w.casefold() in IDENTITY_LEXICON[relation]]
        rule = "lexicon"

    spans = [span for _, span in identity]
    mask = [
        int(overlaps(start, end, spans) and any(ch.isalnum() for ch in target[start:end]))
        for start, end in encoded["offset_mapping"]
    ]
    return {
        "tokens": tokens,
        "mask": mask,
        "identity_words": [word for word, _ in identity],
        "rule": rule,
        "review_note": NEEDS_REVIEW.get((relation, target)),
    }


def build_masks(counts, tokenizer):
    masks = defaultdict(dict)
    uncovered = []
    single_token = Counter()

    for (relation, target), _ in sorted(counts.items()):
        entry = build_entry(relation, target, tokenizer)
        if entry is None:
            single_token[relation] += 1
        elif sum(entry["mask"]) == 0:
            uncovered.append(f"{relation}: {target!r} -> {entry['tokens']}")
        else:
            masks[relation][target] = entry

    if uncovered:
        raise ValueError("Targets without identity tokens:\n" + "\n".join(uncovered))

    unknown_reviews = [key for key in NEEDS_REVIEW if key[1] not in masks.get(key[0], {})]
    unknown_overrides = [key for key in TARGET_OVERRIDES if key[1] not in masks.get(key[0], {})]
    if unknown_reviews or unknown_overrides:
        raise ValueError(f"Stale annotations: {unknown_reviews + unknown_overrides}")

    return dict(masks), single_token


def summarize(masks, counts, single_token):
    per_relation = {}
    removed = Counter()
    for relation, entries in sorted(masks.items()):
        total = kept = examples = 0
        for target, entry in entries.items():
            n = counts[(relation, target)]
            examples += n
            total += n * len(entry["mask"])
            kept += n * sum(entry["mask"])
            for token, value in zip(entry["tokens"], entry["mask"]):
                if not value:
                    removed[token] += n
        per_relation[relation] = {
            "multi_token_targets": len(entries),
            "single_token_targets_skipped": single_token[relation],
            "examples": examples,
            "positions": total,
            "identity_positions": kept,
            "identity_share": round(kept / total, 4),
        }

    total = sum(r["positions"] for r in per_relation.values())
    kept = sum(r["identity_positions"] for r in per_relation.values())
    return {
        "multi_token_targets": sum(r["multi_token_targets"] for r in per_relation.values()),
        "single_token_targets_skipped": sum(single_token.values()),
        "examples": sum(r["examples"] for r in per_relation.values()),
        "positions": total,
        "identity_positions": kept,
        "identity_share": round(kept / total, 4),
        "top_removed_tokens": removed.most_common(15),
        "per_relation": per_relation,
    }


def write_review_csv(path, masks, counts):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["relation", "target", "examples", "tokens", "mask", "kept", "rule", "review_note"])
        for relation, entries in sorted(masks.items()):
            for target, entry in sorted(entries.items()):
                kept = [t for t, m in zip(entry["tokens"], entry["mask"]) if m]
                writer.writerow([
                    relation, target, counts[(relation, target)], " ".join(entry["tokens"]),
                    " ".join(map(str, entry["mask"])), " ".join(kept), entry["rule"],
                    entry["review_note"] or "",
                ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-cased")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    counts = load_examples(args.data_path)
    masks, single_token = build_masks(counts, tokenizer)
    summary = summarize(masks, counts, single_token)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = args.model.split("/")[-1]
    json_path = args.output_dir / f"{name}_identity_mask.json"
    csv_path = args.output_dir / f"{name}_identity_mask_review.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "summary": summary, "masks": masks}, f, indent=2, ensure_ascii=False)
    write_review_csv(csv_path, masks, counts)

    print(json.dumps({k: v for k, v in summary.items() if k != "per_relation"}, indent=2, ensure_ascii=False))
    for relation, stats in summary["per_relation"].items():
        print(relation, stats)
    print(f"Saved: {json_path}\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
