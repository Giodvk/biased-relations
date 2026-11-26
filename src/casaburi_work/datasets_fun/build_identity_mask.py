"""Build a binary identity mask for multi-token targets, grounded in HolisticBias.

For every multi-token (relation, target) pair, each token gets:
    1 = the token expresses the social group targeted by the relation
    0 = generic person noun, other word or punctuation (e.g. "people", "-")

Identity words are chosen in this order:
    1. single_word:   the target is one word split into subwords, all subwords are kept.
    2. holistic_bias: HolisticBias v1.1 descriptors from the axes mapped to the relation
                      (plus HolisticBias gendered nouns for the gender relation).
    3. manual:        explicit annotation, only for targets HolisticBias does not cover.

The relation-to-axis mapping and the manual annotations live in identity_config.json.

Single-token targets are excluded. Words are projected onto tokens through character
offsets, so the mask works with any fast tokenizer.

Usage (from the repository root):
    python src/casaburi_work/datasets_fun/build_identity_mask.py --model bert-base-cased
"""

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
DEFAULT_HOLISTIC_BIAS = DEFAULT_OUTPUT / "holistic_bias" / "v1.1"
DEFAULT_CONFIG = DEFAULT_OUTPUT / "identity_config.json"

WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
UNIT_RE = re.compile(r"\S+")

def load_config(path):
    """Load the annotation config (see identity_config.json).

    relation_axes:           HolisticBias axes per relation, "axis" or "axis/bucket".
    gendered_noun_relations: relations where HolisticBias gendered nouns ("women") are identity.
    manual_identity:         identity words for multi-word targets HolisticBias does not cover.
                             HolisticBias person nouns ("people", "man") are not allowed,
                             except in the gendered-noun relations.
    needs_review:            free-text notes copied into the review CSV.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    def flatten(section, convert):
        return {
            (relation, target): convert(value)
            for relation, targets in raw.get(section, {}).items()
            for target, value in targets.items()
        }

    config = {
        "relation_axes": {
            relation: {tuple(axis.split("/", 1)) if "/" in axis else (axis, None) for axis in axes}
            for relation, axes in raw["relation_axes"].items()
        },
        "gendered_noun_relations": set(raw.get("gendered_noun_relations", [])),
        "manual_identity": flatten("manual_identity", lambda words: {w.casefold() for w in words}),
        "needs_review": flatten("needs_review", str),
    }

    errors = [
        f"{relation} / {target!r}: unknown relation"
        for relation, target in list(config["manual_identity"]) + list(config["needs_review"])
        if relation not in config["relation_axes"]
    ]
    errors += [
        f"{relation} / {target!r}: words {sorted(words - {w.casefold() for w, _ in words_of(target)})} "
        "are not in the target"
        for (relation, target), words in config["manual_identity"].items()
        if words - {w.casefold() for w, _ in words_of(target)}
    ]
    if errors:
        raise ValueError(f"Invalid config {path}:\n" + "\n".join(errors))
    return config


def words_of(text):
    return [(m.group(), m.span()) for m in WORD_RE.finditer(text)]


def units_of(text):
    """Whitespace-separated units without surrounding punctuation: "'Ghetto'" -> "Ghetto".

    Hyphenated compounds stay whole, so "18-year-olds" never matches the descriptor "old".
    """
    units = []
    for match in UNIT_RE.finditer(text):
        word = match.group()
        start = len(word) - len(word.lstrip("'\"()[],.;:!?"))
        end = len(word.rstrip("'\"()[],.;:!?"))
        if start < end:
            units.append((word[start:end], (match.start() + start, match.start() + end)))
    return units


def descriptor_variants(descriptor):
    """"African-American" also matches "African American"."""
    units = tuple(u.casefold() for u, _ in units_of(descriptor))
    spaced = tuple(part for u in units for part in u.split("-") if part)
    return {units, spaced}


def load_holistic_bias(path):
    """Return HolisticBias descriptors as (unit variants, axis, bucket, descriptor) and nouns."""
    path = Path(path)
    with open(path / "descriptors.json", "r", encoding="utf-8") as f:
        raw_descriptors = json.load(f)
    with open(path / "nouns.json", "r", encoding="utf-8") as f:
        raw_nouns = json.load(f)

    descriptors = []

    def walk(node, axis, bucket):
        if isinstance(node, dict) and "descriptor" in node:
            node = node["descriptor"]
        if isinstance(node, str):
            descriptors.append((descriptor_variants(node), axis, bucket, node))
        elif isinstance(node, dict):
            for key, value in node.items():
                walk(value, axis, key if bucket is None else bucket)
        elif isinstance(node, list):
            for value in node:
                walk(value, axis, bucket)

    for axis, content in raw_descriptors.items():
        walk(content, axis, None)

    nouns = {
        gender: {noun.casefold() for pair in pairs for noun in pair}
        for gender, pairs in raw_nouns.items()
    }
    return descriptors, nouns


def axis_allowed(relation_axes, axis, bucket):
    return any(
        axis == allowed_axis and (allowed_bucket is None or bucket == allowed_bucket)
        for allowed_axis, allowed_bucket in relation_axes
    )


def find_descriptor(target_words, descriptor_words):
    """Yield word index ranges matching a descriptor; the last word may be plural."""
    n = len(descriptor_words)
    *head, last = descriptor_words
    for start in range(len(target_words) - n + 1):
        segment = target_words[start : start + n]
        if segment[:-1] == head and segment[-1] in {last, last + "s", last + "es"}:
            yield range(start, start + n)


def holistic_bias_identity(relation, target, holistic_bias, config):
    descriptors, nouns = holistic_bias
    units = units_of(target)
    lowered = [u.casefold() for u, _ in units]
    selected, matched = set(), set()

    for variants, axis, bucket, descriptor in descriptors:
        if not axis_allowed(config["relation_axes"][relation], axis, bucket):
            continue
        for descriptor_units in variants:
            for unit_range in find_descriptor(lowered, descriptor_units):
                selected.update(unit_range)
                matched.add(f"{descriptor} ({axis}/{bucket})")

    if relation in config["gendered_noun_relations"]:
        gendered = nouns["female"] | nouns["male"]
        for idx, unit in enumerate(lowered):
            if unit in gendered:
                selected.add(idx)
                matched.add(f"{unit} (nouns/gendered)")

    return [units[idx] for idx in sorted(selected)], sorted(matched)


def overlaps(start, end, spans):
    return any(start < span_end and span_start < end for span_start, span_end in spans)


def build_entry(relation, target, tokenizer, holistic_bias, config):
    """Return the mask entry, or None for single-token targets."""
    encoded = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    if tokens != tokenizer.tokenize(target):
        raise ValueError(f"Tokenization mismatch for {relation} / {target!r}")
    if len(tokens) < 2:
        return None

    words = words_of(target)
    hb_identity, hb_matches = holistic_bias_identity(relation, target, holistic_bias, config)
    manual = config["manual_identity"].get((relation, target))

    if len(words) == 1:
        identity, source = words, "single_word"
    elif hb_identity:
        if manual is not None:
            raise ValueError(f"Manual annotation shadowed by HolisticBias: {relation} / {target!r}")
        identity, source = hb_identity, "holistic_bias"
    elif manual is not None:
        identity = [(w, s) for w, s in words if w.casefold() in manual]
        source = "manual"
    else:
        identity, source = [], "uncovered"

    spans = [span for _, span in identity]
    mask = [
        int(overlaps(start, end, spans) and any(ch.isalnum() for ch in target[start:end]))
        for start, end in encoded["offset_mapping"]
    ]
    return {
        "tokens": tokens,
        "mask": mask,
        "identity_words": [word for word, _ in identity],
        "source": source,
        "holistic_bias_matches": hb_matches,
        "review_note": config["needs_review"].get((relation, target)),
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


def build_masks(counts, tokenizer, holistic_bias, config):
    manual_identity = config["manual_identity"]
    gendered_relations = config["gendered_noun_relations"]
    masks = defaultdict(dict)
    uncovered = []
    single_token = Counter()

    for (relation, target), _ in sorted(counts.items()):
        entry = build_entry(relation, target, tokenizer, holistic_bias, config)
        if entry is None:
            single_token[relation] += 1
        elif sum(entry["mask"]) == 0:
            uncovered.append(f"{relation}: {target!r} -> {entry['tokens']}")
        else:
            masks[relation][target] = entry

    if uncovered:
        raise ValueError(
            "Targets without identity tokens (add them to manual_identity in the config):\n" + "\n".join(uncovered)
        )

    stale = [
        key for key in list(manual_identity) + list(config["needs_review"])
        if key[1] not in masks.get(key[0], {})
    ]
    stale += [
        key for key in manual_identity
        if key[1] in masks.get(key[0], {}) and masks[key[0]][key[1]]["source"] != "manual"
    ]
    if stale:
        raise ValueError(f"Stale annotations: {stale}")

    person_nouns = set().union(*holistic_bias[1].values())
    noun_violations = [
        (relation, target, sorted(words & person_nouns))
        for (relation, target), words in manual_identity.items()
        if relation not in gendered_relations and words & person_nouns
    ]
    if noun_violations:
        raise ValueError(f"Manual annotations keep HolisticBias person nouns: {noun_violations}")

    return dict(masks), single_token


def summarize(masks, counts, single_token, config):
    per_relation = {}
    removed = Counter()
    sources = Counter()
    for relation, entries in sorted(masks.items()):
        total = kept = examples = 0
        relation_sources = Counter()
        for target, entry in entries.items():
            n = counts[(relation, target)]
            examples += n
            total += n * len(entry["mask"])
            kept += n * sum(entry["mask"])
            relation_sources[entry["source"]] += 1
            for token, value in zip(entry["tokens"], entry["mask"]):
                if not value:
                    removed[token] += n
        sources.update(relation_sources)
        per_relation[relation] = {
            "multi_token_targets": len(entries),
            "single_token_targets_skipped": single_token[relation],
            "targets_by_source": dict(relation_sources),
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
        "targets_by_source": dict(sources),
        "examples": sum(r["examples"] for r in per_relation.values()),
        "positions": total,
        "identity_positions": kept,
        "identity_share": round(kept / total, 4),
        "top_removed_tokens": removed.most_common(15),
        "relation_axes": {r: sorted(f"{a}/{b or '*'}" for a, b in v) for r, v in config["relation_axes"].items()},
        "per_relation": per_relation,
    }


def load_previous(path):
    if path is None or not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        masks = json.load(f).get("masks", {})
    return {
        (relation, target): entry.get("identity_words", [])
        for relation, entries in masks.items()
        for target, entry in entries.items()
    }


def write_review_csv(path, masks, counts, previous):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "relation", "target", "examples", "tokens", "mask", "kept", "source",
            "holistic_bias_matches", "previous_identity_words", "changed", "review_note",
        ])
        for relation, entries in sorted(masks.items()):
            for target, entry in sorted(entries.items()):
                kept = [t for t, m in zip(entry["tokens"], entry["mask"]) if m]
                before = previous.get((relation, target))
                changed = "" if before is None else int(
                    sorted(w.casefold() for w in before)
                    != sorted(w.casefold() for w in entry["identity_words"])
                )
                writer.writerow([
                    relation, target, counts[(relation, target)], " ".join(entry["tokens"]),
                    " ".join(map(str, entry["mask"])), " ".join(kept), entry["source"],
                    "; ".join(entry["holistic_bias_matches"]),
                    "" if before is None else " ".join(before), changed,
                    entry["review_note"] or "",
                ])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="bert-base-cased")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--holistic-bias-dir", type=Path, default=DEFAULT_HOLISTIC_BIAS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    config = load_config(args.config)
    holistic_bias = load_holistic_bias(args.holistic_bias_dir)
    counts = load_examples(args.data_path)
    masks, single_token = build_masks(counts, tokenizer, holistic_bias, config)
    summary = summarize(masks, counts, single_token, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = args.model.split("/")[-1]
    json_path = args.output_dir / f"{name}_identity_mask.json"
    csv_path = args.output_dir / f"{name}_identity_mask_review.csv"
    previous = load_previous(json_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "summary": summary, "masks": masks}, f, indent=2, ensure_ascii=False)
    write_review_csv(csv_path, masks, counts, previous)

    print(json.dumps({k: v for k, v in summary.items() if k != "per_relation"}, indent=2, ensure_ascii=False))
    for relation, stats in summary["per_relation"].items():
        print(relation, stats)
    print(f"Saved: {json_path}\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
