"""Compute neuron attribution scores with Integrated Gradients and an identity mask.

Multi-token targets use only the [MASK] positions whose identity mask is 1.
Single-token targets use their only position. The per-layer score is the mean IG
over the selected positions, and neurons are then filtered at 10% of the maximum.

Usage:
    python src/casaburi_work/utils_functions/my_kn_analysis3.py
"""

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import jsonlines
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.custom_bert import BertForMaskedLM  

DATA_PATH = ROOT / "data" / "biased_relations" / "biased_relations_all_bags.json"
MASK_DIR = ROOT / "data" / "biased_relations" / "identity_masks"
RESULTS_DIR = ROOT / "results" / "identity_mask"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_mask_path(model_name):
    """Cased and uncased BERT models tokenize differently, so each family has its mask."""
    family = "bert-base-uncased" if "uncased" in model_name.lower() else "bert-base-cased"
    return MASK_DIR / f"{family}_identity_mask.json"


def load_identity_masks(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Identity mask not found: {path}. Generate it with build_identity_mask.py."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["masks"]


def select_identity_indices(relation, target, target_tokens, identity_masks):
    """Return the indices of the target tokens used for attribution."""
    if len(target_tokens) == 1:
        return [0]

    entry = identity_masks.get(relation, {}).get(target)
    if entry is None:
        raise KeyError(f"Missing identity mask for {relation} / {target!r}")
    if entry["tokens"] != target_tokens or len(entry["mask"]) != len(target_tokens):
        raise ValueError(
            f"Mask does not match tokenizer for {relation} / {target!r}: "
            f"{entry['tokens']} != {target_tokens}"
        )

    indices = [idx for idx, value in enumerate(entry["mask"]) if value == 1]
    if not indices:
        raise ValueError(f"Empty identity mask for {relation} / {target!r}")
    return indices


def validate_identity_masks(bags_per_relation, tokenizer, identity_masks):
    """Fail before the long IG run if any target cannot be resolved."""
    targets = {
        (relation, example[1])
        for relation, bags in bags_per_relation.items()
        for bag in bags
        for example in bag
    }
    for relation, target in targets:
        select_identity_indices(relation, target, tokenizer.tokenize(target), identity_masks)
    return len(targets)


def example2feature(example, max_seq_length, tokenizer):
    """Convert an example into input features, with one [MASK] per target token."""
    prompt, target, relation = example[0], example[1], example[2]

    target_tokens = tokenizer.tokenize(target)
    if len(target_tokens) == 0:
        raise ValueError(f"Target without tokens: {target}")

    multi_mask = " ".join([tokenizer.mask_token] * len(target_tokens))
    prompt = prompt.replace(tokenizer.mask_token, multi_mask, 1)

    ori_tokens = tokenizer.tokenize(prompt)
    if len(ori_tokens) > max_seq_length - 2:
        ori_tokens = ori_tokens[: max_seq_length - 2]

    tokens = ["[CLS]"] + ori_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    features = {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
    }
    tokens_info = {
        "tokens": tokens,
        "relation": relation,
        "gold_obj": target,
        "gold_tokens": target_tokens,
        "pred_obj": None,
    }
    return features, tokens_info


def scaled_input(emb, batch_size, num_batch):
    """Build the IG path from a zero baseline to the original activation."""
    baseline = torch.zeros_like(emb)
    num_points = batch_size * num_batch
    step = (emb - baseline) / num_points
    res = torch.cat([torch.add(baseline, step * i) for i in range(num_points)], dim=0)
    return res, step[0]


def convert_to_triplet_ig(ig_list):
    """Keep neurons whose attribution is at least 10% of the maximum."""
    ig_triplet = []
    ig = np.array(ig_list)
    max_ig = ig.max()
    for i in range(ig.shape[0]):
        for j in range(ig.shape[1]):
            if ig[i][j] >= max_ig * 0.1:
                ig_triplet.append([i, j, float(ig[i][j])])
    return ig_triplet


def integrated_gradients(model, inputs, tgt_pos, tgt_layer, gold_label, batch_size, num_batch):
    """Compute IG of P(gold_label) for all neurons at one position and layer."""
    with torch.no_grad():
        ffn_weights, _ = model(**inputs, tgt_pos=tgt_pos, tgt_layer=tgt_layer)

    scaled_weights, weights_step = scaled_input(ffn_weights.detach(), batch_size, num_batch)
    scaled_weights.requires_grad_(True)

    total_grad = None
    for batch_idx in range(num_batch):
        start = batch_idx * batch_size
        batch_weights = scaled_weights[start : start + batch_size]
        _, grad = model(
            **inputs,
            tgt_pos=tgt_pos,
            tgt_layer=tgt_layer,
            tmp_score=batch_weights,
            tgt_label=gold_label,
        )
        grad = grad.sum(dim=0)
        total_grad = grad if total_grad is None else total_grad + grad

    return (total_grad * weights_step).detach()


def attribute_example(model, tokenizer, example, relation, identity_masks, args, device):
    features, tokens_info = example2feature(example, args["max_seq_length"], tokenizer)

    tgt_positions = [
        idx for idx, token in enumerate(tokens_info["tokens"]) if token.strip() == tokenizer.mask_token
    ]
    gold_labels = tokenizer.convert_tokens_to_ids(tokens_info["gold_tokens"])
    if len(tgt_positions) != len(gold_labels):
        raise ValueError(
            f"Mismatch: {len(tgt_positions)} masks but {len(gold_labels)} gold tokens "
            f"for target {tokens_info['gold_obj']!r}"
        )

    identity_indices = select_identity_indices(
        relation, tokens_info["gold_obj"], tokens_info["gold_tokens"], identity_masks
    )
    tokens_info["identity_mask"] = [
        int(idx in identity_indices) for idx in range(len(tokens_info["gold_tokens"]))
    ]
    tokens_info["identity_tokens"] = [tokens_info["gold_tokens"][idx] for idx in identity_indices]

    inputs = {
        "input_ids": torch.tensor([features["input_ids"]], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([features["input_mask"]], dtype=torch.long, device=device),
        "token_type_ids": torch.tensor([features["segment_ids"]], dtype=torch.long, device=device),
    }

    ig_layers = []
    for tgt_layer in range(model.bert.config.num_hidden_layers):
        token_ig = [
            integrated_gradients(
                model,
                inputs,
                tgt_positions[idx],
                tgt_layer,
                gold_labels[idx],
                args["batch_size"],
                args["num_batch"],
            )
            for idx in identity_indices
        ]
        ig_layers.append(torch.stack(token_ig, dim=0).mean(dim=0).cpu().tolist())

    res_dict = {
        "pred": [],
        "ig_pred": [],
        "ig_gold": convert_to_triplet_ig(ig_layers),
        "base": [],
    }
    return [tokens_info, res_dict]


def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def run(args):
    if args["no_cuda"] or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args['gpus']}")
    print(f"device: {device}")

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args["seed"])
        torch.cuda.empty_cache()

    tokenizer = load_tokenizer(args["bert_model"])
    identity_masks = load_identity_masks(args["mask_path"])
    with open(args["data_path"], "r", encoding="utf-8") as f:
        bags_per_relation = json.load(f)
    n_targets = validate_identity_masks(bags_per_relation, tokenizer, identity_masks)
    print(f"Identity masks validated for {n_targets} (relation, target) pairs")

    os.makedirs(args["output_dir"], exist_ok=True)
    args_path = os.path.join(args["output_dir"], f"{args['output_prefix']}.args.json")
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(args, f, sort_keys=True, indent=2)

    model = BertForMaskedLM.from_pretrained(args["bert_model"])
    model.to(device)
    model.eval()

    timing_log = []
    for relation, bags in bags_per_relation.items():
        if args.get("pt_relation") is not None and relation != args["pt_relation"]:
            continue

        tic = time.perf_counter()
        limit = args["debug"] if args["debug"] > 0 else len(bags)
        output_path = os.path.join(
            args["output_dir"], f"{args['output_prefix']}-{relation}.rlt.jsonl"
        )
        with jsonlines.open(output_path, "w") as fw:
            for bag in tqdm(bags[:limit], desc=f"Elaborazione [{relation}]"):
                fw.write([
                    attribute_example(model, tokenizer, example, relation, identity_masks, args, device)
                    for example in bag
                ])

        elapsed = time.perf_counter() - tic
        print(f"\n***** Relation: {relation} evaluated. Costing time: {elapsed:0.4f} seconds *****\n")
        timing_log.append({"relation": relation, "costing_time_seconds": round(elapsed, 4)})

    return timing_log


if __name__ == "__main__":
    model_list = [
        # "bert-base-cased",
        "bert-large-cased",
        # "bert-base-uncased",
        # "bert-large-uncased",
        # "aieng-lab/bert-large-cased_requirement-completion",
        # "aieng-lab/bert-large-cased_incivility",
        # "aieng-lab/bert-large-cased_tone-bearing",
        # "aieng-lab/bert-large-cased_sentiment",
        # "aieng-lab/bert-large-cased_requirement-type",
    ]

    full_timing_log = []
    for model_name in model_list:
        print("\n=========================================================")
        print(f" Avvio Elaborazione Modello: {model_name}")
        print("=========================================================\n")

        run_args = {
            "data_path": str(DATA_PATH),
            "mask_path": str(get_mask_path(model_name)),
            "bert_model": model_name,
            "output_dir": str(RESULTS_DIR / model_name.split("/")[-1]),
            "output_prefix": "all",
            "max_seq_length": 128,
            "no_cuda": False,
            "gpus": "0",
            "seed": 42,
            "debug": 100000,
            "batch_size": 20,
            "num_batch": 1,
        }
        full_timing_log.append({"model": model_name, "timing": run(run_args)})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "run_timing_log.json", "w", encoding="utf-8") as f:
        json.dump(full_timing_log, f, indent=2)
