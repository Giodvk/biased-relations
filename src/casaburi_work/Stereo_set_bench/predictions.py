from pathlib import Path
from collections import defaultdict
import gc
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from dataloader import IntrasentenceLoader
from utils_functions.custom_bert import BertForMaskedLM
from compute_kn_neurons_dict import create_mask_foundation
from fine_tuning.union_based_training import flatten_kn_map


STEREOSET_DEV = Path("src/casaburi_work/Stereo_set_bench/dev.json")
OUTPUT_DIR = Path("results/stereoset_predictions")


RELATION_MODEL_PATH = Path(
    "results/bert-base-cased/relation_based_fine_tuned_model.pt"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(
    model_name,
    checkpoint_path=None,
    device="cuda"
):

    model = BertForMaskedLM.from_pretrained(
        model_name
    )

    if checkpoint_path is not None:

        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu"
        )

        model.load_state_dict(
            state_dict
        )

    model = model.to(device)

    model.eval()

    return model

@torch.no_grad()
def forward_stereoset(
    model,
    input_ids,
    attention_mask,
    token_type_ids=None,
    suppression_kn=None
):

    if suppression_kn is None:

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

    else:

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            imp_pos=suppression_kn,
            imp_op="remove"
        )


    if hasattr(outputs, "logits"):
        logits = outputs.logits

    elif isinstance(outputs, tuple):
        logits = outputs[-1]

    else:
        logits = outputs

    return logits



def evaluate_stereoset_intrasentence(
    model,
    tokenizer,
    stereoset_file,
    device,
    batch_size=8,
    max_seq_length=128,
    suppression_kn=None
):
   
    model.eval()


    mask_token_id = tokenizer.mask_token_id

    dataset = IntrasentenceLoader(
        tokenizer,
        max_seq_length=max_seq_length,
        pad_to_max_length=(batch_size > 1),
        input_file=str(stereoset_file)
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    word_probabilities = defaultdict(list)

    for batch in tqdm(
        loader,
        total=len(loader),
        desc="StereoSet"):

        sentence_id, next_token, input_ids, attention_mask, token_type_ids = batch.values()

        input_ids = torch.stack(
            input_ids
        ).to(device).transpose(0, 1)

        attention_mask = torch.stack(
            attention_mask
        ).to(device).transpose(0, 1)

        token_type_ids = torch.stack(
            token_type_ids
        ).to(device).transpose(0, 1)

        next_token = next_token.to(device)

        # -----------------------------------------
        # Forward
        # -----------------------------------------

        logits = forward_stereoset(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            suppression_kn=suppression_kn
        )

        if logits.dim() != 3:
            raise ValueError(
                "StereoSet richiede logits con shape "
                f"[B, L, V], trovata {tuple(logits.shape)}"
            )

        probabilities = torch.softmax(
            logits,
            dim=-1
        )



        mask_positions = (
            input_ids == mask_token_id
        )

  
        masked_probabilities = probabilities[
            mask_positions
        ]

        target_probabilities = (
            masked_probabilities
            .gather(
                1,
                next_token.unsqueeze(1)
            )
            .squeeze(1)
        )


        for sid, prob in zip(
            sentence_id,
            target_probabilities
        ):

            word_probabilities[
                str(sid)
            ].append(
                prob.item()
            )


    predictions = []

    for sentence_id, probs in \
            word_probabilities.items():


        score = float(
            np.mean(probs)
        )

        predictions.append(
            {
                "id": sentence_id,
                "score": score
            }
        )

    return predictions


# ============================================================
# SAVE
# ============================================================

def save_stereoset_predictions(
    predictions,
    output_file
):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result = {
        "intrasentence": predictions
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(description="Preparing predictions fro Stereoset Benchmark")

    parser.add_argument("--model_name", type = str, default = "bert-base-cased")

    parser.add_argument("--kn_neurons_path", type = str, required = True)

    parser.add_argument("--union_based_model", type = str, default=None)

    parser.add_argument("--relation_based_model", type = str, default=None)

    parser.add_argument("--batch_size", type = int, default=8)

    parser.add_argument("--criterion", type = str, default="union")

    parser.add_argument("--max_seq_length", type = int, default=128)

    args = parser.parse_args()

    model_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    kn_dict = create_mask_foundation(args.kn_neurons_path)

    kn_map = flatten_kn_map(kn_dict, args.criterion)
    

    experiments = {
        "original": {
            "checkpoint": None,
            "suppression": False
        },

        "union": {
            "checkpoint": args.union_based_model,
            "suppression": False
        },

        "relation_based": {
            "checkpoint": args.relation_based_model,
            "suppression": False
        },

        "suppression": {
            "checkpoint": None,
            "suppression": True
        }
    }

    # ========================================================
    # Evaluation loop
    # ========================================================

    for experiment_name, config in \
            experiments.items():

        print()
        print("=" * 70)
        print(
            f"Evaluating: {experiment_name}"
        )
        print("=" * 70)

        model = load_model(
            model_name=model_name,
            checkpoint_path=config[
                "checkpoint"
            ],
            device=DEVICE
        )

        suppression_kn = (
            kn_map
            if config["suppression"]
            else None
        )

        predictions = (
            evaluate_stereoset_intrasentence(
                model=model,
                tokenizer=tokenizer,
                stereoset_file=STEREOSET_DEV,
                device=DEVICE,
                batch_size=args.batch_size,
                max_seq_length=args.max_seq_length,
                suppression_kn=suppression_kn
            )
        )

        output_file = (
            OUTPUT_DIR
            / f"{experiment_name}.json"
        )

        save_stereoset_predictions(
            predictions,
            output_file
        )

        print(
            f"Saved {len(predictions)} "
            f"predictions to {output_file}"
        )

        del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()



