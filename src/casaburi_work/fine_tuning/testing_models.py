from functools import partial
import gc
import math
import csv
import os
from collections import defaultdict
import torch
import torch.nn.functional as F
from fine_tuning.train_metrics import compute_bias_diagnostics, symmetric_kl_divergence
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, AutoTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
from fine_tuning.training import flatten_kn_map
from utils_functions.custom_bert import BertForMaskedLM
from utils_functions.utilities import load_dataset_from_disk
from compute_kn_neurons_dict import  create_mask_relation_based
from datasets_fun.pre_processing_datasets import create_dataloaders
from pathlib import Path


def fixed_mlm_collate_fn(batch, tokenizer):

    input_ids = [
        torch.tensor(
            example["input_ids"],
            dtype=torch.long
        )
        for example in batch
    ]

    attention_mask = [
        torch.tensor(
            example["attention_mask"],
            dtype=torch.long
        )
        for example in batch
    ]

    labels = [
        torch.tensor(
            example["labels"],
            dtype=torch.long
        )
        for example in batch
    ]

    input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )

    attention_mask = pad_sequence(
        attention_mask,
        batch_first=True,
        padding_value=0
    )

    labels = pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def create_fixed_mlm_mask(
    example,
    tokenizer,
    mlm_probability=0.15,
    seed=42,
    example_index=0
):

    input_ids = torch.tensor(
        example["input_ids"],
        dtype=torch.long
    )

    labels = input_ids.clone()


    generator = torch.Generator()
    generator.manual_seed(seed + example_index)



    special_tokens_mask = torch.tensor(
        tokenizer.get_special_tokens_mask(
            input_ids.tolist(),
            already_has_special_tokens=True
        ),
        dtype=torch.bool
    )

    eligible_mask = ~special_tokens_mask


    if tokenizer.pad_token_id is not None:
        eligible_mask &= (
            input_ids != tokenizer.pad_token_id
        )



    probabilities = torch.full(
        input_ids.shape,
        mlm_probability
    )

    probabilities.masked_fill_(
        ~eligible_mask,
        0.0
    )

    masked_indices = torch.bernoulli(
        probabilities,
        generator=generator
    ).bool()

    labels[~masked_indices] = -100


    replace_prob = torch.rand(
        input_ids.shape,
        generator=generator
    )


    indices_replaced = (
        masked_indices
        & (replace_prob < 0.8)
    )

    input_ids[indices_replaced] = (
        tokenizer.mask_token_id
    )

    indices_random = (
        masked_indices
        & (replace_prob >= 0.8)
        & (replace_prob < 0.9)
    )

    random_words = torch.randint(
        low=0,
        high=len(tokenizer),
        size=input_ids.shape,
        generator=generator,
        dtype=torch.long
    )

    input_ids[indices_random] = (
        random_words[indices_random]
    )


    example["input_ids"] = input_ids.tolist()
    example["labels"] = labels.tolist()

    return example

def fixed_mask_function(example, idx, tokenizer):
    return create_fixed_mlm_mask(
        example=example,
        tokenizer=tokenizer,
        mlm_probability=0.15,
        seed=42,
        example_index=idx
    )

@torch.no_grad()
def evaluate_neutral(
    model,
    dataloader,
    device,
    original_model=None
):

    model.eval()

    if original_model is not None:
        original_model.eval()

    total_loss = 0.0
    total_correct = 0
    total_masked_tokens = 0

    total_kl = 0.0
    total_kl_tokens = 0

    num_batches = 0

    for batch in dataloader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        _, outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs

        # -----------------------------------------
        # MLM loss
        # -----------------------------------------

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        total_loss += loss.item()
        num_batches += 1


        mask = labels != -100

        masked_logits = logits[mask]
        masked_labels = labels[mask]

        predictions = masked_logits.argmax(dim=-1)

        total_correct += (
            predictions == masked_labels
        ).sum().item()

        total_masked_tokens += masked_labels.numel()



        if original_model is not None:

            _, original_logits = original_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )


            original_masked_logits = (
                original_logits[mask]
            )

            trained_masked_logits = (
                logits[mask]
            )

            kl = symmetric_kl_divergence(
                original_masked_logits,
                trained_masked_logits
            )

            total_kl += kl.sum().item()
            total_kl_tokens += kl.numel()

    avg_loss = (
        total_loss / max(num_batches, 1)
    )

    accuracy = (
        total_correct
        / max(total_masked_tokens, 1)
    )


    pseudo_perplexity_proxy = math.exp(
        min(avg_loss, 50)
    )

    results = {
        "mlm_loss": avg_loss,
        "mlm_accuracy": accuracy,
        "exp_mlm_loss": pseudo_perplexity_proxy,
        "masked_tokens": total_masked_tokens
    }

    if original_model is not None:
        results["symmetric_kl_vs_original"] = (
            total_kl
            / max(total_kl_tokens, 1)
        )

    return results




@torch.no_grad()
def evaluate_biased_relation_based(
    model,
    dataloader,
    device,
    relation_based_kn,
    ablation_forward_fn
):


    model.eval()

    global_metrics = defaultdict(float)
    global_counts = defaultdict(int)

    relation_metrics = defaultdict(
        lambda: defaultdict(float)
    )

    relation_counts = defaultdict(
        lambda: defaultdict(int)
    )

    for batch, relations in dataloader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch[
            "attention_mask"
        ].to(device)

        labels = batch["labels"].to(device)


        # Nel caso la collate restituisca ancora lista
        if isinstance(relations, (list, tuple)):
            unique_relations = set(relations)

            if len(unique_relations) != 1:
                raise ValueError(
                    f"Mixed relations in evaluation batch: "
                    f"{unique_relations}"
                )

            relation = relations[0]

        if relation not in relation_based_kn:
            raise KeyError(
                f"No KN map found for {relation}"
            )

        kn_map = relation_based_kn[relation]

        kn_map = flatten_kn_map(kn_map, criterion="union")


        _, logits_normal = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )


        logits_ablated = ablation_forward_fn(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            kn_map=kn_map
        )


        metrics = compute_bias_diagnostics(logits_normal, logits_ablated, labels)
        metrics = metrics['example']

        for name, values in metrics.items():

            values = values.float()

            global_metrics[name] += (
                values.sum().item()
            )

            global_counts[name] += (
                values.numel()
            )

            relation_metrics[
                relation
            ][name] += values.sum().item()

            relation_counts[
                relation
            ][name] += values.numel()


    global_results = {}

    for name in global_metrics:
        global_results[name] = (
            global_metrics[name]
            / max(global_counts[name], 1)
        )



    per_relation_results = {}

    for relation in relation_metrics:

        per_relation_results[relation] = {}

        for name in relation_metrics[relation]:

            per_relation_results[
                relation
            ][name] = (
                relation_metrics[
                    relation
                ][name]
                /
                max(
                    relation_counts[
                        relation
                    ][name],
                    1
                )
            )

    return {
        "global": global_results,
        "per_relation": per_relation_results
    }

def ablation_forward(model, input_ids, attention_mask, kn_map):

    _, outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        imp_op = "remove",
        imp_pos = kn_map
    )
    return outputs



def append_dict_to_csv(
    filepath,
    row
):
    """
    Appende un dizionario come riga CSV.
    """

    os.makedirs(
        os.path.dirname(filepath) or ".",
        exist_ok=True
    )

    exists = os.path.exists(filepath)

    with open(
        filepath,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=row.keys()
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def save_evaluation_results(
    output_dir,
    model_name,
    neutral_results,
    biased_results
):


    os.makedirs(
        output_dir,
        exist_ok=True
    )


    neutral_row = {
        "model": model_name,
        **neutral_results
    }

    append_dict_to_csv(
        os.path.join(
            output_dir,
            "evaluation_neutral.csv"
        ),
        neutral_row
    )



    biased_global_row = {
        "model": model_name,
        **biased_results["global"]
    }

    append_dict_to_csv(
        os.path.join(
            output_dir,
            "evaluation_biased_global.csv"
        ),
        biased_global_row
    )


    for relation, metrics in \
            biased_results[
                "per_relation"
            ].items():

        row = {
            "model": model_name,
            "relation": relation,
            **metrics
        }

        append_dict_to_csv(
            os.path.join(
                output_dir,
                "evaluation_biased_per_relation.csv"
            ),
            row
        )


def main():
    model_name = "bert-base-cased"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    models_to_evaluate = {
    "relation_based": "results/bert-base-cased/relation_based_fine_tuned_model.pt"
    }

    kn_neurons_path = Path("results/bert-base-cased/kn")

    relation_ids = ["BR0" + str(x) for x in range(1, 10)]

    kn_neurons_map = create_mask_relation_based(kn_neurons_path, relation_ids)

    biased_test_path = "biased_dataset"

    neutral_test_path = "neutral_preservation_dataset"

    biased_dataset = load_dataset_from_disk(biased_test_path)
    neutral_dataset = load_dataset_from_disk(neutral_test_path)

    test_neutral = neutral_dataset['test']

    def tokenize_row(example, idx):
        if 'input_ids' not in example or 'labels' not in example:
            tokenized = tokenizer(
                example['text'],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            example['input_ids'] = tokenized['input_ids'].squeeze(0)
            example['attention_mask'] = tokenized['attention_mask'].squeeze(0)
        return example
    columns_to_remove = test_neutral.column_names
    test_neutral = test_neutral.map(
        tokenize_row,
        with_indices=True, 
        remove_columns=columns_to_remove
    )
    
    

    biased_loader = create_dataloaders(
        biased_dataset,
        tokenizer,
        "biased",
    )


    test_bias_loader = biased_loader['test']

    test_neutral = test_neutral.map(
        fixed_mask_function,
        fn_kwargs={"tokenizer": tokenizer},
        with_indices=True
    )

    test_neutral_loader = DataLoader(
        test_neutral,
        batch_size=8,
        shuffle=False,
        collate_fn=partial(fixed_mlm_collate_fn, tokenizer=tokenizer)
    )

    
    for model_label, model_path in models_to_evaluate.items():
        model = None
        if model_path is None:
            model = BertForMaskedLM.from_pretrained(model_name).to(device)
        else:
            model = BertForMaskedLM.from_pretrained(
                model_name
            ).to(device)

            model.load_state_dict(
            torch.load(model_path, map_location=device)
            )

        neutral_results = evaluate_neutral(
                model=model,
                dataloader=test_neutral_loader,
                device=device,
            )

        biased_results = evaluate_biased_relation_based(
            model=model,
            dataloader=test_bias_loader,
            device=device,
            relation_based_kn=kn_neurons_map,
            ablation_forward_fn=ablation_forward
        )

        save_evaluation_results(
            output_dir=f"results/bert-base-cased/evaluation_results/{model_label}",
            model_name=model_label,
            neutral_results=neutral_results,
            biased_results=biased_results
        )
        del model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
    