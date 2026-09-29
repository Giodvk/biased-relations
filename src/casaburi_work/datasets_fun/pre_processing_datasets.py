import torch
from transformers import BertTokenizer, AutoTokenizer
from pathlib import Path
from torch.utils.data import DataLoader
from functools import partial
from utils_functions.utilities import get_masked_positions, mask_input_tokens, load_dataset_from_disk
from datasets_fun.Relation_sampler import RelationBatchSampler
import os

def neutral_collate_fn(
    batch,
    tokenizer,
    max_length=128,
    mlm_probability=0.15,
):
    """
    Tokenize a batch of neutral examples and apply dynamic MLM masking.
    """

    texts = [example["text"] for example in batch]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]

    masked_positions = get_masked_positions(
        tokenizer=tokenizer,
        input_ids=input_ids,
        mlm_probability=mlm_probability,
    )


    masked_input_ids, labels = mask_input_tokens(
        tokenizer=tokenizer,
        input_ids=input_ids,
        masked_positions=masked_positions,
    )

    result = {
        "input_ids": masked_input_ids,
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }

    if "token_type_ids" in encoded:
        result["token_type_ids"] = encoded["token_type_ids"]

    return result


def biased_collate_fn(
    batch,
    tokenizer,
    max_length=128,
):
    """
    Build deterministic MLM inputs from biased prompts.
    """
    relations = []
    processed_examples = []
    for example in batch:
        processed = preprocess_biased_example(
            tokenizer=tokenizer,
            biased_prompt=example["biased_prompt"],
            target=example["target"],
            max_length=max_length,
        )
        relations.append(example['bias_code'])
        processed_examples.append(processed)

    result = {}

    keys = processed_examples[0].keys()

    for key in keys:
        result[key] = torch.cat(
            [example[key] for example in processed_examples],
            dim=0,
        )

    return result, relations



def preprocess_biased_example(
    tokenizer,
    biased_prompt: str,
    target: str,
    max_length: int = 128,  
):

    if tokenizer is None:
        raise ValueError("Tokenizer must be provided.")

    if tokenizer.mask_token not in biased_prompt:
        raise ValueError(
            f"Prompt does not contain {tokenizer.mask_token}: "
            f"{biased_prompt}"
        )

    # Tokenize only the target, without adding [CLS]/[SEP]
    target_tokens = tokenizer.tokenize(target)

    if not target_tokens:
        raise ValueError(
            f"Target produced no tokens: {target!r}"
        )

    target_ids = tokenizer.convert_tokens_to_ids(
        target_tokens
    )

    # Expand the original single [MASK]
    mask_sequence = " ".join(
        [tokenizer.mask_token] * len(target_tokens)
    )

    expanded_prompt = biased_prompt.replace(
        tokenizer.mask_token,
        mask_sequence,
        1,
    )

    encoding = tokenizer(
        expanded_prompt,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"]

    # Find every [MASK] position
    mask_positions = (
        input_ids[0] == tokenizer.mask_token_id
    ).nonzero(as_tuple=True)[0]

    if len(mask_positions) != len(target_ids):
        raise ValueError(
            f"Number of [MASK] positions ({len(mask_positions)}) "
            f"does not match number of target tokens "
            f"({len(target_ids)}) for target {target!r}."
        )

    # Ignore every position by default
    labels = torch.full_like(
        input_ids,
        fill_value=-100,
    )

    # Put each target token at its corresponding [MASK]
    for position, target_id in zip(
        mask_positions,
        target_ids,
    ):
        labels[0, position] = target_id

    result = {
        "input_ids": input_ids,
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
    }

    if "token_type_ids" in encoding:
        result["token_type_ids"] = (
            encoding["token_type_ids"]
        )

   

    return result


def create_dataloaders(
    dataset,
    tokenizer,
    dataset_type: str,
    batch_size: int = 8,
    max_length: int = 128,
    mlm_probability: float = 0.15,
):

    batch_sampler = None
    if dataset_type == "neutral":

        collate_fn = partial(
            neutral_collate_fn,
            tokenizer=tokenizer,
            max_length=max_length,
            mlm_probability=mlm_probability,
        )

    elif dataset_type == "biased":

        batch_sampler = RelationBatchSampler(
            dataset['train'],
            batch_size = batch_size,
            shuffle=True,
            drop_last=False
        )

        collate_fn = partial(
            biased_collate_fn,
            tokenizer=tokenizer,
            max_length=max_length,
        )

    else:
        raise ValueError(
            f"Unknown dataset_type: {dataset_type}. "
            "Expected 'neutral' or 'biased'."
        )
    if batch_sampler is not None:
        train_loader = DataLoader(
        dataset["train"],
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        )
    else:
        train_loader = DataLoader(
            dataset["train"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            )

    validation_loader = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        dataset["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return {
        "train": train_loader,
        "validation": validation_loader,
        "test": test_loader,
    }




def main():

    print(os.getcwd())
    
    model_name = "bert-base-cased"


    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    # ---------------------------------------------
    # Load raw datasets
    # ---------------------------------------------

    neutral_dataset = load_dataset_from_disk(
        "neutral_preservation_dataset"
    )

    biased_dataset = load_dataset_from_disk(
        "biased_dataset"
    )
    # ---------------------------------------------
    # Create loaders
    # ---------------------------------------------
    
    neutral_loaders = create_dataloaders(
        dataset=neutral_dataset,
        tokenizer=tokenizer,
        dataset_type="neutral",
        batch_size=8,
        max_length=128,
        mlm_probability=0.15,
    )

    
    biased_loaders = create_dataloaders(
        dataset=biased_dataset,
        tokenizer=tokenizer,
        dataset_type="biased",
        batch_size=8,
        max_length=128,
    )





if __name__ == "__main__":
    main()

