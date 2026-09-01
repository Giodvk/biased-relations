import torch
from custom_bert import BertForMaskedLM
import transformers
import huggingface_hub
from transformers import BertTokenizer, AutoTokenizer
from datasets import load_from_disk
from pathlib import Path
from torch.utils.data import DataLoader
from functools import partial

def load_dataset_from_disk(dataset_path: str):
    """
    Load a HuggingFace DatasetDict saved with save_to_disk().
    """

    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    dataset = load_from_disk(str(path))

    required_splits = {"train", "validation", "test"}
    missing_splits = required_splits - set(dataset.keys())

    if missing_splits:
        raise ValueError(
            f"Missing dataset splits: {sorted(missing_splits)}"
        )

    return dataset


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

    processed_examples = []

    for example in batch:
        processed = preprocess_biased_example(
            tokenizer=tokenizer,
            biased_prompt=example["biased_prompt"],
            target=example["target"],
            max_length=max_length,
        )

        processed_examples.append(processed)

    result = {}

    keys = processed_examples[0].keys()

    for key in keys:
        result[key] = torch.cat(
            [example[key] for example in processed_examples],
            dim=0,
        )

    return result



def create_gradient_mask(model:BertForMaskedLM, kn_neurons_mask_generator:dict) -> dict:
    """
    Create a gradient mask for the model based on the specified options.
    """

    def compute_mask(kn_neurons, masks_dict):
         for layer_idx, neurons in kn_neurons.items():
                if layer_idx < 0 or layer_idx >= len(model.bert.encoder.layer):
                    raise ValueError(f"Layer index {layer_idx} is out of bounds for the model's encoder layers.")
                
                layer_module = model.bert.encoder.layer[layer_idx]
        
        
                # Create masks for weights and biases
                weight_mask = torch.zeros_like(layer_module.intermediate.dense.weight)
                bias_mask = torch.zeros_like(layer_module.intermediate.dense.bias)
                output_weight_mask = torch.zeros_like(layer_module.output.dense.weight)
        
                for neuron_idx in neurons:
                    if neuron_idx < 0 or neuron_idx >= layer_module.intermediate.dense.weight.shape[0]:
                        raise ValueError(f"Neuron index {neuron_idx} is out of bounds for layer {layer_idx}.")
                    weight_mask[neuron_idx, :] = 1.0
                    bias_mask[neuron_idx] = 1.0
                    output_weight_mask[:, neuron_idx] = 1.0
        
                masks_dict[f"bert.encoder.layer.{layer_idx}.intermediate.dense.weight"] = weight_mask
                masks_dict[f"bert.encoder.layer.{layer_idx}.intermediate.dense.bias"] = bias_mask
                masks_dict[f"bert.encoder.layer.{layer_idx}.output.dense.weight"] = output_weight_mask

    gradient_mask = {"union": {}, "intersection": {}}

    if kn_neurons_mask_generator.get("union"):
        kn_neurons = kn_neurons_mask_generator["union"]

        compute_mask(kn_neurons, gradient_mask['union'])


    if kn_neurons_mask_generator.get("intersection"):
        kn_neurons = kn_neurons_mask_generator["intersection"]
        compute_mask(kn_neurons, gradient_mask['intersection'])


    return gradient_mask



def get_masked_positions(input_ids: torch.Tensor, tokenizer, mlm_probability=0.15):
    """
    Apply classic MLM masking to the tokenized dataset.
    """
    if not tokenizer:
        raise ValueError("Tokenizer must be provided.")

    if input_ids.numel() == 0 or not isinstance(input_ids, torch.Tensor):
        raise ValueError("Input IDs must be provided.")


    probability_matrix = torch.full(
        input_ids.shape,
        mlm_probability,
        device=input_ids.device
    )

    special_tokens_mask = torch.tensor(
        [
            tokenizer.get_special_tokens_mask(
                sequence.tolist(),
                already_has_special_tokens=True
            )
            for sequence in input_ids
        ],
        dtype=torch.bool,
        device=input_ids.device
    )

    probability_matrix.masked_fill_(
        special_tokens_mask,
        value=0.0
    )

    masked_positions = torch.bernoulli(
        probability_matrix
    ).bool()

    return masked_positions

def mask_input_tokens(input_ids, masked_positions, tokenizer):
    """
    Mask the input tokens based on the masked positions.
    """


    if not tokenizer:
        raise ValueError("Tokenizer must be provided.")

    if input_ids.numel() == 0 or not isinstance(input_ids, torch.Tensor):
        raise ValueError("Input IDs must be provided.")

    if masked_positions.numel() == 0 or not isinstance(masked_positions, torch.Tensor):
        raise ValueError("Masked positions must be provided.")

    labels = input_ids.clone()
    labels[~masked_positions] = -100  # Only compute loss on masked tokens

    masked_inputs = input_ids.clone()

   # 80% delle posizioni selezionate -> [MASK]
    replace_prob = torch.full(
        input_ids.shape,
        0.8,
        device=input_ids.device,
    )

    replaced_positions = (
        torch.bernoulli(replace_prob).bool()
        & masked_positions
    )

    masked_inputs[replaced_positions] = tokenizer.mask_token_id

    # Tra le posizioni selezionate rimaste,
    # metà -> token random
    # cioè circa il 10% totale delle posizioni selezionate
    random_prob = torch.full(
        input_ids.shape,
        0.5,
        device=input_ids.device,
    )

    random_positions = (
        torch.bernoulli(random_prob).bool()
        & masked_positions
        & ~replaced_positions
    )

    random_tokens = torch.randint(
        low=0,
        high=len(tokenizer),
        size=input_ids.shape,
        device=input_ids.device,
    )

    masked_inputs[random_positions] = random_tokens[random_positions]

    # Il restante ~10% delle posizioni selezionate
    # rimane invariato automaticamente

    return masked_inputs, labels



def preprocess_biased_example(
    tokenizer,
    biased_prompt: str,
    target: str,
    max_length: int = 128,
):
    """
    Preprocess one biased prompt for MLM.

    Returns:
        input_ids
        attention_mask
        token_type_ids (if available)
        labels
    """

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
    """
    Create train/validation/test DataLoaders.

    dataset_type:
        "neutral"
        "biased"
    """

    if dataset_type == "neutral":

        collate_fn = partial(
            neutral_collate_fn,
            tokenizer=tokenizer,
            max_length=max_length,
            mlm_probability=mlm_probability,
        )

    elif dataset_type == "biased":

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

    # ---------------------------------------------
    # Sanity checks
    # ---------------------------------------------

    neutral_batch = next(
        iter(neutral_loaders["train"])
    )

    biased_batch = next(
        iter(biased_loaders["train"])
    )

    print("\n=== NEUTRAL ===")
    print("input_ids:", neutral_batch["input_ids"].shape)
    print(
        "attention_mask:",
        neutral_batch["attention_mask"].shape
    )
    print("labels:", neutral_batch["labels"].shape)

    print("\n=== BIASED ===")
    print("input_ids:", biased_batch["input_ids"].shape)
    print(
        "attention_mask:",
        biased_batch["attention_mask"].shape
    )
    print("labels:", biased_batch["labels"].shape)

    mask_positions = (
    biased_batch["input_ids"][0] == tokenizer.mask_token_id
    ).nonzero(as_tuple=True)[0]

    print("Mask positions:", mask_positions.tolist())

    for pos in mask_positions:
        label_id = biased_batch["labels"][0, pos].item()
        print(
        pos.item(),
        tokenizer.convert_ids_to_tokens(label_id)
        )


if __name__ == "__main__":
    main()

