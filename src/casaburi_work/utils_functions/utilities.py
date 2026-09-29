from utils_functions.custom_bert import BertForMaskedLM
import torch
from pathlib import Path
from datasets import load_from_disk
import matplotlib.pyplot as plt

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

def compute_mask(model, kn_neurons, masks_dict):
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


def create_gradient_mask(model:BertForMaskedLM, kn_neurons_mask_generator:dict) -> dict:
    """
    Create a gradient mask for the model based on the specified options.
    """

    gradient_mask = {"union": {}, "intersection": {}}

    if kn_neurons_mask_generator.get("union"):
        kn_neurons = kn_neurons_mask_generator["union"]

        compute_mask(model, kn_neurons, gradient_mask['union'])


    if kn_neurons_mask_generator.get("intersection"):
        kn_neurons = kn_neurons_mask_generator["intersection"]
        compute_mask(model, kn_neurons, gradient_mask['intersection'])


    return gradient_mask


def create_relation_based_mask(model: BertForMaskedLM, kn_neurons_relation_map:dict):
    gradient_mask = {}

    for relation_id, relation_mask in kn_neurons_relation_map.items():
        mask_input = {}
        compute_mask(model, relation_mask, mask_input)
        gradient_mask[relation_id] = mask_input

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
    labels[~masked_positions] = -100  

    masked_inputs = input_ids.clone()

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


    return masked_inputs, labels





def count_single_token_examples(example, split_criterion):
    if len(example) != 3 : 
        print("L'esempio non è valido")
        return None
    targets = []
    for criterion in split_criterion:
        targets = example[1].split(criterion)
        if len(targets) > 1:
            return 0
    print(example[1])
    return 1


