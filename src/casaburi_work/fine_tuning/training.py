from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from utils_functions.custom_bert import BertForMaskedLM
from compute_kn_neurons_dict import create_mask_foundation
from datasets.pre_processing_datasets import create_dataloaders
from utils_functions.utilities import create_gradient_mask, load_dataset_from_disk, create_relation_based_mask
from tqdm import tqdm
from fine_tuning.train_metrics import bias_loss_function, compute_bias_diagnostics
from fine_tuning.Epoch_tracker import EpochMetricTracker

def train_one_epoch(model, neutral_loader, bias_loader, optimizer, device, gradient_mask, kn_neuron_map, biased_loss, lambda_bias = 1.0, criterion = 'union', max_grad_norm = None):
    tracker = EpochMetricTracker()

    neurons_to_deactivate = flatten_kn_map(kn_neuron_map, criterion)

    neutral_loss = nn.CrossEntropyLoss(ignore_index=-100)
    model.train()
    num_steps = 0

    for neutral_batch, bias_batch in zip(neutral_loader, bias_loader):
        neutral_input_ids = neutral_batch['input_ids'].to(device)
        neutral_attention_mask = neutral_batch['attention_mask'].to(device)
        labels_neutral = neutral_batch['labels'].to(device)
        bias_input_ids = bias_batch['input_ids'].to(device)
        bias_attention_mask = bias_batch['attention_mask'].to(device)
        bias_labels = bias_batch['labels'].to(device)

        optimizer.zero_grad()

        _, outputs_logits = model(input_ids=neutral_input_ids, attention_mask=neutral_attention_mask)

        model.eval()
        _, bias_logits = model(input_ids=bias_input_ids, attention_mask=bias_attention_mask)

        loss_neutral = neutral_loss(
        outputs_logits.reshape(-1, outputs_logits.size(-1)),
        labels_neutral.reshape(-1)
)
        with torch.no_grad():

            _, unbiased_logits = model(
                input_ids=bias_input_ids,
                attention_mask=bias_attention_mask,
                tgt_pos=None,
                imp_op="remove",
                imp_pos=neurons_to_deactivate
            )
        model.train()
        loss_bias, bias_contribution = biased_loss(
            logits_normal=bias_logits,
            logits_ablated=unbiased_logits,
            labels=bias_labels
        )


        loss = (
            loss_neutral
            + lambda_bias * loss_bias
        )

        loss.backward()

        apply_gradient_mask(
            model,
            gradient_mask,
            optimizer
        )

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm
            )

        optimizer.step()


        diagnostics = compute_bias_diagnostics(bias_logits, unbiased_logits, bias_labels)

        diag = diagnostics['batch']
        batch_metrics = {
            "bias_loss": loss_bias,
            "bias_contribution":
            diag["logit_contribution"],

            "rank_normal":
            diag["rank_normal"],

            "rank_ablated":
            diag["rank_ablated"],

            "rank_shift":
            diag["rank_shift"],

            "relative_rank_shift":
            diag["relative_rank_shift"],

            "gold_prob_normal":
            diag["gold_prob_normal"],

            "gold_prob_ablated":
            diag["gold_prob_ablated"],

            "prob_drop":
            diag["prob_drop"],

            "margin_normal":
            diag["margin_normal"],

            "margin_ablated":
            diag["margin_ablated"],

            "symmetric_kl":
            diag["symmetric_kl"],

            "top1_preservation":
            diag["top1_preservation_rate"],

            "rank_worsened_fraction":
            diag["rank_worsened_fraction"],
        }
        tracker.update(batch_metrics)

        num_steps+=1
    return tracker.compute()

def evaluate_one_epoch(model, neutral_val_loader, bias_val_loader, neutral_loss, biased_loss, kn_neurons_map, device, criterion = "union"):

    neurons_to_deactivate = flatten_kn_map(kn_neurons_map, criterion)
    total_neutral_loss = 0.0
    total_bias_loss = 0.0
    total_bias_contribution = 0.0
    num_steps = 0

    model.eval()
    with torch.no_grad():
        for neutral_batch, biased_batch in zip(neutral_val_loader, bias_val_loader):
            neutral_input_ids = neutral_batch['input_ids'].to(device)
            neutral_attention_mask = neutral_batch['attention_mask'].to(device)
            labels_neutral = neutral_batch['labels'].to(device)
            bias_input_ids = biased_batch['input_ids'].to(device)
            bias_attention_mask = biased_batch['attention_mask'].to(device)
            bias_labels = biased_batch['labels'].to(device)

            _, neutral_output = model(input_ids = neutral_input_ids, attention_mask = neutral_attention_mask)
            _, bias_output = model(input_ids = bias_input_ids, attention_mask = bias_attention_mask)

            loss_neutral = neutral_loss(
                neutral_output.reshape(-1, neutral_output.size(-1)),
                                       labels_neutral.reshape(-1)
            )

            _, ablated_bias_output = model(input_ids = bias_input_ids, attention_mask = bias_attention_mask, tgt_pos=None,
                            imp_op="remove",
                            imp_pos=neurons_to_deactivate)

            bias_loss, bias_contribution = biased_loss(bias_output, ablated_bias_output, bias_labels)

            total_bias_contribution+=bias_contribution.item()
            total_neutral_loss+=loss_neutral.item()
            total_bias_loss+=bias_loss.item()

            num_steps+=1

    return {
            "neutral_loss": total_neutral_loss / num_steps,
            "bias_loss": total_bias_loss / num_steps,
            "bias_contribution": total_bias_contribution / num_steps,
        }
    

def flatten_kn_map(kn_neuron_map, criterion):
    if criterion in kn_neuron_map:
        kn_neuron_map = kn_neuron_map[criterion]
    return [(layer, neuron) for layer, neurons in kn_neuron_map.items() for neuron in neurons]



def apply_gradient_mask(model, gradient_mask, optimizer):
    for name, param in model.named_parameters():

        if param.grad is None:
            continue

        if name in gradient_mask:

            mask = gradient_mask[name]

            param.grad.mul_(mask)

            state = optimizer.state[param]

        if "exp_avg" in state:
            state["exp_avg"].mul_(mask)

        if "exp_avg_sq" in state:
            state["exp_avg_sq"].mul_(mask)

        else:
            param.grad = None




def surgical_fine_tuning(model_name,
                        num_epoch, 
                        batch_size, 
                        neutral_df, 
                        biased_df, 
                        biased_neurons, 
                        bias_criterion, 
                        learning_rate, 
                        device, criterion, 
                        results_path,
                        ):

    history = []

    gradient_mask = {}

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = BertForMaskedLM.from_pretrained(model_name).to(device)

    gradient_mask = create_gradient_mask(model, biased_neurons)

    for name, param in model.named_parameters():
        param.requires_grad = name in gradient_mask[criterion]

    optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=0.0
        )
    neutral_criterion = nn.CrossEntropyLoss(ignore_index = -100)


    neutral_loader = create_dataloaders(neutral_df, tokenizer, "neutral", batch_size)
    bias_loader = create_dataloaders(biased_df, tokenizer, "biased", batch_size)

    baseline_metrics = evaluate_one_epoch(
                model=model,
                neutral_val_loader=neutral_loader['validation'],
                bias_val_loader=bias_loader['validation'],
                neutral_loss=neutral_criterion,
                biased_loss = bias_criterion,
                kn_neurons_map = biased_neurons,
                device=device
            )

    history.append({
        "epoch": 0,
        "neutral_loss": baseline_metrics["neutral_loss"],
        "bias_loss": baseline_metrics["bias_loss"],
        "bias_contribution": baseline_metrics["bias_contribution"],
        })

    
    for epoch in tqdm(range(num_epoch)):
        train_metrics = train_one_epoch(
            model = model, 
            neutral_loader = neutral_loader['train'], 
            bias_loader = bias_loader['train'], 
            optimizer = optimizer, 
            device = device, 
            gradient_mask = gradient_mask, 
            kn_neuron_map = biased_neurons, 
            biased_loss = bias_criterion)

        print(
            f"\nEpoch {epoch + 1}/{num_epoch}"
        )

        EpochMetricTracker.save_epoch_metrics(results_path, epoch + 1, train_metrics)


        eval_metrics = evaluate_one_epoch(
            model=model,
            neutral_val_loader=neutral_loader['validation'],
            bias_val_loader=bias_loader['validation'],
            neutral_loss=neutral_criterion,
            biased_loss = bias_criterion,
            kn_neurons_map = biased_neurons,
            device=device
        )

        history.append({
            "epoch": epoch + 1,
            "neutral_loss": eval_metrics["neutral_loss"],
            "bias_loss": eval_metrics["bias_loss"],
            "bias_contribution": eval_metrics["bias_contribution"],
            })

        print(
            eval_metrics
        )
    return model


if __name__ == "__main__":
    torch.cuda.empty_cache()

    model_name = "bert-base-cased"

    kn_neurons_path = Path("results/bert-base-cased/kn")

    kn_neurons_map = create_mask_foundation(kn_neurons_path)

    results_path = "results/epoch_stats.csv"

    neutral_data_path = "neutral_preservation_dataset"
    bias_data_path = "biased_dataset"

    neutral_data = load_dataset_from_disk(neutral_data_path)
    biased_data = load_dataset_from_disk(bias_data_path)


    num_epoch = 10
    batch_size = 8
    lr = 0.00001
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = surgical_fine_tuning(
        model_name=model_name,
        num_epoch=num_epoch,
        batch_size=batch_size,
        neutral_df=neutral_data,
        biased_df=biased_data,
        biased_neurons=kn_neurons_map,
        bias_criterion=bias_loss_function,
        learning_rate=lr,
        device=device,
        criterion="union",
        results_path=results_path
    )
    torch.cuda.empty_cache()
