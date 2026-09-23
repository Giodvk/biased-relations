from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer
from custom_bert import BertForMaskedLM
from compute_kn_neurons_dict import create_mask_foundation
from pre_processing_datasets import create_dataloaders
from utilities import create_gradient_mask, load_dataset_from_disk, plot_utility_fairness
from tqdm import tqdm

def train_one_epoch(model, neutral_loader, bias_loader, optimizer, device, gradient_mask, kn_neuron_map, biased_loss, lambda_bias = 1.0, criterion = 'union', max_grad_norm = None):
    total_loss = 0.0
    total_neutral_loss = 0.0
    total_bias_loss = 0.0
    total_bias_contribution = 0.0
    total_margin = 0.0

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
        bias_contribution, margin, loss_bias = biased_loss(
            logits_normal=bias_logits,
            logits_ablated=unbiased_logits,
            labels=bias_labels
        )

        total_bias_contribution+=bias_contribution.item()
        total_margin+=margin.item()

        loss = (
            loss_neutral
            + lambda_bias * loss_bias
        )

        loss.backward()

        apply_gradient_mask(
            model,
            gradient_mask
        )

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm
            )

        optimizer.step()

        total_loss += loss.item()
        total_neutral_loss += loss_neutral.item()
        total_bias_loss += loss_bias.item()

        num_steps+=1
    return {
        "loss": total_loss / num_steps,
        "neutral_loss": total_neutral_loss / num_steps,
        "bias_loss": total_bias_loss / num_steps,
        "bias_contribution": total_bias_contribution / num_steps,
        "margin_best" : total_margin / num_steps
    }

def evaluate_one_epoch(model, neutral_val_loader, bias_val_loader, neutral_loss, biased_loss, kn_neurons_map, device, criterion = "union"):

    neurons_to_deactivate = flatten_kn_map(kn_neurons_map, criterion)
    total_neutral_loss = 0.0
    total_bias_loss = 0.0
    total_bias_contribution = 0.0
    total_margin = 0.0
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

            bias_contribution, margin, bias_loss = biased_loss(bias_output, ablated_bias_output, bias_labels)

            total_margin+=margin.item()
            total_bias_contribution+=bias_contribution.item()
            total_neutral_loss+=loss_neutral.item()
            total_bias_loss+=bias_loss.item()

            num_steps+=1

    return {
            "neutral_loss": total_neutral_loss / num_steps,
            "bias_loss": total_bias_loss / num_steps,
            "bias_contribution": total_bias_contribution / num_steps,
            "margin_best" : total_margin / num_steps
        }
    

def flatten_kn_map(kn_neuron_map, criterion):
    if criterion in kn_neuron_map:
        kn_neuron_map = kn_neuron_map[criterion]
    return [(layer, neuron) for layer, neurons in kn_neuron_map.items() for neuron in neurons]

def bias_loss_function(logits_normal, logits_ablated, labels):
    target_mask = labels != -100
    target_logits = logits_normal[target_mask]
    ablated_logits = logits_ablated[target_mask]
    target_labels = labels[target_mask]

    target_scores_normal = target_logits.gather(1, target_labels.unsqueeze(1)).squeeze(1)
    target_scores_ablated = ablated_logits.gather(1,target_labels.unsqueeze(1)).squeeze(1)

    competitor_logits_normal = target_logits.clone()
    competitor_logits_ablated = ablated_logits.clone()

    competitor_logits_normal.scatter_(1, target_labels.unsqueeze(1), float("-inf"))

    competitor_logits_ablated.scatter_(1, target_labels.unsqueeze(1), float("-inf"))


    competitor_normal_best = torch.max(competitor_logits_normal, dim=1).values
    competitor_ablated_best = torch.max(competitor_logits_ablated, dim=1).values

    margin_normal = (target_scores_normal- competitor_normal_best)
    margin_ablated = (target_scores_ablated - competitor_ablated_best)

    bias_contribution = target_scores_normal - target_scores_ablated


    penalty = F.relu(bias_contribution)

    batch_size = labels.size(0)

    batch_indices = (target_mask.nonzero(as_tuple=False)[:, 0])

    loss = torch.zeros(batch_size, device=labels.device, dtype=penalty.dtype)
    contribution_bias = torch.zeros(batch_size, device=labels.device, dtype=penalty.dtype)
    margins = torch.zeros(batch_size, device=labels.device, dtype=penalty.dtype)


    number_of_examples = torch.zeros(batch_size, device=labels.device, dtype=penalty.dtype)

    loss.scatter_add_(0, batch_indices, penalty)

    contribution_bias.scatter_add_(0, batch_indices, bias_contribution)

    margins.scatter_add_(0, batch_indices, margin_normal)

    number_of_examples.scatter_add_(0, batch_indices, torch.ones_like(penalty))

    number_of_examples = number_of_examples.clamp_min(1.0)

    loss = ( loss / (number_of_examples + 1e-8) )

    contribution_bias = (contribution_bias / number_of_examples)

    margins = (margins / number_of_examples)

    loss = loss.mean()

    bias_contribution = contribution_bias.mean()

    margin_normal = margins.mean()

    return bias_contribution, margin_normal, loss

def apply_gradient_mask(model, gradient_mask):
    for name, param in model.named_parameters():
        if name in gradient_mask and param.grad is not None:
            param.grad.mul_(gradient_mask[name])




def surgical_fine_tuning(model_name, num_epoch, batch_size, neutral_df, biased_df, biased_neurons, bias_criterion, learning_rate, device, criterion):

    history = []

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
        "margin_best": baseline_metrics["margin_best"]
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

        print(
            train_metrics
        )

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
            "neutral_loss": baseline_metrics["neutral_loss"],
            "bias_loss": baseline_metrics["bias_loss"],
            "bias_contribution": baseline_metrics["bias_contribution"],
            "margin_best": baseline_metrics["margin_best"]
            })

        print(
            eval_metrics
        )

    plot_utility_fairness(history, "utility_fairness_trade_off")
    return model


if __name__ == "__main__":


    model_name = "bert-base-cased"

    kn_neurons_path = Path("results/bert-base-cased/kn")

    kn_neurons_map = create_mask_foundation(kn_neurons_path)



    neutral_data_path = "neutral_preservation_dataset"
    bias_data_path = "biased_dataset"

    neutral_data = load_dataset_from_disk(neutral_data_path)
    biased_data = load_dataset_from_disk(bias_data_path)

    print(len(neutral_data['train']))
    print(len(biased_data['train']))

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
        criterion="union"
    )
