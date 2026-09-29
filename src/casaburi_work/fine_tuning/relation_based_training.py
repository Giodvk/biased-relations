from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from utils_functions.custom_bert import BertForMaskedLM
from compute_kn_neurons_dict import create_mask_foundation, create_mask_relation_based
from datasets_fun.pre_processing_datasets import create_dataloaders
from utils_functions.utilities import create_gradient_mask, load_dataset_from_disk, create_relation_based_mask
from fine_tuning.train_metrics import bias_loss_function, compute_bias_diagnostics
from fine_tuning.Epoch_tracker import EpochMetricTracker
from fine_tuning.training import flatten_kn_map

def train_relation_based(
    model,
    biased_loader,
    neutral_loader,
    relation_gradient_masks,
    relation_based_kn,
    optimizer,
    bias_loss_function,
    neutral_loss_function,
    device,
    lambda_bias=1.0,
    num_epochs=1,
):
 

    model.train()

    neutral_iterator = iter(neutral_loader)

    for epoch in range(num_epochs):

        epoch_total_loss = 0.0
        epoch_bias_loss = 0.0
        epoch_neutral_loss = 0.0

        num_batches = 0

        for biased_batch, relation in biased_loader:

            if isinstance(relation, (list, tuple)):
                unique_relations = set(relation)

                if len(unique_relations) != 1:
                    raise ValueError(
                        f"Batch contains multiple relations: "
                        f"{unique_relations}"
                    )

                relation = relation[0]
                

            if relation not in (relation_gradient_masks and relation_based_kn):
                raise KeyError(
                    f"No gradient mask found for relation {relation}"
                )
            
            
            kn_map = relation_based_kn[relation]
            current_mask = relation_gradient_masks[relation]

            ablation_kn_map = flatten_kn_map(kn_map, "union")

            try:
                neutral_batch = next(neutral_iterator)

            except StopIteration:
                neutral_iterator = iter(neutral_loader)
                neutral_batch = next(neutral_iterator)


            biased_input_ids = biased_batch[
                "input_ids"
            ].to(device)

            biased_attention_mask = biased_batch[
                "attention_mask"
            ].to(device)

            biased_labels = biased_batch[
                "labels"
            ].to(device)

            neutral_input_ids = neutral_batch[
                "input_ids"
            ].to(device)

            neutral_attention_mask = neutral_batch[
                "attention_mask"
            ].to(device)

            neutral_labels = neutral_batch[
                "labels"
            ].to(device)


            optimizer.zero_grad(set_to_none=True)


            model.train()

            _, neutral_outputs = model(
                input_ids=neutral_input_ids,
                attention_mask=neutral_attention_mask
            )

            if hasattr(neutral_outputs, "logits"):
                neutral_logits = neutral_outputs.logits
            else:
                neutral_logits = neutral_outputs
            

            neutral_loss = neutral_loss_function(
                neutral_logits.reshape(-1, neutral_logits.size(-1)),
                neutral_labels.reshape(-1)
            )



            model.eval()

            _, normal_outputs = model(
                input_ids=biased_input_ids,
                attention_mask=biased_attention_mask
            )

            if hasattr(normal_outputs, "logits"):
                logits_normal = normal_outputs.logits
            else:
                logits_normal = normal_outputs

    

            with torch.no_grad():

                _, ablated_outputs = model(
                    input_ids=biased_input_ids,
                    attention_mask=biased_attention_mask,
                    imp_op = "remove",
                    imp_pos = ablation_kn_map
                )

                if hasattr(ablated_outputs, "logits"):
                    logits_ablated = ablated_outputs.logits
                else:
                    logits_ablated = ablated_outputs

       
            bias_output = bias_loss_function(
                logits_normal,
                logits_ablated,
                biased_labels
            )



            if isinstance(bias_output, dict):
                bias_loss = bias_output["loss"]
            else:
                bias_loss, _ = bias_output

   
            total_loss = (
                neutral_loss
                + lambda_bias * bias_loss
            )

  

            total_loss.backward()



            named_parameters = dict(
                model.named_parameters()
            )

            apply_gradient_mask(named_parameters, current_mask)
            saved_inactive_states, saved_inactive_values = save_opt_step(named_parameters, current_mask, optimizer=optimizer)
            
            optimizer.step()

            reset_opt_state(named_parameters, current_mask, optimizer, saved_inactive_values, saved_inactive_states)

            epoch_total_loss += (
                total_loss.detach().item()
            )

            epoch_bias_loss += (
                bias_loss.detach().item()
            )

            epoch_neutral_loss += (
                neutral_loss.detach().item()
            )

            num_batches += 1



        avg_total = (
            epoch_total_loss
            / max(num_batches, 1)
        )

        avg_bias = (
            epoch_bias_loss
            / max(num_batches, 1)
        )

        avg_neutral = (
            epoch_neutral_loss
            / max(num_batches, 1)
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"total={avg_total:.6f} | "
            f"neutral={avg_neutral:.6f} | "
            f"bias={avg_bias:.6f}"
        )

def evaluate_model(
    model,
    neutral_val_loader,
    bias_val_loader,
    biased_loss,
    neutral_loss,
    device,
    relation_based_map,

):

    eval_bias_loss = 0.0
    eval_neutral_loss = 0.0
    
    model.eval()
    neutral_iter = iter(neutral_val_loader)
    num_batch = 0
    with torch.no_grad():
        for bias_batch, relations in bias_val_loader:

            try:
                neutral_batch = next(neutral_iter)
            except StopIteration:
                neutral_iter = iter(neutral_val_loader)
                neutral_batch = next(neutral_iter)

            relation = relations[0]

            kn_flatten_map = flatten_kn_map(relation_based_map[relation], "union")

            biased_input = bias_batch['input_ids'].to(device)

            biased_attention = bias_batch['attention_mask'].to(device)

            biased_labels = bias_batch['labels'].to(device)

            neutral_input = neutral_batch['input_ids'].to(device)

            neutral_attention = neutral_batch['attention_mask'].to(device)

            neutral_label = neutral_batch['labels'].to(device)

            _, neutral_logits = model(input_ids = neutral_input, attention_mask = neutral_attention)

            loss_neutral = neutral_loss(
                neutral_logits.reshape(-1, neutral_logits.size(-1)),
                neutral_label.reshape(-1)
            )

            _, normal_logits = model(input_ids = biased_input, attention_mask = biased_attention)

            _, ablated_logits = model(input_ids = biased_input, attention_mask = biased_attention, imp_op = "remove", imp_pos = kn_flatten_map)

            bias_loss, _ = biased_loss(
                normal_logits,
                ablated_logits,
                biased_labels
            )

            eval_neutral_loss += loss_neutral.detach().item()

            eval_bias_loss += bias_loss.detach().item()

            num_batch+=1

    avg_neutral = eval_neutral_loss / num_batch

    avg_bias = eval_bias_loss / num_batch

    print(
        f"Total neutral loss = {avg_neutral:.6f} | "
        f"Total biased loss = {avg_bias:.6f}"
    )






def apply_gradient_mask(named_parameters, gradient_mask_opt):

    for name, param in named_parameters.items():
        if param.grad is None:
            continue

        if name in gradient_mask_opt:
            mask = gradient_mask_opt[name].to(
                device = param.grad.device,
                dtype = param.grad.dtype
            )
            param.grad.mul_(mask)

        else:
            param.grad = None

def save_opt_step(named_parameters, current_mask, optimizer):
    saved_inactive_values = {}
    saved_inactive_states = {}
    with torch.no_grad():

        for name, mask in current_mask.items():

            param = named_parameters[name]

            mask = mask.to(
                device=param.device,
                dtype=param.dtype
            )

            inverse_mask = 1.0 - mask

            saved_inactive_values[name] = (
                param.detach().clone()
            )

            state = optimizer.state.get(
                    param,
                    None
                    )

            if (state is not None and "exp_avg" in state):

                saved_inactive_states[name] = {
                    "exp_avg":
                        state[
                            "exp_avg"
                        ].detach().clone(),

                    "exp_avg_sq":
                        state[
                            "exp_avg_sq"
                        ].detach().clone()
                    }
    return saved_inactive_states, saved_inactive_values

def reset_opt_state(named_parameters, current_mask, optimizer, saved_inactive_values, saved_inactive_states):
    with torch.no_grad():

        for name, old_param in saved_inactive_values.items():

            param = named_parameters[name]

            mask = current_mask[name].to(
                    device=param.device,
                    dtype=param.dtype
                )

            inverse_mask = 1.0 - mask

            param.copy_(
                    param * mask
                        +
                    old_param * inverse_mask
                )

        for name, old_state in saved_inactive_states.items():

            param = named_parameters[name]

            state = optimizer.state[param]

            mask = current_mask[name].to(
                        device=state[
                            "exp_avg"
                        ].device,
                        dtype=state[
                            "exp_avg"
                        ].dtype
                    )

            inverse_mask = 1.0 - mask

            # exp_avg
            state["exp_avg"].copy_(
                    state["exp_avg"] * mask
                        +
                    old_state["exp_avg"]
                        * inverse_mask
                )

            # exp_avg_sq
            state["exp_avg_sq"].copy_(
                    state["exp_avg_sq"] * mask
                        +
                    old_state["exp_avg_sq"]
                        * inverse_mask
                )


def relation_based_fine_tuning(model_name, relation_based_map, general_kn_map, neutral_dataset, biased_dataset, device, num_epochs = 1, batch_size = 8):
    
    model = BertForMaskedLM.from_pretrained(model_name).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    neutral_loss = nn.CrossEntropyLoss(ignore_index=-100)

    neutral_loader = create_dataloaders(neutral_dataset, tokenizer, "neutral")

    bias_loader = create_dataloaders(biased_dataset, tokenizer, "biased")

    gradient_mask = create_gradient_mask(model, general_kn_map)

    for name, param in model.named_parameters():
        param.requires_grad = name in gradient_mask["union"]

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = 1e-5,
        weight_decay = 0.0
    )

    gradient_mask = create_relation_based_mask(model, relation_based_map)


    train_relation_based(model, bias_loader['train'], neutral_loader['train'], gradient_mask, relation_based_map, optimizer, bias_loss_function, neutral_loss, device, num_epochs=num_epochs)



def main():
    model_name = "bert-base-cased"

    kn_neurons_path = Path("results/bert-base-cased/kn")

    relation_ids = ["BR0" + str(x) for x in range(1, 10)]
    
    general_kn_neurons_map = create_mask_foundation(kn_neurons_path)

    relation_based_map = create_mask_relation_based(kn_neurons_path, relation_ids)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    neutral_data_path = "neutral_preservation_dataset"
    bias_data_path = "biased_dataset"
    
    neutral_data = load_dataset_from_disk(neutral_data_path)
    biased_data = load_dataset_from_disk(bias_data_path)

    relation_based_fine_tuning(
        model_name=model_name,
        relation_based_map=relation_based_map,
        general_kn_map=general_kn_neurons_map,
        neutral_dataset=neutral_data,
        biased_dataset=biased_data,
        device=device
    )

if __name__ == "__main__":
    main()
    


