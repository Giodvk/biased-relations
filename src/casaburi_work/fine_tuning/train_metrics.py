import torch
import torch.nn.functional as F


def get_target_tensors(logits, labels):

    target_mask = labels != -100

    target_logits = logits[target_mask]
    target_labels = labels[target_mask]

    batch_indices = target_mask.nonzero(
        as_tuple=False
    )[:, 0]

    return (
        target_logits,
        target_labels,
        batch_indices,
        target_mask
    )


def get_gold_scores(target_logits, target_labels):
 

    return target_logits.gather(
        1,
        target_labels.unsqueeze(1)
    ).squeeze(1)


def get_gold_probabilities(target_logits, target_labels):
 

    probs = F.softmax(
        target_logits,
        dim=-1
    )

    gold_probs = probs.gather(
        1,
        target_labels.unsqueeze(1)
    ).squeeze(1)

    return gold_probs


# ============================================================
# Ranking
# ============================================================

def get_gold_rank(target_logits, target_labels):


    gold_scores = get_gold_scores(
        target_logits,
        target_labels
    )

    rank = (
        target_logits
        > gold_scores.unsqueeze(1)
    ).sum(dim=1) + 1

    return rank


def get_rank_shift(
    logits_normal,
    logits_ablated,
    labels
):
 

    normal_logits, target_labels, _, _ = \
        get_target_tensors(
            logits_normal,
            labels
        )

    ablated_logits, _, _, _ = \
        get_target_tensors(
            logits_ablated,
            labels
        )

    rank_normal = get_gold_rank(
        normal_logits,
        target_labels
    )

    rank_ablated = get_gold_rank(
        ablated_logits,
        target_labels
    )

    rank_shift = (
        rank_ablated.float()
        - rank_normal.float()
    )

    return {
        "rank_normal": rank_normal,
        "rank_ablated": rank_ablated,
        "rank_shift": rank_shift
    }


def get_gold_margin(target_logits, target_labels):
  

    gold_scores = get_gold_scores(
        target_logits,
        target_labels
    )

    competitor_logits = target_logits.clone()

    competitor_logits.scatter_(
        1,
        target_labels.unsqueeze(1),
        float("-inf")
    )

    best_competitor = competitor_logits.max(
        dim=1
    ).values

    margin = (
        gold_scores
        - best_competitor
    )

    return margin


def get_logit_contribution(
    logits_normal,
    logits_ablated,
    labels
):
  

    normal_logits, target_labels, _, _ = \
        get_target_tensors(
            logits_normal,
            labels
        )

    ablated_logits, _, _, _ = \
        get_target_tensors(
            logits_ablated,
            labels
        )

    normal_score = get_gold_scores(
        normal_logits,
        target_labels
    )

    ablated_score = get_gold_scores(
        ablated_logits,
        target_labels
    )

    return normal_score - ablated_score


def get_probability_contribution(
    logits_normal,
    logits_ablated,
    labels
):


    normal_logits, target_labels, _, _ = \
        get_target_tensors(
            logits_normal,
            labels
        )

    ablated_logits, _, _, _ = \
        get_target_tensors(
            logits_ablated,
            labels
        )

    prob_normal = get_gold_probabilities(
        normal_logits,
        target_labels
    )

    prob_ablated = get_gold_probabilities(
        ablated_logits,
        target_labels
    )

    return {
        "prob_normal": prob_normal,
        "prob_ablated": prob_ablated,
        "prob_drop": prob_normal - prob_ablated
    }



def get_topk_predictions(
    target_logits,
    tokenizer=None,
    k=10
):
 
    values, indices = torch.topk(
        target_logits,
        k=k,
        dim=1
    )

    results = []

    for row_idx in range(
        target_logits.size(0)
    ):

        row = []

        for rank_idx in range(k):

            token_id = indices[
                row_idx,
                rank_idx
            ].item()

            item = {
                "rank": rank_idx + 1,
                "token_id": token_id,
                "logit": values[
                    row_idx,
                    rank_idx
                ].item()
            }

            if tokenizer is not None:
                item["token"] = (
                    tokenizer
                    .convert_ids_to_tokens(
                        token_id
                    )
                )

            row.append(item)

        results.append(row)

    return results

def kl_divergence_normal_ablated(
    logits_normal,
    logits_ablated,
    labels=None
):
  
    if labels is not None:

        logits_normal = logits_normal[
            labels != -100
        ]

        logits_ablated = logits_ablated[
            labels != -100
        ]

    log_p = F.log_softmax(
        logits_normal,
        dim=-1
    )

    log_q = F.log_softmax(
        logits_ablated,
        dim=-1
    )

    p = log_p.exp()

    kl = (
        p * (log_p - log_q)
    ).sum(dim=-1)

    return kl


def symmetric_kl_divergence(
    logits_a,
    logits_b,
    labels=None
):
  

    kl_ab = kl_divergence_normal_ablated(
        logits_a,
        logits_b,
        labels
    )

    kl_ba = kl_divergence_normal_ablated(
        logits_b,
        logits_a,
        labels
    )

    return 0.5 * (
        kl_ab + kl_ba
    )


def top1_stability(
    logits_normal,
    logits_ablated,
    labels
):
    

    normal_logits, _, _, _ = \
        get_target_tensors(
            logits_normal,
            labels
        )

    ablated_logits, _, _, _ = \
        get_target_tensors(
            logits_ablated,
            labels
        )

    pred_normal = normal_logits.argmax(
        dim=1
    )

    pred_ablated = ablated_logits.argmax(
        dim=1
    )

    same_prediction = (
        pred_normal == pred_ablated
    )

    return {
        "pred_normal": pred_normal,
        "pred_ablated": pred_ablated,
        "same_prediction": same_prediction
    }



def aggregate_per_example(
    values,
    labels,
    reduction="mean"
):
  

    target_mask = labels != -100

    batch_indices = target_mask.nonzero(
        as_tuple=False
    )[:, 0]

    batch_size = labels.size(0)

    values = values.float()

    result = torch.zeros(
        batch_size,
        device=values.device,
        dtype=values.dtype
    )

    counts = torch.zeros(
        batch_size,
        device=values.device,
        dtype=values.dtype
    )

    result.scatter_add_(
        0,
        batch_indices,
        values
    )

    counts.scatter_add_(
        0,
        batch_indices,
        torch.ones_like(values)
    )

    counts = counts.clamp_min(1.0)

    if reduction == "mean":
        result = result / counts

    elif reduction != "sum":
        raise ValueError(
            f"Unknown reduction: {reduction}"
        )

    return result


def compute_bias_diagnostics(
    logits_normal,
    logits_ablated,
    labels
):
 

    normal_logits, target_labels, _, _ = \
        get_target_tensors(
            logits_normal,
            labels
        )

    ablated_logits, _, _, _ = \
        get_target_tensors(
            logits_ablated,
            labels
        )

    # ---------- logits ----------
    gold_logit_normal = get_gold_scores(
        normal_logits,
        target_labels
    )

    gold_logit_ablated = get_gold_scores(
        ablated_logits,
        target_labels
    )

    logit_contribution = (
        gold_logit_normal
        - gold_logit_ablated
    )

    # ---------- probabilities ----------
    gold_prob_normal = get_gold_probabilities(
        normal_logits,
        target_labels
    )

    gold_prob_ablated = get_gold_probabilities(
        ablated_logits,
        target_labels
    )

    prob_drop = (
        gold_prob_normal
        - gold_prob_ablated
    )

    # ---------- ranks ----------
    rank_normal = get_gold_rank(
        normal_logits,
        target_labels
    ).float()

    rank_ablated = get_gold_rank(
        ablated_logits,
        target_labels
    ).float()

    rank_shift = (
        rank_ablated
        - rank_normal
    )

    relative_rank_shift = (
        rank_shift
        / rank_normal.clamp_min(1.0)
    )

    # ---------- margins ----------
    margin_normal = get_gold_margin(
        normal_logits,
        target_labels
    )

    margin_ablated = get_gold_margin(
        ablated_logits,
        target_labels
    )

    margin_shift = (
        margin_normal
        - margin_ablated
    )

    # ---------- KL ----------
    kl = kl_divergence_normal_ablated(
        logits_normal,
        logits_ablated,
        labels
    )

    sym_kl = symmetric_kl_divergence(
        logits_normal,
        logits_ablated,
        labels
    )

    # ---------- top1 ----------
    top1_info = top1_stability(
        logits_normal,
        logits_ablated,
        labels
    )

    same_top1 = (
        top1_info["same_prediction"]
        .float()
    )


    token_metrics = {
        "gold_logit_normal":
            gold_logit_normal,

        "gold_logit_ablated":
            gold_logit_ablated,

        "logit_contribution":
            logit_contribution,

        "gold_prob_normal":
            gold_prob_normal,

        "gold_prob_ablated":
            gold_prob_ablated,

        "prob_drop":
            prob_drop,

        "rank_normal":
            rank_normal,

        "rank_ablated":
            rank_ablated,

        "rank_shift":
            rank_shift,

        "relative_rank_shift":
            relative_rank_shift,

        "margin_normal":
            margin_normal,

        "margin_ablated":
            margin_ablated,

        "margin_shift":
            margin_shift,

        "kl":
            kl,

        "symmetric_kl":
            sym_kl,

        "same_top1":
            same_top1
    }



    example_metrics = {}

    for name, values in token_metrics.items():

        example_metrics[name] = \
            aggregate_per_example(
                values,
                labels,
                reduction="mean"
            )

  

    batch_metrics = {
        name: values.mean()
        for name, values
        in example_metrics.items()
    }

    batch_metrics[
        "rank_worsened_fraction"
    ] = (
        rank_shift > 0
    ).float().mean()

    batch_metrics[
        "rank_improved_fraction"
    ] = (
        rank_shift < 0
    ).float().mean()

    batch_metrics[
        "top1_preservation_rate"
    ] = same_top1.mean()

    return {
        "token": token_metrics,
        "example": example_metrics,
        "batch": batch_metrics
    }

def bias_loss_function(
    logits_normal,
    logits_ablated,
    labels
):
    """
    Loss:
        ReLU(
            gold_logit_normal
            -
            gold_logit_ablated
        )

    Media prima sui token del target,
    poi sugli esempi.
    """

    normal_logits, target_labels, _, _ = \
        get_target_tensors(
            logits_normal,
            labels
        )

    ablated_logits, _, _, _ = \
        get_target_tensors(
            logits_ablated,
            labels
        )

    target_scores_normal = get_gold_scores(
        normal_logits,
        target_labels
    )

    target_scores_ablated = get_gold_scores(
        ablated_logits,
        target_labels
    )

    bias_contribution = (
        target_scores_normal
        - target_scores_ablated
    )

    penalty = F.relu(
        bias_contribution
    )

    loss_per_example = aggregate_per_example(
        penalty,
        labels,
        reduction="mean"
    )

    contribution_per_example = \
        aggregate_per_example(
            bias_contribution,
            labels,
            reduction="mean"
        )

    loss = loss_per_example.mean()

    bias_contribution_mean = (
        contribution_per_example.mean()
    )

    return loss, bias_contribution_mean


