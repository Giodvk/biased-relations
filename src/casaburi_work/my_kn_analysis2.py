
import logging
import os
import torch
import random
import numpy as np
import json, jsonlines
import time
from tqdm import tqdm  
from transformers import AutoTokenizer
from custom_bert import BertForMaskedLM
import torch.nn.functional as F

# set logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def example2feature(example, max_seq_length, tokenizer):
    """Convert an example into input features"""
    features = []
    tokenslist = []


    prompt = example[0]
    target = example[1]
    relation = example[2]

    target_tokens = tokenizer.tokenize(target)

    if len(target_tokens) == 0:
        raise ValueError(f"Target without tokens: {target}")

    multi_mask = " ".join([tokenizer.mask_token] * len(target_tokens))

    prompt = prompt.replace(tokenizer.mask_token, multi_mask, 1)
    

    ori_tokens = tokenizer.tokenize(prompt)

    if len(ori_tokens) > max_seq_length - 2:
        ori_tokens = ori_tokens[:max_seq_length - 2]

    # add special tokens
    tokens = ["[CLS]"] + ori_tokens + ["[SEP]"]
    base_tokens = ["[UNK]"] + ["[UNK]"] * len(ori_tokens) + ["[UNK]"]
    segment_ids = [0] * len(tokens)

    # Generate id and attention mask
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    baseline_ids = tokenizer.convert_tokens_to_ids(base_tokens)
    input_mask = [1] * len(input_ids)

    # Pad [PAD] tokens (id in BERT-base-cased: 0) up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    baseline_ids += padding
    segment_ids += padding
    input_mask += padding

    assert len(baseline_ids) == max_seq_length
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    features = {
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
        'baseline_ids': baseline_ids,
    }
    tokens_info = {
    "tokens": tokens,
    "relation": relation,
    "gold_obj": target,
    "gold_tokens": target_tokens,
    "pred_obj": None
}
    return features, tokens_info


def scaled_input(emb, batch_size, num_batch):
    # emb: (1, ffn_size)
    baseline = torch.zeros_like(emb)  # (1, ffn_size)

    num_points = batch_size * num_batch
    step = (emb - baseline) / num_points  # (1, ffn_size)

    res = torch.cat([torch.add(baseline, step * i) for i in range(num_points)], dim=0)  # (num_points, ffn_size)
    return res, step[0]


def convert_to_triplet_ig(ig_list):
    ig_triplet = []
    ig = np.array(ig_list)  # 12, 3072
    max_ig = ig.max()
    for i in range(ig.shape[0]):
        for j in range(ig.shape[1]):
            if ig[i][j] >= max_ig * 0.1:
                ig_triplet.append([i, j, ig[i][j]])
    return ig_triplet


def run(args):
    # set device
    if args['no_cuda'] or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    elif len(args['gpus']) == 1:
        device = torch.device("cuda:%s" % args['gpus'])
        n_gpu = 1
    else:
        # !!! to implement multi-gpus
        pass
    print("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))

    # set random seeds
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args['seed'])

    # save args
    os.makedirs(args['output_dir'], exist_ok=True)
    json.dump(args, open(os.path.join(args['output_dir'], args['output_prefix'] + '.args.json'), 'w'), sort_keys=True, indent=2)

    print(args['bert_model'])
    
    # init tokenizer
    tokenizer_name = args['bert_model']
    if "ModernBERT-large" in tokenizer_name:
        tokenizer_name = "answerdotai/ModernBERT-large"
    if "ModernBERT-base" in tokenizer_name:
        tokenizer_name = "answerdotai/ModernBERT-base"
        
    def load_tokenizer(model_name: str):
        try:
            return AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        except Exception:
            # fall back to local snapshot if needed
            from huggingface_hub import snapshot_download
            local_dir = snapshot_download(repo_id=model_name)
            return AutoTokenizer.from_pretrained(local_dir, use_fast=True, trust_remote_code=True)
            
    tokenizer = load_tokenizer(tokenizer_name)
    
    # Load pre-trained BERT
    print("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    if "modernbert" in args['bert_model'].lower():
        print("non pronto")
    else:
        model = BertForMaskedLM.from_pretrained(args['bert_model'])
    model.to(device)

    # data parallel
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    # prepare eval set
    if os.path.exists(args['tmp_data_path']):
        with open(args['tmp_data_path'], 'r') as f:
            eval_bag_list_perrel = json.load(f)
    else:
        with open(args['data_path'], 'r') as f:
            eval_bag_list_all = json.load(f)
        # split bag list into relations
        eval_bag_list_perrel = {}
        for bag_idx, eval_bag in enumerate(eval_bag_list_all):
            bag_rel = eval_bag[0][2].split('(')[0]
            if bag_rel not in eval_bag_list_perrel:
                eval_bag_list_perrel[bag_rel] = []
            if args['debug'] > 0 and len(eval_bag_list_perrel[bag_rel]) >= args['debug']:
                continue
            eval_bag_list_perrel[bag_rel].append(eval_bag)
        with open(args['tmp_data_path'], 'w') as fw:
            json.dump(eval_bag_list_perrel, fw, indent=2)
            
    timing_log = []
    
    # evaluate args.debug bags for each relation
    for relation, eval_bag_list in eval_bag_list_perrel.items():
        if args.get('pt_relation', None) is not None and relation != args['pt_relation']:
            continue
            
        # record running time
        tic = time.perf_counter()
        with jsonlines.open(os.path.join(args['output_dir'], args['output_prefix'] + '-' + relation + '.rlt' + '.jsonl'), 'w') as fw:
            
            # --- INIZIO MODIFICA ---
            # 1. Calcoliamo il limite: se debug è > 0 usiamo quello, altrimenti prendiamo tutta la lista
            limit = args['debug'] if args['debug'] > 0 else len(eval_bag_list)
            
            # 2. Tagliamo la lista in memoria. Il file su disco NON viene toccato!
            eval_bag_list_to_run = eval_bag_list[:limit]
            
            # 3. Avvolgiamo la lista tagliata nella barra di caricamento (tqdm)
            progress_bar = tqdm(eval_bag_list_to_run, desc=f"Elaborazione [{relation}]")
            
            # 4. Facciamo il ciclo sulla nuova progress_bar invece che su eval_bag_list
            for bag_idx, eval_bag in enumerate(progress_bar):
            # --- FINE MODIFICA ---
            
                res_dict_bag = []
                for eval_example in eval_bag:
                    eval_features, tokens_info = example2feature(eval_example, args['max_seq_length'], tokenizer)
                    
                    # convert features to long type tensors
                    baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
                    baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0).to(device)
                    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
                    input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0).to(device)
                    segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0).to(device)

                    # record real input length
                    input_len = int(input_mask[0].sum())

                    # record [MASK]'s position
                    mask_token = tokenizer.mask_token
                    tokens_stripped = [t.strip() for t in tokens_info['tokens']]

                    tgt_positions = [idx for idx, token in enumerate(tokens_stripped) if token == mask_token]

                    if len(tgt_positions) == 0:
                        continue

                    gold_labels = tokenizer.convert_tokens_to_ids(tokens_info['gold_tokens'])

                    if len(tgt_positions) != len(gold_labels):
                        raise ValueError(
                        f"Mismatch: {len(tgt_positions)} masks "
                        f"but {len(gold_labels)} gold tokens "
                        f"for target '{tokens_info['gold_obj']}'"
                        )

                    # record various results
                    res_dict = {
                        'pred': [],
                        'ig_pred': [],
                        'ig_gold': [],
                        'base': []
                    }

                    # original pred prob
                    if args['get_pred']:
                        _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_positions, tgt_layer=0)  # (1, n_vocab)
                        base_pred_prob = F.softmax(logits, dim=1)  # (1, n_vocab)
                        res_dict['pred'].append(base_pred_prob.tolist())

                    # Loop su tutti i layer (es. 12 per base, 24 per large)
                    for tgt_layer in range(model.bert.config.num_hidden_layers):

                        token_ig_gold = []
                        token_base = []

                        for tgt_pos, gold_label in zip(tgt_positions, gold_labels):

                            ffn_weights, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer)  # (1, ffn_size), (1, n_vocab)
                            scaled_weights, weights_step = scaled_input(ffn_weights, args['batch_size'], args['num_batch'])  # (num_points, ffn_size), (ffn_size)
                            scaled_weights.requires_grad_(True)

                            # integrated grad at the gold label for each layer (QUESTO SCOVA I BIAS)
                            if args['get_ig_gold']:
                                ig_gold_token = None
                            for batch_idx in range(args['num_batch']):
                                batch_weights = scaled_weights[batch_idx * args['batch_size']:(batch_idx + 1) * args['batch_size']]
                                _, grad = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer, tmp_score=batch_weights, tgt_label=gold_label)  # (batch, n_vocab), (batch, ffn_size)
                                grad = grad.sum(dim=0)  # (ffn_size)
                                ig_gold_token = grad if ig_gold_token is None else torch.add(ig_gold_token, grad)  # (ffn_size)

                            ig_gold_token = ig_gold_token * weights_step  # (ffn_size)

                            token_ig_gold.append(ig_gold_token)

                        if args['get_ig_gold']:
                            ig_gold_layer = torch.stack(token_ig_gold, dim=0).mean(dim=0)
                            res_dict['ig_gold'].append(ig_gold_layer.tolist())

                        # base ffn_weights for each layer
                        if args['get_base']:
                            res_dict['base'].append(ffn_weights.squeeze().tolist())

                    if args['get_ig_gold']:
                        res_dict['ig_gold'] = convert_to_triplet_ig(res_dict['ig_gold'])
                    if args['get_base']:
                        res_dict['base'] = convert_to_triplet_ig(res_dict['base'])

                    res_dict_bag.append([tokens_info, res_dict])

                fw.write(res_dict_bag)

        # record running time
        toc = time.perf_counter()
        print(f"\n***** Relation: {relation} evaluated. Costing time: {toc - tic:0.4f} seconds *****\n")
        timing_log.append({
                "relation": relation,
                "costing_time_seconds": round(toc - tic, 4)
            })
            
    return timing_log

if __name__ == "__main__":
    
    model_list = [
        #"bert-base-cased",
        "bert-large-cased",
        "bert-base-uncased",
        # "bert-large-uncased",
        # FINETUNED models su base bert-large-cased:
        # "aieng-lab/bert-large-cased_requirement-completion",
        # "aieng-lab/bert-large-cased_incivility",
        # "aieng-lab/bert-large-cased_tone-bearing",
        # "aieng-lab/bert-large-cased_sentiment",
        # "aieng-lab/bert-large-cased_requirement-type",
    ]

    timing_log = []
    
    for model_name in model_list:
        # Crea le directory in base a se è un modello HF o una cartella locale
        r_dir = f"../results/{model_name.split('/')[1] if len(model_name.split('/')) > 1 else model_name}"
        print(f"\n=========================================================")
        print(f" Avvio Elaborazione Modello: {model_name}")
        print(f"=========================================================\n")
        
        args = {
                "data_path": "",
                "tmp_data_path": "../data/biased_relations/biased_relations_all_bags.json",
                "bert_model": model_name,
                "output_dir": r_dir,
                "output_prefix": "all",
                "max_seq_length": 128,
                "do_lower_case": True if 'uncased' in model_name else False,
                "no_cuda": False,
                "gpus": "0",  # <--- CAMBIATO DA "1" A "0" (Usa la prima e unica GPU)
                "seed": 42,
                "debug": 100000, # Per testare senza OOM metti 10, poi ripristina a 100000
                "get_ig_gold": True,
                "get_ig_pred": False,
                "get_pred": False,
                "get_base": False,
                # --- CONFIGURAZIONE GPU BILANCIATA (RTX 5070 Ti) ---
                "batch_size": 20,
                "num_batch": 1  # Step totali = 20
            }
            
        # Start timing
        tic = time.perf_counter()
        timing = run(args)
        toc = time.perf_counter()
        
        timing_log.append({
            "model": model_name,
            "timing": timing
        })
        
    # Save timing log to JSON
    with open("run_timing_log.json", "w", encoding="utf-8") as f:
        json.dump(timing_log, f, indent=2)