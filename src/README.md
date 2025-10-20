# 🧩 Biased Neurons — Source Code

This folder contains the source code used in our study **“Biased Neurons, Fairer Models: Tracing and Suppressing Stereotypes in Transformers.”**
The codebase is **originally forked from the official implementation of [Dai et al., 2022 — *Knowledge Neurons in Pretrained Transformers*](https://github.com/Hunter-DDM/knowledge-neurons)** and has been **extended and modified** to adapt the methodology from *factual knowledge neurons* to **biased neurons**.

---

## ⚖️ Acknowledgment and Credits

Our implementation builds upon the original Knowledge Neurons framework:

> **D. Dai, L. Dong, Y. Hao, Z. Sui, B. Chang, and F. Wei.**
> *Knowledge Neurons in Pretrained Transformers.*
> *ACL 2022, Dublin, Ireland.*
> [https://github.com/Hunter-DDM/knowledge-neurons](https://github.com/Hunter-DDM/knowledge-neurons)

We sincerely thank the authors for releasing their code, which provided the foundation for this work.
All original file structures and execution scripts (`1_run_mlm.sh`, `2_run_kn.sh`, etc.) have been retained and adapted to support bias-oriented analysis.

If you use this modified version, please cite both the original ACL-2022 paper and our work.

---

## ⚙️ Code Usage

> Only the scripts actively used in this work are documented below.
> The remaining original utilities from Dai et al. are retained for completeness but were **not used** in our experiments.

### **Step 1 — Analyze Masked Language Model (Attribution Scores)**

Compute neuron-level attribution scores for each biased relation:

```bash
python 1_analyze_mlm.py
```

This script runs masked-language-model inference for all bias-activating prompts, using Integrated Gradients to compute attribution scores per neuron.
Outputs are stored as `*.rlt.jsonl` files inside each model’s results folder (e.g., `../results/bert-large-cased/`).

---

### **Step 2 — Identify and Refine Biased Neurons**

Aggregate and refine the neurons most responsible for biased predictions:

```bash
python 2_get_kn.py
python 2_analyze_kn.py
```

- `2_get_kn.py` filters and merges neuron-importance scores across prompts to identify candidate **biased neurons**, saving them in:
  - `kn/kn_bag-<relation>.json`
  - `kn/kn_rel-<relation>.json`
- `2_analyze_kn.py` summarizes these results across all relations, generating layer-wise statistics.

---

### **Step 3 — Modify Neuron Activations**

Test the causal role of the identified neurons by directly modifying their activations:

```bash
python 3_modify_activation.py
```

This script simulates neuron editing (as in *Dai et al.*, 2022) and measures how activation changes affect model predictions.

---

### **Step 4 — Edit Knowledge (Bias Transfer Experiment)**

Perform localized editing of neuron weights to evaluate transferability and robustness:

```bash
python 6_edit_knowledge_sampled.py
```

This script replaces biased associations with neutral or counterfactual ones, measuring changes in probability, rank, and perplexity.
It outputs per-relation metrics such as mean reciprocal rank (MRR) and perplexity deltas.

---

### **Step 5 — Erase Biased Knowledge**

Suppress neurons encoding biased associations and evaluate their effect:

```bash
python 7_erase_knowledge.py   --data_path ../data/biased_relations/biased_relations_all_bags.json   --tmp_data_path ../data/biased_relations/tmp.json   --bert_model bert-large-cased   --output_dir ../results/bert-large-cased/   --kn_dir ../results/bert-large-cased/kn/   --pt_relation BR01   --gpus 0
```

This produces performance and perplexity metrics both on **target** (biased) and **control** relations.

---

### **Step 6 — Save and Summarize Results**

Export compact per-relation summaries and **model weights after erasure (suppression checkpoints)**:

```bash
python 7b_save_erased_rels.py
```

Each `erasing_summary_BR*.json` file records:

```json
{
  "relation": "BR01",
  "original_accuracy": 0.82,
  "erased_accuracy": 0.79,
  "ppl_increase_ratio": 1.24,
  "num_kn_rel": 19,
  "time": 265.78
}
```

These outputs are later aggregated for RQ2 and RQ3 analysis.

---

### **Step 7 — Plot Generation and Visualization**

After all suppression experiments are completed, the following scripts produce visual reports and figures.

#### a. Combined Summary and Suppression Metrics
```bash
python 8_new_suppression_plots.py
```
Aggregates all `erasing_summary_BR*.json` files, merges them into per-model summaries, computes statistics, and generates:  
- Perplexity and accuracy before/after suppression  
- Accuracy ratio and perplexity increase plots  
- Correlation between neuron counts and perplexity variation  
All outputs are saved in `../results/<model>/erasing_report/`.

#### b. Activation Effect Visualization (Baseline vs Ours)
```bash
python 8_plot_fig_erase_enhance.py
```
Generates comparative bar charts for **activation suppression** and **amplification**, contrasting our neuron editing method against the baseline.  
Figures are saved in each model’s `/figs/` folder.  
To run across all models:
```bash
bash 8_run_plot.sh
```

#### c. Alternative Visualization Styles
```bash
python 8_alternative_plots.py
```
Produces slope or horizontal bar charts with simplified aesthetics.  
Outputs are stored in `/figs_pretty/`.

#### d. Distant Activation Analysis
```bash
python 8_plot_fig_distant.py
```
Visualizes neuron activations across different prompt types (head-tail, head-only, random).  
Generates per-relation bar plots showing average activation values.

---

### **Step 8 — Downstream SE Task Evaluation**

Once suppression checkpoints (`kn_erase-<BR>.pt`) are generated, evaluate their impact on fairness-sensitive SE tasks.

#### a. Masked Language Modeling (MLM) Tasks
```bash
python 9_evaluate_MLM_task.py
```
Evaluates **requirement completion**, a masked language modeling SE task.  
Computes token-level metrics (accuracy, perplexity) and saves to CSV.

#### b. Binary Classification Tasks
```bash
python 9_evaluate_binary_task.py
```
Evaluates binary SE tasks such as **incivility** and **tone bearing**.  
Logs accuracy, F1-macro, and confusion matrices.

#### c. Multiclass Classification Tasks
```bash
python 9_evaluate_multiclass_task.py
```
Evaluates multiclass SE tasks such as **sentiment analysis** and **requirement type classification**.  
Outputs accuracy, F1, and confusion matrices to CSV.

---

## ⚙️ Environment

Install dependencies (Python 3.10+):

```bash
pip install -r requirements.txt
```

---
