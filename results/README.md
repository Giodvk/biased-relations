# 📊 Results Directory

This folder contains all the experimental results and analysis scripts for the study **“Biased Neurons, Fairer Models: Tracing and Suppressing Stereotypes in Transformers.”**
It includes raw outputs, processed summaries, and per-model artifacts used to evaluate the effect of biased-neuron suppression across bias categories and software engineering (SE) tasks.

---

## 🧠 Overview

Experiments were organized around the **three research questions (RQ1–RQ3)** defined in the paper:

| RQ            | Focus                                           | Main Output                                                     |
| ------------- | ----------------------------------------------- | --------------------------------------------------------------- |
| **RQ1** | Identification of biased neurons                | `kn/` + `analyzed_kn.json`                                  |
| **RQ2** | Effect of neuron suppression on bias expression | `erasing_summary_BR*.json`                                    |
| **RQ3** | Impact of suppression on downstream SE tasks    | Task-specific folders (`*_incivility`, `*_sentiment`, etc.) |

Each subfolder corresponds to a model–task combination or to a specific analysis stage.

---

## 🗂️ Folder Structure

### **Top-level folders**

- **`analysis_rq1/`, `analysis_rq2/`, `analysis_rq3/`** 
  Contain results, visualizations and aggregated CSVs for each research question.
- **`bert-base-cased/`, `bert-base-uncased/`, `bert-large-cased/`, `bert-large-uncased/, bert-large-cased_<task>/`** 
  Store neuron attribution and suppression results for each BERT variant tested.
- **`se_tasks_benchmark/`**
  Contain task-specific evaluation results for **RQ3** on fairness-sensitive SE tasks. Includes baseline results (without suppression) and comparative metrics.

---

## 📁 Folder Contents (Example: `bert-large-cased_requirement-completion/`)

Each task folder contains:

- **`all.args.json`** — configuration of the experiment (hyperparameters, dataset paths, model name, and timestamp).
- **`analyzed_kn.json`** — summary of the biased neurons identified per relation (output of the refinement phase).
- **`erasing_summary_BR0X.json`** — results of neuron suppression for each of the nine bias categories (`BR01`–`BR09`).Each JSON file includes:
  ```json
  {
    "relation": "BR01",
    "original_accuracy": 0.82,
    "erased_accuracy": 0.79,
    "original_ppl": 17.3,
    "erased_ppl": 21.4,
    "ppl_increase_ratio": 1.24,
    "num_kn_rel": 19,
    "time": 265.78
  }
  ```

  These files are used to compute the perplexity increase and selectivity metrics described in RQ2.

- **`erasing_report/`** — detailed results of the erasing experiments, including additional visualizations and CSVs.
- **`figs/ and figs_pretty/`** — additional plots (perplexity trends, neuron distribution, accuracy deltas) describing the results of the biased neuron identification process (RQ1).
- **`kn/`** — raw neuron attribution data (`layer`, `neuron_index`, `attribution_score`) for each relation.

---

## 📜 Analysis Scripts

The root folder includes Python scripts for post-processing and statistical analysis:

- **`RQ1_data_analysis.py`** — Aggregates neuron attribution data (`kn/`) to compute the number and layer distribution of biased neurons per model.
- **`RQ2_data_analysis.py`** — Reads all `erasing_summary_BR*.json` files to compute perplexity increase, selectivity, and generate comparative plots.
- **`RQ3_data_analysis.py`** — Evaluates performance deltas on downstream SE tasks, correlating fairness improvement with task degradation.

Each script outputs both CSV summaries and visual plots (saved under `analysis_rq*/`).

---

## 🧩 Bias Categories (BR01–BR09)

| Code | Category             |
| ---- | -------------------- |
| BR01 | Age                  |
| BR02 | Disability           |
| BR03 | Gender               |
| BR04 | Nationality          |
| BR05 | Physical Appearance  |
| BR06 | Race/Color           |
| BR07 | Religion             |
| BR08 | Sexual Orientation   |
| BR09 | Socioeconomic Status |

---

## 🔁 Reproducibility Notes

- **Models tested:** `bert-base-cased`, `bert-base-uncased`, `bert-large-cased`, `bert-large-uncased`, plus fine-tuned versions.
- **Tasks evaluated:** Incivility, Tone Bearing, Sentiment, Requirement Type, Requirement Completion.
- **Evaluation environment:** Python 3.10, PyTorch 2.2, Transformers 4.40.

All configuration files (`all.args.json`) record seed, GPU, and timestamp for reproducibility.



