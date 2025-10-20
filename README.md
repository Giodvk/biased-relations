# 🧠 Biased Neurons, Fairer Models

This repository contains the replication package for our study  
**“Tracing Stereotypes in Pre-trained Transformers: From Biased Neurons to Fairer Models”**  

The project extends the **Knowledge Neurons** framework (Dai et al., *ACL 2022*) to investigate how **social biases** are internally represented within pretrained transformers and how they can be **localized and mitigated** at the neuron level.  
Our work introduces the concept of **biased neurons**—neurons that encode stereotypical associations—and proposes a reproducible methodology to trace and suppress them, evaluating the trade-off between bias mitigation and performance on fairness-sensitive software engineering tasks.

---

## 🧩 Repository Structure

```
├── data/
│   ├── biased_relations/
│   ├── dataset_extraction/
│   └── README.md
│
├── results/
│   ├── bert-base-cased/
│   ├── bert-large-cased/
│   ├── ....
│   ├── analysis_rq1/
│   └── README.md
│
├── src/
│   ├── 1_analyze_mlm.py
│   ├── 2_get_kn.py
│   ├── ...
│   └── README.md
│
└── README.md
```

---

### 📁 **`data/`**
Contains the datasets of **biased relations** and **bias-activating prompts** used to trace and evaluate biased neurons.  
It includes:
- The nine bias categories (Age, Gender, Race/Color, etc.);
- Scripts for dataset extraction and processing;
- Documentation of the full generation pipeline.

See [`data/README.md`](./data/README.md) for details.

---

### 📊 **`results/`**
Stores all experimental outputs, summaries, and statistical analyses for the three research questions (RQ1–RQ3).  
Includes:
- Attribution and suppression results for each BERT model;
- Per-task evaluation data for SE tasks (e.g., sentiment, tone bearing, requirements classification);
- Scripts for data analysis and visualization.

See [`results/README.md`](./results/README.md) for a detailed description of each file.

---

### ⚙️ **`src/`**
Contains the complete source code implementing our adaptation of the Knowledge Neurons framework for bias tracing and suppression.  
Includes:
- Scripts for neuron attribution, refinement, and suppression;
- Plotting utilities for visual analysis;
- Evaluation scripts for SE downstream tasks.

See [`src/README.md`](./src/README.md) for usage instructions.

---
