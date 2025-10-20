# 📁 Data Folder

This directory contains the dataset of **biased relations** and **bias-activating prompts** used in our study.  
It is organized as follows:

- **`biased_relations/`**  
  Contains the complete dataset and supporting scripts, organized into:
  - **`BR01`–`BR09`** — Nine subfolders corresponding to each bias category (e.g., Age, Gender, Race/Color).  
    Each folder includes the extracted biased relations and their corresponding activating prompts.  
  - **`crowspairs/`** — The base dataset of stereotyped sentences used as the source for biased relation extraction.  
  - **`biased_relations_all_bags/`** — A merged version containing all nine bias categories in a single structured dataset.  
  - **Data processing scripts** — Python utilities for dataset extraction, cleaning, and formatting.

- **`dataset_extraction/`**  
  Contains documentation and intermediate files describing the full extraction workflow used to create the biased relations dataset, including prompt templates and validation steps.

