# ClinSyn

CLINSYN: A Framework for Ontology-Guided
Robustness Evaluation of LLMs on Clinical
Question Answering

This repository provides the research artifact accompanying the ClinSyn paper.
The pipeline evaluates robustness to ontology-guided synonym substitutions of
**disease** mentions using the
[ReMedQA](https://huggingface.co/datasets/disi-unibo-nlp/ReMedQA) benchmark, the
[MONDO](https://mondo.monarchinitiative.org/) disease ontology, and
[MedGemma-4B](https://huggingface.co/google/medgemma-4b-it).

## Setup

Install the environment with `uv` and download MONDO before running the notebooks. 
A CUDA-capable GPU is recommended for model inference.

```bash
uv sync
mkdir -p data
curl -L -o data/mondo.owl https://purl.obolibrary.org/obo/mondo.owl
```

For MedGemma, obtain access to the model on Hugging Face and set `HF_TOKEN`.

```bash
export HF_TOKEN="<your-hugging-face-token>"
```

## Run the pipeline

Execute the following stages in order:

`generate_inputs.ipynb` → `gpu_driver.py` → `calculate_metrics.ipynb`

### 1. Generate inputs

```bash
uv run --with jupyterlab jupyter lab generate_inputs.ipynb
```

The notebook loads the ReMedQA multiple-choice questions, identifies disease
mentions, retrieves and filters MONDO synonyms, and generates transformed
questions by substituting disease mentions. It saves the original questions to
`data/medqa_mcq_with_disease_mentions.json` and the transformed questions to
`data/tr_medqa_mcq_with_disease_mentions.json`, together with a synonym mapping.

### 2. Run inference

Run `gpu_driver.py` once for the original questions and once for the transformed
questions:

```bash
INPUT_PATH=data/medqa_mcq_with_disease_mentions.json uv run python gpu_driver.py
INPUT_PATH=data/tr_medqa_mcq_with_disease_mentions.json uv run python gpu_driver.py
```

The script saves model responses and extracted answers under
`results/medgemma-4b-it/`.

### 3. Compute metrics

```bash
uv run --with jupyterlab jupyter lab calculate_metrics.ipynb
```

The notebook reads the saved predictions and reports original and transformed
accuracy, along with three robustness metrics:

- **Flip rate (FR):** fraction of matched original–transformed pairs with
  different predicted answers.
- **Negative flip rate (NFR):** fraction of matched pairs that change from a
  correct original answer to an incorrect transformed answer.
- **Positive flip rate (PFR):** fraction of matched pairs that change from an
  incorrect original answer to a correct transformed answer.

Predictions whose answers cannot be parsed are counted and excluded before
evaluation. Original accuracy is reported for both the full retained original
dataset and the subset with retained transformed counterparts.

<!--
## Citation

If you use this code, please cite the paper.

```bibtex
@inproceedings{clinsyn,
  author    = {Dola, Swaroopa and Krishnamurthy, Ashok},
  title     = {ClinSyn: A Framework for Ontology-Guided Robustness Evaluation
               of LLMs on Clinical Question Answering},
  booktitle = {IEEE International Conference on Bioinformatics and Biomedicine
               (BIBM) Workshops, Preprint},
  year      = {2026}
}
```
-->