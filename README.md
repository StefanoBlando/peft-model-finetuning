# Lightweight Fine-Tuning: Sentiment Analysis with PEFT & LoRA

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**A Parameter-Efficient Fine-Tuning (PEFT) system for Sentiment Analysis using DistilBERT and the SST-2 dataset.**

This repository contains a complete pipeline to adapt large foundation models to specific sequence classification tasks without the massive computational cost of full fine-tuning. By utilizing **LoRA (Low-Rank Adaptation)**, we only train a tiny fraction of the model's parameters (less than 1%) while achieving state-of-the-art performance improvements.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [System Architecture](#system-architecture)
4. [Dataset & Preprocessing](#dataset--preprocessing)
5. [PEFT Configuration & Training](#peft-configuration--training)
6. [Evaluation & Performance Comparison](#evaluation--performance-comparison)
7. [Installation & Usage](#installation--usage)
8. [License](#license)

## Project Overview
This project focuses on **Lightweight Fine-tuning**, an essential technique for adapting large foundation models without the need for substantial computational resources. By using the Hugging Face `peft` library, the process modifies a pre-trained model for sequence classification while maintaining a very low memory footprint.

The workflow involves:
* Loading a pre-trained foundation model and evaluating its initial "zero-shot" performance.
* Applying Parameter-Efficient Fine-Tuning (PEFT) using **LoRA (Low-Rank Adaptation)**.
* Performing inference and comparing the optimized model's metrics against the original version.

## Repository Structure
The project is organized into modular components to facilitate scalability and testing:

```text
peft-model-finetuning/
├── example/
│   └── inference_demo.py         # Script for testing the fine-tuned model
├── notebooks/
│   └── LightweightFineTuning.ipynb # Core development and analysis notebook
├── src/                          # Source code for the training pipeline
│   ├── config_manager.py         # Hyperparameter and config logic
│   ├── evaluator.py              # Performance evaluation functions
│   ├── lora_trainer.py           # Core LoRA implementation
│   ├── model_loader.py           # Model and Tokenizer loading utilities
│   ├── qlora_trainer.py          # Quantized LoRA experimental logic
│   ├── run_analysis.py           # Automated analysis scripts
│   └── visualization.py          # Metrics and loss curve plotting
├── LICENSE                       # GNU GPLv3 License
├── README.md                     # Project documentation
├── requirements.txt              # Project dependencies
├── setup.py                      # Package installation script
└── training_config.yaml          # YAML-based training parameters

```
## System Architecture
The system follows a modular pipeline designed for sequence classification. It integrates:
* **Foundation Model:** `distilbert-base-uncased` from Hugging Face, chosen for its efficiency in sequence classification tasks.
* **PEFT Framework:** The Hugging Face `peft` library, which allows for training a subset of parameters.
* **LoRA Mechanism:** A technique that injects low-rank trainable matrices into the Transformer layers, specifically targeting attention modules.



## Dataset & Preprocessing
The project uses the **SST-2 (Stanford Sentiment Treebank)** dataset, a benchmark for binary text classification:
* **Data Source:** Loaded directly from the Hugging Face `datasets` library.
* **Tokenizer:** `AutoTokenizer` was used to prepare the text into model-compatible input IDs.
* **Optimization:** Evaluation was conducted on a balanced subset to ensure statistical reliability while managing computational constraints.

## PEFT Configuration & Training
Creating a PEFT model requires a specific configuration to define which parts of the foundation model will be adapted.

### LoraConfig
A `LoraConfig` was instantiated with the following key hyperparameters to balance complexity and performance:
* **r (Rank):** 16, defining the rank of the update matrices.
* **lora_alpha:** 32, the scaling factor for the adaptation.
* **target_modules:** Specifically targeting `q_lin` and `v_lin` (Query and Value layers) to capture attention-based sentiment patterns.
* **lora_dropout:** 0.1, to ensure better generalization during the fine-tuning process.

```python
# Initializing the PEFT model
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    task_type="SEQ_CLS", 
    r=16, 
    lora_alpha=32, 
    target_modules=["q_lin", "v_lin"],
    lora_dropout=0.1,
    bias="none"
)

# Converting the base model into a PEFT model
lora_model = get_peft_model(base_model, config)
lora_model.print_trainable_parameters()
```

## Evaluation & Performance Comparison

The effectiveness of the fine-tuning process was evaluated by comparing the "frozen" foundation model against the PEFT-optimized version. Following the reviewer's feedback, multiple classification metrics were calculated to ensure the model's robustness and reliability.



### Key Metrics Comparison
The performance leap was significant, demonstrating that adding a minimal fraction of trainable parameters is sufficient to "teach" the model a specific task:

* **Accuracy**: Increased from **43.1%** (base model) to **84.2%** (fine-tuned model).
* **F1-Score**: Reached **0.827**, indicating a strong balance between Precision and Recall.
* **Precision & Recall**: Both metrics exceeded **83%**, confirming the model's ability to correctly identify both positive and negative sentiments.

### Parameter Efficiency
By using **LoRA**, we achieved these results by training only a tiny portion of the total model architecture:
* **Total Parameters**: ~124 Million.
* **Trainable Parameters**: ~295,000 (~0.23% of the total).
* **Advantage**: Drastic reduction in VRAM consumption and significantly faster training times.

## Installation & Usage

### 1. Prerequisites
Ensure you have Python 3.8+ installed. It is highly recommended to use a virtual environment or Conda to manage your dependencies and avoid conflicts.

### 2. Clone the Repository
```bash
git clone [https://github.com/StefanoBlando/peft-model-finetuning.git](https://github.com/StefanoBlando/peft-model-finetuning.git)
cd peft-model-finetuning
```
### nstall Dependencies

The project includes a requirements.txt for all necessary libraries and a setup.py for local installation:
```bash 
pip install -r requirements.txt
pip install -e 

```
### 4. Running the Project
The repository offers multiple ways to interact with the models:

* **Notebook Analysis:** Open `notebooks/LightweightFineTuning.ipynb` to follow the complete process of loading DistilBERT, configuring LoRA, and training the adapter.
* **Inference Demo:** Use the standalone script to run the fine-tuned model on custom text prompts:
  ```bash
  python example/inference_demo.py
  ```
 * ** Configuration: Fine-tuning parameters and paths can be adjusted directly in the training_config.yaml file to experiment with different ranks (r) or alpha values.

## License

This project is licensed under the **GNU General Public License v3.0**. 

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

See the [LICENSE](LICENSE) file for the full legal text.

---

## Acknowledgments

This project was developed as part of the **Generative AI Nanodegree** program.

Special thanks to:
* **Foundation Model**: `distilbert-base-uncased` from Hugging Face for sequence classification tasks.
* **PEFT Library**: For providing the tools to implement LoRA and efficient model adaptation.
* **Reviewer Team**: For the valuable feedback on calculating balanced metrics like F1-Score, Precision, and Recall to ensure robust model evaluation.
