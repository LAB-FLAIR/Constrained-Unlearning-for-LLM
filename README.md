# Constrained Unlearning for LLM via Knowledge Isolation.

Code repository for the paper: **SHA256 at SemEval-2025 Task 4: Selective Amnesia – Constrained Unlearning for Large Language Models via Knowledge Isolation**

Authors: Saransh Agrawal, Kuan-Hao Huang (Texas A&M University)

## Abstract

Large language models (LLMs) frequently memorize sensitive information, posing privacy risks. Current unlearning methods often struggle to selectively remove specific data associations without degrading overall model capabilities. This paper presents our solution to SemEval-2025 Task 4 on targeted unlearning, introducing a two-stage methodology combining causal mediation analysis (CMA) with layer-specific constrained optimization. We identify early transformer layers (0-5), specifically MLP modules, as critical for storing subject-attribute associations in OLMo models (1B/7B). We then apply a novel joint loss function to these lower layers while freezing upper layers, maximizing forget set loss and minimizing retain set deviation. Our approach achieved 2nd place in the SemEval 1B model track, demonstrating strong task performance (0.973 aggregate) while maintaining 88% of baseline MMLU accuracy.

## Key Features

*   **Knowledge Isolation:** Uses Causal Mediation Analysis (`causal_trace.py`) to identify layers (specifically early MLP layers 0-5 in OLMo) responsible for storing targeted factual associations.
*   **Constrained Optimization:** Freezes upper layers and applies targeted updates only to the identified lower layers (`unlearn.py`).
*   **Layer-Specific Training:** Allows training only MLP modules (`--train_kind mlp`), Self-Attention modules (`--train_kind self_attn`), or both (`--train_kind all`) within the selected layers.
*   **Joint Loss Function:** Simultaneously maximizes loss on the forget set (via cross-entropy) and minimizes performance degradation on the retain set (via adaptive regularization based on loss deviation).
*   **Efficient Unlearning:** Aims to remove specific information without costly full model retraining.

## Repository Structure
```text
.
├── data/
│ └── sample_data/
├── utils/
│ └── create_data.py
├── causal_trace.py
├── unlearn.py
└── README.md
```
## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/LAB-FLAIR/Constrained_Unlearning_LLM.git
    cd Constrained_Unlearning_LLM
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Model:** Download the OLMo model, 1B or 7B fine-tuned version provided by the SemEval task organizers.

## Data Preparation

This project relies on the dataset format used in the LUME benchmark and SemEval-2025 Task 4.

1.  **Obtain LUME/SemEval Data:** Download the official forget and retain datasets for the task from the LUME repository or the SemEval task page.
    *   LUME GitHub: [https://github.com/amazon-science/lume-llm-unlearning](https://github.com/amazon-science/lume-llm-unlearning)
    *   Organize the data into separate directories for forget, retain, and validation sets as required by `unlearn.py`.

2.  **Generate Sample Data for Causal Trace (Optional but Recommended):**
    The `causal_trace.py` script benefits from a smaller, targeted dataset (like the QA pairs mentioned in the paper) to efficiently run CMA. Use `utils/create_data.py` to generate this sample dataset from the larger LUME dataset, or use the sample provided in `data/`.
    *   You may need to inspect `utils/create_data.py` for its specific arguments. Assuming it takes an input LUME file and an output path:
        ```bash
        python utils/create_data.py \
            --data_dir <path_to_downloaded_lume_data_file_or_dir>
        ```
        
## Usage

### 1. Causal Mediation Analysis (Knowledge Isolation)

Run CMA to identify the layers crucial for recalling the information in your sample fact file.

```bash
python causal_trace.py \
    --model_dir "semeval25-unlearning-model" \
    --fact_file "Constrained_Unlearning_LLM/data/known_QA_T2.json" \
    --output_dir "results/causal_trace_output"
```
The analysis helps confirm which layers (e.g., 0-5) and component types (e.g., MLP) are most influential, guiding the parameters for the unlearning step.



2. Constrained Unlearning
Run the main unlearning script using the insights from CMA.
```bash
python unlearn.py \
    --model_dir "semeval25-unlearning-model" \
    --output_dir "results/unlearned_model" \
    --forget_dir "semeval25-unlearning-data/data" \
    --retain_dir "semeval25-unlearning-data/data" \
    --trainable_layers "[0,1,2,3,4,5]" \
    --train_kind "mlp" \
    --epochs 8 \
    --data_ratio 1.0
```
The script will apply the joint loss function to the specified layers and components, saving the resulting unlearned model to the --output_dir.

## Citation
If you use the code or ideas from this repository, please cite our paper:

```bibtex
@inproceedings{agrawal2025selective,
  title={SHA256 at {SemEval}-2025 Task 4: Selective Amnesia -- Constrained Unlearning for Large Language Models via Knowledge Isolation},
  author={Agrawal, Saransh and Huang, Kuan-Hao},
  booktitle={Proceedings of the The 19th International Workshop on Semantic Evaluation (SemEval), 2025},
  year={2025}
}
```
## Acknowledgments
This work was developed for SemEval-2025 Task 4: Unlearning Sensitive Content from Large Language Models.
Portions of this research were conducted with the advanced computing resources provided by Texas A&M High Performance Research Computing.

## License
Apache 2.0
