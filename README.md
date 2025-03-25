
This repo uses GPT-3.5 to predict ADM spectra.

---

## Installation

### Environment Setup
To set up the required Python environment:

conda env create -f environment.yaml

conda activate llm_ft


### Access to Datasets

Download the datasets from the following Google Drive link:  
🔗 [ADM Dataset](https://drive.google.com/drive/folders/1L53bAP2vT3V_DyiwCOjSDX-SaRhrjLlk?usp=drive_link)

---

## Python Scripts

| File              | Description                              |
|-------------------|------------------------------------------|
| `src/fine_tune.py`     | Fine-tune GPT-3.5 on the ADM dataset      |
| `src/eval.py`          | Evaluate model performance                |
| `src/create_prompt.py` | Generate prompts from input data          |
| `Tutorial.ipynb`   | Quick-start notebook                          |


---

## Baseline

Switch to the Baseline branch.
