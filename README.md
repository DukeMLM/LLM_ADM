
This repo uses GPT-3.5 to predict ADM spectra. Currently, it is not structured as a standalone library.

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

## Iore Python Scripts

| File              | Description                              |
|-------------------|------------------------------------------|
| `fine_tune.py`     | Fine-tune GPT-3.5 on the ADM dataset      |
| `eval.py`          | Evaluate model performance                |
| `create_prompt.py` | Generate prompts from input data          |
| `Tutorial.ipynb`   | Quick-start notebook                      |


---

## Baseline

Switch to the Baseline branch.
