# Amharic Ecommerce Data Extractor

A Python-based pipeline for scraping Amharic Telegram channels, preprocessing text, labeling data for Named Entity Recognition (NER), fine-tuning and comparing multilingual NER models, interpreting model predictions, and generating vendor scorecards for micro-lending.

## Features
- **Task 1:** Data ingestion from Telegram channels (Telethon).
- **Task 2:** Amharic text preprocessing (normalization, punctuation & emoji removal).
- **Task 3:** Fine-tune NER models (XLM-Roberta, bert-tiny-amharic, etc.) using Hugging Face.
- **Task 4:** Model comparison & selection based on evaluation metrics.
- **Task 5:** Model interpretability with SHAP and LIME.
- **Task 6:** Vendor scorecard generation (posting frequency, engagement, pricing, lending score).

## Installation
```bash
git clone https://github.com/yohannesalex/Amharic_Ecommerce_Data_Extractor.git
cd Amharic_Ecommerce_Data_Extractor
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage
- **Task 1 & 2** (scraping and preprocessing):
  ```bash
  python -m src.main
  ```
- **Task 2** (prepare CoNLL data):
  ```bash
  python -m src.labeling.conll_utils
  ```
- **Task 3** (fine-tune NER):
  ```bash
  python -m src.models.ner_model_trainer
  ```
- **Task 4** (compare models):
  Use the functions in `src/models/model_comparison.py` in a script or notebook.
- **Task 5** (interpretability):
  Use `src/models/interpretability.py` in a script or notebook.
- **Task 6** (vendor scorecards):
  Use `src/models/vendor_scorecard.py` in a script or notebook.

## Directory Structure
```
configs/             # Telegram API configuration
data/                # Raw, processed, labeled data
src/                 # Source code modules
notebooks/           # Jupyter notebooks for exploration and tutorials
models/              # Trained model checkpoints and utilities
reports/             # Final reports and figures
``` 

## Author
[yohannesalex](https://github.com/yohannesalex)

## License
MIT License