# EthioMart Data Pipeline

A comprehensive Python pipeline for extracting, processing, and analyzing Amharic-language ecommerce data from Telegram channels. The project supports end-to-end workflows including data scraping, text preprocessing, NER data labeling, multilingual model fine-tuning, model evaluation, interpretability, and vendor analytics for micro-lending applications.

## Key Capabilities
- **Data Collection:** Scrape Amharic Telegram channels using Telethon.
- **Text Preprocessing:** Normalize Amharic text, remove punctuation and emojis.
- **NER Model Training:** Fine-tune models (XLM-Roberta, bert-tiny-amharic, etc.) with Hugging Face for Named Entity Recognition.
- **Model Evaluation:** Compare and select models based on robust evaluation metrics.
- **Interpretability:** Analyze model predictions using SHAP and LIME.
- **Vendor Analytics:** Generate vendor scorecards (activity, engagement, pricing, lending suitability).

## Getting Started
### Installation
Clone the repository and set up the environment:
```bash
git clone https://github.com/yohannesalex/Amharic_Ecommerce_Data_Extractor.git
cd Amharic_Ecommerce_Data_Extractor
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Usage
- **Data Ingestion & Preprocessing:**
  ```bash
  python -m src.main
  ```
- **CoNLL Data Preparation:**
  ```bash
  python -m src.labeling.conll_utils
  ```
- **NER Model Fine-tuning:**
  ```bash
  python -m src.models.ner_model_trainer
  ```
- **Model Comparison:**
  Use the functions in `src/models/model_comparison.py` within your scripts or notebooks.
- **Model Interpretability:**
  Use `src/models/interpretability.py` in your analysis workflows.
- **Vendor Scorecard Generation:**
  Use `src/models/vendor_scorecard.py` for vendor analytics.

## Project Structure
```
configs/     # Telegram API credentials and settings
data/        # Raw, processed, and labeled datasets
src/         # Core source code modules
notebooks/   # Jupyter notebooks for exploration and tutorials
models/      # Trained model checkpoints and scripts
reports/     # Reports and visualizations
```

## Maintainer
[abdulmunimjemal](https://github.com/abdulmunimjemal)

## License
MIT License