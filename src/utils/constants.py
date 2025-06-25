# src/utils/constants.py

# Define your NER labels based on the CoNLL format
# Ensure the order matches how you'll map them to model output IDs
# O should always be 0, B- and I- tags follow.
# The actual mapping will be done by the model's config.
# For now, this list helps in understanding and converting.
NER_LABELS =[]

# Paths
LABELED_DATA_PATH = "data/labeled/amharic_ner_labeled_data.conll"
# If you have multiple labeled files, you might want to adjust this or load them all.
# For this example, we'll assume one primary labeled file.

MODEL_OUTPUT_DIR = "models/amharic_ner_model"