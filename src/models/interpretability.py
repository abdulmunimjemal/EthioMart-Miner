# filepath: src/models/interpretability.py
"""
Task 5: Model Interpretability utilities (SHAP and LIME explanations for NER predictions).
"""
import numpy as np
import torch
import shap
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForTokenClassification, AutoTokenizer
from ..utils.constants import NER_LABELS


def shap_explanation(model_checkpoint: str, sample_texts: list) -> shap.Explanation:
    """
    Generates SHAP explanations for a list of texts using a pre-trained NER model.
    Returns a SHAP Explanation object.
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
    model.eval()

    # Prepare input for SHAP: encode texts
    encoded = tokenizer(sample_texts, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded['input_ids'].detach().numpy()

    # Define prediction function for SHAP
    def predict_fn(texts):
        enc = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            logits = model(**enc).logits.detach().numpy()
        # return max token-level probabilities aggregated by mean
        probs = np.max(logits, axis=-1)
        return probs

    explainer = shap.Explainer(predict_fn, input_ids)
    shap_values = explainer(sample_texts)
    return shap_values

def lime_explanation(model_checkpoint: str, text: str) -> LimeTextExplainer:
    """
    Generates a LIME explanation for a single text input using a pre-trained NER model.
    Returns a LIME explanation object.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
    model.eval()

    explainer = LimeTextExplainer(class_names=NER_LABELS)

    def predict_fn(texts: list):
        enc = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            logits = model(**enc).logits.detach().numpy()
        # aggregate token logits into a single feature vector (mean probabilities)
        probs = np.mean(logits, axis=1)
        return probs

    explanation = explainer.explain_instance(text, predict_fn, num_features=10)
    return explanation
