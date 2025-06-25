# filepath: src/models/model_comparison.py
"""
Compare multiple fine-tuned NER models on a validation dataset.
"""
import os
from datasets import DatasetDict, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from src.models.ner_data_processor import NERDataProcessor
from src.utils.constants import NER_LABELS

def evaluate_model(checkpoint: str, dataset: DatasetDict) -> dict:
    """Load model and evaluate on the validation split."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint)

    # Initialize data processor to align labels
    processor = NERDataProcessor(checkpoint, NER_LABELS)
    tokenized = dataset.map(
        processor.tokenize_and_align_labels,
        batched=True,
        remove_columns=['tokens', 'ner_tags']
    )
    # Use Trainer for evaluation
    trainer = Trainer(model=model, tokenizer=tokenizer)
    predictions, labels, _ = trainer.predict(tokenized['validation'])
    preds = predictions.argmax(-1)

    seqeval = load_metric('seqeval')
    # Convert label ids back to label strings
    true_labels = [[processor.id_to_label[l] for l in seq if l != -100] for seq in labels]
    pred_labels = [[processor.id_to_label[p] for (p, l) in zip(seq_pred, seq_label) if l != -100]
                   for seq_pred, seq_label in zip(preds, labels)]

    results = seqeval.compute(predictions=pred_labels, references=true_labels)
    return results

def compare_models(checkpoints: list, dataset: DatasetDict) -> dict:
    """Compare multiple checkpoints and return evaluation metrics."""
    metrics = {}
    for ckpt in checkpoints:
        print(f"Evaluating model {ckpt}...")
        metrics[ckpt] = evaluate_model(ckpt, dataset)
    return metrics
