# src/models/ner_data_processor.py
import os
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer
from src.utils.constants import NER_LABELS # Import your defined labels

class NERDataProcessor:
    def __init__(self, model_checkpoint, label_list):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.label_list = label_list
        self.label_to_id = {label: i for i, label in enumerate(label_list)}
        self.id_to_label = {i: label for i, label in enumerate(label_list)}

    def load_conll_data(self, file_path):
        """
        Loads data from a CoNLL formatted text file.
        The datasets library can load CoNLL files directly.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Labeled data not found at {file_path}. Please ensure Task 2 is complete.")
        
        # datasets.load_dataset can load text files and infer structure
        # For CoNLL, it's often easier to read line by line and then convert
        # or use a custom loading script if the format is very specific.
        # However, for simple token\tlabel per line, 'text' loader might work
        # but 'conll2003' or similar custom script is better for NER.
        # Let's simulate a simple CoNLL loading for demonstration.
        
        # A more robust way to load CoNLL-like format:
        # We'll read the file and parse it into a list of dictionaries
        # where each dict has 'tokens' and 'ner_tags' (list of strings)
        
        # Initialize containers for CoNLL parsing
        raw_texts = []
        raw_labels = []
        current_tokens = []
        current_labels = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line: # Not an empty line
                    parts = line.split('\t')
                    if len(parts) == 2:
                        current_tokens.append(parts)
                        current_labels.append(parts[1])
                    else:
                        # Handle lines that might be comments or malformed
                        # For simplicity, we'll skip them or raise an error
                        print(f"Skipping malformed line: {line}")
                else: # Empty line, end of a sentence/message
                    if current_tokens: # If there are tokens collected
                        raw_texts.append(current_tokens)
                        raw_labels.append(current_labels)
                    # Reset for next sentence
                    current_tokens = []
                    current_labels = []
            # Add any remaining tokens/labels after loop if file doesn't end with blank line
            if current_tokens:
                raw_texts.append(current_tokens)
                raw_labels.append(current_labels)

        # Convert string labels to their corresponding IDs per sentence
        processed_labels = [
            [self.label_to_id[label] for label in sentence_labels]
            for sentence_labels in raw_labels
        ]

        # Create a Hugging Face Dataset
        from datasets import Dataset
        dataset = Dataset.from_dict({
            'tokens': raw_texts,
            'ner_tags': processed_labels
        })
        
        # For demonstration, we'll split into train/validation.
        # In a real scenario, you'd have separate train/val/test files.
        train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
        return DatasetDict({
            'train': train_test_split['train'],
            'validation': train_test_split['test'] # Using 'test' as validation
        })

    def tokenize_and_align_labels(self, examples):
        """
        Tokenizes the input text and aligns the labels with the new tokens.
        Handles subword tokenization and special tokens.
        """
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True # Important for pre-tokenized input
        )

        labels = []  # aligned label IDs for each example
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []  # label IDs for current tokenized sequence
            for word_idx in word_ids:
                if word_idx is None: # Special tokens (CLS, SEP, PAD)
                    label_ids.append(-100)
                elif word_idx!= previous_word_idx: # Start of a new word
                    label_ids.append(label[word_idx])
                else: # Subsequent sub-token of the same word
                    label_ids.append(-100) # Assign -100 to ignore in loss calculation
                previous_word_idx = word_idx
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs