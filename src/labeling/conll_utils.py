# src/labeling/conll_utils.py (utility functions for CoNLL)
import pandas as pd
import re
import os

def load_processed_data(filepath='data/processed/telegram_messages_processed.parquet'):
    """Loads the processed data from Task 1."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed data not found at {filepath}. Please run Task 1 first.")
    df = pd.read_parquet(filepath)
    print(f"Loaded {len(df)} messages from {filepath}")
    return df

def tokenize_amharic_text(text):
    """
    A simple tokenizer for Amharic text.
    For actual NER fine-tuning, you'd use a sub-word tokenizer from a pre-trained LLM.
    This is for manual labeling preparation.
    """
    # Split by spaces and common Amharic sentence delimiters (e.g., '።', '፣', '፤')
    # This is a basic tokenization. More advanced tokenization might be needed.
    tokens = re.findall(r'\S+|\s+', text) # Keep spaces for now to reconstruct sentences
    tokens = [t for t in tokens if t.strip()] # Remove empty tokens
    return tokens

def save_to_conll(tokens_and_labels, output_filepath):
    """
    Saves a list of (token, label) pairs to a CoNLL formatted file.
    Each inner list represents a sentence.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for sentence_data in tokens_and_labels:
            for token, label in sentence_data:
                f.write(f"{token}\t{label}\n")
            f.write("\n") # Blank line separates sentences
    print(f"Labeled data saved to {output_filepath}")

# Example usage (can be run from a notebook for interactive labeling)
if __name__ == '__main__':
    # This part would typically be in a Jupyter notebook for interactive labeling
    # notebooks/02_data_labeling_guide.ipynb

    # Load data
    try:
        df_processed = load_processed_data()
    except FileNotFoundError as e:
        print(e)
        print("Please run `python main.py` to generate processed data first.")
        exit()

    # Select a subset for labeling (e.g., the first 50 messages)
    # The project requires 30-50 messages to be labeled.[1]
    messages_to_label = df_processed['processed_text'].head(50).tolist()

    # --- Manual Labeling Process (Conceptual) ---
    # This is where you would manually go through each message and assign labels.
    # For a real project, you would use an annotation tool like Doccano, Prodigy, or Label Studio.
    # The following is a *manual example* to illustrate the CoNLL format.

    labeled_sentences_example = []  # Initialize list for labeled sentences

    # Example 1: "ይህ አዲስ የህፃን ጠርሙስ ነው። ዋጋው ፲፭፻ ብር ነው። አዲስ አበባ ይገኛል።"
    # Simplified tokenization for manual example
    sentence1_text = "ይህ አዲስ የህፃን ጠርሙስ ነው። ዋጋው ፲፭፻ ብር ነው። አዲስ አበባ ይገኛል።"
    tokens1 = tokenize_amharic_text(sentence1_text)
    labels1 = [(token, 'O') for token in tokens1]  # Placeholder 'O' labels for each token
    # Ensure tokens and labels match length and order for actual use
    # For this example, we'll just use the pre-defined labels1 for illustration
    labeled_sentences_example.append(labels1)

    # Example 2: "ጥሩ ጥራት ያለው ስልክ በ10000 ብር ብቻ! አድራሻ: ቦሌ።"
    sentence2_text = "ጥሩ ጥራት ያለው ስልክ በ10000 ብር ብቻ! አድራሻ: ቦሌ።"
    tokens2 = tokenize_amharic_text(sentence2_text)
    labels2 = [(token, 'O') for token in tokens2]  # Placeholder 'O' labels
    labeled_sentences_example.append(labels2)

    # Save the manually labeled data to a CoNLL file
    output_conll_filepath = 'data/labeled/amharic_ner_labeled_data.conll'
    save_to_conll(labeled_sentences_example, output_conll_filepath)

    print(f"\nExample of CoNLL content saved to {output_conll_filepath}:")
    with open(output_conll_filepath, 'r', encoding='utf-8') as f:
        print(f.read())