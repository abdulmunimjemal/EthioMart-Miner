# src/preprocessing/amharic_processor.py
import re
import pandas as pd
import os
# For more advanced Amharic NLP, consider libraries like 'app_toolkit'
# from app_toolkit import stemmer, stopwords_remover # Example import

class AmharicTextProcessor:
    def __init__(self):
        # Define common Amharic character variations for normalization
        # This is a simplified example; a comprehensive list would be larger.
        self.char_normalization_map = {
            'ሀ': 'ሃ', 'ሁ': 'ሁ', 'ሂ': 'ሂ', 'ሃ': 'ሃ', 'ሄ': 'ሄ', 'ህ': 'ህ', 'ሆ': 'ሆ',
            'ኀ': 'ሃ', 'ኁ': 'ሁ', 'ኂ': 'ሂ', 'ኃ': 'ሃ', 'ኄ': 'ሄ', 'ኅ': 'ህ', 'ኆ': 'ሆ',
            'ሐ': 'ሃ', 'ሑ': 'ሁ', 'ሒ': 'ሂ', 'ሓ': 'ሃ', 'ሔ': 'ሄ', 'ሕ': 'ህ', 'ሖ': 'ሆ',
            'ኸ': 'ሃ', 'ኹ': 'ሁ', 'ኺ': 'ሂ', 'ኻ': 'ሃ', 'ኼ': 'ሄ', 'ኽ': 'ህ', 'ኾ': 'ሆ',
            'ሰ': 'ሠ', 'ሠ': 'ሠ', # Normalize 'ሰ' to 'ሠ' or vice versa consistently
            'ጸ': 'ፀ', 'ፀ': 'ፀ', # Normalize 'ጸ' to 'ፀ' or vice versa consistently
            'ው': 'ዉ', 'ዉ': 'ዉ', # Normalize 'ው' to 'ዉ' or vice versa consistently
            'አ': 'ዓ', 'ዓ': 'ዓ', # Normalize 'አ' to 'ዓ' or vice versa consistently
            # Add more as identified
        }
        # Amharic punctuation marks
        self.amharic_punctuation = r'[።፣፤፥፧፨ᓭᙅᙆ]' # Amharic full stop, comma, semicolon, etc.
        self.english_punctuation = r'[.,!?;:()\[\]{}"]'
        self.emojis = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+",
            flags=re.UNICODE
        )

    def normalize_characters(self, text):
        """Normalizes Amharic characters with common variations."""
        return ''.join(self.char_normalization_map.get(char, char) for char in text)

    def remove_punctuation(self, text):
        """Removes Amharic and English punctuation."""
        text = re.sub(self.amharic_punctuation, '', text)
        text = re.sub(self.english_punctuation, '', text)
        return text

    def remove_emojis(self, text):
        """Removes emojis from text."""
        return self.emojis.sub(r'', text)

    def normalize_whitespace(self, text):
        """Replaces multiple whitespaces with a single space and strips leading/trailing spaces."""
        return re.sub(r'\s+', ' ', text).strip()

    def preprocess_text(self, text):
        """Applies a sequence of preprocessing steps to Amharic text."""
        if not isinstance(text, str):
            return "" # Handle non-string inputs gracefully
        text = self.normalize_characters(text)
        text = self.remove_punctuation(text)
        text = self.remove_emojis(text)
        text = self.normalize_whitespace(text)
        # Add more advanced steps here if using app_toolkit, e.g.:
        # text = stopwords_remover.remove_stopwords(text)
        # text = stemmer.stem(text) # Stemming might be too aggressive for NER
        return text

    def process_dataframe(self, df, text_column='text'):
        """Applies preprocessing to a specified text column in a DataFrame."""
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")
        print(f"Preprocessing text in column '{text_column}'...")
        df['processed_text'] = df[text_column].apply(self.preprocess_text)
        print("Text preprocessing complete.")
        return df

# Example usage (can be run from main.py or a notebook)
if __name__ == '__main__':
    # Create a dummy DataFrame for demonstration
    data = {
        'id': '234567890',
        'text': [
            "ይህ አዲስ የህፃን ጠርሙስ ነው። ዋጋው ፲፭፻ ብር ነው። አዲስ አበባ ይገኛል። 😊",
            "ጥሩ ጥራት ያለው ስልክ በ10000 ብር ብቻ! አድራሻ: ቦሌ።",
            "የተለያዩ ልብሶች አሉን። ዋጋ ከ500 ብር ጀምሮ።"
        ],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'views': 3
    }
    df_raw = pd.DataFrame(data)

    processor = AmharicTextProcessor()
    df_processed = processor.process_dataframe(df_raw.copy()) # Use.copy() to avoid modifying original df_raw

    print("\nOriginal DataFrame:")
    print(df_raw)
    print("\nProcessed DataFrame:")
    print(df_processed[['id', 'text', 'processed_text', 'views']])

    # Save processed data
    output_filepath = 'data/processed/telegram_messages_processed.parquet'
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df_processed.to_parquet(output_filepath, index=False)
    print(f"\nProcessed data saved to {output_filepath}")
