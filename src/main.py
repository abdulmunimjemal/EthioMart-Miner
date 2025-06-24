# main.py
import asyncio
import os
import pandas as pd
from .data_ingestion.telegram_scraper import TelegramScraper
from .preprocessing.amharic_processor import AmharicTextProcessor

async def run_task1_pipeline():
    print("--- Starting Task 1: Data Ingestion and Preprocessing ---")

    # 1. Data Ingestion
    scraper = TelegramScraper()
    print("Initiating Telegram data scraping...")
    df_raw = await scraper.scrape_all_channels(limit_per_channel=500) # Adjust limit as needed
    
    if df_raw.empty:
        print("No data scraped. Exiting Task 1.")
        return

    raw_output_path = os.path.join(scraper.output_dir, 'telegram_messages_raw.parquet')
    scraper.save_data(df_raw, filename='telegram_messages_raw.parquet')
    print(f"Raw data saved to {raw_output_path}")

    # 2. Data Preprocessing
    print("\nInitiating Amharic text preprocessing...")
    processor = AmharicTextProcessor()
    df_processed = processor.process_dataframe(df_raw.copy(), text_column='text')

    processed_output_path = 'data/processed/telegram_messages_processed.parquet'
    os.makedirs(os.path.dirname(processed_output_path), exist_ok=True)
    df_processed.to_parquet(processed_output_path, index=False)
    print(f"Processed data saved to {processed_output_path}")

    print("--- Task 1 Complete ---")
    return df_processed

if __name__ == '__main__':
    # Run the Task 1 pipeline
    asyncio.run(run_task1_pipeline())