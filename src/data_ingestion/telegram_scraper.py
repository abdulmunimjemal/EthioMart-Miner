# src/data_ingestion/telegram_scraper.py
import asyncio
import configparser
import os
from datetime import datetime
import pandas as pd
from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
from dotenv import load_dotenv

# Load environment variables from.env file
load_dotenv()

class TelegramScraper:
    def __init__(self, config_path='configs/telegram_config.ini'):
        self.api_id = int(os.getenv('API_ID'))
        self.api_hash = os.getenv('API_HASH')
        self.client = TelegramClient('ethio_mart_session', self.api_id, self.api_hash)
        self.channels = self._load_channels(config_path)
        self.output_dir = 'data/raw'
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'media'), exist_ok=True)

    def _load_channels(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        channels_str = config.get('TELEGRAM', 'CHANNELS')
        return [c.strip() for c in channels_str.split('\n') if c.strip()]

    async def connect(self):
        print("Connecting to Telegram...")
        await self.client.start()
        print("Client Created")

    async def disconnect(self):
        await self.client.disconnect()
        print("Client Disconnected")

    async def fetch_messages_from_channel(self, channel_entity, limit=None):
        """
        Fetches messages from a single Telegram channel.
        """
        print(f"Fetching messages from channel: {channel_entity.title} ({channel_entity.username})")
        records =[]
        try:
            async for msg in self.client.iter_messages(channel_entity, limit=limit):
                message_data = {
                    'id': msg.id,
                    'date': msg.date.isoformat() if msg.date else None,
                    'channel_id': channel_entity.id,
                    'channel_username': channel_entity.username,
                    'sender_id': getattr(msg.sender, 'id', None),
                    'text': msg.message or '',
                    'views': msg.views or 0,
                    'media_type': None,
                    'media_path': None
                }

                if msg.media:
                    media_path = os.path.join(self.output_dir, 'media', f"{channel_entity.id}_{msg.id}")
                    if isinstance(msg.media, MessageMediaPhoto):
                        message_data['media_type'] = 'photo'
                        media_path += '.jpg'
                    elif isinstance(msg.media, MessageMediaDocument):
                        message_data['media_type'] = 'document'
                        # Attempt to get file extension
                        if msg.document and msg.document.mime_type:
                            ext = msg.document.mime_type.split('/')[-1]
                            media_path += f'.{ext}'
                        else:
                            media_path += '.bin' # Default binary extension
                    
                    try:
                        # Download media if it exists and path is set
                        if message_data['media_type']:
                            await self.client.download_media(msg.media, file=media_path)
                            message_data['media_path'] = media_path
                    except Exception as e:
                        print(f"Error downloading media for message {msg.id} in {channel_entity.username}: {e}")
                        message_data['media_path'] = None # Mark as failed download

                records.append(message_data)
        except FloodWaitError as e:
            print(f"⏱ Flood wait—sleeping {e.seconds + 5}s for {channel_entity.username}")
            await asyncio.sleep(e.seconds + 5)
            # Recursively call to continue fetching after wait
            records.extend(await self.fetch_messages_from_channel(channel_entity, limit))
        except Exception as e:
            print(f"Error fetching messages from {channel_entity.username}: {e}")
        return records

    async def scrape_all_channels(self, limit_per_channel=None):
        """
        Scrapes messages from all configured channels.
        """
        all_messages =[]
        await self.connect()
        for channel_url in self.channels:
            try:
                channel_entity = await self.client.get_entity(channel_url)
                messages = await self.fetch_messages_from_channel(channel_entity, limit=limit_per_channel)
                all_messages.extend(messages)
            except Exception as e:
                print(f"Could not access channel {channel_url}: {e}")
        await self.disconnect()
        return pd.DataFrame(all_messages)

    def save_data(self, dataframe, filename='telegram_messages_raw.parquet'):
        """
        Saves the collected data to a Parquet file in the raw data directory.
        """
        filepath = os.path.join(self.output_dir, filename)
        dataframe.to_parquet(filepath, index=False)
        print(f"Data saved to {filepath}")

# Example usage (can be run from main.py or a notebook)
async def run_scraper():
    scraper = TelegramScraper()
    df_raw = await scraper.scrape_all_channels(limit_per_channel=100) # Fetch 100 messages per channel for testing
    if not df_raw.empty:
        scraper.save_data(df_raw)
    else:
        print("No messages scraped.")

if __name__ == '__main__':
    asyncio.run(run_scraper())