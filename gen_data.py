import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import logging
import re
import json
from langdetect import detect
from hashlib import md5

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
visited_urls = set()
url_queue = deque()
data_queue = deque()
MAX_PAGES = 10000
CONCURRENT_TASKS = 500
MIN_WORDS_PER_LINE = 5
MAX_WORDS_PER_LINE = 100
SIMILARITY_THRESHOLD = 0.8  # Adjust this value to control similarity detection

# New global variable for content hashes
content_hashes = set()

async def fetch(session, url):
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                return await response.text()
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
    return None

def clean_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def is_valid_sentence(sentence):
    words = sentence.split()
    return MIN_WORDS_PER_LINE <= len(words) <= MAX_WORDS_PER_LINE

def compute_similarity_hash(text):
    # Compute a hash of the text for similarity checking
    return md5(text.encode()).hexdigest()

def is_similar(hash_value):
    # Check if the hash is similar to any existing hash
    return hash_value in content_hashes

async def process_url(session, url):
    if url in visited_urls or len(visited_urls) >= MAX_PAGES:
        return

    visited_urls.add(url)
    logging.info(f"Visiting: {url}")

    html = await fetch(session, url)
    if html:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract text
        text_data = soup.get_text(separator='\n', strip=True)
        
        # Process and clean the text
        sentences = text_data.split('\n')
        cleaned_sentences = [clean_text(sentence) for sentence in sentences if is_valid_sentence(sentence)]
        
        # Compute similarity hash for the cleaned content
        content = ' '.join(cleaned_sentences)
        content_hash = compute_similarity_hash(content)
        
        # Check if the content is similar to previously seen content
        if not is_similar(content_hash):
            content_hashes.add(content_hash)
            
            # Detect language
            try:
                language = detect(' '.join(cleaned_sentences[:10]))  # Use first 10 sentences for language detection
            except:
                language = 'unknown'
            
            # Create a structured data point
            data_point = {
                'url': url,
                'language': language,
                'content': cleaned_sentences,
                'metadata': {
                    'title': soup.title.string if soup.title else '',
                    'timestamp': time.time()
                }
            }
            
            data_queue.append(json.dumps(data_point))
        else:
            logging.info(f"Skipping similar content from {url}")

        # Find new URLs
        for link in soup.find_all('a', href=True):
            new_url = urljoin(url, link['href'])
            parsed_url = urlparse(new_url)
            if parsed_url.scheme in ['http', 'https'] and new_url not in visited_urls:
                url_queue.append(new_url)

async def crawler():
    async with aiohttp.ClientSession() as session:
        while url_queue and len(visited_urls) < MAX_PAGES:
            tasks = []
            for _ in range(min(CONCURRENT_TASKS, len(url_queue))):
                if url_queue:
                    url = url_queue.popleft()
                    task = asyncio.ensure_future(process_url(session, url))
                    tasks.append(task)
            await asyncio.gather(*tasks)

def save_data():
    with open('content/dataset.jsonl', 'a', encoding='utf-8') as file:
        while data_queue:
            json_data = data_queue.popleft()
            file.write(json_data + '\n')

def main(starting_url):
    url_queue.append(starting_url)
    
    start_time = time.time()
    
    # Run the crawler
    asyncio.run(crawler())
    
    # Save collected data
    with ThreadPoolExecutor() as executor:
        executor.submit(save_data)
    
    end_time = time.time()
    logging.info(f"Crawling completed. Visited {len(visited_urls)} pages in {end_time - start_time:.2f} seconds.")
    logging.info(f"Unique content pieces: {len(content_hashes)}")

if __name__ == "__main__":
    starting_url = "https://en.wikipedia.org/wiki/Main_Page"
    main(starting_url)