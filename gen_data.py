import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
visited_urls = set()
url_queue = deque()
data_queue = deque()
MAX_PAGES = 10000
CONCURRENT_TASKS = 100

async def fetch(session, url):
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                return await response.text()
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
    return None

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
        data_queue.append(text_data)

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
    with open('dataset.txt', 'a', encoding='utf-8') as file:
        while data_queue:
            text_data = data_queue.popleft()
            lines = text_data.split('\n')
            for line in lines:
                if len(line.split()) >= 3:
                    file.write(line + '\n')

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

if __name__ == "__main__":
    starting_url = "https://www.github.com/"
    main(starting_url)