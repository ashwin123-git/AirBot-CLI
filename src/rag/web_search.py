"""Web search module for RAG"""
import requests
from typing import List, Optional
from bs4 import BeautifulSoup
import time
import os
import re


class WebSearch:
    """Web search using DuckDuckGo"""

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

    def search(self, query: str, num_results: int = 5) -> List[dict]:
        """Search the web for information"""
        results = []
        url = "https://html.duckduckgo.com/html/"
        data = {'q': query, 'b': ''}

        try:
            response = requests.post(url, data=data, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all links that look like search results
            # The new DuckDuckGo HTML format uses various link patterns
            seen_urls = set()

            # Method 1: Look for links in result containers
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                title = link.get_text(strip=True)

                # Skip empty or navigation links
                if not title or len(title) < 5:
                    continue
                if href.startswith('/'):
                    continue
                if 'duckduckgo' in href.lower():
                    continue

                # Skip if we've seen this URL
                if href in seen_urls:
                    continue
                seen_urls.add(href)

                # Look for nearby snippet text
                snippet = ""
                parent = link.find_parent(['div', 'li'])
                if parent:
                    # Try to find description-like text
                    for p in parent.find_all(['p', 'span', 'a']):
                        text = p.get_text(strip=True)
                        if text and text != title and len(text) > 20:
                            snippet = text[:300]
                            break

                results.append({
                    'title': title[:200],
                    'link': href,
                    'snippet': snippet
                })

                if len(results) >= num_results:
                    break

            # Method 2: If method 1 failed, try more aggressive parsing
            if not results:
                # Look for links with specific patterns (result__a class was deprecated)
                for result in soup.find_all('a', attrs={'data-testid': True}):
                    href = result.get('href')
                    title = result.get_text(strip=True)
                    if href and title and len(title) > 5:
                        results.append({
                            'title': title[:200],
                            'link': href,
                            'snippet': ''
                        })
                        if len(results) >= num_results:
                            break

        except Exception as e:
            print(f"Search error: {e}")

        return results

    def fetch_page_content(self, url: str, max_length: int = 2000) -> Optional[str]:
        """Fetch and extract main content from a webpage"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text content
            text = soup.get_text(separator=' ', strip=True)

            # Clean up whitespace
            text = ' '.join(text.split())

            # Limit length
            if len(text) > max_length:
                text = text[:max_length] + "..."

            return text

        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def search_and_fetch(self, query: str, num_results: int = 3) -> List[str]:
        """Search and fetch content from top results"""
        results = self.search(query, num_results)
        contents = []

        for result in results:
            content = self.fetch_page_content(result['link'])
            if content:
                # Combine title, snippet, and content
                combined = f"Title: {result['title']}\n\n{result['snippet']}\n\n{content}"
                contents.append(combined)

            # Be respectful to servers
            time.sleep(1)

        return contents
