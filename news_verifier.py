from duckduckgo_search import DDGS
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# A broad list of trusted mainstream media and news agencies
TRUSTED_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "cnn.com", 
    "bloomberg.com", "nytimes.com", "washingtonpost.com", "npr.org",
    "wsj.com", "theguardian.com", "ft.com", "aljazeera.com", "cnbc.com",
    "abcnews.go.com", "cbsnews.com", "nbcnews.com", "foxnews.com", 
    "time.com", "forbes.com", "thehindu.com", "timesofindia.indiatimes.com",
    "ndtv.com", "indianexpress.com", "hindustantimes.com", "thewire.in",
    "politico.com", "axios.com", "usatoday.com", "newsweek.com",
    "theatlantic.com", "economist.com"
}

def verify_news_live(headline: str, max_results: int = 5) -> dict:
    """
    Perform a live web search for the headline using DuckDuckGo.
    If the news is currently being reported by mainstream media,
    return a strong real/verified signal.
    """
    if not headline or len(headline.split()) < 3:
        return {"found": False, "signal": 0.0, "sources": []}
        
    try:
        results = DDGS().news(headline, max_results=max_results)
        
        matched_sources = []
        for r in results:
            url = r.get("url", "")
            try:
                # Extract base domain (e.g., www.reuters.com -> reuters.com)
                domain = urlparse(url).netloc.lower()
                if domain.startswith("www."):
                    domain = domain[4:]
                    
                if domain in TRUSTED_DOMAINS:
                    matched_sources.append({"domain": domain, "url": url, "title": r.get("title")})
            except Exception:
                continue
                
        if matched_sources:
            logger.info(f"Live Verification matched {len(matched_sources)} trusted sources for: {headline}")
            # Return a negative signal (meaning REAL in our prob space)
            # The more sources verify it, the stronger the signal
            strength = -0.7 if len(matched_sources) > 1 else -0.5
            return {
                "found": True,
                "signal": strength, 
                "sources": matched_sources
            }
            
        return {"found": False, "signal": 0.0, "sources": []}
        
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed for verification: {e}")
        return {"found": False, "signal": 0.0, "sources": []}

import requests
import xml.etree.ElementTree as ET
from datetime import datetime

def get_latest_news(query: str = "world news", max_results: int = 15) -> list:
    """
    Fetch the latest news from Global and India RSS Feeds.
    """
    feeds = [
        {"url": "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en", "label": "GLOBAL"},
        {"url": "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en", "label": "INDIA"}
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    news_list = []
    
    for feed in feeds:
        try:
            response = requests.get(feed["url"], headers=headers, timeout=10)
            if response.status_code != 200:
                continue
                
            root = ET.fromstring(response.content)
            
            # Parse up to half of max_results from each feed
            items_per_feed = max_results // len(feeds)
            
            for item in root.findall(".//item")[:items_per_feed]:
                title_full = item.find("title").text if item.find("title") is not None else "No Title"
                
                if " - " in title_full:
                    title, source = title_full.rsplit(" - ", 1)
                else:
                    title, source = title_full, "Google News"
                    
                pub_date = item.find("pubDate").text if item.find("pubDate") is not None else "Just now"
                display_date = pub_date.replace("GMT", "").strip()
                
                news_list.append({
                    "title": title.strip(),
                    "source": f"{source.strip()} ({feed['label']})",
                    "url": item.find("link").text if item.find("link") is not None else "#",
                    "date": display_date,
                    "snippet": f"Latest update from {feed['label']} news node. Click to verify integrity."
                })
        except Exception as e:
            logger.error(f"Error fetching {feed['label']} news: {e}")
            
    # Shuffle the news so global and india news are mixed
    import random
    random.shuffle(news_list)
    
    if not news_list:
        return [{
            "title": "System Alert: Live News Feed Handshake Error",
            "source": "TruthLens System",
            "url": "#",
            "date": "Now",
            "snippet": "The real-time news node is experiencing connection issues. Retrying link..."
        }]
        
    return news_list

if __name__ == "__main__":
    # Test script
    res = verify_news_live("India GDP grows 6.5 percent")
    print("Test 1 (Real):", res)
    
    res = verify_news_live("Aliens land in Times Square and eat pizza")
    print("Test 2 (Fake):", res)
