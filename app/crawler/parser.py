from bs4 import BeautifulSoup
from datetime import datetime
import re
from typing import Optional, Tuple, List, Dict
from app.crawler.utils import normalize_url

class NJUParser:
    BASE_URL = "https://is.nju.edu.cn"
    
    @staticmethod
    def parse_list_page(html: str, base_url: str) -> List[Dict]:
        """
        Parse list page to extract articles.
        Returns list of dicts: {url, title, date, is_top}
        """
        soup = BeautifulSoup(html, "lxml")
        articles = []
        
        # Try common WebPLUS selectors
        # .news_list, .wp_article_list, etc.
        # Based on previous analysis or standard templates
        
        # Strategy: Find the main list container
        # Usually ul.news_list or similar
        container = soup.select_one(".news_list") or soup.select_one(".wp_article_list")
        
        if not container:
            # Fallback: look for any ul/table with many links and dates
            return []

        items = container.find_all("li") # Assuming li structure
        
        for item in items:
            link_tag = item.find("a")
            if not link_tag:
                continue
                
            url = normalize_url(base_url, link_tag.get("href"))
            title = link_tag.get("title") or link_tag.get_text(strip=True)
            
            # Date extraction
            date_tag = item.select_one(".Article_PublishDate") or item.select_one(".news_meta") or item.find("span", class_="date")
            date_str = ""
            if date_tag:
                date_str = date_tag.get_text(strip=True)
            else:
                # Try regex on text
                match = re.search(r"\d{4}-\d{2}-\d{2}", item.get_text())
                if match:
                    date_str = match.group(0)
            
            # Top detection (heuristic)
            is_top = False
            if item.select_one(".top") or item.select_one("img[src*='top']"):
                is_top = True
                
            if url and title:
                articles.append({
                    "url": url,
                    "title": title,
                    "date": date_str,
                    "is_top": is_top
                })
                
        return articles

    @staticmethod
    def parse_detail_page(html: str) -> Dict:
        """
        Parse detail page to extract content and metadata.
        """
        soup = BeautifulSoup(html, "lxml")
        
        # Title
        title_tag = soup.select_one(".arti_title")
        title = title_tag.get_text(strip=True) if title_tag else ""
        
        # Content
        content_tag = soup.select_one(".arti_content") or soup.select_one(".wp_articlecontent")
        # Remove noise inside content before extraction
        if content_tag:
            # Remove scripts/styles inside content
            for tag in content_tag(["script", "style"]):
                tag.decompose()
            content_html = str(content_tag)
        else:
            content_html = ""
            
        # Meta (Date, Source)
        meta_tag = soup.select_one(".arti_metas") or soup.select_one(".arti_update")
        meta_text = meta_tag.get_text(strip=True) if meta_tag else ""
        
        # Extract date from meta text if possible
        publish_date = ""
        match = re.search(r"\d{4}-\d{2}-\d{2}", meta_text)
        if match:
            publish_date = match.group(0)
            
        return {
            "title": title,
            "content_html": content_html,
            "publish_date": publish_date,
            "meta_text": meta_text
        }
