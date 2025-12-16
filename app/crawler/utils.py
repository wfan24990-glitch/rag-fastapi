import hashlib
import re
from bs4 import BeautifulSoup

def compute_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def clean_text(html_content: str) -> str:
    """
    Clean HTML content to extract pure text for RAG.
    Removes scripts, styles, navs, footers, etc.
    """
    if not html_content:
        return ""
    
    soup = BeautifulSoup(html_content, "lxml")
    
    # Remove unwanted tags
    for tag in soup(["script", "style", "nav", "footer", "iframe", "noscript"]):
        tag.decompose()
        
    # Remove common noise classes/ids (heuristic)
    noise_selectors = [
        ".header", ".footer", ".nav", ".sidebar", ".menu", 
        ".breadcrumb", ".pagination", ".prev-next", ".related-posts",
        "#header", "#footer", "#nav", "#sidebar"
    ]
    for selector in noise_selectors:
        for tag in soup.select(selector):
            tag.decompose()

    text = soup.get_text(separator="\n")
    
    # Normalize whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    
    return text

def normalize_url(base_url: str, link: str) -> str:
    if not link:
        return ""
    if link.startswith("http"):
        return link
    if link.startswith("/"):
        # Handle root relative
        from urllib.parse import urljoin
        return urljoin(base_url, link)
    # Handle relative
    from urllib.parse import urljoin
    return urljoin(base_url, link)
