import asyncio
import aiohttp
import time
import logging
import random
import ssl
from typing import List, Optional, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import chardet

from app.crawler.state import CrawlerState, RunStats, update_run_state, load_state
from app.crawler.parser import NJUParser
from app.crawler.utils import compute_hash, clean_text
from app.utils.chunker import chunk_text
from app.embeddings import get_embeddings
from app.vectorstore import add_embeddings

logger = logging.getLogger(__name__)

class NJUSpider:
    def __init__(self, run_id: str, mode: str = "incremental", max_pages: int = 50, dry_run: bool = False):
        self.run_id = run_id
        self.mode = mode
        self.max_pages = max_pages
        self.dry_run = dry_run
        self.state = load_state()
        self.stats = RunStats(run_id=run_id, start_time=time.time(), mode=mode)
        
        # Concurrency control
        self.sem = asyncio.Semaphore(3) # Max 3 concurrent requests
        self.session = None
        
        # Target Config
        self.list_url_template = "https://is.nju.edu.cn/57162/list{}.htm" # News list
        self.first_page_url = "https://is.nju.edu.cn/57162/list.htm"
        
        # SSL Context for legacy servers
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        try:
            self.ssl_context.set_ciphers('DEFAULT@SECLEVEL=1')
        except Exception:
            pass # Ignore if set_ciphers fails (e.g. on some systems)

    async def run(self):
        logger.info(f"Starting crawler run {self.run_id} in {self.mode} mode")
        try:
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                self.session = session
                await self._crawl_loop()
                
            self.stats.status = "completed"
        except Exception as e:
            logger.error(f"Crawler run failed: {e}", exc_info=True)
            self.stats.status = "failed"
            self.stats.errors.append(str(e))
        finally:
            self.stats.end_time = time.time()
            # Update history
            self.state.history.append(self.stats)
            # Trim history
            if len(self.state.history) > 10:
                self.state.history = self.state.history[-10:]
            
            # Update last sync date if successful and not dry run
            if self.stats.status == "completed" and not self.dry_run:
                # Find the latest date in this run (if any)
                # This logic needs to be robust. For now, we rely on the loop to have found the latest.
                pass 
                
            self.state.save()
            logger.info(f"Crawler run finished. Stats: {self.stats}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _fetch(self, url: str, method: str = "GET", check_content: bool = False) -> Optional[bytes]:
        async with self.sem:
            # Random delay
            await asyncio.sleep(random.uniform(0.3, 0.6))
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            
            if check_content:
                # Use GET but check headers before reading body
                async with self.session.get(url, headers=headers, timeout=10) as resp:
                    resp.raise_for_status()
                    
                    # Check Content-Type
                    ctype = resp.headers.get("Content-Type", "").lower()
                    if "text/html" not in ctype:
                        logger.warning(f"Skipping non-HTML content: {url} ({ctype})")
                        return None
                    
                    # Check Content-Length (if present)
                    cl = resp.headers.get("Content-Length")
                    if cl and int(cl) > 5 * 1024 * 1024: # 5MB limit
                        logger.warning(f"Skipping large file: {url} ({cl} bytes)")
                        return None
                        
                    return await resp.read()
            else:
                async with self.session.request(method, url, headers=headers, timeout=10) as resp:
                    resp.raise_for_status()
                    return await resp.read()

    async def _crawl_loop(self):
        page = 1
        stop_crawling = False
        latest_date_seen = None

        while page <= self.max_pages and not stop_crawling:
            url = self.first_page_url if page == 1 else self.list_url_template.format(page)
            logger.info(f"Fetching list page: {url}")
            
            try:
                content_bytes = await self._fetch(url)
                if not content_bytes:
                    logger.warning(f"Failed to fetch list page content: {url}")
                    break

                # Auto-detect encoding
                encoding = chardet.detect(content_bytes)['encoding'] or 'utf-8'
                html = content_bytes.decode(encoding, errors='replace')
                
                articles = NJUParser.parse_list_page(html, url)
                
                if not articles:
                    logger.warning(f"No articles found on page {page}. Stopping.")
                    break
                
                for article in articles:
                    # Check if we should stop (Incremental logic)
                    if self.mode == "incremental" and self.state.last_sync_date:
                        if not article["is_top"]: # Ignore top posts for date check
                            if article["date"] < self.state.last_sync_date:
                                logger.info(f"Found old article ({article['date']} < {self.state.last_sync_date}). Stopping.")
                                stop_crawling = True
                                break
                    
                    # Track latest date for state update
                    if not latest_date_seen or (article["date"] > latest_date_seen):
                        latest_date_seen = article["date"]

                    # Process article
                    await self._process_article(article)
                    
                page += 1
                
            except Exception as e:
                logger.error(f"Error processing list page {url}: {e}")
                self.stats.error_count += 1
                self.stats.errors.append(f"List page {url}: {str(e)}")
                # Don't stop entire run for one page error, but maybe break if critical
                if page == 1: # If first page fails, probably critical
                    raise e
        
        # Update state last_sync_date if we found something new
        if latest_date_seen and not self.dry_run:
            if not self.state.last_sync_date or latest_date_seen > self.state.last_sync_date:
                self.state.last_sync_date = latest_date_seen

    async def _process_article(self, article_meta: Dict):
        url = article_meta["url"]
        url_hash = compute_hash(url)
        
        # Deduplication (URL level)
        if url_hash in self.state.seen_url_hashes:
            logger.info(f"Skipping seen URL: {url}")
            self.stats.skipped_count += 1
            return
            
        logger.info(f"Processing article: {url}")
        try:
            # Fetch detail with content check
            content_bytes = await self._fetch(url, check_content=True)
            if not content_bytes:
                self.stats.skipped_count += 1
                return

            encoding = chardet.detect(content_bytes)['encoding'] or 'utf-8'
            html = content_bytes.decode(encoding, errors='replace')
            
            detail = NJUParser.parse_detail_page(html)
            clean_content = clean_text(detail["content_html"])
            
            if not clean_content or len(clean_content) < 50:
                logger.warning(f"Content too short for {url}, skipping.")
                self.stats.skipped_count += 1
                return

            # Ingestion
            if not self.dry_run:
                # Chunking
                chunks = chunk_text(clean_content, chunk_size=512, overlap=64)
                
                # Embedding & Storage
                embeddings = get_embeddings(chunks)
                metas = [{
                    "source": "is.nju.edu.cn",
                    "url": url,
                    "title": detail["title"],
                    "publish_date": detail["publish_date"] or article_meta["date"],
                    "id": f"{url_hash}_{idx}",
                    "text": c
                } for idx, c in enumerate(chunks)]
                
                add_embeddings(embeddings, metas)
                self.stats.ingested_count += 1
                
                # Update seen hashes
                self.state.seen_url_hashes.append(url_hash)
                # Trim seen hashes if too big
                if len(self.state.seen_url_hashes) > 10000:
                    self.state.seen_url_hashes = self.state.seen_url_hashes[-10000:]
            else:
                logger.info(f"[DRY RUN] Would ingest {url} ({len(clean_content)} chars)")
                self.stats.fetched_count += 1

        except Exception as e:
            logger.error(f"Error processing article {url}: {e}")
            self.stats.error_count += 1
            self.stats.errors.append(f"Article {url}: {str(e)}")
