from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid
from app.crawler.spider import NJUSpider
from app.crawler.state import load_state

router = APIRouter()

class CrawlerRunRequest(BaseModel):
    mode: str = "incremental" # incremental, full
    max_pages: int = 50
    dry_run: bool = False

@router.post("/run")
async def run_crawler(req: CrawlerRunRequest, background_tasks: BackgroundTasks):
    run_id = str(uuid.uuid4())
    spider = NJUSpider(
        run_id=run_id,
        mode=req.mode,
        max_pages=req.max_pages,
        dry_run=req.dry_run
    )
    
    # Run in background
    background_tasks.add_task(spider.run)
    
    return {"status": "accepted", "run_id": run_id, "message": "Crawler started in background"}

@router.get("/status")
async def get_crawler_status():
    state = load_state()
    
    last_run = state.history[-1] if state.history else None
    
    return {
        "last_sync_date": state.last_sync_date,
        "total_seen_urls": len(state.seen_url_hashes),
        "last_run": last_run
    }
