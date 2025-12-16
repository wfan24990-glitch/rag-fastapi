from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import json
import os
import time
from datetime import datetime

STATE_FILE = os.path.expanduser("~/projects/rag-fastapi/data/crawler_state.json")

class RunStats(BaseModel):
    run_id: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"  # running, completed, failed, stopped
    mode: str = "incremental"
    fetched_count: int = 0
    ingested_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    errors: List[str] = []

class CrawlerState(BaseModel):
    last_sync_date: Optional[str] = None  # YYYY-MM-DD
    last_sync_ts: float = 0.0
    seen_url_hashes: List[str] = Field(default_factory=list) # Keep recent N
    history: List[RunStats] = Field(default_factory=list) # Keep recent N runs

    def save(self):
        # Atomic write
        temp_file = STATE_FILE + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        os.replace(temp_file, STATE_FILE)

def load_state() -> CrawlerState:
    if not os.path.exists(STATE_FILE):
        return CrawlerState()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CrawlerState(**data)
    except Exception:
        return CrawlerState()

def update_run_state(run_id: str, **kwargs):
    state = load_state()
    for run in state.history:
        if run.run_id == run_id:
            updated = run.model_copy(update=kwargs)
            # Update in place
            idx = state.history.index(run)
            state.history[idx] = updated
            state.save()
            return
