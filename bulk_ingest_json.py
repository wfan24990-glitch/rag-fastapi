"""
批量导入 JSON，并避免重复处理已经向量化过的文本（按 source 去重）。
支持：单个对象包含 text/content，或数组中对象包含 text/content。
"""
import asyncio
import json
import os
import pathlib
import pickle
from typing import Iterable, List, Tuple

import httpx
from app.config import FAISS_INDEX_PATH

# 配置区域
INGEST_URL = "http://localhost:8001/ingest"           # 服务地址
DATA_DIR = pathlib.Path("/Users/water/Desktop/docs")  # JSON 目录
TEXT_KEYS = ("text", "content")                       # 文本字段名
META_PATH = pathlib.Path(str(pathlib.Path(os.path.expanduser(FAISS_INDEX_PATH))) + ".meta.pkl")
# -------------


def extract_payloads(obj, source_prefix: str) -> List[dict]:
    payloads = []
    if isinstance(obj, dict):
        for k in TEXT_KEYS:
            if k in obj and obj[k]:
                payloads.append({"text": obj[k], "source": source_prefix})
                break
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if not isinstance(item, dict):
                continue
            for k in TEXT_KEYS:
                if k in item and item[k]:
                    payloads.append({"text": item[k], "source": f"{source_prefix}_{i}"})
                    break
    return payloads


async def send_payloads(client: httpx.AsyncClient, payloads: Iterable[dict]) -> Tuple[int, int, list]:
    ok = fail = 0
    success_sources = []
    for p in payloads:
        try:
            r = await client.post(INGEST_URL, json=p, timeout=30)
            r.raise_for_status()
            ok += 1
            success_sources.append(p["source"])
        except Exception as e:
            fail += 1
            print(f"[FAIL] source={p.get('source')} error={e}")
    return ok, fail, success_sources


async def ingest_directory():
    processed_sources = set()
    # 仅以已持久化的元数据作为“已处理”判定，避免只请求成功但向量化失败时误判
    if META_PATH.exists():
        try:
            with META_PATH.open("rb") as f:
                meta = pickle.load(f)
            for v in meta.values():
                if isinstance(v, dict) and "source" in v:
                    processed_sources.add(v["source"])
        except Exception as e:
            print(f"[WARN] 读取已存在元数据失败，无法用于去重：{e}")
    print(f"[INFO] 已载入去重源数: {len(processed_sources)}")

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")

    # 预扫描：计算去重后计划发送的 payload 总数，方便进度展示
    files = sorted(DATA_DIR.glob("*.json"))
    staged = []
    total_todo = 0
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[SKIP] {path.name} 加载失败: {e}")
            continue

        extracted = extract_payloads(data, source_prefix=path.stem)
        if not extracted:
            print(f"[SKIP] {path.name} 缺少文本字段({', '.join(TEXT_KEYS)})或结构不支持")
            continue

        payloads = [p for p in extracted if p["source"] not in processed_sources]
        if not payloads:
            print(f"[SKIP] {path.name} 已存在于索引（source 去重）")
            continue

        staged.append((path, payloads))
        total_todo += len(payloads)

    if total_todo == 0:
        print("Nothing to ingest. All files skipped or already processed.")
        return

    total_ok = total_fail = 0
    processed_count = 0
    async with httpx.AsyncClient() as client:
        for path, payloads in staged:
            ok, fail, success_sources = await send_payloads(client, payloads)
            total_ok += ok
            total_fail += fail
            processed_count += ok + fail
            remaining = max(total_todo - processed_count, 0)
            print(f"[DONE] {path.name} sent={ok} failed={fail}")
            print(f"[PROGRESS] processed={processed_count}/{total_todo} remaining={remaining}")

            if ok > 0:
                processed_sources.update(success_sources)

    print(f"Finished. sent={total_ok} failed={total_fail}")


if __name__ == "__main__":
    asyncio.run(ingest_directory())
