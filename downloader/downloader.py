"""
Downloader ‑ poll mode
Re‑reads URL_FILE every POLL_SECONDS, fetches any images that do not
yet exist in OUTPUT_DIR, then sleeps.
"""

import asyncio, aiohttp, aiofiles, hashlib, os, sys, time
from pathlib import Path

# ── tunables ────────────────────────────────────────────────────────────
DATASET_PATH  = os.getenv("URL_FILE",      "/urls.txt")
OUT_DIR       = Path(os.getenv("OUTPUT_DIR", "/data/images"))
CONCURRENCY   = int(os.getenv("MAX_CONCURRENCY", 32))
POLL_SECONDS  = int(os.getenv("POLL_SECONDS", 30))
TIMEOUT       = aiohttp.ClientTimeout(total=30)
# ────────────────────────────────────────────────────────────────────────

def fname_from_url(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16] + ".jpg"

async def fetch(session, url):
    try:
        async with session.get(url) as resp:
            resp.raise_for_status()
            return await resp.read(), None
    except Exception as e:
        return None, e

async def download_round():
    """Return number of newly downloaded files in this cycle."""
    urls = [u.strip() for u in open(DATASET_PATH) if u.strip()]
    queue = asyncio.Queue()
    for u in urls:
        if not (OUT_DIR / fname_from_url(u)).exists():
            queue.put_nowait(u)

    if queue.empty():
        return 0  # nothing new to do

    conn = aiohttp.TCPConnector(limit=CONCURRENCY)
    async with aiohttp.ClientSession(timeout=TIMEOUT, connector=conn) as s:

        async def worker(name):
            while True:
                url = await queue.get()
                data, err = await fetch(s, url)
                if err:
                    print(f"[{name}] ❌ {url} – {err}")
                else:
                    out_file = OUT_DIR / fname_from_url(url)
                    async with aiofiles.open(out_file, "wb") as f:
                        await f.write(data)
                    print(f"[{name}] ✅ {url}")
                queue.task_done()

        tasks = [asyncio.create_task(worker(f"W{i}"))
                 for i in range(CONCURRENCY)]

        await queue.join()
        for t in tasks:
            t.cancel()

    return queue.qsize()  # how many we processed

async def run_forever():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        new_count = await download_round()
        if new_count:
            print(f"✓ download round complete – {new_count} new file(s)")
        await asyncio.sleep(POLL_SECONDS)

if __name__ == "__main__":
    try:
        asyncio.run(run_forever())
    except KeyboardInterrupt:
        sys.exit(0)
