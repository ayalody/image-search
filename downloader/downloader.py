import asyncio, aiohttp, aiofiles, hashlib, os, sys
from pathlib import Path

# configurable via env-vars so we don’t rebuild the image for each run
DATASET_PATH = os.getenv("URL_FILE", "/urls.txt")
OUT_DIR      = Path(os.getenv("OUTPUT_DIR", "/data/images"))
CONCURRENCY  = int(os.getenv("MAX_CONCURRENCY", 32))
TIMEOUT      = aiohttp.ClientTimeout(total=30)

# Deduplicate by SHA-1 of URL string (cheap) and of bytes (optional)
def fname_from_url(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16] + ".jpg"

async def fetch(session, url):
    try:
        async with session.get(url) as resp:
            resp.raise_for_status()
            content = await resp.read()
            return url, content, None
    except Exception as e:
        return url, None, e

async def worker(name, q, session):
    while True:
        url = await q.get()
        file_name = fname_from_url(url)
        out_file  = OUT_DIR / file_name
        if out_file.exists():            # skip if already downloaded
            q.task_done(); continue

        url, data, err = await fetch(session, url)
        if err:
            print(f"[{name}] ❌ {url} – {err}")
        else:
            async with aiofiles.open(out_file, "wb") as f:
                await f.write(data)
            print(f"[{name}] ✅ {url}")
        q.task_done()

async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    url_list = [u.strip() for u in open(DATASET_PATH) if u.strip()]
    q = asyncio.Queue()
    for u in url_list: q.put_nowait(u)

    conn = aiohttp.TCPConnector(limit=CONCURRENCY)
    async with aiohttp.ClientSession(timeout=TIMEOUT,
                                     connector=conn) as session:
        tasks = [asyncio.create_task(worker(f"W{i}", q, session))
                 for i in range(CONCURRENCY)]
        await q.join()
        for t in tasks: t.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
