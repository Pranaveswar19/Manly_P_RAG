#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlparse, urljoin

import requests

SITE = "http://manlyphall.info"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
TIMEOUT = 30  

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "data" / "pdfs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def safe_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.netloc or "site").replace(":", "_")
    leaf = (parsed.path.rsplit("/", 1)[-1] or "document").strip()
    leaf = "".join(c for c in leaf if c.isalnum() or c in ("-", "_", "."))
    if not leaf:
        leaf = "document"
    if not leaf.lower().endswith(".pdf"):
        leaf += ".pdf"
    if len(leaf) > 100:
        root, ext = (leaf.rsplit(".", 1) + [""])[:2]
        leaf = root[:95] + ("." + ext if ext else "")
    return f"{host}__{leaf}"

def extract_links(page_url: str, session: requests.Session) -> list[str]:
    """Extract all href links from a page."""
    try:
        r = session.get(page_url, timeout=TIMEOUT)
        r.raise_for_status()
    except Exception:
        return []
    html = r.text
    hrefs = re.findall(r'href=[\'\"]?([^\'\" >]+)', html, flags=re.IGNORECASE)
    links = [urljoin(page_url, href) for href in hrefs]
    return links

def save_pdf(content: bytes, out_dir: Path, suggested_name: str, digest: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{digest}__{suggested_name}"
    final = out_dir / fname
    if final.exists():
        return final
    tmp = final.with_suffix(final.suffix + ".part")
    with open(tmp, "wb") as f:
        f.write(content)
    tmp.replace(final)
    return final

def main() -> int:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    base_host = urlparse(SITE).netloc
    visited: set[str] = set()
    queue: deque[str] = deque([SITE])
    pdfs: set[str] = set()

    manifest = OUTPUT_DIR / "manifest.csv"
    errors = OUTPUT_DIR / "errors.csv"
    new_manifest = not manifest.exists()

    success = 0
    start = time.time()

    with open(manifest, "a", newline="", encoding="utf-8") as mf, open(errors, "a", newline="", encoding="utf-8") as ef:
        mwr = csv.writer(mf)
        ewr = csv.writer(ef)
        if new_manifest:
            mwr.writerow(["url", "saved_path", "sha256", "size_bytes", "status_code", "content_type", "fetched_at"])
        if errors.stat().st_size == 0:
            ewr.writerow(["url", "error", "when_ts"])

        while queue:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            try:
                if url.lower().endswith(".pdf"):
                    if url in pdfs:
                        continue
                    resp = session.get(url, timeout=TIMEOUT)
                    resp.raise_for_status()
                    blob = resp.content
                    digest = sha256_bytes(blob)
                    suggested = safe_name_from_url(url)
                    saved = save_pdf(blob, OUTPUT_DIR, suggested, digest)
                    mwr.writerow([
                        url,
                        str(saved),
                        digest,
                        len(blob),
                        resp.status_code,
                        resp.headers.get("content-type", ""),
                        int(time.time()),
                    ])
                    success += 1
                    pdfs.add(url)
                    print(f"[{success}] Saved: {saved.name}")
                else:
                    if urlparse(url).netloc == base_host:
                        links = extract_links(url, session)
                        for link in links:
                            if link not in visited:
                                queue.append(link)
            except Exception as e:
                ewr.writerow([url, repr(e), int(time.time())])
                print(f"ERROR: {url} -> {e}")

    elapsed = time.time() - start
    print(f"Done. {success} PDFs saved in {elapsed:.1f}s -> {OUTPUT_DIR}")
    return 0 if success > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())