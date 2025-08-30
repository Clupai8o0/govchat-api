#!/usr/bin/env python3
"""
RAG-friendly web crawler for GovHack-style sites.

Features:
- BFS crawl with domain scoping and depth limit
- Optional JS rendering via Playwright (--js)
- Progress prints + basic robots.txt compliance
- Extracts title, meta description, clean text
- Outputs CSV with one row per chunk (with overlap)
- Records non-HTML resources (pdf/csv/xlsx) with metadata

Usage examples:
  python rag_crawler.py --seeds https://hackerspace.govhack.org/challenges/making-ai-decisions-understandable-and-clear \
                        --output out.csv --max-pages 300 --concurrency 6 --chunk-size 1400 --chunk-overlap 200

  # With JS rendering (heavier but handles dynamic content)
  python rag_crawler.py --seeds https://hackerspace.govhack.org/challenges/... \
                        --output out.csv --js --max-pages 150 --concurrency 3
"""

import argparse
import asyncio
import csv
import html
import re
import sys
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode

import aiohttp
from bs4 import BeautifulSoup
import urllib.robotparser as robotparser

# ---------- Optional (only used if --js is enabled) ----------
try:
    from playwright.async_api import async_playwright
except Exception:
    async_playwright = None
# ------------------------------------------------------------

BINARY_EXTS = {".pdf", ".csv", ".xls", ".xlsx"}
TEXT_MIME_HINTS = ("text/html", "application/xhtml+xml")
DEFAULT_HEADERS = {
    "User-Agent": "GovHack-RAG-Crawler/1.0 (+https://govhack.org)"
}

def now():
    return time.strftime("%H:%M:%S")

def normalize_url(u: str) -> str:
    """Normalize URL: strip fragments, remove common tracking params, resolve ../, etc."""
    try:
        parsed = urlparse(u)
        if not parsed.scheme or not parsed.netloc:
            return u  # we'll join later if needed

        # strip fragment
        parsed = parsed._replace(fragment="")

        # remove common tracking params
        q = parse_qsl(parsed.query, keep_blank_values=True)
        filtered = [(k, v) for (k, v) in q if not k.lower(
        ).startswith(("utm_", "gclid", "fbclid"))]
        parsed = parsed._replace(query=urlencode(filtered, doseq=True))

        # remove trailing slash duplication for root
        norm = urlunparse(parsed)
        return norm
    except Exception:
        return u

def same_registered_domain(a: str, b: str) -> bool:
    pa, pb = urlparse(a), urlparse(b)
    return pa.netloc == pb.netloc

def extract_title_and_description(soup: BeautifulSoup):
    title = (soup.title.get_text(strip=True) if soup.title else "") or ""
    desc = ""
    # meta description
    m = soup.find("meta", attrs={"name": "description"})
    if m and m.get("content"):
        desc = m["content"].strip()
    if not desc:
        # fall back: first meaningful <p>
        p = soup.find("p")
        if p:
            desc = p.get_text(" ", strip=True)
    return title, desc

def visible_text_from_html(soup: BeautifulSoup) -> str:
    # Remove non-contenty tags
    for tag in soup(["script", "style", "noscript", "template", "header", "footer", "nav", "aside"]):
        tag.decompose()
    # Keep headings, paragraphs, list items, table cells
    text_parts = []
    selectors = ["h1", "h2", "h3", "h4", "h5",
                 "h6", "p", "li", "dt", "dd", "th", "td"]
    for sel in selectors:
        for el in soup.select(sel):
            t = el.get_text(" ", strip=True)
            if t:
                text_parts.append(t)
    text = "\n".join(text_parts)
    # Clean multiple spaces, decode entities
    text = html.unescape(re.sub(r"[ \t]+", " ", text))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, max_chars: int = 1400, overlap: int = 200):
    """
    Simple character-based chunking with overlap for RAG.
    Keeps chunk boundaries on whitespace when possible.
    """
    if not text:
        return []

    if max_chars <= 0:
        return [text]

    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + max_chars, n)
        # try not to cut in the middle of a word/sentence
        if end < n:
            # backtrack to nearest whitespace/newline within 120 chars
            window_start = max(i, end - 120)
            window = text[window_start:end]
            m = max(window.rfind("\n"), window.rfind(". "),
                    window.rfind(" "), window.rfind("\t"))
            if m != -1:
                end = window_start + m + 1
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        i = max(0, end - overlap)
    return chunks

def is_binary_link(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in BINARY_EXTS)

def extract_links(base_url: str, soup: BeautifulSoup):
    hrefs = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:")):
            continue
        abs_url = urljoin(base_url, href)
        hrefs.add(normalize_url(abs_url))
    return hrefs

async def fetch_html_aiohttp(session, url, timeout=20):
    try:
        async with session.get(url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True) as resp:
            ct = resp.headers.get("content-type", "")
            status = resp.status
            if any(h in ct for h in TEXT_MIME_HINTS):
                txt = await resp.text(errors="ignore")
                return status, ct, txt, str(resp.url)
            else:
                return status, ct, None, str(resp.url)
    except Exception as e:
        return 0, "", None, url

async def render_html_playwright(browser, url, timeout_ms=25000):
    page = await browser.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        # Try to settle network a bit
        try:
            await page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass

        # Opportunistic expansion (buttons like "Load more", "Show more")
        candidates = [
            "button:has-text('Load more')",
            "button:has-text('Show more')",
            "button:has-text('Accept')",
            "button:has-text('I agree')",
            "button[aria-expanded='false']",
        ]
        for sel in candidates:
            btns = await page.locator(sel).all()
            for b in btns[:3]:
                try:
                    await b.click(timeout=2000)
                    await page.wait_for_timeout(300)
                except Exception:
                    pass

        content = await page.content()
        # Content-type isn't provided via Playwright easily; assume HTML
        return 200, "text/html", content, page.url
    except Exception:
        return 0, "", None, url
    finally:
        await page.close()

async def crawl(args):
    seeds = [normalize_url(s.strip())
             for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        print("No seeds provided.", file=sys.stderr)
        return

    allowed_domains = set()
    if args.allowed_domains:
        for d in args.allowed_domains.split(","):
            allowed_domains.add(d.strip().lower())
    else:
        # default: seed domains
        for s in seeds:
            allowed_domains.add(urlparse(s).netloc.lower())

    # robots.txt (best-effort single robots per domain)
    robots = {}

    def allowed_by_robots(url: str) -> bool:
        netloc = urlparse(url).netloc
        if netloc not in robots:
            rp = robotparser.RobotFileParser()
            rp.set_url(f"{urlparse(url).scheme}://{netloc}/robots.txt")
            try:
                rp.read()
            except Exception:
                pass
            robots[netloc] = rp
        try:
            return robots[netloc].can_fetch(DEFAULT_HEADERS["User-Agent"], url)
        except Exception:
            return True

    csv_file = open(args.output, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow([
        "url", "final_url", "status", "content_type", "source_type",
        "title", "description", "chunk_index", "chunk_total", "chunk_text",
        "depth"
    ])
    csv_lock = asyncio.Lock()

    visited = set()
    q = deque([(s, 0) for s in seeds])
    total_processed_pages = 0

    session_timeout = aiohttp.ClientTimeout(total=args.http_timeout)
    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        pw = None
        browser = None
        if args.js:
            if async_playwright is None:
                print(
                    "[!] Playwright not installed. Run: pip install playwright && playwright install", file=sys.stderr)
                return
            pw = await async_playwright().start()
            browser = await pw.chromium.launch(headless=True)

        sem = asyncio.Semaphore(args.concurrency)

        async def process_url(url: str, depth: int):
            nonlocal total_processed_pages
            norm = normalize_url(url)
            if norm in visited:
                return []
            visited.add(norm)

            if args.same_domain_only:
                if urlparse(norm).netloc.lower() not in allowed_domains:
                    return []

            if not allowed_by_robots(norm):
                print(f"[{now()}] ROBOTS blocked: {norm}")
                return []

            status, ct, html_text, final_url = (0, "", None, norm)

            # Binary resource? (record and skip parsing)
            if is_binary_link(norm):
                status = 200  # optimistic
                ct = "binary"
                source_type = "binary"
                async with csv_lock:
                    writer.writerow([norm, final_url, status, ct, source_type,
                                     "", "", "", "", "", depth])
                print(f"[{now()}] FILE: {norm}")
                return []

            # Fetch HTML (requests or JS)
            async with sem:
                if args.js:
                    status, ct, html_text, final_url = await render_html_playwright(browser, norm)
                else:
                    status, ct, html_text, final_url = await fetch_html_aiohttp(session, norm, timeout=args.http_timeout)

            if html_text is None or status == 0:
                # Not HTML or failed
                if ct and not any(h in ct for h in TEXT_MIME_HINTS):
                    # record non-HTML link (e.g., JSON/image)
                    async with csv_lock:
                        writer.writerow([norm, final_url, status, ct, "non-html",
                                         "", "", "", "", "", depth])
                print(f"[{now()}] SKIP ({status} {ct}): {norm}")
                return []

            soup = BeautifulSoup(html_text, "html.parser")
            title, desc = extract_title_and_description(soup)
            text = visible_text_from_html(soup)
            chunks = chunk_text(
                text, max_chars=args.chunk_size, overlap=args.chunk_overlap)
            total_chunks = len(chunks) if chunks else 0

            # Write chunks
            async with csv_lock:
                if total_chunks == 0:
                    writer.writerow([norm, final_url, status, ct, "html",
                                     title, desc, 0, 0, "", depth])
                else:
                    for i, ch in enumerate(chunks):
                        writer.writerow([norm, final_url, status, ct, "html",
                                         title, desc, i, total_chunks, ch, depth])

            total_processed_pages += 1
            print(
                f"[{now()}] OK  ({total_processed_pages}/{args.max_pages}) {norm}  chunks={total_chunks} depth={depth}")

            # Next links
            next_links = []
            if depth < args.max_depth:
                for link in extract_links(final_url, soup):
                    if args.same_domain_only and urlparse(link).netloc.lower() not in allowed_domains:
                        continue
                    # skip obvious binaries but still enqueue non-visited
                    next_links.append((link, depth + 1))
            return next_links

        try:
            pending = set()
            while q and total_processed_pages < args.max_pages:
                url, depth = q.popleft()
                # schedule
                pending.add(asyncio.create_task(process_url(url, depth)))

                # drain in batches to keep memory steady
                if len(pending) >= args.concurrency * 2 or not q:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        try:
                            results = task.result()
                            for nxt in results:
                                if nxt[0] not in visited and len(visited) + len(q) < args.max_pages * 4:
                                    q.append(nxt)
                        except Exception as e:
                            print(f"[{now()}] Worker error: {e}",
                                  file=sys.stderr)

            # finalize any remaining tasks
            if pending:
                done, _ = await asyncio.wait(pending)
                for task in done:
                    try:
                        results = task.result()
                        for nxt in results:
                            if nxt[0] not in visited and len(visited) + len(q) < args.max_pages * 4:
                                q.append(nxt)
                    except Exception as e:
                        print(f"[{now()}] Worker error: {e}", file=sys.stderr)
        finally:
            csv_file.flush()
            csv_file.close()
            if args.js and browser:
                await browser.close()
            if args.js and pw:
                await pw.stop()

def main():
    ap = argparse.ArgumentParser(description="RAG-friendly crawler → CSV")
    ap.add_argument("--seeds", required=True, help="Comma-separated seed URLs")
    ap.add_argument("--allowed-domains", default="",
                    help="Comma-separated allowed domains (defaults to seeds' domains)")
    ap.add_argument("--same-domain-only", action="store_true", default=True,
                    help="Restrict crawl to allowed domains (default: True)")
    ap.add_argument("--output", default="out.csv", help="CSV output path")
    ap.add_argument("--max-pages", type=int, default=300,
                    help="Max HTML pages to process")
    ap.add_argument("--max-depth", type=int, default=4,
                    help="Max crawl depth from seeds")
    ap.add_argument("--concurrency", type=int,
                    default=6, help="Concurrent fetches")
    ap.add_argument("--http-timeout", type=int, default=25,
                    help="Per-request timeout (seconds)")
    ap.add_argument("--chunk-size", type=int, default=1400,
                    help="Chunk size (characters)")
    ap.add_argument("--chunk-overlap", type=int, default=200,
                    help="Chunk overlap (characters)")
    ap.add_argument("--js", action="store_true",
                    help="Use Playwright to render JavaScript")
    args = ap.parse_args()

    # Ensure default for same-domain-only is True even if flag omitted
    if "--same-domain-only" not in sys.argv and "--no-same-domain-only" not in sys.argv:
        args.same_domain_only = True

    print(f"[{now()}] Starting crawl")
    print(f" Seeds: {args.seeds}")
    print(f" Output: {args.output}")
    print(f" JS rendering: {'ON' if args.js else 'OFF'}")
    try:
        asyncio.run(crawl(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)


if __name__ == "__main__":
    main()
