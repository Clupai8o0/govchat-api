#!/usr/bin/env python3
"""
gov_data_scraper_js.py

A crawler for *.gov.au that can render JavaScript pages with Playwright,
interact with common UI controls (e.g., "Explore", "Show more"), and extract
links to CSV/XLS/XLSX/PDF files. It records file URL plus page context
(title, description, tags, anchor text) and prints progress while crawling.

Outputs: <outfile>.csv and <outfile>.jsonl

Usage:
    # Install deps:
    pip install requests beautifulsoup4 playwright
    playwright install

    # Crawl ATO on data.gov.au with JS rendering and dropdown clicking
    python gov_data_scraper_js.py --seeds https://data.gov.au/organisation/ato \
      --use-js --same-domain-only --allow-zip --max-pages 400 --max-files 300 \
      --outfile results_ato_js

    # Mixed crawl across ABS + Education, JS only on configured domains
    python gov_data_scraper_js.py --seeds https://www.abs.gov.au/ https://www.education.gov.au/higher-education-statistics \
      --use-js --js-domains data.gov.au abs.gov.au education.gov.au \
      --max-pages 800 --max-files 300 --outfile results_abs_edu_js

"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import traceback
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from html import unescape
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup  # type: ignore
from urllib import robotparser

# Try to import Playwright (optional)
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError  # type: ignore
    HAVE_PLAYWRIGHT = True
except Exception:
    HAVE_PLAYWRIGHT = False

DEFAULT_USER_AGENT = "GovHack-DataScraper-JS/1.0 (+https://govhack.org)"
DATA_FILE_EXTS = {".csv", ".xls", ".xlsx", ".pdf"}
ARCHIVE_FILE_EXTS = {".zip"}  # gated by --allow-zip
HTML_MIME_TYPES = {"text/html", "application/xhtml+xml"}
DATA_MIME_HINTS = {
    "text/csv": ".csv",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/pdf": ".pdf",
    "application/zip": ".zip",
}

@dataclass
class FileHit:
    file_url: str
    file_ext: str
    page_url: str
    page_title: str
    page_description: str
    page_tags: List[str]
    anchor_text: str
    content_type: str
    content_length: Optional[int]
    discovered_at: str  # ISO timestamp


def is_gov_au(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.endswith(".gov.au") or netloc == "gov.au"
    except Exception:
        return False


def normalize_url(url: str, base: Optional[str] = None) -> str:
    if base:
        url = urljoin(base, url)
    url, _ = urldefrag(url)
    return url


def guess_ext_from_url(url: str, allow_zip: bool) -> Optional[str]:
    path = urlparse(url).path.lower()
    for ext in sorted(DATA_FILE_EXTS | (ARCHIVE_FILE_EXTS if allow_zip else set()), key=len, reverse=True):
        if path.endswith(ext):
            return ext
    return None


def looks_like_data_file(url: str, allow_zip: bool) -> bool:
    ext = guess_ext_from_url(url, allow_zip=allow_zip)
    return ext is not None


def get_robot_parser(cache: Dict[str, robotparser.RobotFileParser], url: str) -> robotparser.RobotFileParser:
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin in cache:
        return cache[origin]
    rp_url = origin.rstrip("/") + "/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(rp_url)
        rp.read()
    except Exception:
        pass
    cache[origin] = rp
    return rp


def allowed_by_robots(rp: robotparser.RobotFileParser, ua: str, url: str) -> bool:
    try:
        return rp.can_fetch(ua, url)
    except Exception:
        return True


def extract_text(s: Optional[str]) -> str:
    return unescape((s or "").strip())


def extract_page_metadata(soup: BeautifulSoup) -> Tuple[str, str, List[str]]:
    title = ""
    if soup.title and soup.title.string:
        title = extract_text(soup.title.string)

    desc = ""
    for sel in [
        ("meta", {"name": "description"}),
        ("meta", {"property": "og:description"}),
        ("meta", {"name": "twitter:description"}),
    ]:
        m = soup.find(sel[0], attrs=sel[1])
        if m and m.get("content"):
            desc = extract_text(m["content"])
            break
    if not desc:
        p = soup.find("p")
        if p:
            desc = extract_text(p.get_text(" ", strip=True))[:500]

    # tags
    tags: List[str] = []

    def add(vals: Iterable[str]):
        for v in vals:
            v = extract_text(v)
            if v and v.lower() not in [t.lower() for t in tags]:
                tags.append(v)

    mkw = soup.find("meta", attrs={"name": "keywords"})
    if mkw and mkw.get("content"):
        add([t for t in mkw["content"].split(",") if t.strip()])

    for prop in ["article:tag", "og:video:tag", "og:audio:tag"]:
        for m in soup.find_all("meta", attrs={"property": prop}):
            if m.get("content"):
                add([m["content"]])

    for n in ["dcterms.subject", "dc.subject"]:
        for m in soup.find_all("meta", attrs={"name": n}):
            if m.get("content"):
                add([m["content"]])

    for t in soup.find_all("a", attrs={"rel": lambda v: v and "tag" in v}):
        add([t.get_text(" ", strip=True)])

    for cls in ["tag", "tags", "tag-list", "keywords", "facet", "topic"]:
        for el in soup.select(f".{cls} a, .{cls} li, .{cls} span"):
            txt = el.get_text(" ", strip=True)
            if txt:
                add([txt])

    for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(s.string or "")
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for it in items:
            if not isinstance(it, dict):
                continue
            if "keywords" in it:
                kw = it["keywords"]
                if isinstance(kw, str):
                    add([k for k in kw.split(",") if k.strip()])
                elif isinstance(kw, list):
                    add([str(k) for k in kw if str(k).strip()])
            if "about" in it:
                ab = it["about"]
                if isinstance(ab, list):
                    add([a.get("name") if isinstance(a, dict) else str(a) for a in ab])
                elif isinstance(ab, dict):
                    add([ab.get("name", "")])

    return title, desc, tags


def get_with_head_fallback(session: requests.Session, url: str, timeout: float = 15.0) -> Optional[requests.Response]:
    try:
        r = session.head(url, allow_redirects=True, timeout=timeout)
        if r.status_code >= 400 or not r.headers.get("Content-Type"):
            raise requests.RequestException()
        return r
    except Exception:
        try:
            r = session.get(url, allow_redirects=True, timeout=timeout, stream=True)
            return r
        except Exception:
            return None


# ----------------- Playwright helpers -----------------

def should_use_js(url: str, use_js: bool, js_domains: Set[str]) -> bool:
    if not use_js or not HAVE_PLAYWRIGHT:
        return False
    host = urlparse(url).netloc.lower()
    return any(host == d or host.endswith("." + d) for d in js_domains)


def click_expanders(page) -> int:
    """Try to expand common controls to reveal hidden download links."""
    labels = [
        "Show more",
        "Load more",
        "Explore",
        "More information",
        "More info",
        "View more",
        "Resources",
        "Expand",
        "Open",
    ]
    clicked = 0
    # Open all <details> elements
    try:
        page.evaluate("""() => { document.querySelectorAll('details').forEach(d => d.open = true); }""")
    except Exception:
        pass

    for lab in labels:
        try:
            loc = page.locator(f"text={lab}")
            count = loc.count()
            for i in range(count):
                try:
                    loc.nth(i).click(timeout=1000)
                    page.wait_for_timeout(150)
                    clicked += 1
                except Exception:
                    pass
        except Exception:
            pass

    # Specifically handle data.gov.au "Explore" toggles (buttons with dropdowns)
    try:
        toggles = page.locator("[aria-haspopup='true'], button:has-text('Explore')")
        for i in range(toggles.count()):
            try:
                toggles.nth(i).click(timeout=1000)
                page.wait_for_timeout(150)
                clicked += 1
            except Exception:
                pass
    except Exception:
        pass
    return clicked


def render_and_collect_links(play, browser, url: str, timeout_ms: int = 30000) -> Tuple[str, List[Tuple[str, str]]]:
    """Return page HTML and list of (href, anchor_text) after basic interactions."""
    page = browser.new_page()
    try:
        page.set_default_timeout(timeout_ms)
        page.goto(url, wait_until="domcontentloaded")
        try:
            page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except Exception:
            pass

        # Gentle scroll to bottom to trigger lazy loads
        try:
            for _ in range(12):
                page.mouse.wheel(0, 2000)
                page.wait_for_timeout(150)
        except Exception:
            pass

        expanded = click_expanders(page)
        if expanded:
            page.wait_for_timeout(300)

        html = page.content()
        anchors = page.locator("a[href]")
        links: List[Tuple[str, str]] = []
        count = anchors.count()
        for i in range(min(count, 4000)):  # safety cap
            try:
                href = anchors.nth(i).get_attribute("href") or ""
                txt = anchors.nth(i).inner_text().strip()
                if href:
                    links.append((href, txt))
            except Exception:
                pass
        return html, links
    finally:
        try:
            page.close()
        except Exception:
            pass


# ----------------- Crawler -----------------

def crawl(
    seeds: List[str],
    max_pages: int,
    max_files: int,
    same_domain_only: bool,
    allow_zip: bool,
    per_domain_delay: float,
    user_agent: str,
    request_timeout: float,
    use_js: bool,
    js_domains: Set[str],
    headless: bool,
) -> List[FileHit]:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept": "*/*"})

    q: Deque[str] = deque()
    visited: Set[str] = set()
    enqueued: Set[str] = set()
    files_found: List[FileHit] = []

    for s in seeds:
        s_norm = normalize_url(s)
        if s_norm not in enqueued:
            q.append(s_norm); enqueued.add(s_norm)

    rp_cache: Dict[str, robotparser.RobotFileParser] = {}
    next_allowed: Dict[str, float] = defaultdict(float)
    seeds_domains = {urlparse(s).netloc for s in seeds}
    pages_processed = 0

    # Init Playwright if requested
    pw = None
    browser = None
    if use_js and HAVE_PLAYWRIGHT:
        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=headless)

    try:
        while q and pages_processed < max_pages and len(files_found) < max_files:
            url = q.popleft()
            if url in visited:
                continue

            if not is_gov_au(url):
                continue

            if same_domain_only and urlparse(url).netloc not in seeds_domains:
                continue

            rp = get_robot_parser(rp_cache, url)
            if not allowed_by_robots(rp, user_agent, url):
                print(f"[skip robots] {url}")
                continue

            domain = urlparse(url).netloc
            now = time.time()
            next_ok = next_allowed.get(domain, 0.0)
            if now < next_ok:
                time.sleep(max(0.0, next_ok - now))
            next_allowed[domain] = time.time() + per_domain_delay

            pages_processed += 1
            print(f"[page {pages_processed}/{max_pages}] Crawling: {url}", flush=True)

            html = None
            data_links: List[Tuple[str, str]] = []
            page_links: List[str] = []

            try:
                if should_use_js(url, use_js, js_domains) and browser is not None:
                    html, rendered_links = render_and_collect_links(pw, browser, url, timeout_ms=int(request_timeout*1000))
                    for href, txt in rendered_links:
                        full = normalize_url(href, base=url)
                        data_links.append((full, txt))
                        page_links.append(full)
                else:
                    r = session.get(url, timeout=request_timeout, allow_redirects=True)
                    ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                    if ctype not in HTML_MIME_TYPES and ctype != "":
                        visited.add(url)
                        continue
                    html = r.text
                    soup = BeautifulSoup(html, "html.parser")
                    for a in soup.find_all("a", href=True):
                        full = normalize_url(a["href"], base=url)
                        txt = a.get_text(" ", strip=True) or ""
                        page_links.append(full)
                        data_links.append((full, txt))
            except Exception as e:
                print(f"[warn] Failed to fetch: {url} ({e})", flush=True)
                visited.add(url)
                continue

            if not html:
                visited.add(url)
                continue

            soup = BeautifulSoup(html, "html.parser")
            title, desc, tags = extract_page_metadata(soup)

            # Enqueue new page links
            for link in page_links:
                if looks_like_data_file(link, allow_zip):
                    pass
                else:
                    if is_gov_au(link):
                        if same_domain_only and urlparse(link).netloc not in seeds_domains:
                            continue
                        if link not in visited and link not in enqueued:
                            enqueued.add(link); q.append(link)

            # Process data file links
            for link, anchor_text in data_links:
                if not looks_like_data_file(link, allow_zip):
                    continue

                fr = get_with_head_fallback(session, link, timeout=request_timeout)
                if fr is None:
                    continue

                ctype = (fr.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                clen = fr.headers.get("Content-Length")
                try:
                    clen_int = int(clen) if clen is not None else None
                except ValueError:
                    clen_int = None

                ext = guess_ext_from_url(link, allow_zip=allow_zip) or DATA_MIME_HINTS.get(ctype, "")
                if not ext:
                    continue

                print(f"[file {len(files_found)+1}] {link}  (from {url})", flush=True)

                hit = FileHit(
                    file_url=link,
                    file_ext=ext,
                    page_url=url,
                    page_title=title,
                    page_description=desc,
                    page_tags=tags,
                    anchor_text=anchor_text,
                    content_type=ctype,
                    content_length=clen_int,
                    discovered_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
                files_found.append(hit)
                if len(files_found) >= max_files:
                    break

            visited.add(url)

    finally:
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass
        if pw is not None:
            try:
                pw.stop()
            except Exception:
                pass

    return files_found


def save_results(results: List[FileHit], outfile_prefix: str) -> None:
    csv_path = f"{outfile_prefix}.csv"
    jsonl_path = f"{outfile_prefix}.jsonl"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "file_url",
            "file_ext",
            "page_url",
            "page_title",
            "page_description",
            "page_tags",
            "anchor_text",
            "content_type",
            "content_length",
            "discovered_at",
        ])
        for h in results:
            w.writerow([
                h.file_url,
                h.file_ext,
                h.page_url,
                h.page_title,
                h.page_description,
                "; ".join(h.page_tags),
                h.anchor_text,
                h.content_type,
                h.content_length if h.content_length is not None else "",
                h.discovered_at,
            ])
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for h in results:
            f.write(json.dumps(asdict(h), ensure_ascii=False) + "\n")
    print(f"[done] Files found: {len(results)} -> {csv_path}, {jsonl_path}")


def main():
    ap = argparse.ArgumentParser(description="Crawl *.gov.au (with optional JS rendering) to collect CSV/XLS/XLSX/PDF links.")
    ap.add_argument("--seeds", nargs="+", required=True, help="Seed URLs to start crawling from (space-separated).")
    ap.add_argument("--max-pages", type=int, default=500, help="Maximum number of HTML pages to crawl.")
    ap.add_argument("--max-files", type=int, default=500, help="Maximum number of file links to collect.")
    ap.add_argument("--outfile", type=str, default="gov_data_results_js", help="Output file prefix (without extension).")
    ap.add_argument("--same-domain-only", action="store_true", help="Only follow links within the exact seed domains.")
    ap.add_argument("--allow-zip", action="store_true", help="Also collect .zip archives that may contain CSVs.")
    ap.add_argument("--delay", type=float, default=1.0, help="Minimum delay per domain between requests (seconds).")
    ap.add_argument("--timeout", type=float, default=50.0, help="HTTP/Render timeout (seconds).")
    ap.add_argument("--user-agent", type=str, default=DEFAULT_USER_AGENT, help="Custom User-Agent string.")
    ap.add_argument("--use-js", action="store_true", help="Enable Playwright JS rendering + UI interactions.")
    ap.add_argument("--js-domains", nargs="*", default=["data.gov.au", "abs.gov.au", "ato.gov.au", "education.gov.au", "aihw.gov.au"],
                    help="Domains to render with Playwright (exact or subdomains).")
    ap.add_argument("--no-headless", action="store_true", help="Run browser in headed mode for debugging.")
    args = ap.parse_args()

    if args.use_js and not HAVE_PLAYWRIGHT:
        print("⚠️ Playwright not installed. Install with:\n  pip install playwright\n  playwright install", file=sys.stderr)

    try:
        results = crawl(
            seeds=args.seeds,
            max_pages=args.max_pages,
            max_files=args.max_files,
            same_domain_only=args.same_domain_only,
            allow_zip=args.allow_zip,
            per_domain_delay=args.delay,
            user_agent=args.user_agent,
            request_timeout=args.timeout,
            use_js=args.use_js and HAVE_PLAYWRIGHT,
            js_domains=set(args.js_domains),
            headless=not args.no_headless,
        )
        save_results(results, args.outfile)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
    except Exception as e:
        print("Error during crawl:", e, file=sys.stderr)
        traceback.print_exc()


if __name__ == "__main__":
    main()
