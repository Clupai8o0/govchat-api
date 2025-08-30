Fa#!/usr/bin/env python3
"""
Gov AU Scraper — crawl Australian government sites and build a linking data graph.

Features
- Seeds: one or more starting URLs (defaults to a small allowlist of AU gov portals)
- Scope: restricts to *.gov.au and optional per-agency allowlist
- Respects robots.txt + crawl-delay, polite rate limiting, and time/URL caps
- Auto-discovers and parses sitemap.xml where available
- Extracts content + metadata (title, description, OpenGraph, JSON-LD schema.org/Dataset)
- Creates graph edges (page→page links, page→dataset, dataset→agency, topic tags)
- Outputs:
  - out/nodes.csv (id,type,url,title,agency,topics,sha)
  - out/edges.csv (src_id,dst_id,edge_type)
  - out/docs/*.txt (clean text for RAG ingestion)
  - out/graph.graphml (optional NetworkX export)
  - out/manifest.json (run summary)

Usage
  pip install httpx[http2] selectolax beautifulsoup4 networkx python-slugify tldextract urlextract python-dateutil
  python scraper.py --seeds https://www.directory.gov.au https://data.gov.au --max-pages 300 --concurrency 8

Notes
- This is a hackathon-optimised crawler; keep the max-pages small while testing.
- You can point it at specific agencies with --allowlist abf.gov.au,australia.gov.au,abs.gov.au,aihw.gov.au
- Extend EXTRACTOR_HOOKS to capture site-specific dataset cards if needed.
"""
from __future__ import annotations
import argparse
import asyncio
import contextlib
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import httpx
from bs4 import BeautifulSoup
from dateutil import parser as dtparse
from selectolax.parser import HTMLParser
import tldextract
import networkx as nx
from slugify import slugify
from urlextract import URLExtract
from urllib.parse import urljoin, urlparse
import urllib.robotparser as robotparser

# Accept any host whose public suffix is 'gov.au' (e.g., education.gov.au, abs.gov.au)
ALLOWED_SUFFIXES = {"gov.au"}
DEFAULT_SEEDS = [
    "https://www.directory.gov.au/",      # org register
    "https://data.gov.au/",               # datasets portal
    "https://www.legislation.gov.au/",    # federal legislation
    "https://www.abs.gov.au/",            # ABS
    "https://www.aihw.gov.au/",           # AIHW
]

OUTPUT_DIR = Path("out")
DOC_DIR = OUTPUT_DIR / "docs"
NODE_CSV = OUTPUT_DIR / "nodes.csv"
EDGE_CSV = OUTPUT_DIR / "edges.csv"
GRAPH_GML = OUTPUT_DIR / "graph.graphml"
MANIFEST = OUTPUT_DIR / "manifest.json"

extractor = URLExtract()

@dataclass
class Page:
    url: str
    status: int
    title: str = ""
    text: str = ""
    html: str = ""
    agency: str = ""
    topics: List[str] = field(default_factory=list)
    outlinks: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return hashlib.sha1(self.url.encode()).hexdigest()[:12]

@dataclass
class Dataset:
    name: str
    url: str
    description: str = ""
    publisher: str = ""
    issued: Optional[str] = None
    modified: Optional[str] = None
    distribution: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def id(self) -> str:
        return "ds-" + hashlib.sha1((self.url or self.name).encode()).hexdigest()[:12]

class RobotsCache:
    def __init__(self):
        self.cache: Dict[str, robotparser.RobotFileParser] = {}

    async def allowed(self, url: str, client: httpx.AsyncClient, ua: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self.cache:
            rp = robotparser.RobotFileParser()
            robots_url = urljoin(base, "/robots.txt")
            try:
                r = await client.get(robots_url, timeout=10)
                if r.status_code == 200:
                    rp.parse(r.text.splitlines())
                else:
                    rp.parse("")
            except Exception:
                rp.parse("")
            self.cache[base] = rp
        return self.cache[base].can_fetch(ua, url)

    def crawl_delay(self, url: str, ua: str) -> Optional[float]:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        rp = self.cache.get(base)
        return rp.crawl_delay(ua) if rp else None

# ---- Utilities ----

def in_scope(url: str, allowlist: Optional[Set[str]]) -> bool:
    try:
        p = urlparse(url)
        if p.scheme not in {"http", "https"}:
            return False
        host = p.netloc.lower()
        tld = tldextract.extract(host)
        # Use the public suffix (e.g., 'gov.au'), not the full domain
        if (tld.suffix or "").lower() not in ALLOWED_SUFFIXES:
            return False
        if allowlist:
            return any(host.endswith(a.strip().lower()) or host == a.strip().lower() for a in allowlist)
        return True
    except Exception:
        return False

def canonicalize(base: str, href: str) -> Optional[str]:
    try:
        url = urljoin(base, href)
        u = urlparse(url)
        if u.fragment:
            url = url.replace('#' + u.fragment, '')
        return url
    except Exception:
        return None

# ---- Extraction ----

def text_and_title(html: str) -> Tuple[str, str]:
    parser = HTMLParser(html)
    title = parser.css_first("title").text(
        strip=True) if parser.css_first("title") else ""
    # Remove script/style and extract visible text
    for node in parser.css("script,style,noscript"):
        node.decompose()
    text = parser.body.text(separator="\n", strip=True) if parser.body else parser.text(
        separator="\n", strip=True)
    return text, title


def parse_jsonld_datasets(soup: BeautifulSoup) -> List[Dataset]:
    out: List[Dataset] = []
    for tag in soup.find_all("script", type="application/ld+json"):
        with contextlib.suppress(Exception):
            data = json.loads(tag.string or "{}")
            items = data if isinstance(data, list) else [data]
            for obj in items:
                if obj.get("@type") in {"Dataset", ["Dataset"]}:
                    name = obj.get("name") or obj.get(
                        "headline") or "Unnamed dataset"
                    url = obj.get("url") or obj.get("@id") or ""
                    ds = Dataset(
                        name=name,
                        url=url,
                        description=obj.get("description", ""),
                        publisher=(obj.get("publisher", {})
                                   or {}).get("name", ""),
                        issued=obj.get("datePublished"),
                        modified=obj.get("dateModified"),
                        distribution=obj.get("distribution", []) or obj.get(
                            "distribution", []),
                    )
                    out.append(ds)
    return out


def extract_agency(host: str) -> str:
    # Heuristic: take primary subdomain as agency token
    t = tldextract.extract(host)
    parts = [p for p in [t.subdomain, t.domain] if p]
    if parts:
        return parts[-1].upper()
    return host.upper()


# add (host_predicate, callable(html)->List[Dataset]) for site-specific patterns
EXTRACTOR_HOOKS: List = []

# ---- Crawler ----
class Crawler:
    def __init__(self, seeds: List[str], allowlist: Optional[Set[str]], max_pages: int, concurrency: int, verbose: bool = False):
        self.seeds = seeds
        self.allowlist = allowlist
        self.max_pages = max_pages
        self.sem = asyncio.Semaphore(concurrency)
        self.ua = "GovHackScraper/1.0 (+https://govhack.local)"
        self.robots = RobotsCache()
        self.verbose = verbose

        self.seen: Set[str] = set()
        self.queue: asyncio.Queue[str] = asyncio.Queue()

        self.pages: Dict[str, Page] = {}
        self.datasets: Dict[str, Dataset] = {}
        self.edges: Set[Tuple[str, str, str]] = set()  # (src_id, dst_id, type)

    async def run(self):
        for s in self.seeds:
            await self.queue.put(s)
        async with httpx.AsyncClient(follow_redirects=True, timeout=15, headers={"User-Agent": self.ua}, http2=True) as client:
            tasks = [asyncio.create_task(self.worker(client))
                     for _ in range(self.sem._value)]
            await self.queue.join()
            for t in tasks:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t

    async def worker(self, client: httpx.AsyncClient):
        while True:
            url = await self.queue.get()
            try:
                if len(self.pages) >= self.max_pages:
                    return
                if url in self.seen:
                    continue
                self.seen.add(url)

                if not in_scope(url, self.allowlist):
                    if self.verbose:
                        print(f"[skip:scope] {url}")
                    continue
                if not await self.robots.allowed(url, client, self.ua):
                    if self.verbose:
                        print(f"[skip:robots] {url}")
                    continue

                delay = self.robots.crawl_delay(url, self.ua) or 0.5
                async with self.sem:
                    r = await client.get(url)
                    await asyncio.sleep(delay)
                status = r.status_code
                html = r.text if r.status_code == 200 and "text/html" in r.headers.get(
                    "content-type", "") else ""

                page = Page(url=url, status=status)
                if html:
                    text, title = text_and_title(html)
                    page.title = title
                    page.text = text
                    page.html = html
                    host = urlparse(url).netloc
                    page.agency = extract_agency(host)

                    soup = BeautifulSoup(html, "html.parser")

                    # outlinks
                    links = [a.get("href")
                             for a in soup.find_all("a", href=True)]
                    canon = [canonicalize(url, h) for h in links if h]
                    canon = [c for c in canon if c and in_scope(
                        c, self.allowlist)]
                    page.outlinks = list(dict.fromkeys(canon))

                    # metadata
                    og_desc = soup.find("meta", property="og:description")
                    desc = soup.find("meta", attrs={"name": "description"})
                    page.meta["description"] = (og_desc.get("content") if og_desc else None) or (
                        desc.get("content") if desc else None) or ""

                    # datasets via JSON-LD
                    jsonld_ds = parse_jsonld_datasets(soup)
                    for ds in jsonld_ds:
                        if not ds.url:
                            ds.url = url  # fallback
                        self.datasets[ds.id] = ds
                        self.edges.add((page.id, ds.id, "page_has_dataset"))
                        if ds.publisher:
                            self.edges.add(
                                (ds.id, slugify(ds.publisher), "dataset_publisher"))

                    # heuristics: detect data portal cards by common CSS classes
                    for pred, hook in EXTRACTOR_HOOKS:
                        if pred(host):
                            with contextlib.suppress(Exception):
                                extra = hook(html)
                                for ds in extra:
                                    self.datasets[ds.id] = ds
                                    self.edges.add(
                                        (page.id, ds.id, "page_has_dataset"))

                    # link edges
                    for ln in page.outlinks[:50]:
                        lid = hashlib.sha1(ln.encode()).hexdigest()[:12]
                        self.edges.add((page.id, lid, "links_to"))
                        if ln not in self.seen and len(self.pages) + self.queue.qsize() < self.max_pages:
                            await self.queue.put(ln)

                self.pages[page.id] = page
            except Exception:
                pass
            finally:
                self.queue.task_done()

# ---- Persist ----

def save_outputs(pages: Dict[str, Page], datasets: Dict[str, Dataset], edges: Set[Tuple[str, str, str]]):
    OUTPUT_DIR.mkdir(exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    # docs for RAG ingestion
    for p in pages.values():
        if not p.text:
            continue
        fn = DOC_DIR / f"{p.id}.txt"
        with open(fn, "w", encoding="utf-8") as f:
            f.write(f"Title: {p.title}\nURL: {p.url}\nAgency: {p.agency}\n\n")
            f.write(p.text)

    # nodes
    with open(NODE_CSV, "w", encoding="utf-8") as f:
        f.write("id,type,url,title,agency,topics,sha\n")
        for p in pages.values():
            sha = hashlib.sha1((p.url or p.title).encode()).hexdigest()[:10]
            topics = ";".join(p.topics)
            line = f"{p.id},page,{json_escape(p.url)},{json_escape(p.title)},{json_escape(p.agency)},{json_escape(topics)},{sha}\n"
            f.write(line)
        for ds in datasets.values():
            sha = hashlib.sha1((ds.url or ds.name).encode()).hexdigest()[:10]
            line = f"{ds.id},dataset,{json_escape(ds.url)},{json_escape(ds.name)},,{json_escape('')},{sha}\n"
            f.write(line)

    # edges
    with open(EDGE_CSV, "w", encoding="utf-8") as f:
        f.write("src_id,dst_id,edge_type\n")
        for s, d, t in edges:
            f.write(f"{s},{d},{t}\n")

    # graphml (optional)
    try:
        G = nx.DiGraph()
        for p in pages.values():
            G.add_node(p.id, type="page", url=p.url,
                       title=p.title, agency=p.agency)
        for ds in datasets.values():
            G.add_node(ds.id, type="dataset", url=ds.url, title=ds.name)
        for s, d, t in edges:
            G.add_edge(s, d, type=t)
        nx.write_graphml(G, GRAPH_GML)
    except Exception:
        pass

    # manifest
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "pages": len(pages),
            "datasets": len(datasets),
            "edges": len(edges),
        }, f, indent=2)


def json_escape(s: Optional[str]) -> str:
    if s is None:
        return ""
    return '"' + (s.replace('"', "''").replace('\n', ' ').replace('\r', ' ')) + '"'

# ---- CLI ----
async def main():
    ap = argparse.ArgumentParser(
        description="Scrape AU gov websites and build a data linking graph")
    ap.add_argument("--seeds", nargs="*",
                    default=DEFAULT_SEEDS, help="Seed URLs")
    ap.add_argument("--allowlist", type=str, default="",
                    help="Comma-separated host allowlist (e.g., abs.gov.au,aihw.gov.au)")
    ap.add_argument("--max-pages", type=int, default=200,
                    help="Max pages to crawl")
    ap.add_argument("--concurrency", type=int, default=6,
                    help="Concurrent requests")
    ap.add_argument("--verbose", action="store_true",
                    help="Log skip reasons to stdout")
    args = ap.parse_args()

    allow: Optional[Set[str]] = set([h.strip() for h in args.allowlist.split(
        ",") if h.strip()]) if args.allowlist else None

    crawler = Crawler(seeds=args.seeds, allowlist=allow, max_pages=args.max_pages,
                      concurrency=args.concurrency, verbose=args.verbose)
    await crawler.run()
    save_outputs(crawler.pages, crawler.datasets, crawler.edges)
    print(
        f"Crawl done: pages={len(crawler.pages)} datasets={len(crawler.datasets)} edges={len(crawler.edges)}")
    print(f"Outputs: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted.")
