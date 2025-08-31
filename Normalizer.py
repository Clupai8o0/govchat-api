# Takes the data from the ABS API dataflows endpoint and stores the relevant data to the .csv file. 
from __future__ import annotations
import argparse, json, re, sys, hashlib
from pathlib import Path
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional
import pandas as pd

FREQ_MAP = {
    "A": "annual",
    "Q": "quarterly",
    "M": "monthly",
    "W": "weekly",
    "D": "daily"
}

def _load_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        return json.loads(text)
    except Exception as e:
        print(f"[WARN] Skipping {path.name}: not valid JSON ({e})", file=sys.stderr)
        return None

def _extract_dataflows(obj: Any) -> List[Dict[str, Any]]:
    """
    Expected ABS shape: { "data": { "dataflows": [ ... ] } }
    Fall back to known list keys if needed.
    """
    if isinstance(obj, dict):
        if isinstance(obj.get("data"), dict) and isinstance(obj["data"].get("dataflows"), list):
            return obj["data"]["dataflows"]
        for k in ("dataflows", "datasets", "results", "items", "records"):
            v = obj.get(k)
            if isinstance(v, list):
                return v
    if isinstance(obj, list):
        return obj
    return []

def _first(*vals) -> Optional[str]:
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None

def _infer_freq(annotations: List[Dict[str, Any]]) -> str:
    """
    Look for hints like:
      - "DEFAULT" with "...FREQUENCY=A..."
      - any annotation title containing 'FREQUENCY=Q/M/A'
    """
    for a in annotations or []:
        t = _first(a.get("title"), a.get("text"))
        if not t:
            continue
        m = re.search(r"FREQUENCY\s*=\s*([AQMWD])", t, re.I)
        if m:
            return FREQ_MAP.get(m.group(1).upper(), "")
    return ""

def _collect_tags(annotations: List[Dict[str, Any]]) -> List[str]:
    """
    Extract all annotation information as tags, including titles, text, and types.
    """
    tags: set[str] = set()
    for a in annotations or []:
        # Add annotation type if present
        if a.get("type"):
            tags.add(str(a["type"]).strip())
        
        # Add title content
        title = _first(a.get("title"))
        if title:
            # Split comma-separated values and add each as a tag
            for part in [p.strip() for p in title.split(",") if p.strip()]:
                # For key=value pairs, add both the key and the full pair
                if "=" in part:
                    key = part.split("=")[0].strip()
                    if key and len(key) <= 64:
                        tags.add(key)
                    if len(part) <= 64:
                        tags.add(part)
                else:
                    if len(part) <= 64:
                        tags.add(part)
        
        # Add text content if different from title
        text = _first(a.get("text"))
        if text and text != title and len(text) <= 100:
            tags.add(text)
    
    return sorted(tags)

def _stable_id(native_id: str, agency: str) -> str:
    """
    Keep native ABS id if present; otherwise sha1 fallback.
    """
    nid = (native_id or "").strip()
    if nid:
        return nid
    base = f"{native_id}|{agency}|{datetime.now(UTC).isoformat()}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def normalise_flow(flow: Dict[str, Any]) -> Dict[str, Any]:
    native_id = _first(flow.get("id"))
    title = _first(flow.get("name"),
                   (flow.get("names") or {}).get("en"))
    description = _first(flow.get("description"),
                         (flow.get("descriptions") or {}).get("en"))
    agency = _first(flow.get("agencyID"), "ABS") or ""
    annotations = flow.get("annotations") or []
    freq = _infer_freq(annotations)
    tags = _collect_tags(annotations)
    
    # Generate API URL using the format: http://data.api.abs.gov.au/data/{id}
    api_url = f"http://data.api.abs.gov.au/data/{native_id}" if native_id else ""

    return {
        "id": _stable_id(native_id, agency),
        "title": title or "",
        "description": description or "",
        "agency": agency,
        "collected": "",            # keep as empty as requested
        "freq": freq,               # inferred from annotations
        "api_url": api_url,
        "download_url": "",
        "tags": ";".join(tags)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True,
                    help="Input JSON file or directory containing ABS dataflows JSON")
    ap.add_argument("--out", default="datasets.csv", help="Output CSV path")
    args = ap.parse_args()

    p = Path(args.in_path)
    files: List[Path] = []
    if p.is_dir():
        files = [*p.glob("**/*.json")]
    elif p.is_file():
        files = [p]
    else:
        print(f"[ERROR] Input path not found: {p}", file=sys.stderr)
        sys.exit(2)

    rows: List[Dict[str, Any]] = []
    for f in files:
        obj = _load_json(f)
        if obj is None:
            continue
        flows = _extract_dataflows(obj)
        if not flows:
            # try top-level as a single flow
            if isinstance(obj, dict):
                flows = [obj]
        for flow in flows:
            try:
                rows.append(normalise_flow(flow))
            except Exception as e:
                print(f"[WARN] Skipping one flow in {f.name}: {e}", file=sys.stderr)

    if not rows:
        print("[ERROR] No dataflows found. Check your input structure.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows, columns=[
        "id","title","description","agency","collected",
        "freq","api_url","download_url","tags"
    ]).drop_duplicates(subset=["id"], keep="first")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[OK] Wrote {len(df)} rows → {args.out}")

if __name__ == "__main__":
    main()
