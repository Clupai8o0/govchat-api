# abs_scrape_codes_v3.py
import json
import time
import requests
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin

BASE = "https://api.data.abs.gov.au/"  # <-- no /rest here

HEADERS_STRUCT = {
    "accept": "application/vnd.sdmx.structure+json"  # structure JSON per ABS guide
}
TIMEOUT = 30

def get_json(url: str, params: Dict[str, Any] = None, retries=3) -> Dict[str, Any]:
    for attempt in range(1, retries + 1):
        r = requests.get(url, headers=HEADERS_STRUCT,
                         params=params, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 502, 503, 504):
            time.sleep(1.5 * attempt)
            continue
        r.raise_for_status()
    raise RuntimeError(f"GET failed: {url}")

def find_dataflows(j: Dict[str, Any]) -> List[Dict[str, Any]]:
    # ABS shape per docs: j["data"]["structure"]["dataflows"]
    try:
        dfs = j["data"]["structure"]["dataflows"]
        if isinstance(dfs, list):
            return dfs
        if isinstance(dfs, dict):
            return [dfs]
    except KeyError:
        pass
    # a few fallbacks just in case
    for path in (("data", "dataflows", "dataflow"), ("data", "dataflows"), ("dataflows",)):
        cur = j
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok:
            return cur if isinstance(cur, list) else [cur]
    return []

def sdmx_items(obj: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    try:
        blk = obj["data"]["structure"][key]
        if isinstance(blk, list):
            return blk
        if isinstance(blk, dict):
            return [blk]
    except KeyError:
        pass
    return []

def normalize_annotations(annos) -> List[Dict[str, Any]]:
    if not annos:
        return []
    if isinstance(annos, dict):
        annos = [annos]
    out = []
    for a in annos:
        out.append({
            "title": a.get("title"),
            "type": a.get("type"),
            "text": a.get("text"),
            "url": a.get("url"),
        })
    return out

def text_or_none(x: Optional[str]) -> Optional[str]:
    return x if isinstance(x, str) and x.strip() else None

def map_dataflow_to_dsd(df: Dict[str, Any]) -> Optional[str]:
    s = df.get("structure")
    if isinstance(s, dict) and s.get("id"):
        return s["id"]
    # fallbacks
    for k in ("dataStructure", "datastructure", "DataStructure"):
        v = df.get(k)
        if isinstance(v, dict) and v.get("id"):
            return v["id"]
        if isinstance(v, str):
            return v
    return None

def fetch_dsd_with_children(dsd_id: str) -> Dict[str, Any]:
    # /datastructure/ABS/{dsd_id}?references=children&detail=referencepartial
    url = urljoin(BASE, f"datastructure/ABS/{dsd_id}")
    return get_json(url, params={"references": "children", "detail": "referencepartial"})

def collect_codelists(dsd_json: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cl in sdmx_items(dsd_json, "codelists"):
        cl_id = cl.get("id")
        out[cl_id] = {
            "name": cl.get("name"),
            "description": text_or_none(cl.get("description")),
            "annotations": normalize_annotations(cl.get("annotations")),
            "codes": []
        }
        items = cl.get("items") or []
        if isinstance(items, dict):
            items = [items]
        for it in items:
            out[cl_id]["codes"].append({
                "id": it.get("id"),
                "name": it.get("name"),
                "description": text_or_none(it.get("description")),
                "annotations": normalize_annotations(it.get("annotations")),
            })
    return out

def main():
    # 1) List all dataflows
    df_url = urljoin(BASE, "dataflow/ABS")
    j = get_json(df_url, params={
                 "references": "none", "detail": "allcompletestubs"})
    dataflows = find_dataflows(j)

    if not dataflows:
        with open("abs_dataflow_raw_debug.json", "w", encoding="utf-8") as f:
            json.dump(j, f, ensure_ascii=False, indent=2)
        print("No dataflows found. Wrote abs_dataflow_raw_debug.json for inspection.")
        with open("abs_dataflows_codes.json", "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
        print("Saved abs_dataflows_codes.json")
        print("Dataflows processed: 0")
        return

    results: Dict[str, Any] = {}
    for df in dataflows:
        agency = df.get("agencyID") or df.get("agency") or "ABS"
        df_id = df.get("id") or df.get("identifier")
        version = df.get("version") or "latest"
        key = f"{agency},{df_id},{version}"

        entry = {
            "name": df.get("name"),
            "description": text_or_none(df.get("description")),
            "datastructure_id": None,
            "codelists": {}
        }

        dsd_id = map_dataflow_to_dsd(df)
        entry["datastructure_id"] = dsd_id

        if dsd_id:
            try:
                dsd_json = fetch_dsd_with_children(dsd_id)
                entry["codelists"] = collect_codelists(dsd_json)
            except Exception as e:
                entry["error_fetching_dsd"] = str(e)

        results[key] = entry
        time.sleep(0.15)  # gentle on gateway

    with open("abs_dataflows_codes.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Saved abs_dataflows_codes.json")
    print(f"Dataflows processed: {len(dataflows)}")


if __name__ == "__main__":
    main()
