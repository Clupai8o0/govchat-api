import os, math, pickle
import pandas as pd
from dotenv import load_dotenv
# Saves embeedings for a dataset to embeddings.pkl (which basically acts as a knowledge base)
# --- NEW: force valid certificate bundle ---
import certifi, httpx
for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
    os.environ[var] = certifi.where()

from openai import OpenAI

# --- Config ---
CSV_PATH = "datasets.csv"
MODEL = "text-embedding-3-small"
BATCH_SIZE = 128

# Load API key from .env
load_dotenv()

# --- NEW: give OpenAI client a verified httpx client ---
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=httpx.Client(verify=certifi.where())
)

# 1) Load data
df = pd.read_csv(CSV_PATH)

# 2) Build text input for embeddings
def row_to_text(r):
    parts = [
        f"Title: {str(r.get('title', '') or '')}",
        f"Description: {str(r.get('description', '') or '')}",
        f"Agency: {str(r.get('agency', '') or '')}",
        f"Collected: {str(r.get('collected', '') or '')}",
        f"Tags: {str(r.get('tags', '') or '')}"
    ]
    # Filter out empty parts and join with separators
    filtered_parts = [p.strip() for p in parts if p and str(p).strip() and not p.endswith(": ")]
    return " | ".join(filtered_parts)

df["text"] = df.apply(row_to_text, axis=1)
df = df[df["text"].str.len() > 0].copy()

ids = df["id"].astype(str).tolist()
texts = df["text"].astype(str).tolist()

# 3) Batch embeddings
embeddings = {}
num_batches = math.ceil(len(texts) / BATCH_SIZE)

for b in range(num_batches):
    start, end = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
    batch_ids, batch_texts = ids[start:end], texts[start:end]

    resp = client.embeddings.create(model=MODEL, input=batch_texts)
    vecs = [d.embedding for d in resp.data]

    for k, v in zip(batch_ids, vecs):
        embeddings[k] = v

print(f"Embedded {len(embeddings)} rows; dim={len(next(iter(embeddings.values())))}")

# 4) Save to pickle
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("Saved embeddings to embeddings.pkl")
