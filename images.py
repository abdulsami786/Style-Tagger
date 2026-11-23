"""Step 2: FashionSigLIP image vectors, saved in resumable parts under data/image_vectors/.

Slow on a laptop CPU (about 1 image a second), so it works incrementally: each run embeds only products
that have no vector yet and saves a part file every `part_size` products, so stopping it loses little.
Photos that fail to download are listed in failed.json and skipped next time (use retry_failed to retry).
Products without a vector are still tagged, from their text, price and the desi rules.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from itertools import batched
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

import config

FAILED = config.VECTORS / "failed.json"


def load_vectors() -> dict[str, np.ndarray]:
    out = {}
    for part in sorted(config.VECTORS.glob("part_*.npz")):
        with np.load(part) as d:
            out.update(zip(d["ids"].tolist(), d["vectors"]))
    return out


def sized(url: str, width: int = 512) -> str:
    """Shopify serves multi-MB originals but resizes on request; the model only needs 224 px."""
    if ("cdn.shopify.com/" in url or "/cdn/shop/" in url) and "width=" not in url:
        return f"{url}{'&' if '?' in url else '?'}width={width}"
    return url


def fetch(url: str, tries: int = 2) -> Image.Image | None:
    for _ in range(tries):
        try:
            resp = requests.get(sized(url), timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            im = Image.open(BytesIO(resp.content)).convert("RGB")
            im.thumbnail((512, 512))
            return im
        except Exception:
            continue
    return None


def load_model():
    model_dir = "models--" + config.IMAGE_MODEL.removeprefix("hf-hub:").replace("/", "--")
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    if (hub / model_dir).exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights already downloaded: skip the network check
    import open_clip  # imported here: torch is only needed for this step

    model, _, preprocess = open_clip.create_model_and_transforms(config.IMAGE_MODEL)
    return model.eval(), preprocess


def embed(model, preprocess, images: list[Image.Image]) -> np.ndarray:
    import torch

    with torch.no_grad():
        vectors = model.encode_image(torch.stack([preprocess(im) for im in images]))
        return torch.nn.functional.normalize(vectors, dim=-1).numpy()


def embed_missing(products: list[dict], limit: int | None = None, retry_failed: bool = False,
                  workers: int = 8, batch_size: int = 64, part_size: int = 1000) -> None:
    config.VECTORS.mkdir(parents=True, exist_ok=True)
    have = set(load_vectors())
    failed = set() if retry_failed or not FAILED.exists() else set(json.loads(FAILED.read_text()))
    todo = [p for p in products if p["imageUrl"] and p["id"] not in have and p["id"] not in failed]
    # photos matter most where the text says little: no description and a short name ("Glam 05") first
    todo = sorted(todo, key=lambda p: (bool(p["desc"]), len(p["name"].split()) > 3))[:limit]
    print(f"{len(have):,} products have vectors; embedding {len(todo):,} more")
    if not todo:
        return
    model, preprocess = load_model()
    part_no = len(list(config.VECTORS.glob("part_*.npz")))
    ids, vectors = [], []

    def save_part() -> None:
        nonlocal part_no, ids, vectors
        if ids:
            np.savez(config.VECTORS / f"part_{part_no:05d}.npz", ids=np.array(ids),
                     vectors=np.concatenate(vectors).astype(np.float16))
            part_no, ids, vectors = part_no + 1, [], []
        FAILED.write_text(json.dumps(sorted(failed)))

    chunks = list(batched(todo, batch_size))
    with ThreadPoolExecutor(workers) as pool, tqdm(total=len(todo), unit="img") as bar:
        pending = pool.map(fetch, [p["imageUrl"] for p in chunks[0]])
        for n, chunk in enumerate(chunks):
            images = list(pending)
            if n + 1 < len(chunks):  # download the next chunk while this one is embedded
                pending = pool.map(fetch, [p["imageUrl"] for p in chunks[n + 1]])
            ok = [(p, im) for p, im in zip(chunk, images) if im is not None]
            failed.update(p["id"] for p, im in zip(chunk, images) if im is None)
            if ok:
                vectors.append(embed(model, preprocess, [im for _, im in ok]))
                ids.extend(p["id"] for p, _ in ok)
            bar.update(len(chunk))
            if len(ids) >= part_size:
                save_part()
    save_part()
    print(f"done: {len(failed):,} photos could not be downloaded (listed in {FAILED.name})")
