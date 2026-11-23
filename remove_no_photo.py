"""Remove products with no working photo from the catalog export (liveProducts.cleaned.json).

    python remove_no_photo.py

A product has no photo when the image the Lookflock app shows doesn't download, doesn't open as an image,
or is a placeholder (smaller than MIN_SIDE px, e.g. a 100x97 "out of stock" ribbon). The app shows
imageUrl and falls back to originalImageUrl only when imageUrl fails to load, so this checks them the same way.

One slow pass over every product (about 2 hours here), then every failure is checked again more gently, so a
network blip or a rate limit can't remove a product. Results are saved as they come in (data/no_photo_*.jsonl),
so if it stops, run it again and it continues. The removed products are saved in full to
removed_no_photo_<date_time>/ (removed_products.json to restore them, removed_products.csv with the reason).
Afterwards run `python run.py prep` and `python run.py tag`.
"""

import csv
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from urllib.parse import urlparse

import requests
from PIL import Image
from tqdm import tqdm

import config
from images import sized

MIN_SIDE = 150  # px; real product photos are far larger, placeholder icons and ribbons are ~100
Image.MAX_IMAGE_PIXELS = None  # a browser opens huge photos, so don't reject them
local = threading.local()      # one connection pool per thread: a new TLS handshake per image was 5x slower


def session() -> requests.Session:
    if not hasattr(local, "s"):
        local.s = requests.Session()
    return local.s


def probe(url: str, tries: int = 1, timeout: int = 15) -> str | None:
    """Why this image can't be shown, or None if it's a real photo."""
    if not (url or "").strip():
        return "no image url"
    why = "?"
    for _ in range(tries):
        try:
            resp = session().get(sized(url.strip()), timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                why = f"HTTP {resp.status_code}"
                continue
            im = Image.open(BytesIO(resp.content))
            im.load()  # a full decode, so a truncated or corrupt file counts as broken
            if min(im.size) < MIN_SIDE:
                return f"placeholder {im.size[0]}x{im.size[1]}"
            return None
        except Exception as e:
            why = type(e).__name__
    return why


def problem(p: dict, **kw) -> str | None:
    """Why the app shows no photo for this product, or None if it shows one."""
    why = probe(p.get("imageUrl"), **kw)
    # the fallback is used only when imageUrl fails to load; a placeholder that loads is what users see
    if why and not why.startswith("placeholder") and probe(p.get("originalImageUrl"), **kw) is None:
        return None
    return why


def check(products: list[dict], name: str, workers: int, **kw) -> dict[str, str]:
    """Problems by product id. Each result is appended to data/<name>.jsonl as it comes in, so a stopped run
    picks up where it left off."""
    path = config.DATA / f"{name}.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done[r["id"]] = r["why"]
            except json.JSONDecodeError:  # a run killed mid-write leaves a broken line
                pass
    todo = [p for p in products if p["id"] not in done]
    with ThreadPoolExecutor(workers) as pool, open(path, "a", encoding="utf-8") as f:
        results = pool.map(lambda p: problem(p, **kw), todo)
        for p, why in tqdm(zip(todo, results), total=len(todo), unit="product", desc=name):
            done[p["id"]] = why
            f.write(json.dumps({"id": p["id"], "why": why}, ensure_ascii=False) + "\n")
    return {p["id"]: done[p["id"]] for p in products if done[p["id"]]}


def clear_progress() -> None:
    for f in config.DATA.glob("no_photo_pass*.jsonl"):
        f.unlink()


def main() -> None:
    products = json.load(open(config.CATALOG, encoding="utf-8"))
    order = random.sample(products, len(products))  # mix the hosts so no one site gets hammered
    print(f"checking {len(products):,} products")
    failed = check(order, "no_photo_pass1", workers=8)
    print(f"{len(failed):,} failed; checking them again in 60 s, slower")
    time.sleep(60)
    failed = check([p for p in order if p["id"] in failed], "no_photo_pass2", workers=4, tries=2, timeout=30)
    if not failed:
        print("every product has a photo; nothing removed")
        clear_progress()
        return

    removed = [p for p in products if p["id"] in failed]
    out = config.ROOT / f"removed_no_photo_{datetime.now():%Y-%m-%d_%H%M}"
    out.mkdir()
    (out / "removed_products.json").write_text(json.dumps(removed, ensure_ascii=False, indent=2),
                                               encoding="utf-8", newline="\n")
    with open(out / "removed_products.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "supplier", "name", "reason", "imageUrl", "originalImageUrl", "url"])
        for p in removed:
            w.writerow([p["id"], p.get("supplier"), p.get("name"), failed[p["id"]], p.get("imageUrl"),
                        p.get("originalImageUrl"), p.get("url")])

    kept = [p for p in products if p["id"] not in failed]
    tmp = config.CATALOG.with_suffix(".tmp")
    tmp.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")  # keep LF on Windows
    os.replace(tmp, config.CATALOG)  # atomic: a crash can't leave a half-written catalog
    clear_progress()                 # so the next run checks everything afresh

    by = {}
    for p in removed:
        k = (p.get("supplier"), urlparse(p.get("imageUrl") or "").netloc, failed[p["id"]].split(" ")[0])
        by[k] = by.get(k, 0) + 1
    print(f"\nremoved {len(removed):,} products, kept {len(kept):,} -> backup in {out.name}/")
    for (brand, host, why), n in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"{n:>6}  {brand:<20} {host:<40} {why}")
    print("\nnext: python run.py prep && python run.py tag")


if __name__ == "__main__":
    main()
