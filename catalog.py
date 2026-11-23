"""Step 1: read the raw catalog export and write one clean row per product (data/products.parquet).

Cleaning: strip HTML from descriptions, repair broken prices (corrupt newPrice -> oldPrice -> "Rs 22,785"
in the text; one brand prices in USD), cap prices far above the brand's own top end, and add each product's
brand median price and its price percentile within the brand. Price within the brand matters: Rs 20k is
the top of Khaadi's range but the bottom of Deepak Perwani's.
"""

import html
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import config

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
RS_RE = re.compile(r"(?:rs\.?|pkr)\s*([\d,]{3,9})", re.I)
MAX_PKR = 3_000_000   # above this the field is corrupt (e.g. two prices concatenated)
FOREIGN_MEDIAN = 500  # a brand whose median is below this is not pricing in PKR
USD_TO_PKR = 280      # fixed rate, only used to rescue the one brand that prices in USD


def clean(s) -> str:
    if s is None:
        return ""
    if isinstance(s, list):
        s = " ".join(map(str, s))
    s = html.unescape(TAG_RE.sub(" ", str(s))).replace("�", "'")
    s = WS_RE.sub(" ", s).strip()
    return "" if s.lower() == "none" else s


def norm_key(s) -> str:
    return re.sub(r"[\s_\-]", "", str(s or "").lower())


def raw_price(p: dict, desc: str) -> float | None:
    for field in ("newPrice", "oldPrice"):
        try:
            price = float(p.get(field) or 0)
        except (TypeError, ValueError):
            continue
        if 0 < price <= MAX_PKR:
            return price
    m = RS_RE.search(desc)
    return float(m.group(1).replace(",", "")) if m else None


def group_key(r: dict) -> str:
    """Colour variants share a description; cross-validation keeps them in the same fold."""
    body = r["desc"][:120].lower() or re.sub(r"[\d\W]+", " ", r["name"].lower())
    return f'{r["supplier"]}|{body}'


def prep(src: Path = config.CATALOG, dst: Path = config.PRODUCTS) -> list[dict]:
    rows = []
    for p in json.load(open(src, encoding="utf-8")):
        desc = clean(p.get("description"))
        secs = p.get("secondaryImages") or []
        rows.append({
            "id": p["id"],
            "supplier": p.get("supplier") or "",
            "gender": norm_key(p.get("category")),
            "subCategory": clean(p.get("subCategory")),
            "subSub": norm_key(p.get("subSubCategory")),
            "subSubRaw": clean(p.get("subSubCategory")),  # as the main app stores it; "" when the catalog says None
            "piece": clean(p.get("piece")),
            "name": clean(p.get("name")),
            "desc": desc,
            "details": clean(p.get("product_details_text")),
            "keywords": clean(p.get("keywords")),
            "price": raw_price(p, desc),
            "imageUrl": (p.get("imageUrl") or "").strip() or (str(secs[0]).strip() if secs else ""),
            "url": p.get("url") or "",
        })

    by_brand = defaultdict(list)
    for r in rows:
        if r["price"]:
            by_brand[r["supplier"]].append(r["price"])
    foreign = {b for b, ps in by_brand.items() if np.median(ps) < FOREIGN_MEDIAN}
    for r in rows:
        # that brand also lists some items already in PKR; only rescale the tiny ones
        if r["price"] and r["supplier"] in foreign and r["price"] < 10 * FOREIGN_MEDIAN:
            r["price"] *= USD_TO_PKR

    # a price far above the brand's own top end is corrupt; cap it there rather than drop it,
    # because couture lines legitimately sit 30-50x above a lawn-heavy brand's median
    p99 = {b: float(np.percentile(ps, 99)) for b, ps in by_brand.items()}
    for r in rows:
        if r["price"] and r["price"] > 3 * p99[r["supplier"]]:
            r["price"] = p99[r["supplier"]]

    groups, brand_prices = defaultdict(list), defaultdict(list)
    for i, r in enumerate(rows):
        if r["price"]:
            groups[(r["supplier"], r["gender"])].append(i)
            brand_prices[r["supplier"]].append(r["price"])
    for idx in groups.values():
        ranks = np.array([rows[i]["price"] for i in idx]).argsort().argsort()
        for i, rank in zip(idx, ranks):
            rows[i]["pricePct"] = float(rank / max(len(idx) - 1, 1))
    brand_median = {b: float(np.median(ps)) for b, ps in brand_prices.items()}
    for r in rows:
        r.setdefault("pricePct", None)
        r["brandMedian"] = brand_median.get(r["supplier"])
        r["group"] = group_key(r)

    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), dst)
    print(f"{len(rows):,} products -> {dst}")
    return rows


def load() -> list[dict]:
    return pq.read_table(config.PRODUCTS).to_pylist()
