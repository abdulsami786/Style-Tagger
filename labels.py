"""Hand labels: one JSON object per line in labels/labels.jsonl.

    {"id": "<product id>", "tags": ["wedding", "festive_wear"], "formality": 4}

tags: 1 to 3 of desi_rules.TAGS, most important first. formality: 1 daily, 2 semi-formal, 3 festive/formal,
4 wedding formal, 5 bridal. Every label trains the model.

labels.original.jsonl is the starting set. It also carries "split" (the original train/test split that
`run.py eval --benchmark` uses to check code changes). The web app never writes it.

The category tagger's labels work differently: see the section at the end of this file.
"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import config
from desi_rules import TAGS


def problem(r: dict, tags=TAGS, formality: bool = True) -> str | None:
    """Why a label is invalid, or None if it is fine. tags=None accepts any tag name."""
    if not isinstance(r.get("id"), str) or not r["id"]:
        return "missing product id"
    picked = r.get("tags")
    if not isinstance(picked, list) or not 1 <= len(picked) <= 3:
        return f"{r['id']}: needs 1 to 3 tags"
    if len(set(picked)) != len(picked):
        return f"{r['id']}: repeated tag"
    unknown = [t for t in picked if tags is not None and t not in tags]
    if unknown:
        return f"{r['id']}: unknown tag {unknown[0]!r}"
    if formality:
        f = r.get("formality")
        if not isinstance(f, int) or isinstance(f, bool) or not 1 <= f <= 5:
            return f"{r['id']}: formality must be 1 to 5"
    return None


def load(path: Path = config.LABELS, check=problem) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                r = json.loads(line)
                if err := check(r):
                    raise ValueError(f"{path.name} line {n}: {err}")
                out[r["id"]] = r
    return out


def save(labels: dict[str, dict], path: Path = config.LABELS) -> None:
    """Write atomically, so a crash mid-write can never leave a half-written label file."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in labels.values():
            f.write(json.dumps({k: r[k] for k in ("id", "tags", "formality", "split", "hand", "removed") if k in r},
                               ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ---- exclusions: products that get no tags and never train the tagger ------------------------------------
# labels/exclusions.json is a list of rules. A rule matches a product when every field it names is equal,
# for example {"supplier": "Lakhany", "gender": "men", "subCategory": "Unstitched", "subSub": "suits"}.

RULE_FIELDS = ("id", "supplier", "gender", "subCategory", "subSub")


def rule_problem(rule) -> str | None:
    """Why an exclusion rule is invalid, or None if it is fine."""
    if not isinstance(rule, dict) or not rule:
        return "an exclusion rule must name at least one field"
    extra = set(rule) - set(RULE_FIELDS)
    if extra:
        return f"unknown rule field {sorted(extra)[0]!r}"
    if not all(isinstance(v, str) for v in rule.values()):
        return "rule values must be text"
    return None


def load_exclusions(path: Path = config.EXCLUSIONS) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_exclusions(rules: list[dict], path: Path = config.EXCLUSIONS) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rules, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def matches(rule: dict, product: dict) -> bool:
    return all(product[k] == v for k, v in rule.items())


def excluded(products: list[dict], rules: list[dict]) -> set[str]:
    """Ids of the products any rule matches."""
    ids = {r["id"] for r in rules if set(r) == {"id"}}   # one-product rules: a lookup, not a scan per rule
    rest = [r for r in rules if set(r) != {"id"}]
    return {p["id"] for p in products if p["id"] in ids or any(matches(r, p) for r in rest)} if rules else set()


def trainable(labels: dict[str, dict], excluded_ids: set[str]) -> dict[str, dict]:
    """The labels that train the tagger: every label except those on excluded products."""
    return {i: r for i, r in labels.items() if i not in excluded_ids}


def differences(a: dict[str, dict], b: dict[str, dict]) -> set[str]:
    """Products whose label was added, removed or changed between two label sets."""
    return {i for i in a.keys() | b.keys()
            if any((a.get(i) or {}).get(k) != (b.get(i) or {}).get(k) for k in ("tags", "formality", "hand"))}


# ---- category tagger: each product's subSubCategory is its starting label ---------------------------------
# Starting labels come from the catalog and only train the model. labels/category_labels.jsonl holds your edits:
#     {"id": "<product id>", "tags": ["Shirt"], "hand": true}     your label: trains, and decides that product's output
#     {"id": "<product id>", "removed": true}                     you removed its starting label
# Each merged spelling below counts as one category, named by the spelling the catalog uses most.

CATEGORY_MERGE = {"abaya": "abayas", "suit": "suits", "shawl": "shawls", "top": "tops", "sweatshirts": "sweatshirt",
                  "chino": "chinos", "culotte": "cullote", "culottes": "cullote", "peshwas": "pishwas", "joggers": "jogger"}
# Not clothing, so not a category. labels/category_exclusions.json also leaves these products out, so they get no category.
NOT_CLOTHING = {"earrings", "fragrance", "necklace"}


def category_seeds(products: list[dict]) -> dict[str, dict]:
    """A starting label for every product with a clothing subSubCategory. Products the catalog marks None get none."""
    key = lambda p: CATEGORY_MERGE.get(p["subSub"], p["subSub"])
    spellings = defaultdict(Counter)
    for p in products:
        if p["subSubRaw"]:
            spellings[key(p)][p["subSubRaw"]] += 1
    names = {k: min(c, key=lambda s: (-c[s], s)) for k, c in spellings.items()}
    return {p["id"]: {"id": p["id"], "tags": [names[key(p)]]} for p in products
            if p["subSubRaw"] and key(p) not in NOT_CLOTHING}


def edit_problem(r: dict) -> str | None:
    return None if r.get("removed") and isinstance(r.get("id"), str) else problem(r, tags=None, formality=False)


def with_edits(seeds: dict[str, dict], edits: dict[str, dict]) -> dict[str, dict]:
    """The working label set: starting labels with your edits on top."""
    out = dict(seeds)
    for pid, r in edits.items():
        if r.get("removed"):
            out.pop(pid, None)
        else:
            out[pid] = r
    return out


def edits_of(labels: dict[str, dict], seeds: dict[str, dict]) -> dict[str, dict]:
    """What makes a working label set differ from the starting labels: the lines to keep on disk."""
    return {**{i: r for i, r in labels.items() if r != seeds.get(i)},
            **{i: {"id": i, "removed": True} for i in seeds if i not in labels}}
