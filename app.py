"""Step 5: the web app. Look up any product's tags, correct it, retrain, and see what changed.

    python run.py serve   ->   http://127.0.0.1:8765   (only reachable from this computer)

The page talks to this server over a small JSON API. Labels are saved to labels/labels.jsonl as you edit;
the model only changes when you press Retrain, which also rewrites data/predictions.jsonl.
"""

import base64
import hmac
import json
import os
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib

import catalog
import config
import desi_rules as R
import labels as L
import model as M
from images import sized

PAGE = config.ROOT / "web" / "index.html"
MAX_BODY = 5_000_000
TRAIN_LOCK = threading.Lock()   # one retrain at a time, across both taggers (they share this computer's memory)
PASSWORD = os.environ.get("TAGGER_PASSWORD", "").strip()   # set only on the online copy: every request must send it


class App:
    """The style tagger (occasion tags). CategoryApp and BridalApp below change what they learn from, not how the app works."""
    name = "style"
    files = config.STYLE
    tags = R.TAGS
    hand_only = False   # True: starting labels only train; only your own labels decide a product's output
    tag_catalog = staticmethod(M.tag_catalog)

    def __init__(self, products: list[dict] | None = None, vectors: tuple | None = None):
        self.products = catalog.load() if products is None else products
        self.by_id = {p["id"]: p for p in self.products}
        self.feats = self.make_features(vectors)
        self.label_lock = threading.Lock()   # one label write at a time
        self.retraining = False
        on_disk, before_disk = M.load_predictions(self.files.predictions), M.load_predictions(self.files.previous)
        self.current = on_disk or {}
        self._fit()
        causes = {}
        if on_disk and all(on_disk.get(i, {}).get("tags") == r["tags"] for i, r in self.fresh.items()):
            # nothing changed since the last run: keep showing the last retrain's changes
            self.previous, self.current = before_disk or {}, self.fresh
            causes = json.loads(self.files.causes.read_text()) if self.files.causes.exists() else {}
        else:
            self.previous, self.current = self.current, self.fresh
            M.save_predictions(self.previous, self.files.previous)
            M.save_predictions(self.current, self.files.predictions)
            self.files.causes.write_text("{}")
        self.change_list = M.changes(self.previous, self.current, set(causes.get("edited", [])))

    # ---- what differs between the taggers ----------------------------------------------------------------

    def labels(self) -> dict[str, dict]:
        return L.load()

    def original(self) -> dict[str, dict]:
        return L.load(config.LABELS_ORIGINAL)

    def write_labels(self, labels: dict[str, dict]) -> None:
        L.save(labels)

    def make_features(self, vectors: tuple | None) -> M.Features:
        return M.Features(self.products, vectors=vectors)

    def make_tagger(self, labels: dict[str, dict]) -> M.Tagger:
        return M.Tagger(self.feats, labels)

    def clean_label(self, r: dict) -> dict:
        return {"id": r.get("id"), "tags": r.get("tags"), "formality": r.get("formality")}

    def label_problem(self, r: dict) -> str | None:
        return L.problem(r)

    def drop_label(self, labels: dict[str, dict], pid: str) -> None:
        labels.pop(pid, None)

    def pins(self, label: dict | None) -> bool:
        """Does this label decide its product's output?"""
        return bool(label) and (not self.hand_only or bool(label.get("hand")))

    # ---- training ----------------------------------------------------------------------------------------

    def _fit(self) -> None:
        t0 = time.time()
        labels, rules = self.labels(), L.load_exclusions(self.files.exclusions)
        excluded = L.excluded(self.products, rules)
        tagger = self.make_tagger(L.trainable(labels, excluded))
        self.fresh = self.tag_catalog(tagger, labels, excluded)
        self.tagger, self.trained_labels, self.trained_rules = tagger, labels, rules
        self._load_rules()
        self.trained_at, self.train_seconds = time.time(), time.time() - t0

    def _load_rules(self) -> None:
        """The exclusions as saved now; they reach the output at the next retrain."""
        self.rules = L.load_exclusions(self.files.exclusions)
        self.excluded_now = L.excluded(self.products, self.rules)

    def retrain(self) -> dict:
        before_agreement = self.tagger.agreement
        edited = L.differences(self.labels(), self.trained_labels)   # labels changed since the last training
        self._fit()
        self.previous, self.current = self.current, self.fresh
        M.save_predictions(self.previous, self.files.previous)
        M.save_predictions(self.current, self.files.predictions)
        self.files.causes.write_text(json.dumps({"edited": sorted(edited)}))
        self.change_list = M.changes(self.previous, self.current, edited)
        return {"seconds": round(self.train_seconds), "changed": len(self.change_list), "causes": self.cause_counts(),
                "agreementBefore": slim(before_agreement), "agreement": slim(self.tagger.agreement)}

    def cause_counts(self) -> dict:
        return {cause: sum(c["cause"] == cause for c in self.change_list) for cause in M.CAUSES}

    # ---- views -------------------------------------------------------------------------------------------

    def card(self, pid: str, labels: dict) -> dict:
        rec = self.current[pid]
        lab = labels.get(pid)
        return {"id": pid, "name": rec["name"], "brand": rec["supplier"], "price": rec["price"],
                "thumb": sized(rec["imageUrl"], 360), "url": rec["url"], "tags": rec["tags"],
                "source": rec["source"], "modelTags": rec.get("modelTags"),
                "ruleAdded": [n["tag"] for n in rec["ruleNotes"] if n["change"] == "added"],
                "label": lab["tags"] if lab else None, "pinned": self.pins(lab), "excluded": pid in self.excluded_now}

    def state(self) -> dict:
        labels, original = self.labels(), self.original()
        gained, lost = {}, {}
        for c in self.change_list:
            for t in c["gained"]:
                gained[t] = gained.get(t, 0) + 1
            for t in c["lost"]:
                lost[t] = lost.get(t, 0) + 1
        return {
            "mode": self.name, "tags": self.tags, "bridePrice": R.BRIDE_PRICE, "brideMinPrice": R.BRIDE_MIN_PRICE,
            "products": len(self.products), "withVectors": len(self.feats.vector_row),
            "genders": sorted({p["gender"] for p in self.products}),
            "labels": sum(i in self.by_id for i in labels),  # a label whose product left the catalog doesn't train
            "editedFromOriginal": len(L.differences(labels, original)),
            "untrained": len(L.differences(labels, self.trained_labels)),
            "trainedLabels": len(self.tagger.ids), "excludedProducts": len(self.excluded_now),
            "untrainedExclusions": self.rules != self.trained_rules,
            "agreement": slim(self.tagger.agreement), "trainedAt": self.trained_at,
            "trainSeconds": round(self.train_seconds),
            "changes": {"count": len(self.change_list), "gained": gained, "lost": lost, "causes": self.cause_counts()},
            "tagCounts": Counter(t for r in self.current.values() for t in r["tags"]),  # products per output tag
            "retraining": self.retraining,
        }

    def catalog_page(self, q: dict) -> dict:
        text, tag, only_labelled = q.get("q", "").strip().lower(), q.get("tag", ""), q.get("labelled") == "1"
        gender, only_excluded = q.get("gender", ""), q.get("excluded") == "1"
        labels = self.labels()
        hits = [p["id"] for p in self.products
                if (text in f'{p["name"]} {p["supplier"]} {p["id"]}'.lower() if text  # bridal: other products only by search
                    else only_excluded or self.current[p["id"]]["source"] != "not bridal")
                and (not tag or tag in self.current[p["id"]]["tags"])
                and (not gender or p["gender"] == gender)
                and (not only_labelled or self.pins(labels.get(p["id"])))
                and (not only_excluded or p["id"] in self.excluded_now)]
        if tag:  # most confident first: the model's probability for that tag (for a label too, as in its panel)
            hits.sort(key=lambda i: -self.current[i]["tagScores"][tag])
        offset, limit = int(q.get("offset", 0)), min(int(q.get("limit", 48)), 200)
        return {"total": len(hits), "items": [self.card(i, labels) for i in hits[offset:offset + limit]],
                "sortedBy": tag or None}

    def label_list(self) -> dict:
        labels, original = self.labels(), self.original()
        edited = L.differences(labels, original)
        items = []
        for pid, lab in labels.items():
            if pid not in self.current:
                continue
            item = self.card(pid, labels)
            cv = self.tagger.cv_tags.get(pid)  # None for labels added since the last retrain
            item.update({"formality": lab.get("formality"), "cvTags": cv, "edited": pid in edited,
                         "disagrees": bool(cv) and cv[0] not in lab["tags"]})
            items.append(item)
        return {"items": items}

    def change_page(self, q: dict) -> dict:
        tag, kind = q.get("tag", ""), q.get("kind", "")
        kinds = [kind] if kind in ("gained", "lost") else ["gained", "lost"]
        hits = [c for c in self.change_list if (not tag or any(tag in c[k] for k in kinds))
                and (kind != "related" or c["cause"] in M.CAUSES[:3])]  # your labels, what followed, your exclusions
        offset, limit = int(q.get("offset", 0)), min(int(q.get("limit", 48)), 200)
        labels = self.labels()
        return {"total": len(hits),
                "items": [{**self.card(c["id"], labels), "before": c["before"], "after": c["after"], "cause": c["cause"]}
                          for c in hits[offset:offset + limit]]}

    def product(self, pid: str) -> dict:
        if pid not in self.by_id:
            raise KeyError(pid)
        p, rec, row = self.by_id[pid], self.current[pid], self.feats.index[pid]
        P, _, T, _ = self.tagger.predict([row])
        s = self.feats.signals(row)
        labels = self.labels()
        model_tags, notes = self.tagger.choose(P[0], T[0], row)   # the model's own answer, before any label
        by_rule = {n["tag"] for n in notes if n["change"] == "added"}
        taught = [{"tag": t, "p": round(float(P[0][self.tags.index(t)]), 3), "byRule": t in by_rule,
                   "labels": [] if t in by_rule else
                   [{**self.card(i, labels), "push": round(v, 2)} for i, v in self.tagger.taught_by(row, t)]}
                  for t in model_tags]
        return {
            "id": pid, "name": p["name"], "brand": p["supplier"], "price": p["price"], "url": p["url"],
            "image": sized(p["imageUrl"], 720), "gender": p["gender"], "category": p["subCategory"],
            "shopCategory": p["subSub"], "desc": p["desc"], "details": p["details"],
            "tags": rec["tags"], "tagScores": rec["tagScores"], "formality": rec.get("formality"),
            "source": rec["source"], "modelTags": rec.get("modelTags"),
            "formalityLabel": rec.get("formalityLabel"), "usedImage": rec["usedImage"], "ruleNotes": rec["ruleNotes"],
            "why": [] if self.hand_only else [R.SIGNAL_NAMES[k] for k in R.evidence(s, limit=8)],
            "probabilities": sorted(({"tag": t, "p": round(float(P[0][k]), 3), "threshold": round(float(T[0][k]), 2)}
                                     for k, t in enumerate(self.tags)), key=lambda x: -x["p"]),
            "before": (self.previous.get(pid) or {}).get("tags"),
            "label": labels.get(pid),
            "flags": {"bride": bool(s["bride"]), "men": bool(s["men"]), "modestOk": bool(s["g_modest"] or s["modest_cat"])},
            "similar": [self.card(self.products[r]["id"], labels) for r in self.feats.similar(row)],
            "taughtBy": taught,
            "exclusion": self.exclusion_options(p, labels),
        }

    def describe(self, rule: dict) -> str:
        if "id" in rule:
            p = self.by_id.get(rule["id"])
            return f'only "{p["name"]}" ({p["supplier"]})' if p else f"only product {rule['id']}"
        return " · ".join([rule.get("supplier") or "all brands"] +
                          [rule[k] or "(none)" for k in ("gender", "subCategory", "subSub") if k in rule])

    def matching(self, rule: dict, labels: dict) -> dict:
        if set(rule) == {"id"}:  # one product: a lookup, not a scan (the category tagger has hundreds of these)
            hits = [self.by_id[rule["id"]]] if rule["id"] in self.by_id else []
        else:
            hits = [p for p in self.products if L.matches(rule, p)]
        return {"rule": rule, "text": self.describe(rule), "count": len(hits), "labels": sum(p["id"] in labels for p in hits),
                "hits": hits}

    def exclusion_options(self, p: dict, labels: dict) -> dict:
        """The rules that exclude this product, and three ways to exclude it: alone, with its brand's products
        in the same shop category, or with every brand's products in that category."""
        scopes = [{"id": p["id"]}, {k: p[k] for k in ("supplier", "gender", "subCategory", "subSub")},
                  {k: p[k] for k in ("gender", "subCategory", "subSub")}]
        options = []
        for rule in scopes:
            m = self.matching(rule, labels)
            others = [q for q in m.pop("hits") if q["id"] != p["id"]]
            brands, sample = set(), []
            for q in others:   # one per brand first, so a category that mixes brands shows its mix
                if q["supplier"] not in brands and len(sample) < 4:
                    brands.add(q["supplier"]); sample.append(q)
            sample += [q for q in others if q not in sample][:4 - len(sample)]
            options.append({**m, "sample": [f'{q["name"]} ({q["supplier"]})' for q in sample]})
        return {"by": [{"rule": r, "text": self.describe(r)} for r in self.rules if L.matches(r, p)], "options": options}

    def exclusion_list(self) -> dict:
        labels = self.labels()
        items = []
        for rule in self.rules:
            m = self.matching(rule, labels)
            m.pop("hits")
            items.append(m)
        return {"items": items}

    # ---- edits -------------------------------------------------------------------------------------------

    def save_labels(self, body: dict) -> dict:
        updates, removals = body.get("set", []), body.get("remove", [])
        if not isinstance(updates, list) or not isinstance(removals, list):
            raise ValueError('expected {"set": [...], "remove": [...]}')
        clean = []
        for r in updates:
            if not isinstance(r, dict):
                raise ValueError("each label must be an object")
            r = self.clean_label(r)
            if err := self.label_problem(r):
                raise ValueError(err)
            if r["id"] not in self.by_id:
                raise ValueError(f"unknown product id {r['id']!r}")
            clean.append(r)
        with self.label_lock:
            labels = self.labels()
            for r in clean:
                labels[r["id"]] = {**labels.get(r["id"], {}), **r}  # keeps the hidden benchmark split
            for pid in removals:
                self.drop_label(labels, pid)
            self.write_labels(labels)
        return {"saved": len(clean), "removed": len(removals),
                "untrained": len(L.differences(labels, self.trained_labels))}

    def edit_exclusions(self, body: dict) -> dict:
        """{"add": rule} or {"remove": rule}. Takes effect at the next retrain."""
        add, remove = body.get("add"), body.get("remove")
        if add is not None and (err := L.rule_problem(add)):
            raise ValueError(err)
        with self.label_lock:
            rules = L.load_exclusions(self.files.exclusions)
            if add is not None and add not in rules:
                rules.append(add)
            if remove is not None:
                if remove not in rules:
                    raise ValueError("that exclusion was already removed")
                rules.remove(remove)
            L.save_exclusions(rules, self.files.exclusions)
            self._load_rules()
        return self.exclusion_list()

    def restore(self) -> dict:
        with self.label_lock:
            self.write_labels(self.original())
        return self.state()


class CategoryApp(App):
    """Learns subSubCategory. Every product's brand category is a starting label: it trains the model, and the
    product outputs the model's answer unless you labelled it yourself. Your edits are the only labels on disk."""
    name = "category"
    files = config.CATEGORY
    hand_only = True
    tag_catalog = staticmethod(M.tag_categories)
    tagger_class = M.CategoryTagger

    def __init__(self, products: list[dict] | None = None, vectors: tuple | None = None):
        products = catalog.load() if products is None else products
        self.seeds = L.category_seeds(products)
        # a hand label can name a category the catalog no longer has, so the list includes both
        self.tags = sorted({t for r in (*self.seeds.values(), *self.edits().values()) for t in r.get("tags", ())})
        super().__init__(products, vectors)

    def edits(self) -> dict[str, dict]:
        return L.load(self.files.labels, L.edit_problem) if self.files.labels.exists() else {}

    def labels(self) -> dict[str, dict]:
        return L.with_edits(self.seeds, self.edits())

    def original(self) -> dict[str, dict]:
        return self.seeds

    def write_labels(self, labels: dict[str, dict]) -> None:
        L.save(L.edits_of(labels, self.seeds), self.files.labels)

    def make_features(self, vectors: tuple | None) -> M.Features:
        return M.Features(self.products, self.files.features, blind=True, vectors=vectors)

    def make_tagger(self, labels: dict[str, dict]) -> M.Tagger:
        """A full training takes about 10 minutes (Bridal's about one), so the fitted tagger is saved, and only tags whose
        labels changed are refitted. Adding or removing a training row (an exclusion, a removed label) refits everything."""
        code = (config.PRODUCTS, Path(R.__file__), Path(M.__file__), Path(config.__file__))
        key = [f.stat().st_mtime for f in code] + [len(self.feats.vector_row)]  # what makes a saved tagger stale
        reuse = getattr(self, "tagger", None)
        if reuse is None and self.files.tagger.exists():
            saved = joblib.load(self.files.tagger)
            reuse = saved["tagger"] if saved["key"] == key else None
        tagger = self.tagger_class(self.feats, labels, self.tags, reuse=reuse)
        tmp = self.files.tagger.with_suffix(".tmp")   # replaced whole, so a crash mid-write can't leave half a file
        joblib.dump({"key": key, "tagger": tagger}, tmp)
        os.replace(tmp, self.files.tagger)
        return tagger

    def clean_label(self, r: dict) -> dict:
        return {"id": r.get("id"), "tags": r.get("tags"), "hand": True}

    def label_problem(self, r: dict) -> str | None:
        return L.problem(r, self.tags, formality=False)

    def drop_label(self, labels: dict[str, dict], pid: str) -> None:
        """Removing your own label goes back to the brand's category; removing that takes the product out of training."""
        if labels.get(pid, {}).get("hand") and pid in self.seeds:
            labels[pid] = self.seeds[pid]
        else:
            labels.pop(pid, None)


class BridalApp(CategoryApp):
    """Which of her wedding functions (mehendi, barat, valima, nikah, dholki) a bride's outfit is for: the app's bridal
    substyles. Only a bride's outfit gets these tags (desi_rules.bridal_scope, read from the style tagger's output at
    every training, so a style retrain moves them at the next bridal one). Starting labels come from the desi rules (the
    function a listing names, else its main colour). Like brand categories they train the model but don't decide a
    product's output; your labels do, even on a product that isn't a bride's outfit."""
    name = "bridal"
    files = config.BRIDAL
    tags = R.EVENTS
    tagger_class = M.BridalTagger

    def __init__(self, style: App):
        self.style, self.seeds = style, {}
        scope = self.scope()
        for row, p in enumerate(style.products):  # catalog order, so My labels lists them the same way every run
            if p["id"] in scope and (events := R.bridal_events(p, R.formality(style.feats.signals(row)))):
                self.seeds[p["id"]] = {"id": p["id"], "tags": events}
        App.__init__(self, style.products)   # CategoryApp's own __init__ would make brand-category starting labels

    def scope(self) -> set[str]:
        return {p["id"] for p in self.style.products if R.bridal_scope(p, self.style.current[p["id"]])}

    def make_features(self, vectors: tuple | None) -> M.Features:
        return self.style.feats

    def tag_catalog(self, tagger: M.Tagger, labels: dict[str, dict], excluded: set[str]) -> dict[str, dict]:
        out, scope = M.tag_categories(tagger, labels, excluded), self.scope()
        for pid, rec in out.items():
            if rec["source"] == "model" and pid not in scope:   # your label still decides
                out[pid] = {**rec, "tags": [], "tagScores": {}, "modelTags": rec["tags"], "source": "not bridal",
                            "needsReview": False}
        return out


def slim(m: dict) -> dict:
    return {k: round(m[k], 4) for k in ("n", "micro_f1", "top_tag_right", "top_tag_acceptable")}


def password_ok(authorization: str, password: str) -> bool:
    """Does an HTTP Basic "Authorization" header carry this password? Any user name is accepted."""
    scheme, _, sent = authorization.partition(" ")
    try:
        given = base64.b64decode(sent).decode().partition(":")[2]
    except ValueError:  # not base64, or not UTF-8
        return False
    return scheme == "Basic" and hmac.compare_digest(given.encode(), password.encode())


class Handler(BaseHTTPRequestHandler):
    apps: dict[str, App]
    build_locks = {"category": threading.Lock(), "bridal": threading.Lock()}   # Bridal needn't wait for a Category build

    def pick(self, q: dict) -> App:
        """The tagger a request is for. Category and Bridal get ready the first time someone asks for them."""
        mode = q.get("mode", "style")
        if mode not in ("style", "category", "bridal"):
            raise ValueError(f"unknown mode {mode!r}")
        if mode not in self.apps:  # checked outside the lock, so style keeps working while category trains
            with self.build_locks[mode]:
                if mode not in self.apps:
                    style = self.apps["style"]
                    self.apps[mode] = (BridalApp(style) if mode == "bridal" else
                                       CategoryApp(style.products, (style.feats.vector_row, style.feats.vectors)))
        return self.apps[mode]

    def allowed(self) -> bool:
        """With a password set, the browser asks for it once and then sends it with every request."""
        if not PASSWORD or password_ok(self.headers.get("Authorization", ""), PASSWORD):
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Desi Tagger"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def do_GET(self) -> None:
        if not self.allowed():
            return
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/":
            return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        routes = {"/api/state": lambda: self.pick(q).state(), "/api/catalog": lambda: self.pick(q).catalog_page(q),
                  "/api/labels": lambda: self.pick(q).label_list(), "/api/changes": lambda: self.pick(q).change_page(q),
                  "/api/product": lambda: self.pick(q).product(q.get("id", "")),
                  "/api/exclusions": lambda: self.pick(q).exclusion_list()}
        self.respond(routes.get(url.path))

    def do_POST(self) -> None:
        if not self.allowed():
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self.send_json(413, {"error": "request too large"})
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self.send_json(400, {"error": "request body is not valid JSON"})
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        routes = {"/api/labels": lambda: self.pick(q).save_labels(body), "/api/retrain": lambda: self.retrain(self.pick(q)),
                  "/api/restore": lambda: self.pick(q).restore(), "/api/exclusions": lambda: self.pick(q).edit_exclusions(body)}
        self.respond(routes.get(url.path))

    def retrain(self, app: App) -> dict:
        if not TRAIN_LOCK.acquire(blocking=False):
            raise RuntimeError("a retrain is already running")
        try:
            app.retraining = True
            return app.retrain()
        finally:
            app.retraining = False
            TRAIN_LOCK.release()

    def respond(self, handler) -> None:
        if handler is None:
            return self.send_json(404, {"error": "not found"})
        try:
            self.send_json(200, handler())
        except (ValueError, KeyError) as e:
            self.send_json(400, {"error": str(e)})
        except RuntimeError as e:
            self.send_json(409, {"error": str(e)})
        except Exception as e:  # keep the server alive; show the problem in the page
            self.send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def send_json(self, status: int, obj) -> None:
        self.send(status, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def send(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass  # keep the console quiet


class Server(ThreadingHTTPServer):
    allow_reuse_address = False  # with it, Windows lets a second copy bind the same port silently


def serve(port: int = config.PORT) -> None:
    try:  # bind before training, so a second copy stops before it rewrites the predictions
        server = Server((os.environ.get("TAGGER_HOST", "127.0.0.1"), port), Handler)  # this computer only, unless set (online copy)
    except OSError:
        raise SystemExit(f"Port {port} is busy: Only Tagger is probably already running at http://127.0.0.1:{port}")
    print("loading products and training the tagger (about a minute)...")
    Handler.apps = {"style": App()}
    print(f"Only Tagger is running: open http://127.0.0.1:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
