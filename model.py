"""Step 3: features, training, prediction and scoring. The desi knowledge itself lives in desi_rules.py.

Each product becomes: TF-IDF of its text (plus brand and category tokens), the desi signals and rule scores,
and, when it has one, its FashionSigLIP image vector. One logistic regression per tag learns from the hand
labels, and each tag's cut-off comes from cross-validation.

Two models learn from the same labels: one that also uses the image vector, for products that have one,
and one from text alone, for the rest (photo not embedded yet, or no longer online). Training is
deterministic, so retraining on unchanged labels gives identical tags.
"""

import json
import os
import zlib
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.preprocessing import StandardScaler

import config
import desi_rules as R
import images
import labels as L

TAGS = R.TAGS
FORMALITY_NAMES = {1: "daily", 2: "semi_formal", 3: "festive_formal", 4: "wedding_formal", 5: "bridal"}


def doc(p: dict) -> str:
    cat = p["subCategory"].lower().replace(" ", "_")
    return f'{R.product_text(p)} brand_{p["supplier"].lower()} sub_{p["subSub"]} cat_{cat} g_{p["gender"]} pc_{p["piece"]}'


def hide_category(p: dict) -> dict:
    """A product without its shop category. The category tagger has to predict that, and both doc() and the
    SUBSUB_FAMILY rule in desi_rules would otherwise hand it the answer."""
    return {**p, "subSub": ""}


class Features:
    """Model inputs for every product. Text and rule features are cached in `cache` and rebuilt when the
    catalog or the rules change. Image vectors are read fresh, so a new `embed` run is picked up."""

    def __init__(self, products: list[dict], cache: Path = config.FEATURES_CACHE, blind: bool = False,
                 vectors: tuple | None = None):
        self.products = products
        self.index = {p["id"]: i for i, p in enumerate(products)}
        key = [f.stat().st_mtime for f in (config.PRODUCTS, Path(R.__file__), Path(__file__))]
        cached = joblib.load(cache) if cache.exists() else None
        if not cached or cached["key"] != key:
            cached = self._build(key, [hide_category(p) for p in products] if blind else products)
            joblib.dump(cached, cache)
        self.signal_keys, self.signal_matrix = cached["signal_keys"], cached["signal_matrix"]
        self.tfidf, self.dense = cached["tfidf"], cached["dense"]
        if vectors:  # the same for every tagger, so a second one can share the first one's
            self.vector_row, self.vectors = vectors
            return
        vectors = images.load_vectors()
        rows = sorted(self.index[i] for i in vectors if i in self.index)
        self.vector_row = {r: k for k, r in enumerate(rows)}
        raw = np.array([vectors[products[r]["id"]] for r in rows], dtype=np.float32)
        self.vectors = StandardScaler().fit_transform(raw) if rows else np.zeros((0, 0), dtype=np.float32)

    def _build(self, key: list[float], products: list[dict]) -> dict:
        print("building text and rule features for every product (about a minute, cached afterwards)...")
        sigs = [R.signals(p) for p in products]
        keys = sorted(sigs[0])
        dense = np.array([[s[k] for k in keys] + [sc[t] for t in TAGS] + [R.formality(s)]
                          for s, sc in ((s, R.rule_scores(s)) for s in sigs)], dtype=np.float32)
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000, sublinear_tf=True,
                                token_pattern=r"(?u)\b[\w\-]+\b").fit_transform(map(doc, products))
        return {"key": key, "signal_keys": keys,
                "signal_matrix": np.array([[s[k] for k in keys] for s in sigs], dtype=np.float32),
                "tfidf": tfidf.tocsr(), "dense": StandardScaler().fit_transform(dense).astype(np.float32)}

    def signals(self, row: int) -> dict[str, float]:
        return dict(zip(self.signal_keys, self.signal_matrix[row].tolist()))

    def has_vector(self, row: int) -> bool:
        return row in self.vector_row

    def matrix(self, rows: list[int], with_image: bool) -> sparse.csr_matrix:
        blocks = [self.tfidf[rows], sparse.csr_matrix(self.dense[rows])]
        if with_image:
            blocks.append(sparse.csr_matrix(self.vectors[[self.vector_row[r] for r in rows]] * config.IMAGE_WEIGHT))
        return sparse.hstack(blocks).tocsr()

    def similar(self, row: int, k: int = 12) -> list[int]:
        """The products whose text (name, description, brand, category) is closest to this one."""
        sims = (self.tfidf[row] @ self.tfidf.T).toarray().ravel()
        sims[row] = -1
        top = np.argpartition(-sims, k)[:k]
        return [int(i) for i in top[np.argsort(-sims[top])]]


def fit_one(X, y: np.ndarray):
    """A logistic regression for one tag; a tag with too few (or too many) examples gets a constant instead."""
    pos = int(y.sum())
    if pos < 2 or pos > len(y) - 2:
        return pos / len(y)
    return LogisticRegression(C=config.C, class_weight="balanced", max_iter=3000, solver="liblinear",
                              random_state=0).fit(X, y)


def representer(m, X, y: np.ndarray) -> np.ndarray:
    """Each training label's weight in a fitted tag model (zeros for a constant model)."""
    if isinstance(m, float):
        return np.zeros(len(y))
    if isinstance(m, SGDClassifier):  # SGD stops near the optimum, so these pushes are close to the score, not exact
        return (y - m.predict_proba(X)[:, 1]) / (m.alpha * len(y))
    n, pos = len(y), y.sum()
    weight = np.where(y == 1, n / (2 * pos), n / (2 * (n - pos)))  # class_weight="balanced"
    return config.C * weight * (y - m.predict_proba(X)[:, 1])


def fit_per_tag(X, Y: np.ndarray) -> list:
    # liblinear releases the GIL, so threads use every core (about 3x faster on a 4-core laptop).
    # Each thread gets its own copy: scikit-learn tidies a sparse matrix in place, which is not thread-safe.
    return Parallel(n_jobs=-1, prefer="threads")(delayed(fit_one)(X.copy(), Y[:, j]) for j in range(Y.shape[1]))


def fit_sgd_one(X, y: np.ndarray):
    pos = int(y.sum())
    if pos < 2 or pos > len(y) - 2:
        return float(pos >= 2)  # a tag with one example can't be learned, so it must never win as the likeliest
    # averaged weights: with SGD's last weights one edit moved about 190 other products; averaged, about 35, same accuracy
    return SGDClassifier(loss="log_loss", alpha=config.CATEGORY_ALPHA, max_iter=config.CATEGORY_EPOCHS, tol=None,
                         average=True, random_state=0).fit(X, y)


def fit_sgd(X, Y: np.ndarray) -> list:
    """The per-tag fit when every product is a label: liblinear would take hours on 56k labels and 90 categories,
    and a copy of the matrix per tag would not fit in memory. SGD reads one shared matrix, so its indices are
    tidied here once and no thread needs to touch them."""
    X.sort_indices()
    return Parallel(n_jobs=4, prefer="threads")(delayed(fit_sgd_one)(X, Y[:, j]) for j in range(Y.shape[1]))


def probabilities(models: list, X) -> np.ndarray:
    return np.column_stack([np.full(X.shape[0], m) if isinstance(m, float) else m.predict_proba(X)[:, 1]
                            for m in models])


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def fold_of(group: str, n_folds: int = config.FOLDS) -> int:
    """A product group's cross-validation fold, fixed by its name. Adding or editing one label never
    reshuffles the other folds, which would shift every tag's cut-off and flip thousands of products."""
    return zlib.crc32(group.encode()) % n_folds


def best_threshold(p: np.ndarray, y: np.ndarray) -> float:
    """The cut-off with the best F1, read from a smoothed F1 curve: for rare tags the raw curve is bumpy,
    and its single highest point can jump a long way when one label changes."""
    grid = np.linspace(0.15, 0.85, 29)
    scores = np.array([f1((p >= t)[y].sum(), (p >= t)[~y].sum(), (p < t)[y].sum()) for t in grid])
    smooth = np.convolve(np.pad(scores, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    return float(grid[np.argmax(smooth)])


class TagModel:
    """Per-tag logistic regressions, a cut-off per tag, and a formality regressor, all from one label set."""

    def __init__(self, X, Y: np.ndarray, groups: list[str], formality: list[int | None], fit=fit_per_tag,
                 n_folds: int = config.FOLDS, reuse: "TagModel | None" = None):
        # `reuse` is an earlier model of these same rows: tags whose labels did not change keep its work
        todo = [j for j in range(Y.shape[1]) if reuse is None or not np.array_equal(reuse.Y[:, j], Y[:, j])]
        # cross-validated probabilities: each product scored by a model that never saw its label.
        # Colour variants (same group) always land in the same fold.
        self.cv_probs = np.zeros(Y.shape) if reuse is None else reuse.cv_probs.copy()
        folds = np.array([fold_of(g, n_folds) for g in groups])
        for k in range(n_folds if todo else 0):
            test = folds == k
            if test.any() and not test.all():
                self.cv_probs[np.ix_(test, todo)] = probabilities(fit(X[~test], Y[~test][:, todo]), X[test])
        self.thresholds = np.array([best_threshold(self.cv_probs[:, j], Y[:, j] == 1) if Y[:, j].sum() >= 3 else 0.5
                                    for j in range(Y.shape[1])])
        self.models = [None] * Y.shape[1] if reuse is None else list(reuse.models)
        for j, m in zip(todo, fit(X, Y[:, todo]) if todo else ()):
            self.models[j] = m
        self.formality = Ridge(alpha=3.0).fit(X, formality) if formality[0] is not None else None
        # liblinear penalises the intercept like any weight, so each fitted model is a weighted sum of its training
        # rows: a tag's score for product x is exactly sum(alpha) + sum(alpha_i * (x_i . x)), one push per label
        self.X, self.Y = X, Y
        self.alpha = np.zeros(Y.shape) if reuse is None else reuse.alpha.copy()
        for j in todo:
            self.alpha[:, j] = representer(self.models[j], X, Y[:, j])

    def __getstate__(self) -> dict:  # saved without X, which rebuilds in seconds and is most of the size
        return {k: v for k, v in self.__dict__.items() if k != "X"}

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        level = np.clip(self.formality.predict(X), 1, 5) if self.formality is not None else np.zeros(X.shape[0])
        return probabilities(self.models, X), level


class Tagger:
    """The trained tagger: an image model and a text-only model, with products routed by whether they
    have an image vector."""

    def __init__(self, feats: Features, labels: dict[str, dict], tags: list[str] = TAGS, fit=fit_per_tag,
                 n_folds: int = config.FOLDS, reuse: "Tagger | None" = None):
        self.feats, self.tags = feats, tags
        self.ids = [i for i in labels if i in feats.index]
        rows = [feats.index[i] for i in self.ids]
        Y = np.array([[t in labels[i]["tags"] for t in tags] for i in self.ids], dtype=int)
        F = [labels[i].get("formality") for i in self.ids]
        groups = [feats.products[r]["group"] for r in rows]

        def matrix(old: TagModel | None, part: list[int], with_image: bool):
            x = getattr(old, "X", None)   # an earlier model in memory has the same matrix; a saved one has none
            return x if x is not None else feats.matrix(part, with_image)

        same = reuse is not None and reuse.ids == self.ids and reuse.tags == tags  # same rows and same columns
        old_text, old_image = (reuse.text, reuse.image) if same else (None, None)
        self.text = TagModel(matrix(old_text, rows, False), Y, groups, F, fit, n_folds, old_text)
        with_vec = [k for k, r in enumerate(rows) if feats.has_vector(r)]
        self.image, self.image_ids = None, [self.ids[k] for k in with_vec]
        if len(with_vec) >= 50:
            self.image = TagModel(matrix(old_image, [rows[k] for k in with_vec], True), Y[with_vec],
                                  [groups[k] for k in with_vec], [F[k] for k in with_vec], fit, n_folds, old_image)
        # what the tagger says about each labelled product when it has not seen that product's label
        image_pos = {k: n for n, k in enumerate(with_vec)} if self.image else {}
        self.cv_tags = {}
        for k, (pid, row) in enumerate(zip(self.ids, rows)):
            if k in image_pos:
                p, thr = self.image.cv_probs[image_pos[k]], self.image.thresholds
            else:
                p, thr = self.text.cv_probs[k], self.text.thresholds
            self.cv_tags[pid] = self.choose(p, thr, row)[0]
        self.agreement = evaluate(self.cv_tags, labels, tags)

    def __getstate__(self) -> dict:  # saved without the shared features
        return {k: v for k, v in self.__dict__.items() if k != "feats"}

    def choose(self, p: np.ndarray, thresholds: np.ndarray, row: int) -> tuple[list[str], list[dict]]:
        return R.choose_tags(dict(zip(self.tags, p.tolist())), dict(zip(self.tags, thresholds.tolist())),
                             self.feats.signals(row), config.MAX_TAGS)

    def taught_by(self, row: int, tag: str, k: int = 3) -> list[tuple[str, float]]:
        """The labels that pushed this product's score for `tag` up the most: similar labelled products whose label
        has the tag. Each push is an exact part of the score (see TagModel), though the list is not all of it."""
        use_image = self.image is not None and self.feats.has_vector(row)
        m, ids = (self.image, self.image_ids) if use_image else (self.text, self.ids)
        j = self.tags.index(tag)
        overlap = m.X @ self.feats.matrix([row], use_image).toarray().ravel()
        push = m.alpha[:, j] * overlap
        own = self.feats.products[row]["id"]
        best = [i for i in np.argsort(-push) if m.Y[i, j] == 1 and overlap[i] > 0 and ids[i] != own]
        return [(ids[i], float(push[i])) for i in best[:k]]

    def predict(self, rows: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, set[int]]:
        """Probabilities, formality and thresholds per row, plus which rows used the image model."""
        n = len(rows)
        P, F, T = np.zeros((n, len(self.tags))), np.zeros(n), np.zeros((n, len(self.tags)))
        img = [k for k, r in enumerate(rows) if self.image and self.feats.has_vector(r)]
        img_set = set(img)
        txt = [k for k in range(n) if k not in img_set]
        for part, model, with_image in ((txt, self.text, False), (img, self.image, True)):
            if part:
                P[part], F[part] = model.predict(self.feats.matrix([rows[k] for k in part], with_image))
                T[part] = model.thresholds
        return P, F, T, {rows[k] for k in img}


class CategoryTagger(Tagger):
    """Learns subSubCategory. The same model shape as the style tagger, without its house rules or formality."""

    def __init__(self, feats: Features, labels: dict[str, dict], tags: list[str], reuse: Tagger | None = None):
        super().__init__(feats, labels, tags, fit_sgd, config.CATEGORY_FOLDS, reuse)

    def choose(self, p: np.ndarray, thresholds: np.ndarray, row: int) -> tuple[list[str], list[dict]]:
        hits = np.flatnonzero(p >= thresholds)
        top = hits[np.argsort(-p[hits])][:config.MAX_TAGS] if hits.size else [int(np.argmax(p))]
        return [self.tags[j] for j in top], []


class BridalTagger(Tagger):
    """Learns which wedding function (desi_rules.EVENTS) an outfit is for, from the style tagger's features, without
    its house rules or formality. Its few thousand labels need no faster solver: about a minute with the style one's."""
    choose = CategoryTagger.choose


def tag_categories(tagger: Tagger, labels: dict[str, dict], excluded: set[str] = frozenset()) -> dict[str, dict]:
    """The category tagger's output, one record per product. A starting label (the brand's category) only trains
    the model, so the product outputs the model's answer; a hand label also decides its own product's output."""
    feats = tagger.feats
    rows = list(range(len(feats.products)))
    P, _, T, used_image = tagger.predict(rows)
    out = {}
    for r in rows:
        p = feats.products[r]
        tags = tagger.choose(P[r], T[r], r)[0]
        rec = {"id": p["id"], "name": p["name"], "supplier": p["supplier"], "gender": p["gender"],
               "subCategory": p["subCategory"], "subSubCategory": p["subSubRaw"], "price": p["price"],
               "imageUrl": p["imageUrl"], "url": p["url"], "tags": tags,
               "tagScores": {t: round(float(P[r][tagger.tags.index(t)]), 3) for t in tags},
               "ruleNotes": [], "usedImage": r in used_image, "needsReview": not (P[r] >= T[r]).any(), "source": "model"}
        if p["id"] in excluded:
            rec = excluded_record(rec)
        elif labels.get(p["id"], {}).get("hand"):
            rec = labelled_record(rec, labels[p["id"]], dict(zip(tagger.tags, P[r].tolist())))
        out[p["id"]] = rec
    return out


def tag_catalog(tagger: Tagger, labels: dict[str, dict], excluded: set[str] = frozenset()) -> dict[str, dict]:
    """Tags for every product, in the output format of data/predictions.jsonl."""
    feats = tagger.feats
    rows = list(range(len(feats.products)))
    P, F, T, used_image = tagger.predict(rows)
    out = {}
    for r in rows:
        p, s = feats.products[r], feats.signals(r)
        tags, notes = tagger.choose(P[r], T[r], r)
        level = int(np.rint(F[r]))
        rec = {
            "id": p["id"], "name": p["name"], "supplier": p["supplier"], "gender": p["gender"],
            "subCategory": p["subCategory"], "subSubCategory": p["subSub"], "price": p["price"],
            "imageUrl": p["imageUrl"], "url": p["url"],
            "tags": tags, "tagScores": {t: round(float(P[r][TAGS.index(t)]), 3) for t in tags},
            "formality": level, "formalityLabel": FORMALITY_NAMES[level], "formalityScore": round(float(F[r]), 2),
            "evidence": R.evidence(s), "ruleNotes": notes, "usedImage": r in used_image,
            "needsReview": bool(not (P[r] >= T[r]).any() or (s["no_text"] and r not in used_image)),
            "source": "model",
        }
        if p["id"] in excluded:
            rec = excluded_record(rec)
        elif p["id"] in labels:
            rec = labelled_record(rec, labels[p["id"]], dict(zip(TAGS, P[r].tolist())))
        out[p["id"]] = rec
    return out


def excluded_record(rec: dict) -> dict:
    """An excluded product outputs no tags. The model's own answer is kept in modelTags."""
    return {**rec, "tags": [], "tagScores": {}, "modelTags": rec["tags"], "source": "excluded", "ruleNotes": [],
            "needsReview": False}


def labelled_record(rec: dict, label: dict, probs: dict[str, float]) -> dict:
    """A hand-labelled product outputs its label, so a correction always shows on that product.
    The model's own choice is kept in modelTags: it is what similar, unlabelled products get."""
    out = {**rec, "tags": list(label["tags"]), "tagScores": {t: round(probs[t], 3) for t in label["tags"]},
           "modelTags": rec["tags"], "source": "label", "ruleNotes": [], "needsReview": False}
    if "formality" in label:
        level = label["formality"]
        out.update({"formality": level, "formalityLabel": FORMALITY_NAMES[level], "formalityScore": float(level)})
    return out


def evaluate(pred: dict[str, list[str]], labels: dict[str, dict], tags: list[str] = TAGS) -> dict:
    ids = [i for i in labels if pred.get(i)]
    per_tag = {}
    for t in tags:
        tp = sum(t in pred[i] and t in labels[i]["tags"] for i in ids)
        fp = sum(t in pred[i] and t not in labels[i]["tags"] for i in ids)
        fn = sum(t not in pred[i] and t in labels[i]["tags"] for i in ids)
        per_tag[t] = {"n": tp + fn, "precision": tp / (tp + fp) if tp + fp else 0.0,
                      "recall": tp / (tp + fn) if tp + fn else 0.0, "f1": f1(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn}
    tp, fp, fn = (sum(m[k] for m in per_tag.values()) for k in ("tp", "fp", "fn"))
    supported = [m["f1"] for m in per_tag.values() if m["n"]]
    return {
        "n": len(ids),
        "micro_f1": f1(tp, fp, fn),
        "macro_f1": float(np.mean(supported)) if supported else 0.0,
        "top_tag_right": float(np.mean([pred[i][0] == labels[i]["tags"][0] for i in ids])) if ids else 0.0,
        "top_tag_acceptable": float(np.mean([pred[i][0] in labels[i]["tags"] for i in ids])) if ids else 0.0,
        "per_tag": per_tag,
    }


def benchmark(feats: Features) -> dict:
    """The fixed check: train on the original training labels, score on the original held-out labels
    (from labels.original.jsonl, unaffected by edits). Run it after changing code or rules."""
    original = L.load(config.LABELS_ORIGINAL)
    train = {i: r for i, r in original.items() if r.get("split") == "train"}
    # products dropped from the catalog can't be scored; Tagger skips their training labels the same way
    test = {i: r for i, r in original.items() if r.get("split") == "test" and i in feats.index}
    tagger = Tagger(feats, train)
    rows = [feats.index[i] for i in test]
    P, _, T, _ = tagger.predict(rows)
    return evaluate({i: tagger.choose(P[k], T[k], rows[k])[0] for k, i in enumerate(test)}, test)


def save_predictions(records: dict[str, dict], path: Path = config.PREDICTIONS) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records.values())
    os.replace(tmp, path)


def load_predictions(path: Path = config.PREDICTIONS) -> dict[str, dict] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


CAUSES = ["your label", "follows your change", "your exclusions", "photo added", "nudged at a cut-off"]


def changes(before: dict[str, dict], after: dict[str, dict], edited: set[str] = frozenset()) -> list[dict]:
    """Products whose tags differ between two prediction sets, with the likely cause, most direct first.

    A product "follows your change" when it gains a tag your edits added or loses one they removed: the
    tagger learned the pattern. "your exclusions" means it was excluded from tagging or included again.
    "photo added" means it got an image vector since (`run.py embed`), so it is now tagged from photo and text.
    Every retrain also nudges some products that sit right at a tag's cut-off in other directions; those are
    listed last as "nudged at a cut-off"."""
    pairs = [(before[p]["tags"], after[p]["tags"]) for p in edited if p in before and p in after]
    added = {t for old, new in pairs for t in new if t not in old}
    removed = {t for old, new in pairs for t in old if t not in new}
    out = []
    for pid, rec in after.items():
        old = (before.get(pid) or {}).get("tags")
        if old is not None and old != rec["tags"]:
            gained = [t for t in rec["tags"] if t not in old]
            lost = [t for t in old if t not in rec["tags"]]
            flipped = (before[pid].get("source") == "excluded") != (rec.get("source") == "excluded")
            cause = (CAUSES[0] if pid in edited else
                     CAUSES[2] if flipped else
                     CAUSES[1] if added.intersection(gained) or removed.intersection(lost) else
                     CAUSES[3] if rec.get("usedImage") and not before[pid].get("usedImage") else CAUSES[4])
            out.append({"id": pid, "before": old, "after": rec["tags"], "cause": cause, "gained": gained,
                        "lost": lost, "topChanged": old[:1] != rec["tags"][:1]})  # an excluded product has no tags
    return sorted(out, key=lambda c: (CAUSES.index(c["cause"]), not c["topChanged"]))
