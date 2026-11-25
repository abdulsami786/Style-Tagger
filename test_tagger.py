"""Checks for the desi rules and label handling.  Run:  python test_tagger.py   (or python -m pytest)"""

import desi_rules as R
import labels as L


def product(**kw) -> dict:
    base = dict(name="", desc="", details="", keywords="", subSub="", subCategory="Ready to Wear",
                gender="women", price=None, brandMedian=5000.0, pricePct=0.5, url="", imageUrl="")
    return {**base, **kw}


def top_rule(p: dict) -> str:
    scores = R.rule_scores(R.signals(p))
    return max(scores, key=scores.get)


def test_rules_read_products_like_a_shopper():
    assert top_rule(product(name="2 Piece Printed Lawn Suit", desc="Printed lawn shirt dyed trouser", price=2800)) == "casuals"
    assert top_rule(product(name="Zoryana", desc="signature lehenga for a modern bride, gota patti, sequins", price=890000)) == "bridal_wear"
    assert top_rule(product(name="NM-150", desc="embroidered gold sherwani", gender="men", subSub="sherwani")) == "wedding"
    assert top_rule(product(name="Graphic Boxy Tee", desc="streetwear tee 100% cotton", subCategory="Western Wear", price=1500)) == "streetwear"
    assert top_rule(product(name="Saudi Jilbab", subCategory="Modest Wear", subSub="jilbab", price=2700)) == "modest"
    assert top_rule(product(name="Pink Plain Formal Shirt", desc="formal gatherings", gender="men", subCategory="Western Wear", price=3300)) == "office_wear"
    assert top_rule(product(name="3 Piece Embroidered Lawn Suit", desc="embroidered neckline, chiffon dupatta, Eid", price=8000)) == "festive_wear"
    assert R.formality(R.signals(product(name="Bridal Lehenga", price=500000))) == 5.0


def test_bride_rule_needs_a_couture_price():
    assert not R.signals(product(name="Maharani 2 piece", desc="bridal inspired print", price=11500))["bride"]
    assert R.signals(product(name="Maharani 2 piece", desc="bridal inspired print", price=320000))["bride"]
    couture = product(name="D-402", desc="dripping with pearls and sequins for upcoming events", price=475000)
    assert not R.signals(couture)["bride"]                            # couture formal, not the bride's
    assert R.signals({**couture, "name": "Zoryana Lehenga"})["bride"]  # bridal silhouette at couture price
    assert not R.signals(product(name="Gown", desc="perfect for a bridal shower", price=90000))["bride"]


def choose(s: dict, **probs) -> tuple[list[str], list[dict]]:
    p = {t: probs.get(t, 0.0) for t in R.TAGS}
    return R.choose_tags(p, {t: 0.5 for t in R.TAGS}, s)


def test_hard_rules():
    plain = R.signals(product(name="Embroidered Suit", price=20000))
    bride = {**plain, "bride": 1.0}
    tags, notes = choose(plain, wedding=0.9, bridal_wear=0.6)
    assert tags == ["wedding"] and notes[0]["tag"] == "bridal_wear" and notes[0]["change"] == "removed"
    assert choose(bride, festive_wear=0.9)[0][:2] == ["bridal_wear", "wedding"]   # rule adds both
    assert choose({**plain, "men": 1.0}, bridal_wear=0.9, wedding=0.7)[0] == ["wedding"]
    assert choose(plain, casuals=0.9, wedding=0.6)[0] == ["casuals"]             # never both
    assert choose(plain, modest=0.9, party_wear=0.7)[0] == ["party_wear"]        # no modest garment
    assert choose({**plain, "modest_cat": 1.0}, modest=0.9)[0] == ["modest"]
    assert choose(plain, modest=0.4, festive_wear=0.3)[0] == ["festive_wear"]   # fallback skips blocked tags
    many, _ = choose(bride, wedding=0.9, festive_wear=0.8, party_wear=0.7, casuals=0.6)
    assert len(many) == 3 and len(set(many)) == 3


def test_label_checks():
    ok = {"id": "x", "tags": ["wedding", "festive_wear"], "formality": 4}
    assert L.problem(ok) is None
    for bad in ({**ok, "tags": []}, {**ok, "tags": ["wedding", "casuals", "modest", "party_wear"]},
                {**ok, "tags": ["gala"]}, {**ok, "tags": ["wedding", "wedding"]},
                {**ok, "formality": 6}, {**ok, "formality": True}, {**ok, "id": ""}):
        assert L.problem(bad), bad
    assert L.differences({"a": ok}, {"a": {**ok, "formality": 3}, "b": ok}) == {"a", "b"}


def test_a_labelled_product_outputs_its_label():
    import model as M
    rec = {"tags": ["casuals"], "tagScores": {"casuals": 0.9}, "formality": 1, "formalityLabel": "daily",
           "formalityScore": 1.2, "ruleNotes": [{"tag": "wedding", "change": "removed", "rule": "x"}],
           "needsReview": True, "source": "model"}
    out = M.labelled_record(rec, {"id": "a", "tags": ["festive_wear", "casuals"], "formality": 2}, {t: 0.2 for t in R.TAGS})
    assert out["tags"] == ["festive_wear", "casuals"] and out["modelTags"] == ["casuals"] and out["source"] == "label"
    assert out["formality"] == 2 and out["formalityLabel"] == "semi_formal" and not out["ruleNotes"] and not out["needsReview"]


def test_change_causes():
    import model as M
    before = {"a": {"tags": ["wedding"]}, "b": {"tags": ["festive_wear", "wedding"]}, "c": {"tags": ["casuals", "office_wear"]}}
    after = {"a": {"tags": ["sportswear"]}, "b": {"tags": ["festive_wear", "sportswear"]}, "c": {"tags": ["casuals"]}}
    causes = {c["id"]: c["cause"] for c in M.changes(before, after, {"a"})}
    assert causes == {"a": "your label", "b": "follows your change", "c": "nudged at a cut-off"}
    before["d"], after["d"] = {"tags": ["casuals"], "usedImage": False}, {"tags": ["party_wear"], "usedImage": True}
    assert {c["id"]: c["cause"] for c in M.changes(before, after)}["d"] == "photo added"


def test_exclusions():
    import model as M
    fabric = product(name="Dupont Shadow", supplier="Lakhany", gender="men", subCategory="Unstitched", subSub="suits", id="a")
    suit = {**fabric, "subCategory": "Ready to Wear", "id": "b"}
    rule = {"supplier": "Lakhany", "gender": "men", "subCategory": "Unstitched", "subSub": "suits"}
    assert L.rule_problem(rule) is None and L.rule_problem({}) and L.rule_problem({"price": "1"}) and L.rule_problem({"id": 5})
    assert L.excluded([fabric, suit], [rule]) == {"a"} and L.excluded([fabric], []) == set()
    assert L.excluded([fabric, suit], [{"id": "b"}, {"id": "gone"}]) == {"b"}
    assert L.excluded([fabric, suit], [{"id": "b"}, rule]) == {"a", "b"}
    assert L.trainable({"a": {}, "b": {}}, {"a"}) == {"b": {}}
    out = M.excluded_record({"tags": ["casuals"], "tagScores": {"casuals": 0.9}, "ruleNotes": [], "needsReview": True, "source": "model"})
    assert out["tags"] == [] and out["modelTags"] == ["casuals"] and out["source"] == "excluded" and not out["needsReview"]
    before, after = {"a": {"tags": ["casuals"], "source": "model"}}, {"a": out}
    assert M.changes(before, after)[0]["cause"] == "your exclusions"   # no tags left must not crash
    assert M.changes(after, before)[0]["cause"] == "your exclusions"   # included again


def test_label_pushes_add_up_to_the_score():
    import numpy as np
    from scipy import sparse
    import model as M
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.normal(size=(80, 6)))
    Y = np.column_stack([X[:, 0].toarray().ravel() > 0, X[:, 1].toarray().ravel() > 0.5]).astype(int)
    m = M.TagModel(X, Y, [str(i) for i in range(80)], [1] * 80)
    x = sparse.csr_matrix(rng.normal(size=(1, 6)))
    for j, fitted in enumerate(m.models):
        pushes = m.alpha[:, j] * (m.X @ x.T).toarray().ravel()
        assert abs(m.alpha[:, j].sum() + pushes.sum() - fitted.decision_function(x)[0]) < 0.01


def test_a_new_tag_needs_no_rule():
    saved = R.TAGS
    try:
        R.TAGS = saved + ["mehndi"]
        assert R.rule_scores(R.signals(product(name="Yellow Gharara Set")))["mehndi"] == 0.0
    finally:
        R.TAGS = saved


def test_category_starting_labels():
    rows = [("a", "abayas", "Abayas"), ("b", "abayas", "Abayas"), ("c", "abaya", "Abaya"), ("d", "waistcoat", "Waistcoat"),
            ("e", "waistcoat", "Waistcoat"), ("f", "waistcoat", "WaistCoat"), ("g", "none", ""), ("h", "necklace", "Necklace")]
    seeds = L.category_seeds([product(id=i, subSub=s, subSubRaw=r) for i, s, r in rows])
    assert seeds["c"]["tags"] == ["Abayas"]        # merged into the commoner spelling
    assert seeds["f"]["tags"] == ["Waistcoat"]     # capitals and spacing never split a category
    assert "g" not in seeds                        # the catalog says None: no starting label
    assert "h" not in seeds                        # not clothing: not a category


def test_category_label_file_keeps_only_your_edits():
    seeds = {i: {"id": i, "tags": [t]} for i, t in (("a", "Shirt"), ("b", "Kurta"), ("c", "Pant"))}
    labels = {**seeds, "a": {"id": "a", "tags": ["Kurta", "Shirt"], "hand": True}, "new": {"id": "new", "tags": ["Shirt"], "hand": True}}
    del labels["c"]
    edits = L.edits_of(labels, seeds)
    assert set(edits) == {"a", "new", "c"} and edits["c"] == {"id": "c", "removed": True}
    assert L.with_edits(seeds, edits) == labels
    assert L.edits_of(seeds, seeds) == {}          # restoring leaves nothing on disk
    assert L.edit_problem({"id": "c", "removed": True}) is None and L.edit_problem({"id": "x", "tags": []})
    assert L.differences({**seeds, "b": {**seeds["b"], "hand": True}}, seeds) == {"b"}  # confirming a brand category counts


def test_removing_your_category_label_goes_back_to_the_brand():
    import app as A
    a = A.CategoryApp.__new__(A.CategoryApp)   # no training needed for this
    a.seeds = {"a": {"id": "a", "tags": ["Shirt"]}, "b": {"id": "b", "tags": ["Kurta"]}}
    labels = {**a.seeds, "a": {"id": "a", "tags": ["Coat"], "hand": True}, "c": {"id": "c", "tags": ["Pant"], "hand": True}}
    a.drop_label(labels, "a")                  # your label on a product with a brand category: back to the brand
    assert labels["a"] == a.seeds["a"] and L.edits_of(labels, a.seeds) == {"c": labels["c"]}
    a.drop_label(labels, "c")                  # your label on a product the catalog marks None: no label left
    a.drop_label(labels, "b")                  # a brand category removed: out of training, and remembered on disk
    assert "b" not in labels and "c" not in labels and L.edits_of(labels, a.seeds) == {"b": {"id": "b", "removed": True}}


def test_category_features_cannot_see_the_answer():
    import model as M
    p = product(id="x", name="Silk Set", supplier="B", piece="", subSub="lehenga")
    other = {**p, "subSub": "abayas"}
    assert "sub_lehenga" in M.doc(p)               # the style tagger's input carries it, so hiding it matters
    assert M.doc(M.hide_category(p)) == M.doc(M.hide_category(other))
    assert R.signals(M.hide_category(p)) == R.signals(M.hide_category(other))


def test_a_retrain_refits_only_the_tags_that_changed():
    import numpy as np
    from scipy import sparse
    import model as M
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.normal(size=(120, 6)))
    Y = np.column_stack([X[:, k].toarray().ravel() > 0 for k in range(3)]).astype(int)
    groups = [str(i) for i in range(120)]
    fits = []

    def counting_fit(x, y):
        fits.append(y.shape[1])
        return M.fit_per_tag(x, y)

    first = M.TagModel(X, Y, groups, [None] * 120, counting_fit, 3)
    assert fits == [3, 3, 3, 3]                                              # three folds and the final model
    fits.clear()
    edited = Y.copy()
    edited[0, 1] = 1 - edited[0, 1]
    again = M.TagModel(X, edited, groups, [None] * 120, counting_fit, 3, first)
    assert fits == [1, 1, 1, 1]                                              # only the edited tag
    assert again.models[0] is first.models[0] and again.models[1] is not first.models[1]
    assert (again.cv_probs[:, 0] == first.cv_probs[:, 0]).all() and (again.alpha[:, 2] == first.alpha[:, 2]).all()
    fresh = M.TagModel(X, edited, groups, [None] * 120, M.fit_per_tag, 3)    # the shortcut must not change the answer
    assert np.allclose(again.cv_probs, fresh.cv_probs) and np.allclose(again.alpha, fresh.alpha)
    fits.clear()
    M.TagModel(X, edited, groups, [None] * 120, counting_fit, 3, again)
    assert fits == []                                                        # nothing edited, nothing fitted


def test_one_example_can_never_be_the_answer():
    import numpy as np
    from scipy import sparse
    import model as M
    X = sparse.csr_matrix(np.eye(6))
    assert M.fit_sgd_one(X, np.array([1, 0, 0, 0, 0, 0])) == 0.0


def test_category_tagger_picks_and_pins():
    import numpy as np
    import model as M
    tagger = M.CategoryTagger.__new__(M.CategoryTagger)   # choose() and tag_categories() need no fitted model
    tagger.tags = ["A", "B", "C", "D"]
    p, half = np.array([.9, .8, .7, .6]), np.full(4, .5)
    assert tagger.choose(p, half, 0)[0] == ["A", "B", "C"]                               # at most three, strongest first
    assert tagger.choose(np.array([.2, .4, .1, .3]), half, 0)[0] == ["B"]                # none clear their cut-off: the likeliest
    assert tagger.choose(np.array([.9, .6, .2, .1]), np.array([.95, .5, .5, .5]), 0)[0] == ["B"]
    tagger.feats = type("F", (), {"products": [product(id=i, supplier="x", subSubRaw="A") for i in "abc"]})
    tagger.predict = lambda rows: (np.tile([.9, .1, 0, 0], (len(rows), 1)), None, np.full((len(rows), 4), .5), set())
    labels = {"a": {"id": "a", "tags": ["B"]}, "b": {"id": "b", "tags": ["B"], "hand": True}}
    out = M.tag_categories(tagger, labels, {"c"})
    assert out["a"]["tags"] == ["A"] and out["a"]["source"] == "model"                    # a brand category only trains
    assert out["b"]["tags"] == ["B"] and out["b"]["modelTags"] == ["A"] and out["b"]["source"] == "label"  # your label decides
    assert out["c"]["tags"] == [] and out["c"]["source"] == "excluded"


def test_bridal_starting_labels():
    guess = lambda formality=4.5, **kw: R.bridal_events(product(**kw), formality)
    assert guess(name="Zaib Lehenga", desc="a red lehenga for your mehndi") == ["mehendi"]   # a named function beats colour
    assert guess(name="Mehendi Gharara") == ["mehendi"]                                       # the app's spelling
    assert guess(name="Gul", desc="for your barat, walima or nikkah") == []                   # three named say nothing...
    assert guess(name="Red Gul", desc="for your barat, walima or nikkah") == ["barat"]        # ...so the colour decides
    assert guess(name="Ivory Gharara", desc="with a mehndi green dupatta") == ["nikah"]       # a colour, not the function
    assert guess(name="Red and Gold Lehenga") == ["barat"] and guess(name="Gold and Red Lehenga") == ["valima"]  # first named
    assert guess(name="Lehenga", url="https://x.pk/products/maroon-bridal-lehenga") == ["barat"]                 # URL slug
    assert guess(name="Lilac Maxi") == ["nikah", "valima"] and guess(name="Embroidered Lehenga") == []          # not "red"
    assert guess(name="Hot Pink Gharara") == guess(name="Teal Lehenga") == ["mehendi", "dholki"]                # the mehndi palette
    assert guess(name="Black Net Saree") == [] and guess(name="Navy Lehenga") == []
    assert guess(name="Yellow Gharara") == ["mehendi", "dholki"] and guess(5.0, name="Yellow Gharara") == ["mehendi"]  # her heaviest


def test_bridal_scope_is_the_brides_outfits():
    scope = R.bridal_scope
    rec = lambda tags=("wedding",), notes=(), gender="women": dict(gender=gender, tags=list(tags), ruleNotes=list(notes))
    outfit = lambda price=500000, name="Zaib Lehenga", sub="": product(name=name, price=price, subSub=sub)
    assert scope(outfit(name="Zaib"), rec(["bridal_wear", "wedding"]))                      # Style's bride
    assert scope(outfit(name="Zaib"), rec(notes=[{"tag": "bridal_wear", "change": "removed"}]))  # Style's model said bride
    assert scope(outfit(60000), rec(["bridal_wear"])) and not scope(outfit(60000, "Zaib Kurta"), rec(["bridal_wear"]))
    assert scope(outfit(None, "Zaib"), rec(["bridal_wear"]))           # couture with no price
    assert scope(outfit(120000), rec()) and not scope(outfit(80000), rec())                 # a lehenga at a bride's price
    assert scope(outfit(120000, "Zaib", "gharara"), rec())                                  # ...its category says so too
    assert scope(outfit(80000, "Mustard Lehenga"), rec()) and not scope(outfit(60000, "Mustard Lehenga"), rec())  # mehndi
    assert not scope(outfit(300000, "Ivory Kurta Set"), rec())          # not a bridal silhouette
    assert not scope(outfit(), rec(["festive_wear"]))                   # not wedding wear
    assert not scope(outfit(), {**rec(), "source": "label"})            # your Style label: a guest's wedding outfit
    assert not scope(outfit(name="Black Lehenga"), rec(["bridal_wear"]))  # brides don't wear black
    assert not scope(outfit(), rec(["bridal_wear"], gender="men"))
    assert not scope(outfit(sub="sherwani"), rec(["bridal_wear"])) and not scope(outfit(name="Ivory Sherwani"), rec(["bridal_wear"]))


def test_bridal_tags_only_brides_outfits():
    import numpy as np
    import app as A
    import model as M
    a = A.BridalApp.__new__(A.BridalApp)   # no training needed for this
    ps = [product(id=i, name=n, price=300000, supplier="x", subSubRaw="") for i, n in (("a", "Lehenga"), ("b", "Kurta"), ("c", "Lehenga"))]
    a.style = type("S", (), {"products": ps, "current": {i: {"gender": g, "tags": ["wedding"], "ruleNotes": []}
                                                         for i, g in (("a", "women"), ("b", "women"), ("c", "men"))}})
    tagger = M.BridalTagger.__new__(M.BridalTagger)
    tagger.tags = R.EVENTS
    tagger.feats = type("F", (), {"products": ps})
    tagger.predict = lambda rows: (np.tile([.9, .1, .1, .1, .1], (len(rows), 1)), None, np.full((len(rows), 5), .5), set())
    out = a.tag_catalog(tagger, {"c": {"id": "c", "tags": ["barat"], "hand": True}}, set())
    assert out["a"]["tags"] == ["mehendi"] and out["a"]["source"] == "model"
    assert out["b"]["tags"] == [] and out["b"]["modelTags"] == ["mehendi"] and out["b"]["source"] == "not bridal"
    assert out["c"]["tags"] == ["barat"] and out["c"]["source"] == "label"   # your label decides, even outside the scope


def test_no_photo_follows_what_the_app_shows():
    import remove_no_photo as N
    saved = N.probe
    try:
        N.probe = lambda url, **kw: {"ok": None, "ribbon": "placeholder 100x97", "gone": "HTTP 404"}.get(url, "no image url")
        assert N.problem({"imageUrl": "ok", "originalImageUrl": "gone"}) is None
        assert N.problem({"imageUrl": "gone", "originalImageUrl": "ok"}) is None          # the app falls back
        assert N.problem({"imageUrl": "ribbon", "originalImageUrl": "ok"}) == "placeholder 100x97"  # loads, so no fallback
        assert N.problem({"imageUrl": "gone", "originalImageUrl": "ribbon"}) == "HTTP 404"
        assert N.problem({"imageUrl": "gone"}) == "HTTP 404"
        assert N.problem({}) == "no image url"
    finally:
        N.probe = saved


def test_password_check():
    import base64
    import app as A
    basic = lambda s: "Basic " + base64.b64encode(s.encode()).decode()
    assert A.password_ok(basic("anyone:s3cret"), "s3cret") and A.password_ok(basic(":s3cret"), "s3cret")
    assert not A.password_ok(basic("anyone:wrong"), "s3cret")
    assert not A.password_ok(basic("s3cret"), "s3cret")                       # no colon: that is a user name
    assert not A.password_ok("Bearer " + basic("x:s3cret")[6:], "s3cret")
    assert not A.password_ok("Basic !!notbase64", "s3cret")
    assert not A.password_ok("Basic " + base64.b64encode(b"\xff:s3cret").decode(), "s3cret")  # not UTF-8
    assert not A.password_ok("", "s3cret")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all checks passed")
