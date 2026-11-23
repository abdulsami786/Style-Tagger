# Desi Tagger

Two taggers in one local web app, switched with **Style | Category** at the top of the page. Most of this
file is about Style; see [Category tagger](#category-tagger) for what differs.

Tags every Lookflock product with the occasions a Pakistani shopper would wear it to (wedding, bridal,
festive, party, casual, office and so on), plus a formality level from 1 (daily) to 5 (bridal).
You can correct any product in a local web app, retrain in about 20 seconds, and see which other
products changed and why.

**How it decides:** it reads the product's text, fabric, embroidery, price and brand through desi fashion
rules, adds the product photo through FashionSigLIP, and learns from 887 hand labels. On a fixed held-out
test of 288 products it scores
0.85 micro-F1 (0.83 macro-F1), and its top tag is acceptable 93% of the time. The old CLIP system scored 0.52. Tag meanings are in [TAGS.md](TAGS.md).

## Set up once

```powershell
cd D:\Lookflock\DesiTagger
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Use it

```powershell
python run.py prep     # 1. clean the catalog export (reads liveProducts.cleaned.json in this folder; see config.py)
python run.py tag      # 2. tag every product -> data\predictions.jsonl
python run.py serve    # 3. web app: open http://127.0.0.1:8765
```

Optional, slow: `python run.py embed` adds photo vectors (see "Photo vectors" below).

To drop products with no working photo from the catalog export, run `python remove_no_photo.py` before
`prep`. It counts a product as having no photo when the image the app shows won't download, won't open, or
is a placeholder (under 150 px, like the 100×97 "out of stock" ribbon). The app falls back to
`originalImageUrl` when `imageUrl` fails to load, and the script checks the same way. It takes about 2
hours and checks every failure a second time before it removes anything. If it stops, run it again: it
continues from where it was. The removed products are saved
in full to `removed_no_photo_<date_time>/`.

## Fix a wrong tag

In the web app:

1. **Catalog** tab: search for the product and click it.
2. In the panel, under "Correct label", pick the right tags (1 to 3, the first is the main one) and a
   formality, then **Save label**. The product is now a label, saved in `labels/labels.jsonl`.
3. Press **Retrain** (about 20 s). The product now shows your label; a labelled product always outputs its
   label. The **What changed** tab lists every product whose tags moved, before and after, marked with the
   cause:
   - **your label**: a product you labelled;
   - **follows your change**: it gained a tag you added or lost one you removed, so the tagger learned the pattern;
   - **photo added**: it got a photo vector since the last run, so it is now tagged from photo and text;
   - **nudged at a cut-off**: every retrain shifts some products that sit right at a tag's cut-off.

**One correction teaches little; ten to twenty similar ones teach a pattern.** In our test (before the
catalog had photo vectors), labelling one
embroidered Lakhany shirt as festive fixed that shirt. Only 4 other products followed, and the model
alone still called the shirt casual. The 907 nudged products were mostly second tags flipping at the
festive cut-off.

The **My labels** tab has bulk edits (replace, add or remove a tag on every label shown) and a "Tagger
disagrees" filter. That filter lists labels the tagger would predict differently without seeing them,
which is the quickest way to find mislabelled products.

**Try it safely.** In My labels, pick "wedding" in the tag filter, replace wedding with sportswear, and
press Retrain. In our run (before the catalog had photo vectors):
- 99% of the 8,621 unlabelled wedding products switched to sportswear (8,626 "follows your change",
  10 nudged);
- the real sportswear stayed sportswear;
- 408 bride outfits kept wedding, because a rule adds it (shown with a dashed border).

Press **Restore original labels** and Retrain to undo. Our run came back with zero differences.

You can also edit `labels/labels.jsonl` by hand, one product per line, then run `python run.py tag`:

```json
{"id": "<product id>", "tags": ["party_wear", "wedding"], "formality": 4}
```

## See which labels taught a tag

Open any product. **Which labels taught it these tags** lists, for each tag the model gives it, the hand labels
that pushed it there the most: similar labelled products whose label has that tag, with the size of each push.
These come straight from the model (its score for a tag is exactly a starting value plus one push per training
label), but the list shows only the biggest pushes. A wrong tag usually traces back to one or two labels here:
open one and fix its label, or exclude it, then press Retrain.

## Exclude products from tagging

Some products shouldn't get occasion tags at all, for example men's 4.5-meter suit pieces sold as fabric. Open
one and use **Exclude from tagging**. You choose the scope, with a count and examples for each:

- only this product;
- this brand's products in the same category (brand, gender, category and shop category all match);
- every brand's products in that category.

Excluded products output no tags (`source` is `excluded`, the model's answer stays in `modelTags`) and never
train the tagger, even when they are labelled. Only the labelled ones among them change what the tagger learns,
so the panel says how many that is. Exclusions apply when you press Retrain; the **What changed** tab marks those
products **your exclusions**. The **Excluded** tab lists every exclusion with **Include again**. They are saved in
`labels/exclusions.json`.

## Category tagger

Press **Category** at the top of the page. It predicts each product's `subSubCategory` from its name, text, brand
and photo, on the same screens as Style (catalog, My labels, What changed, Excluded, Retrain, Restore).

- **Starting labels.** Every product's own `subSubCategory` is a label: 56,205 products, 77 categories. These train
  the model but don't decide the output, so a product shows the model's answer.
- **Your labels decide.** A label you save trains the model and decides that product's output, even if you keep the
  brand's category. Only your edits are stored, in `labels/category_labels.jsonl`; **Restore** clears them.
- **Same editing as Style:** 1 to 3 tags, the first is the main one, plus bulk edits. There is no formality and none
  of Style's house rules. Exclusions are separate from Style's (`labels/category_exclusions.json`).
- **Merged spellings.** Abaya/Abayas, Suit/Suits, Shawl/Shawls, Top/Tops, Sweatshirt/Sweatshirts, Chino/Chinos,
  Cullote/Culotte/Culottes, Pishwas/Peshwas and Jogger/Joggers each count as one category, named by the catalog's
  most common spelling. Spellings that differ only in capitals or spacing are joined too. 405 products therefore
  start under a different name from the one the main app stores. Products marked `None` start with no label. The
  list is `CATEGORY_MERGE` in `labels.py`.
- **Clothing only.** Earrings, Fragrance and Necklace are not categories (`NOT_CLOTHING` in `labels.py`). Their 9
  products are excluded, and so are 346 accessories that brands filed under clothing categories: Outfitters'
  sunglasses, sneakers, belts, wallets, rings and hair clips under Casuals; Chicophicial's jewellery under Kurta;
  EdenRobe's earrings and necklaces under Casuals and Kurta. Excluded products get no category and don't train.
  They are listed in the Excluded tab, where **Include again** undoes one.

**How good it is.** On products the model hasn't seen, its main tag matches the label 85% of the time (F1 0.85). It
disagrees with the brand on about 8,200 products; find them with **Tagger disagrees** in My labels. The output's main
tag differs from the brand's category on about 10% of products, sometimes because the brand was wrong and sometimes
because the model is.

**Fix a whole pattern, not a few examples.** Similar products keep their own brand labels, and those teach against
your correction. Correcting 20 of 67 hijabs filed as Scarf moved only 6 of the other 47. To fix a group, filter it
in My labels (search plus tag filter) and use bulk **Replace**: every product shown becomes your label.

| On this laptop | Time |
|---|---|
| Full training (first run, or after code or catalog changes) | 7 to 10 minutes, plus about 2 if the features need rebuilding |
| Retrain after a few edits | about 1 minute: only the categories you touched are refitted |
| Retrain after an exclusion, or after removing a brand category | a full training, because the training rows changed |
| First Category click after a server start | about 45 seconds: it loads `data/category_tagger.joblib` |

Keep the laptop awake during a full training; sleep pauses it. `python run.py tag --mode category` writes
`data/category_predictions.jsonl` without the web app. Each line has the brand's `subSubCategory`, the output `tags`,
`tagScores`, `usedImage`, `needsReview` and `source` (`label`, `excluded` or `model`).

**Which labels taught it** works here too, but its push numbers are close to the score rather than exact, because this
tagger is trained with a faster method than Style's.

## Rules labels can't change

Some decisions are rules in `desi_rules.py`, because the desi meaning is fixed. The web app warns you when
a label conflicts with one.

| Rule | To change it |
|---|---|
| `bridal_wear` only when the bride rule fires: "bridal"/"bride" in the text at Rs 40k+, or Rs 250k+ for a lehenga-type outfit or a listing with no description | `BRIDE_PRICE`, `BRIDE_MIN_PRICE` and the bridal words in `desi_rules.py` |
| A bride's outfit is also tagged `wedding` | `choose_tags()` in `desi_rules.py` |
| Menswear is never `bridal_wear` (groom wear is `wedding`) | `choose_tags()` |
| `casuals` and `wedding` never appear together | `choose_tags()` |
| `modest` needs an abaya / hijab-type garment or the Modest Wear category | `choose_tags()` |

## Add a new tag

1. Add it to `TAGS` in `desi_rules.py` and describe it in `TAGS.md`.
2. Label 20 to 30 products with it. Also check the existing labels: a labelled product without the new tag
   teaches "this is not that tag".
3. Retrain. A rule score in `rule_scores()` is optional and helps when labels are few.

## Photo vectors

Products with a FashionSigLIP photo vector are tagged from photo and text; the rest from text alone. On the
311-product benchmark that is 0.84 against 0.80 micro-F1. Image gains are biggest on party, smart-casual and
office wear, and on designer pieces with no description. The text-only model also covers products whose
photo is offline.

`python run.py embed` adds vectors for products that don't have one yet:

- **Order.** Products with no description and a short name ("Glam 05") go first, because photos help
  them most.
- **Speed.** It runs at about 1 image a second on this laptop's CPU (the model is the limit, not the
  download), so the full catalog takes about 18 hours. Leave it running overnight, or run it on a GPU machine.
- **Resumable.** It saves every 1,000 products, so stopping or a crash loses at most about 20 minutes of
  work. Running the same command again continues from there.
- **Failed downloads.** Photos that fail to download are listed in `data/image_vectors/failed.json` and
  skipped next time. `python run.py embed --retry-failed` tries them again.

New vectors are used the next time you run `python run.py tag` or restart `python run.py serve`; the
**What changed** tab then marks the products whose tags moved as **photo added**.

The full run (October 2026) gave 62,744 of 64,105 products a vector; 1,361 photos failed to download. It
changed the tags of 23,344 products, mostly casuals giving way to smart casual, office wear, festive and
wedding. Products flagged `needsReview` fell from 7.8% to 1.6%, and the benchmark stayed at 0.84.

## Check a code or rule change

```powershell
python test_tagger.py             # rule and label checks
python run.py eval --benchmark    # the fixed held-out test; should stay near 0.85 micro-F1 and 0.83 macro-F1
python run.py eval                # agreement with your current labels (cross-validated)
```

## Run it online

The app also runs on Google Cloud, in the `lookflock-qa2` project, so people without this laptop can use it:
**https://desitagger-193093379794.asia-east1.run.app**. It asks for a password (any user name works). To read the password:

```powershell
gcloud secrets versions access latest --secret desitagger-password --project lookflock-qa2
```

- **Its labels are a separate copy.** It started from this folder's `labels/` and `data/` on 12 Nov 2026. From then on, edits
  made online stay online, in the bucket `gs://lookflock-qa2-desitagger`, and edits made here stay here. To bring the online
  labels to this laptop (this replaces the laptop's): stop `run.py serve`, back up `labels/`, then run
  `gcloud storage cp "gs://lookflock-qa2-desitagger/labels/*" labels/`.
- **It sleeps when nobody uses it.** About 15 minutes after the last request it shuts down. The next visit waits about 45
  seconds while it starts and trains Style, and the first Category click after that about 50 seconds more.
- **Cost:** about $0.60 an hour while it is awake (8 CPUs, 8 GB), nothing while it sleeps. Training is held to 4 threads
  (`LOKY_MAX_CPU_COUNT`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` on the service), so half the CPUs stay free to answer
  the page: with all of them busy, Cloud Run stops sending it requests and the page shows "Rate exceeded".
- **Not a web.app address,** because Firebase Hosting cuts every request off after 60 seconds, and a Category retrain takes
  longer.

To put a code change online, run this in this folder. It keeps all the settings above:

```powershell
gcloud run deploy desitagger --source . --project lookflock-qa2 --region asia-east1
```

After a code deploy, the first visit also rebuilds the text features (about a minute more), and the first Category click
retrains Category in full (about 8½ minutes). The saved features and tagger in the bucket are checked against the code
files' dates, and every build gives those files new dates. Open Category once yourself right after deploying, so nobody
else waits for it.

## Files

| File | What it does |
|---|---|
| `config.py` | Paths and the few settings (model, image weight, port) |
| `catalog.py` | Cleans the raw export: prices, text, brand stats |
| `desi_rules.py` | All desi knowledge: word lists, signals, rule scores, hard rules |
| `images.py` | FashionSigLIP photo vectors (`run.py embed`) |
| `model.py` | Features, training, prediction, scoring |
| `labels.py` | Reads, checks and saves `labels/labels.jsonl` |
| `app.py`, `web/index.html` | The web app |
| `run.py` | The commands above |
| `Dockerfile`, `.gcloudignore` | The online copy (see [Run it online](#run-it-online)): what goes into it, and how it finds `labels/` and `data/` in the bucket |
| `remove_no_photo.py` | Removes products with no working photo from the catalog export; backs them up to `removed_no_photo_<date_time>/` |
| `labels/labels.jsonl` | Your labels (the web app edits this) |
| `labels/labels.original.jsonl` | The starting labels, used by "Restore" and the benchmark. The web app never edits it. Trimmed to 886 on 3 Nov 2026, when 85 of its products left the catalog |
| `labels/exclusions.json` | Your exclusions (the web app edits this): products that get no tags and never train. Created on the first exclusion |
| `labels/category_labels.jsonl`, `labels/category_exclusions.json` | The Category tagger's edits and exclusions. Each is created on the first one |
| `removed_products_2026-11-03/` | The 6,583 products dropped from the catalog on 3 Nov 2026 (`removed_products.csv`), with the labels, photo vectors and failed-download ids that were removed with them, in case they are needed again |
| `data/` | Generated: cleaned catalog, photo vectors, feature cache, `predictions.jsonl` (the output), and the `category_*` files for the Category tagger |

The output `data/predictions.jsonl` has one product per line: `id, name, supplier, ..., tags, tagScores,
formality, formalityLabel, evidence, ruleNotes, usedImage, needsReview, source`. `source` is `label` for
hand-labelled products, which output their label and keep the model's own answer in `modelTags`, `excluded`
for excluded products, which output no tags (also with `modelTags`), and `model` for all other products. The `tags` field uses the same keys as the app's Algolia facets.
