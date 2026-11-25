"""Paths and the few settings worth changing. Everything else is derived from these."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent

# Input: the raw catalog export. Only remove_no_photo.py writes it (dropping products with no working photo).
CATALOG = ROOT / "liveProducts.cleaned.json"

# Hand labels. labels.jsonl is the working copy the web app edits;
# labels.original.jsonl is the untouched starting set that "Restore original labels" copies back.
LABELS = ROOT / "labels" / "labels.jsonl"
LABELS_ORIGINAL = ROOT / "labels" / "labels.original.jsonl"
# Products left out of tagging and training: a list of rules that match on product fields (see labels.py).
EXCLUSIONS = ROOT / "labels" / "exclusions.json"

# Generated files (safe to delete; `run.py prep` and `run.py tag` rebuild them).
DATA = ROOT / "data"
PRODUCTS = DATA / "products.parquet"            # cleaned catalog
VECTORS = DATA / "image_vectors"                # FashionSigLIP vectors, part_*.npz (+ failed.json)
FEATURES_CACHE = DATA / "features.joblib"       # text + desi-rule features for every product
PREDICTIONS = DATA / "predictions.jsonl"        # the output: tags for every product
PREDICTIONS_PREVIOUS = DATA / "predictions.previous.jsonl"  # output before the last retrain


@dataclass(frozen=True)
class Files:
    """Where one tagger keeps its labels and generated files."""
    labels: Path
    exclusions: Path
    features: Path
    predictions: Path
    previous: Path
    causes: Path        # which labels changed in the last retrain
    tagger: Path | None = None  # the fitted tagger, for a tagger too slow to train at every start


STYLE = Files(LABELS, EXCLUSIONS, FEATURES_CACHE, PREDICTIONS, PREDICTIONS_PREVIOUS, DATA / "last_changes.json")
# The category tagger starts from each product's subSubCategory, so its label file holds only your edits.
CATEGORY = Files(ROOT / "labels" / "category_labels.jsonl", ROOT / "labels" / "category_exclusions.json",
                 DATA / "category_features.joblib", DATA / "category_predictions.jsonl",
                 DATA / "category_predictions.previous.jsonl", DATA / "category_last_changes.json",
                 DATA / "category_tagger.joblib")
# The bridal tagger (mehendi, barat...) uses the style tagger's features. Like the category tagger, its label file
# holds only your edits: its starting labels come from desi_rules.bridal_events.
BRIDAL = Files(ROOT / "labels" / "bridal_labels.jsonl", ROOT / "labels" / "bridal_exclusions.json", FEATURES_CACHE,
               DATA / "bridal_predictions.jsonl", DATA / "bridal_predictions.previous.jsonl",
               DATA / "bridal_last_changes.json", DATA / "bridal_tagger.joblib")

# Image model: Marqo-FashionSigLIP, the best open fashion model in our trial.
IMAGE_MODEL = "hf-hub:Marqo/marqo-fashionSigLIP"

# Model settings, picked by cross-validation in the Styles/fashion_model_trial experiment.
C = 0.3              # logistic regression strength (higher = fits the labels more tightly)
IMAGE_WEIGHT = 0.15  # how much the image vector counts next to the text
FOLDS = 5            # cross-validation folds used for thresholds and the agreement score
MAX_TAGS = 3         # tags per product, strongest first

# Category tagger: every product is a label, so it trains with a faster solver. Picked on held-out products.
CATEGORY_ALPHA = 1e-5   # SGD regularisation (1e-4 and 1e-6 scored lower)
CATEGORY_EPOCHS = 8     # passes over the products
CATEGORY_FOLDS = 3      # fewer than FOLDS: each fold costs about a minute

PORT = 8765          # web app: http://127.0.0.1:8765
