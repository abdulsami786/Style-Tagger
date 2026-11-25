"""Desi Tagger commands (run from this folder with the project's virtualenv):

  python run.py prep                1. clean the catalog export -> data/products.parquet
  python run.py embed [--limit N]   2. add FashionSigLIP image vectors (slow on CPU, resumable, optional)
  python run.py eval                3. how well the tagger agrees with your labels (cross-validated)
  python run.py eval --benchmark       the fixed check against the original held-out labels
  python run.py tag                 4. tag every product -> data/predictions.jsonl
  python run.py tag --mode category    the same with subSubCategory -> data/category_predictions.jsonl (minutes)
  python run.py tag --mode bridal      wedding functions (mehendi, barat...) of a bride's outfits -> data/bridal_predictions.jsonl
  python run.py serve               5. web app at http://127.0.0.1:8765
"""

import argparse
import time
from pathlib import Path

import config


def print_scores(m: dict) -> None:
    print(f"micro-F1 {m['micro_f1']:.3f} | macro-F1 {m['macro_f1']:.3f} | top tag right {m['top_tag_right']:.0%} | "
          f"top tag acceptable {m['top_tag_acceptable']:.0%} | products scored {m['n']}")
    print(f"\n{'tag':<14}{'n':>5}{'precision':>11}{'recall':>8}{'F1':>7}")
    for tag, t in m["per_tag"].items():
        print(f"{tag:<14}{t['n']:>5}{t['precision']:>11.2f}{t['recall']:>8.2f}{t['f1']:>7.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prep", help="clean the catalog export")
    p.add_argument("--catalog", type=Path, default=config.CATALOG)
    p = sub.add_parser("embed", help="add FashionSigLIP vectors for products that have none")
    p.add_argument("--limit", type=int, default=None, help="embed at most N products this run")
    p.add_argument("--retry-failed", action="store_true", help="retry photos that failed to download before")
    p = sub.add_parser("eval", help="score the tagger")
    p.add_argument("--benchmark", action="store_true", help="original train/test split instead of your labels")
    p = sub.add_parser("tag", help="tag every product")
    p.add_argument("--mode", choices=("style", "category", "bridal"), default="style")
    p = sub.add_parser("serve", help="start the web app")
    p.add_argument("--port", type=int, default=config.PORT)
    args = parser.parse_args()

    if args.command == "prep":
        import catalog
        catalog.prep(args.catalog)
        return
    if args.command == "serve":
        import app
        app.serve(args.port)
        return
    if args.command == "tag":
        import app
        t0 = time.time()
        # the web app's startup: tags every product and keeps the old output for "What changed"
        a = app.CategoryApp() if args.mode == "category" else app.App()
        if args.mode == "bridal":   # its scope is read from the style output, so style comes first
            a = app.BridalApp(a)
        print(f"tagged {len(a.current):,} products ({len(a.feats.vector_row):,} with an image vector) -> "
              f"{a.files.predictions} ({time.time() - t0:.0f} s)")
        return

    import catalog
    products = catalog.load()
    if args.command == "embed":
        import images
        images.embed_missing(products, limit=args.limit, retry_failed=args.retry_failed)
        return

    import labels
    import model
    feats = model.Features(products)
    print(f"{len(products):,} products, {len(feats.vector_row):,} with an image vector")
    t0 = time.time()
    if args.benchmark:
        print("benchmark: trained on the original training labels, scored on the original held-out labels")
        print_scores(model.benchmark(feats))
    else:
        excluded = labels.excluded(products, labels.load_exclusions())
        tagger = model.Tagger(feats, labels.trainable(labels.load(), excluded))
        print(f"agreement with your {len(tagger.ids)} labels (each product scored by a model that never saw its label)")
        print_scores(tagger.agreement)
    print(f"\n({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
