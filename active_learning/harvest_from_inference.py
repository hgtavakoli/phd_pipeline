"""
Inference-time Active Learning harvesting.

Runs a trained OD model over a folder of REAL, unlabeled images, scores each
image's uncertainty (the same difficulty-calibrated entropy used by the AL
selection strategies), and saves the most-uncertain images together with the
model's predictions as YOLO pre-labels into a review folder:

    active_learning/harvest/<name>/
        images/   the selected real images
        labels/   model predictions as YOLO .txt (pre-labels to verify/fix)
        harvest_manifest.json   per-image uncertainty + box count

The user then reviews/accepts these in the dashboard (or refines the boxes in
any YOLO labeller — the pre-labels are standard format) to turn them into a
labeled dataset for further training / AL. This closes the human-in-the-loop:
the model itself tells you which real frames are worth labelling.

Usage:
    python harvest_from_inference.py --model <best.pt> --source <folder>
        --out <name> --top 50 --conf 0.10 --num_classes 9
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
HARVEST_DIR = ROOT / "harvest"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def image_uncertainty(result, num_classes):
    """Difficulty-calibrated entropy of one image's detections (matches the
    AL selection uncertainty). Higher = the model is less sure = more useful
    to label."""
    if result.boxes is None or len(result.boxes) == 0:
        return 0.0, 0
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    class_unc = {i: 1.0 for i in range(num_classes)}
    entropy = 0.0
    for s, c in zip(scores, classes):
        if 0 <= c < num_classes:
            u = -s * math.log2(s + 1e-9)
            class_unc[c] += u * 0.1
            entropy += u * class_unc[c]
    return float(entropy), int(len(scores))


def main():
    ap = argparse.ArgumentParser(description="Harvest AL candidates from inference on real images")
    ap.add_argument("--model", required=True, help="trained OD weights (best.pt)")
    ap.add_argument("--source", required=True, help="folder of real unlabeled images")
    ap.add_argument("--out", required=True, help="harvest name (folder under active_learning/harvest)")
    ap.add_argument("--top", type=int, default=50, help="number of most-uncertain images to keep")
    ap.add_argument("--conf", type=float, default=0.10, help="detection confidence threshold")
    ap.add_argument("--num_classes", type=int, default=9)
    args = ap.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        print(f"ERROR: source folder not found: {source}", file=sys.stderr)
        sys.exit(1)
    images = [p for p in sorted(source.iterdir()) if p.suffix.lower() in IMG_EXTS]
    if not images:
        print(f"ERROR: no images in {source}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    print(f"Scoring {len(images)} real images (conf>={args.conf})...")

    scored = []
    for i, img in enumerate(images, 1):
        res = model(str(img), verbose=False, conf=args.conf)[0]
        unc, nboxes = image_uncertainty(res, args.num_classes)
        scored.append((unc, nboxes, img, res))
        if i % 25 == 0:
            print(f"  scored {i}/{len(images)}")

    # most uncertain first
    scored.sort(key=lambda x: x[0], reverse=True)
    keep = scored[:args.top]

    out_dir = HARVEST_DIR / args.out
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    manifest = []
    for unc, nboxes, img, res in keep:
        shutil.copy2(img, out_dir / "images" / img.name)
        # write predictions as YOLO pre-labels
        lines = []
        if res.boxes is not None:
            for cls, xywhn in zip(res.boxes.cls.cpu().numpy().astype(int),
                                  res.boxes.xywhn.cpu().numpy()):
                lines.append(f"{int(cls)} " + " ".join(f"{v:.6f}" for v in xywhn))
        (out_dir / "labels" / f"{img.stem}.txt").write_text("\n".join(lines))
        manifest.append({"image": img.name, "uncertainty": round(unc, 4), "boxes": nboxes})

    (out_dir / "harvest_manifest.json").write_text(json.dumps({
        "model": args.model, "source": str(source),
        "scored": len(images), "kept": len(keep),
        "num_classes": args.num_classes, "items": manifest,
    }, indent=2))

    print(f"\nHARVEST COMPLETE: kept {len(keep)} most-uncertain of {len(images)} images")
    print(f"  -> {out_dir}")
    print(f"  review/accept them in the dashboard, then train on the accepted set.")


if __name__ == "__main__":
    main()
