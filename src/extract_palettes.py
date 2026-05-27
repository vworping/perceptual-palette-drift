from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color
from sklearn.cluster import MiniBatchKMeans

try:
    import sklearn.cluster._kmeans as _sklearn_kmeans

    _sklearn_kmeans.threadpool_info = lambda: []
    _sklearn_kmeans.threadpool_limits = lambda *args, **kwargs: nullcontext()
except Exception:
    pass


RAW_DIR = Path("data/raw")
MANIFEST_PATH = Path("data/manifest.csv")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def discover_images_from_folders(raw_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not raw_dir.exists():
        return rows

    for year_dir in sorted(raw_dir.glob("*")):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue

        year = int(year_dir.name)
        for img_path in sorted(year_dir.rglob("*")):
            if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
                rows.append(
                    {
                        "relpath": img_path.relative_to(Path("data")).as_posix(),
                        "year": year,
                        "shoot_id": "",
                        "subject_id": "",
                        "notes": "",
                    }
                )

    return rows


def load_manifest_or_discover(manifest_path: Path, raw_dir: Path) -> list[dict]:
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        if "relpath" not in df.columns or "year" not in df.columns:
            raise ValueError("manifest.csv must include at least columns: relpath, year")

        if "include" in df.columns:
            df = df[df["include"] == 1].copy()

        for col in ["shoot_id", "subject_id", "notes"]:
            if col not in df.columns:
                df[col] = ""

        df["year"] = df["year"].astype(int)
        df["relpath"] = df["relpath"].astype(str)
        return df[["relpath", "year", "shoot_id", "subject_id", "notes"]].to_dict("records")

    return discover_images_from_folders(raw_dir)


def sample_lab_pixels(
    img_path: Path,
    n_sample: int,
    rng: np.random.Generator,
    tau_low: float,
    tau_high: float,
    min_after_clip: int,
    max_image_edge: int,
) -> tuple[np.ndarray, dict]:
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        orig_width, orig_height = img.size

        if max_image_edge > 0:
            img.thumbnail((max_image_edge, max_image_edge), Image.Resampling.LANCZOS)

        width, height = img.size
        rgb = np.asarray(img, dtype=np.uint8)

    rgb_norm = rgb.astype(np.float32) / 255.0
    lab = color.rgb2lab(rgb_norm).astype(np.float32)
    lab_flat = lab.reshape(-1, 3)

    n_pixels = lab_flat.shape[0]
    n = min(n_sample, n_pixels)
    idx = rng.choice(n_pixels, size=n, replace=False)
    lab_sample = lab_flat[idx]

    lightness = lab_sample[:, 0]
    keep = (lightness >= tau_low) & (lightness <= tau_high)
    lab_clip = lab_sample[keep]
    used_clip = lab_clip.shape[0] >= min_after_clip
    if used_clip:
        lab_sample = lab_clip

    return lab_sample.astype(np.float32), {
        "orig_width": int(orig_width),
        "orig_height": int(orig_height),
        "sample_width": int(width),
        "sample_height": int(height),
        "n_sample_before_clip": int(n),
        "n_sample_used": int(lab_sample.shape[0]),
        "used_lightness_clip": int(used_clip),
    }


def extract_palette(
    lab_sample: np.ndarray,
    k: int,
    batch_size: int,
    max_iter: int,
    n_init: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
    )
    labels = kmeans.fit_predict(lab_sample)
    centroids = kmeans.cluster_centers_.astype(np.float32)

    counts = np.bincount(labels, minlength=k).astype(np.float32)
    weights = counts / counts.sum()

    order = np.argsort(-weights)
    return weights[order], centroids[order]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract fixed-seed LAB palettes from raw images.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--out-root", type=Path, default=Path("data/processed/seed_0/palettes_k5"))
    parser.add_argument("--out-name", default="palettes_all_k5_seed_0.csv")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n-sample", type=int, default=100_000)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--max-image-edge", type=int, default=0)
    parser.add_argument("--tau-low", type=float, default=10.0)
    parser.add_argument("--tau-high", type=float, default=95.0)
    parser.add_argument("--min-after-clip", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--n-init", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_manifest_or_discover(args.manifest, args.raw_dir)
    rows = sorted(rows, key=lambda row: (int(row["year"]), str(row["relpath"])))
    if not rows:
        return

    args.out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.rng_seed)
    fieldnames = [
        "year",
        "relpath",
        "filename",
        "height",
        "width",
        "sample_height",
        "sample_width",
        "n_sample",
        "n_sample_before_clip",
        "used_lightness_clip",
        "k",
        "cluster_rank",
        "weight",
        "L_star",
        "a_star",
        "b_star",
        "shoot_id",
        "subject_id",
        "notes",
    ]

    per_year: dict[int, list[dict]] = {}
    for index, row in enumerate(rows, start=1):
        year = int(row["year"])
        relpath = str(row["relpath"])
        img_path = Path("data") / relpath
        if not img_path.exists() or img_path.suffix.lower() not in IMG_EXTS:
            continue

        lab_sample, sample_meta = sample_lab_pixels(
            img_path=img_path,
            n_sample=args.n_sample,
            rng=rng,
            tau_low=args.tau_low,
            tau_high=args.tau_high,
            min_after_clip=args.min_after_clip,
            max_image_edge=args.max_image_edge,
        )
        weights, centroids = extract_palette(
            lab_sample=lab_sample,
            k=args.k,
            batch_size=args.batch_size,
            max_iter=args.max_iter,
            n_init=args.n_init,
            random_state=args.random_state,
        )

        for rank, (weight, lab) in enumerate(zip(weights, centroids), start=1):
            per_year.setdefault(year, []).append(
                {
                    "year": year,
                    "relpath": relpath,
                    "filename": img_path.name,
                    "height": sample_meta["orig_height"],
                    "width": sample_meta["orig_width"],
                    "sample_height": sample_meta["sample_height"],
                    "sample_width": sample_meta["sample_width"],
                    "n_sample": sample_meta["n_sample_used"],
                    "n_sample_before_clip": sample_meta["n_sample_before_clip"],
                    "used_lightness_clip": sample_meta["used_lightness_clip"],
                    "k": args.k,
                    "cluster_rank": rank,
                    "weight": float(weight),
                    "L_star": float(lab[0]),
                    "a_star": float(lab[1]),
                    "b_star": float(lab[2]),
                    "shoot_id": str(row.get("shoot_id", "")),
                    "subject_id": str(row.get("subject_id", "")),
                    "notes": str(row.get("notes", "")),
                }
            )

        print(
            f"[{index}/{len(rows)}] {relpath} n={sample_meta['n_sample_used']} k={args.k}",
            flush=True,
        )

    all_rows = []
    for year in sorted(per_year):
        all_rows.extend(per_year[year])
        year_dir = args.out_root / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        with (year_dir / f"palettes_{year}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_year[year])

    with (args.out_root / args.out_name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == "__main__":
    main()
