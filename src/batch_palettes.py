# src/batch_palettes.py

from __future__ import annotations

from pathlib import Path
import csv
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color
from sklearn.cluster import MiniBatchKMeans


# -----------------------------
# Config
# -----------------------------
RAW_DIR = Path("data/raw")
MANIFEST_PATH = Path("data/manifest.csv")

OUT_ROOT = Path("data/processed/seed_0/palettes_medclip")
OUT_ALL = OUT_ROOT / "palettes_all_medclip_seed_0.csv"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

K = 10                # number of palette colors
N_SAMPLE = 100_000     # pixel sample size per image
MIN_AFTER_CLIP = 10_000

# LAB Lightness clipping (both shadows and near-white backgrounds)
TAU_LOW = 10
TAU_HIGH = 95

RNG_SEED = 0

# MiniBatchKMeans settings
MB_BATCH_SIZE = 2048
MB_MAX_ITER = 200
MB_N_INIT = 5
MB_RANDOM_STATE = 0


# -----------------------------
# Helpers
# -----------------------------
def discover_images_from_folders() -> list[dict]:
    """
    Returns list of dict rows:
      {relpath, year, shoot_id, subject_id, notes}
    relpath is relative to data/ (e.g., raw/2021/DSC1234.jpg)
    """
    rows: list[dict] = []

    if not RAW_DIR.exists():
        return rows

    for year_dir in sorted(RAW_DIR.glob("*")):
        if not year_dir.is_dir():
            continue
        if not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)

        for p in sorted(year_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                relpath = p.relative_to(Path("data")).as_posix()
                rows.append(
                    {
                        "relpath": relpath,
                        "year": year,
                        "shoot_id": "",
                        "subject_id": "",
                        "notes": "",
                    }
                )
    return rows


def load_manifest_if_present() -> list[dict]:
    """
    If manifest exists, uses it (include==1).
    Otherwise falls back to folder discovery.
    """
    if MANIFEST_PATH.exists():
        df = pd.read_csv(MANIFEST_PATH)

        # Required minimum columns
        if "relpath" not in df.columns or "year" not in df.columns:
            raise ValueError("manifest.csv must include at least columns: relpath, year")

        if "include" in df.columns:
            df = df[df["include"] == 1].copy()

        # Add optional columns if missing
        for col in ["shoot_id", "subject_id", "notes"]:
            if col not in df.columns:
                df[col] = ""

        # Normalize types
        df["year"] = df["year"].astype(int)
        df["relpath"] = df["relpath"].astype(str)

        return df[["relpath", "year", "shoot_id", "subject_id", "notes"]].to_dict("records")

    # fallback
    return discover_images_from_folders()


def sample_lab_pixels(img_path: Path, n_sample: int, rng: np.random.Generator) -> np.ndarray:
    """
    Load image -> RGB -> LAB -> sample pixels -> clip by L* -> return (n',3) LAB.
    """
    img = Image.open(img_path).convert("RGB")
    rgb = np.asarray(img, dtype=np.uint8)

    # Normalize to [0,1] for skimage
    rgb_norm = rgb.astype(np.float32) / 255.0
    lab = color.rgb2lab(rgb_norm).astype(np.float32)  # (H,W,3)

    H, W, _ = lab.shape
    N = H * W

    n = min(n_sample, N)
    idx = rng.choice(N, size=n, replace=False)

    lab_flat = lab.reshape(-1, 3)
    lab_sample = lab_flat[idx]

    # Clip shadows + highlights (studio whites) based on L*
    L = lab_sample[:, 0]
    mask = (L >= TAU_LOW) & (L <= TAU_HIGH)
    lab_clip = lab_sample[mask]

    # Fall back if too aggressive
    if lab_clip.shape[0] >= MIN_AFTER_CLIP:
        return lab_clip

    return lab_sample


def extract_palette(lab_sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Run MiniBatchKMeans on (n,3) LAB sample.
    Returns:
      weights_sorted: (K,)
      centroids_sorted: (K,3)
    """
    kmeans = MiniBatchKMeans(
        n_clusters=K,
        batch_size=MB_BATCH_SIZE,
        max_iter=MB_MAX_ITER,
        n_init=MB_N_INIT,
        random_state=MB_RANDOM_STATE,
    )

    labels = kmeans.fit_predict(lab_sample)
    centroids = kmeans.cluster_centers_  # (K,3)

    counts = np.bincount(labels, minlength=K).astype(np.float32)
    weights = counts / counts.sum()

    order = np.argsort(-weights)
    weights_sorted = weights[order]
    centroids_sorted = centroids[order]

    return weights_sorted, centroids_sorted


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    rows = load_manifest_if_present()
    if not rows:
        # no prints per your request; just exit silently
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Prepare global output
    fieldnames = [
        "year",
        "relpath",
        "filename",
        "height",
        "width",
        "n_sample",
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

    # We'll buffer per-year rows so we can write year subfiles
    per_year: dict[int, list[dict]] = {}

    rng = np.random.default_rng(RNG_SEED)

    for row in rows:
        year = int(row["year"])
        relpath = str(row["relpath"])
        img_path = Path("data") / relpath

        if not img_path.exists():
            continue
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        # Load to get dimensions (cheap enough)
        img = Image.open(img_path)
        width, height = img.size
        img.close()

        # Sample LAB pixels + extract palette
        lab_sample = sample_lab_pixels(img_path, N_SAMPLE, rng)
        weights_sorted, centroids_sorted = extract_palette(lab_sample)

        filename = img_path.name

        # Emit K rows per image
        for rank in range(K):
            Ls, a, b = centroids_sorted[rank]
            out_row = {
                "year": year,
                "relpath": relpath,
                "filename": filename,
                "height": int(height),
                "width": int(width),
                "n_sample": int(lab_sample.shape[0]),
                "k": int(K),
                "cluster_rank": int(rank + 1),
                "weight": float(weights_sorted[rank]),
                "L_star": float(Ls),
                "a_star": float(a),
                "b_star": float(b),
                "shoot_id": str(row.get("shoot_id", "")),
                "subject_id": str(row.get("subject_id", "")),
                "notes": str(row.get("notes", "")),
            }

            per_year.setdefault(year, []).append(out_row)

    # Write global file
    all_rows = []
    for y in sorted(per_year.keys()):
        all_rows.extend(per_year[y])

    OUT_ALL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_ALL.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Write per-year files into subfolders
    for y, y_rows in per_year.items():
        year_dir = OUT_ROOT / str(y)
        year_dir.mkdir(parents=True, exist_ok=True)
        out_path = year_dir / f"palettes_{y}.csv"

        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(y_rows)


if __name__ == "__main__":
    main()