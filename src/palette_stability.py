from __future__ import annotations

import argparse
import csv
import hashlib
import re
from contextlib import nullcontext
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
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
OUT_ROOT = Path("data/processed/palette_stability")
CACHE_ROOT = Path("data/processed/cache/lab_samples")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def stable_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


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


def filter_rows(rows: list[dict], years: set[int] | None, image_limit: int | None) -> list[dict]:
    if years is not None:
        rows = [row for row in rows if int(row["year"]) in years]
    rows = sorted(rows, key=lambda row: (int(row["year"]), str(row["relpath"])))
    if image_limit is not None:
        rows = rows[:image_limit]
    return rows


def cache_path_for_image(
    img_path: Path,
    relpath: str,
    n_sample: int,
    sample_seed: int,
    max_image_edge: int,
    tau_low: float,
    tau_high: float,
) -> Path:
    stat = img_path.stat()
    key = "|".join(
        [
            relpath,
            str(stat.st_size),
            str(int(stat.st_mtime)),
            str(n_sample),
            str(sample_seed),
            str(max_image_edge),
            str(tau_low),
            str(tau_high),
        ]
    )
    return CACHE_ROOT / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()[:24]}.npz"


def load_or_create_lab_sample(
    img_path: Path,
    relpath: str,
    n_sample: int,
    sample_seed: int,
    max_image_edge: int,
    tau_low: float,
    tau_high: float,
    min_after_clip: int,
    force_resample: bool,
) -> tuple[np.ndarray, dict]:
    cache_path = cache_path_for_image(
        img_path=img_path,
        relpath=relpath,
        n_sample=n_sample,
        sample_seed=sample_seed,
        max_image_edge=max_image_edge,
        tau_low=tau_low,
        tau_high=tau_high,
    )

    if cache_path.exists() and not force_resample:
        cached = np.load(cache_path)
        lab_sample = cached["lab_sample"].astype(np.float32)
        meta = {key: cached[key].item() for key in cached.files if key != "lab_sample"}
        meta["cache_hit"] = True
        return lab_sample, meta

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
    rng_seed = (sample_seed + stable_int(relpath)) % (2**32 - 1)
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(n_pixels, size=n, replace=False)
    lab_sample = lab_flat[idx]

    lightness = lab_sample[:, 0]
    keep = (lightness >= tau_low) & (lightness <= tau_high)
    lab_clip = lab_sample[keep]
    used_clip = lab_clip.shape[0] >= min_after_clip
    if used_clip:
        lab_sample = lab_clip

    meta = {
        "orig_width": int(orig_width),
        "orig_height": int(orig_height),
        "sample_width": int(width),
        "sample_height": int(height),
        "n_pixels_after_resize": int(n_pixels),
        "n_sample_requested": int(n_sample),
        "n_sample_before_clip": int(n),
        "n_sample_used": int(lab_sample.shape[0]),
        "used_lightness_clip": int(used_clip),
        "cache_hit": False,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, lab_sample=lab_sample.astype(np.float32), **meta)
    return lab_sample.astype(np.float32), meta


def extract_palette(
    lab_sample: np.ndarray,
    k: int,
    seed: int,
    batch_size: int,
    max_iter: int,
    n_init: int,
) -> tuple[np.ndarray, np.ndarray]:
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        max_iter=max_iter,
        n_init=n_init,
        random_state=seed,
    )
    labels = kmeans.fit_predict(lab_sample)
    centroids = kmeans.cluster_centers_.astype(np.float32)

    counts = np.bincount(labels, minlength=k).astype(np.float32)
    weights = counts / counts.sum()

    order = np.argsort(-weights)
    return weights[order], centroids[order]


def palette_cost_matrix(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    diff = lab_a[:, None, :] - lab_b[None, :, :]
    return np.linalg.norm(diff, axis=2)


def lab_to_rgb01(L: float, a: float, b: float) -> np.ndarray:
    lab = np.array([[[L, a, b]]], dtype=np.float32)
    rgb = color.lab2rgb(lab)
    return np.clip(rgb[0, 0, :], 0.0, 1.0)


def matched_palette_distance(
    weights_a: np.ndarray,
    lab_a: np.ndarray,
    weights_b: np.ndarray,
    lab_b: np.ndarray,
) -> dict:
    cost = palette_cost_matrix(lab_a, lab_b)
    row_idx, col_idx = linear_sum_assignment(cost)
    delta_e = cost[row_idx, col_idx]
    match_weights = (weights_a[row_idx] + weights_b[col_idx]) / 2.0
    weighted_mean_delta_e = float(np.average(delta_e, weights=match_weights))
    unweighted_mean_delta_e = float(delta_e.mean())
    rms_delta_e = float(np.sqrt(np.mean(delta_e**2)))
    weighted_rms_delta_e = float(np.sqrt(np.average(delta_e**2, weights=match_weights)))
    max_matched_delta_e = float(delta_e.max())
    weight_l1 = float(np.abs(weights_a[row_idx] - weights_b[col_idx]).sum())
    return {
        "weighted_mean_delta_e": weighted_mean_delta_e,
        "unweighted_mean_delta_e": unweighted_mean_delta_e,
        "rms_delta_e": rms_delta_e,
        "weighted_rms_delta_e": weighted_rms_delta_e,
        "max_matched_delta_e": max_matched_delta_e,
        "weight_l1": weight_l1,
    }


def reorder_to_reference(reference_lab: np.ndarray, labs: list[np.ndarray]) -> np.ndarray:
    reordered = []
    for lab in labs:
        cost = palette_cost_matrix(reference_lab, lab)
        _, col_idx = linear_sum_assignment(cost)
        reordered.append(lab[col_idx])
    return np.stack(reordered, axis=0)


def extract_four_digits(filename: str) -> str:
    matches = re.findall(r"\d{4}", Path(filename).stem)
    return matches[-1] if matches else Path(filename).stem[-4:]


def summarize_stability(
    palettes: list[dict],
    row: dict,
    sample_meta: dict,
) -> dict:
    pair_metrics = []
    for left, right in combinations(palettes, 2):
        pair_metrics.append(
            matched_palette_distance(
                left["weights"],
                left["labs"],
                right["weights"],
                right["labs"],
            )
        )

    labs_by_seed = [palette["labs"] for palette in palettes]
    reordered = reorder_to_reference(labs_by_seed[0], labs_by_seed)
    mean_lab = reordered.mean(axis=0)
    slot_delta_e = np.linalg.norm(reordered - mean_lab[None, :, :], axis=2)

    weights_by_seed = np.stack([palette["weights"] for palette in palettes], axis=0)
    weighted_pair_dists = [metric["weighted_mean_delta_e"] for metric in pair_metrics]
    unweighted_pair_dists = [metric["unweighted_mean_delta_e"] for metric in pair_metrics]
    rms_pair_dists = [metric["rms_delta_e"] for metric in pair_metrics]
    weighted_rms_pair_dists = [metric["weighted_rms_delta_e"] for metric in pair_metrics]
    max_matched_dists = [metric["max_matched_delta_e"] for metric in pair_metrics]
    pair_weight_l1 = [metric["weight_l1"] for metric in pair_metrics]

    return {
        "year": int(row["year"]),
        "relpath": row["relpath"],
        "filename": Path(row["relpath"]).name,
        "n_seeds": len(palettes),
        "k": int(len(palettes[0]["weights"])),
        "mean_pairwise_delta_e": float(np.mean(weighted_pair_dists)) if weighted_pair_dists else 0.0,
        "median_pairwise_delta_e": float(np.median(weighted_pair_dists)) if weighted_pair_dists else 0.0,
        "max_pairwise_delta_e": float(np.max(weighted_pair_dists)) if weighted_pair_dists else 0.0,
        "mean_weighted_delta_e": float(np.mean(weighted_pair_dists)) if weighted_pair_dists else 0.0,
        "median_weighted_delta_e": float(np.median(weighted_pair_dists)) if weighted_pair_dists else 0.0,
        "p95_weighted_delta_e": float(np.percentile(weighted_pair_dists, 95)) if weighted_pair_dists else 0.0,
        "mean_unweighted_delta_e": float(np.mean(unweighted_pair_dists)) if unweighted_pair_dists else 0.0,
        "median_unweighted_delta_e": float(np.median(unweighted_pair_dists)) if unweighted_pair_dists else 0.0,
        "p95_unweighted_delta_e": float(np.percentile(unweighted_pair_dists, 95)) if unweighted_pair_dists else 0.0,
        "mean_rms_delta_e": float(np.mean(rms_pair_dists)) if rms_pair_dists else 0.0,
        "mean_weighted_rms_delta_e": float(np.mean(weighted_rms_pair_dists)) if weighted_rms_pair_dists else 0.0,
        "mean_max_matched_delta_e": float(np.mean(max_matched_dists)) if max_matched_dists else 0.0,
        "max_matched_delta_e": float(np.max(max_matched_dists)) if max_matched_dists else 0.0,
        "mean_pairwise_weight_l1": float(np.mean(pair_weight_l1)) if pair_weight_l1 else 0.0,
        "median_pairwise_weight_l1": float(np.median(pair_weight_l1)) if pair_weight_l1 else 0.0,
        "p95_pairwise_weight_l1": float(np.percentile(pair_weight_l1, 95)) if pair_weight_l1 else 0.0,
        "mean_slot_delta_e": float(slot_delta_e.mean()),
        "max_slot_delta_e": float(slot_delta_e.max()),
        "mean_weight_sd": float(weights_by_seed.std(axis=0).mean()),
        "max_weight_sd": float(weights_by_seed.std(axis=0).max()),
        **sample_meta,
    }


def palette_rows_for_csv(row: dict, palettes: list[dict], sample_meta: dict) -> list[dict]:
    out_rows = []
    for palette in palettes:
        for rank, (weight, lab) in enumerate(zip(palette["weights"], palette["labs"]), start=1):
            out_rows.append(
                {
                    "year": int(row["year"]),
                    "relpath": row["relpath"],
                    "filename": Path(row["relpath"]).name,
                    "seed": int(palette["seed"]),
                    "cluster_rank": rank,
                    "weight": float(weight),
                    "L_star": float(lab[0]),
                    "a_star": float(lab[1]),
                    "b_star": float(lab[2]),
                    "n_sample_used": int(sample_meta["n_sample_used"]),
                    "sample_width": int(sample_meta["sample_width"]),
                    "sample_height": int(sample_meta["sample_height"]),
                }
            )
    return out_rows


def plot_stability_bars(summary_csv: Path, out_path: Path, top_n: int) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(summary_csv)
    if df.empty:
        return

    metric = "mean_pairwise_delta_e"
    df = df.sort_values(metric, ascending=False).head(top_n).copy()
    df["label"] = df["year"].astype(str) + "  " + df["filename"].map(extract_four_digits)

    fig_h = max(4.0, 0.28 * len(df) + 1.2)
    fig, ax = plt.subplots(figsize=(9.5, fig_h), dpi=180)
    y = np.arange(len(df))
    ax.barh(y, df[metric], color="#3f4652")
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean pairwise matched Delta E across seeds")
    ax.set_title("Most K-means-sensitive palettes")
    ax.grid(axis="x", color=(0, 0, 0, 0.12), linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_seed_gallery(summary_csv: Path, palettes_csv: Path, out_path: Path, top_n: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    summary = pd.read_csv(summary_csv)
    palettes = pd.read_csv(palettes_csv)
    if summary.empty or palettes.empty:
        return

    summary = summary.sort_values("mean_pairwise_delta_e", ascending=False).head(top_n)
    relpaths = summary["relpath"].tolist()
    palettes = palettes[palettes["relpath"].isin(relpaths)].copy()

    seeds = sorted(palettes["seed"].unique())
    n_seed_rows = len(relpaths) * len(seeds)
    row_gap = 0.45
    fig_h = max(4.5, 0.23 * n_seed_rows + row_gap * len(relpaths) + 1.1)

    fig, ax = plt.subplots(figsize=(12.5, fig_h), dpi=180)
    ax.set_xlim(-0.22, 1.0)
    ax.set_ylim(-0.8, n_seed_rows + row_gap * len(relpaths))
    ax.axis("off")
    ax.set_title("Repeated K-means palettes for most sensitive images", pad=12)

    y = 0.0
    for relpath in relpaths:
        image_summary = summary[summary["relpath"] == relpath].iloc[0]
        label = f"{int(image_summary['year'])} {extract_four_digits(image_summary['filename'])}"
        metric = float(image_summary["mean_pairwise_delta_e"])

        block_mid = y + (len(seeds) - 1) / 2
        ax.text(
            -0.215,
            block_mid,
            f"{label}\nDelta E {metric:.1f}",
            ha="left",
            va="center",
            fontsize=8,
            color=(0.15, 0.15, 0.15),
        )

        image_palettes = palettes[palettes["relpath"] == relpath]
        for seed in seeds:
            seed_rows = image_palettes[image_palettes["seed"] == seed].sort_values("cluster_rank")
            ax.text(-0.035, y, str(seed), ha="right", va="center", fontsize=6, color=(0.35, 0.35, 0.35))

            x = 0.0
            weights = seed_rows["weight"].to_numpy(dtype=float)
            if weights.sum() > 0:
                weights = weights / weights.sum()

            for weight, (_, palette_row) in zip(weights, seed_rows.iterrows()):
                rgb = lab_to_rgb01(
                    float(palette_row["L_star"]),
                    float(palette_row["a_star"]),
                    float(palette_row["b_star"]),
                )
                ax.add_patch(
                    Rectangle(
                        (x, y - 0.34),
                        float(weight),
                        0.68,
                        facecolor=rgb,
                        edgecolor="none",
                    )
                )
                x += float(weight)

            y += 1.0

        ax.plot([-0.22, 1.0], [y - 0.5, y - 0.5], color=(0, 0, 0, 0.08), linewidth=0.8)
        y += row_gap

    ax.invert_yaxis()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_stability_extremes(
    summary_csv: Path,
    palettes_csv: Path,
    out_path: Path,
    n_each: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    summary = pd.read_csv(summary_csv)
    palettes = pd.read_csv(palettes_csv)
    if summary.empty or palettes.empty:
        return

    metric = "mean_pairwise_delta_e"
    most_stable = summary.sort_values(metric, ascending=True).head(n_each).copy()
    least_stable = summary.sort_values(metric, ascending=False).head(n_each).copy()
    selected = pd.concat(
        [
            least_stable.assign(group="Highest seed sensitivity"),
            most_stable.assign(group="Lowest seed sensitivity"),
        ],
        ignore_index=True,
    )

    relpaths = selected["relpath"].tolist()
    palettes = palettes[palettes["relpath"].isin(relpaths)].copy()
    seeds = sorted(palettes["seed"].unique())

    row_gap = 0.45
    section_gap = 1.1
    n_seed_rows = len(selected) * len(seeds)
    fig_h = max(5.0, 0.23 * n_seed_rows + row_gap * len(selected) + section_gap + 1.3)

    fig, ax = plt.subplots(figsize=(12.5, fig_h), dpi=180)
    ax.set_xlim(-0.23, 1.0)
    ax.set_ylim(-1.0, n_seed_rows + row_gap * len(selected) + section_gap)
    ax.axis("off")
    ax.set_title("K-means seed stability: least stable vs most stable palettes", pad=12)

    y = 0.0
    previous_group = None
    for _, image_summary in selected.iterrows():
        group = image_summary["group"]
        if group != previous_group:
            if previous_group is not None:
                y += section_gap
            ax.text(
                -0.23,
                y - 0.35,
                group,
                ha="left",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=(0.12, 0.12, 0.12),
            )
            previous_group = group

        relpath = image_summary["relpath"]
        label = f"{int(image_summary['year'])} {extract_four_digits(image_summary['filename'])}"
        metric_value = float(image_summary[metric])
        block_mid = y + (len(seeds) - 1) / 2

        ax.text(
            -0.215,
            block_mid,
            f"{label}\nDelta E {metric_value:.1f}",
            ha="left",
            va="center",
            fontsize=8,
            color=(0.15, 0.15, 0.15),
        )

        image_palettes = palettes[palettes["relpath"] == relpath]
        for seed in seeds:
            seed_rows = image_palettes[image_palettes["seed"] == seed].sort_values("cluster_rank")
            ax.text(-0.035, y, str(seed), ha="right", va="center", fontsize=6, color=(0.35, 0.35, 0.35))

            x = 0.0
            weights = seed_rows["weight"].to_numpy(dtype=float)
            if weights.sum() > 0:
                weights = weights / weights.sum()

            for weight, (_, palette_row) in zip(weights, seed_rows.iterrows()):
                rgb = lab_to_rgb01(
                    float(palette_row["L_star"]),
                    float(palette_row["a_star"]),
                    float(palette_row["b_star"]),
                )
                ax.add_patch(
                    Rectangle(
                        (x, y - 0.34),
                        float(weight),
                        0.68,
                        facecolor=rgb,
                        edgecolor="none",
                    )
                )
                x += float(weight)

            y += 1.0

        ax.plot([-0.23, 1.0], [y - 0.5, y - 0.5], color=(0, 0, 0, 0.08), linewidth=0.8)
        y += row_gap

    ax.invert_yaxis()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def write_summary_stats(summary_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(summary_csv)
    if df.empty:
        return

    metrics = [
        "mean_weighted_delta_e",
        "mean_unweighted_delta_e",
        "mean_rms_delta_e",
        "mean_weighted_rms_delta_e",
        "mean_max_matched_delta_e",
        "mean_pairwise_weight_l1",
    ]
    rows = []
    for metric in metrics:
        values = df[metric].dropna()
        rows.append(
            {
                "metric": metric,
                "n_images": int(values.shape[0]),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std()),
                "p05": float(values.quantile(0.05)),
                "p25": float(values.quantile(0.25)),
                "p75": float(values.quantile(0.75)),
                "p95": float(values.quantile(0.95)),
                "max": float(values.max()),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated K-means palette extraction to estimate palette stability."
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--years", nargs="*", type=int)
    parser.add_argument("--image-limit", type=int)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-sample", type=int, default=30_000)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--max-image-edge", type=int, default=1800)
    parser.add_argument("--tau-low", type=float, default=10.0)
    parser.add_argument("--tau-high", type=float, default=95.0)
    parser.add_argument("--min-after-clip", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-iter", type=int, default=160)
    parser.add_argument("--n-init", type=int, default=1)
    parser.add_argument("--force-resample", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--gallery-top-n", type=int, default=12)
    parser.add_argument("--extremes-n", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global CACHE_ROOT
    CACHE_ROOT = args.cache_root

    rows = load_manifest_or_discover(args.manifest, args.raw_dir)
    rows = filter_rows(rows, set(args.years) if args.years else None, args.image_limit)
    if not rows:
        return

    args.out_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_root / f"stability_k{args.k}_seeds{args.n_seeds}.csv"
    palettes_path = args.out_root / f"palettes_by_seed_k{args.k}_seeds{args.n_seeds}.csv"

    summary_rows = []
    all_palette_rows = []
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    for row_index, row in enumerate(rows, start=1):
        img_path = Path("data") / row["relpath"]
        if not img_path.exists() or img_path.suffix.lower() not in IMG_EXTS:
            continue

        lab_sample, sample_meta = load_or_create_lab_sample(
            img_path=img_path,
            relpath=row["relpath"],
            n_sample=args.n_sample,
            sample_seed=args.sample_seed,
            max_image_edge=args.max_image_edge,
            tau_low=args.tau_low,
            tau_high=args.tau_high,
            min_after_clip=args.min_after_clip,
            force_resample=args.force_resample,
        )

        palettes = []
        for seed in seeds:
            weights, labs = extract_palette(
                lab_sample=lab_sample,
                k=args.k,
                seed=seed,
                batch_size=args.batch_size,
                max_iter=args.max_iter,
                n_init=args.n_init,
            )
            palettes.append({"seed": seed, "weights": weights, "labs": labs})

        summary_rows.append(summarize_stability(palettes, row, sample_meta))
        all_palette_rows.extend(palette_rows_for_csv(row, palettes, sample_meta))

        print(
            f"[{row_index}/{len(rows)}] {row['relpath']} "
            f"mean Delta E={summary_rows[-1]['mean_pairwise_delta_e']:.2f} "
            f"cache={'hit' if sample_meta['cache_hit'] else 'new'}",
            flush=True,
        )

    if summary_rows:
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        write_summary_stats(
            summary_csv=summary_path,
            out_path=args.out_root / f"summary_stats_k{args.k}_seeds{args.n_seeds}.csv",
        )

    if all_palette_rows:
        with palettes_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_palette_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_palette_rows)

    if summary_rows and not args.no_figures:
        plot_stability_bars(
            summary_csv=summary_path,
            out_path=args.out_root / f"stability_top_k{args.k}_seeds{args.n_seeds}.png",
            top_n=args.top_n,
        )
        plot_seed_gallery(
            summary_csv=summary_path,
            palettes_csv=palettes_path,
            out_path=args.out_root / f"seed_gallery_top_k{args.k}_seeds{args.n_seeds}.png",
            top_n=args.gallery_top_n,
        )
        plot_stability_extremes(
            summary_csv=summary_path,
            palettes_csv=palettes_path,
            out_path=args.out_root / f"seed_gallery_extremes_k{args.k}_seeds{args.n_seeds}.png",
            n_each=args.extremes_n,
        )


if __name__ == "__main__":
    main()
