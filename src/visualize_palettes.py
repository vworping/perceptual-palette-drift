# src/visualize_palettes.py

from __future__ import annotations

from pathlib import Path
import csv
from collections import defaultdict
from datetime import datetime
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ExifTags
from skimage import color


# -----------------------------
# Paths
# -----------------------------
PALETTES_ROOT = Path("data/processed/seed_0/palettes_medclip")
PALETTES_ALL = PALETTES_ROOT / "palettes_all_medclip_seed_0.csv"
RAW_DATA_ROOT = Path("data")  # relpath in CSV is relative to data/

OUT_DIR = Path("figures/seed_0/palettes_medclip")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Style knobs
# -----------------------------
DPI = 240
K_EXPECTED = 10

LEFT_LABEL_W = 0.20     # width reserved for left labels (axes fraction)
ROW_H_IN = 0.42         # row height per image (inches)
TOP_MARGIN_IN = 0.65
BOTTOM_MARGIN_IN = 0.30

TITLE_SIZE = 22
LABEL_SIZE = 9
PCT_SIZE = 9

# Percent text padding inside each swatch (axes coords)
PCT_PAD_AX = 0.006


# -----------------------------
# Color helpers
# -----------------------------
def lab_to_rgb01(L: float, a: float, b: float) -> np.ndarray:
    lab = np.array([[[L, a, b]]], dtype=np.float32)
    rgb = color.lab2rgb(lab)
    return np.clip(rgb[0, 0, :], 0.0, 1.0)


def text_color_for_rgb(rgb01: np.ndarray) -> str:
    r, g, b = rgb01
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.55 else "black"


# -----------------------------
# Loading palettes
# -----------------------------
def find_palette_csvs() -> list[Path]:
    if PALETTES_ALL.exists():
        return [PALETTES_ALL]
    return sorted(PALETTES_ROOT.glob("*/palettes_*.csv"))


def load_palettes(csv_paths: list[Path]):
    """
    Returns:
      by_year: dict[int, dict[relpath, list[rows]]]
    Each row contains cluster_rank, weight, L_star, a_star, b_star, filename, relpath.
    """
    by_year = defaultdict(lambda: defaultdict(list))

    for csv_path in csv_paths:
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                year = int(row["year"])
                relpath = row.get("relpath") or row.get("filename") or row["filename"]

                by_year[year][relpath].append(
                    {
                        "cluster_rank": int(row["cluster_rank"]),
                        "weight": float(row["weight"]),
                        "L_star": float(row["L_star"]),
                        "a_star": float(row["a_star"]),
                        "b_star": float(row["b_star"]),
                        "filename": row.get("filename", Path(relpath).name),
                        "relpath": relpath,
                    }
                )

    for year in by_year:
        for relpath in by_year[year]:
            by_year[year][relpath].sort(key=lambda d: d["cluster_rank"])

    return by_year


# -----------------------------
# EXIF date extraction (robust)
# -----------------------------
_TAG_NAME_BY_ID = {k: v for k, v in ExifTags.TAGS.items()}


def _parse_exif_datetime(s: str) -> datetime | None:
    s = s.strip()
    # Common formats:
    # "YYYY:MM:DD HH:MM:SS"
    # "YYYY-MM-DD HH:MM:SS"
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    # Sometimes date only
    for fmt in ("%Y:%m:%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return None


def get_capture_datetime(img_path: Path) -> datetime | None:
    """
    Try hard to get a capture datetime:
      1) EXIF tags: DateTimeOriginal, DateTimeDigitized, DateTime
      2) other EXIF variants if present
      3) fallback: filesystem mtime
    """
    dt = None

    try:
        with Image.open(img_path) as img:
            exif = img.getexif()
            if exif:
                # Standard EXIF tag IDs:
                # 36867 DateTimeOriginal
                # 36868 DateTimeDigitized
                # 306   DateTime
                candidate_ids = [36867, 36868, 306]

                # Also try name-based lookup in case PIL gives different mapping
                candidate_names = {"DateTimeOriginal", "DateTimeDigitized", "DateTime"}

                # First: direct by ID
                for tid in candidate_ids:
                    val = exif.get(tid)
                    if isinstance(val, str):
                        dt = _parse_exif_datetime(val)
                        if dt:
                            return dt

                # Second: scan all EXIF for any of those names
                for tid, val in exif.items():
                    name = _TAG_NAME_BY_ID.get(tid, "")
                    if name in candidate_names and isinstance(val, str):
                        dt = _parse_exif_datetime(val)
                        if dt:
                            return dt

    except Exception:
        dt = None

    # Fallback: filesystem modified time (gives you *a* consistent ordering)
    try:
        return datetime.fromtimestamp(img_path.stat().st_mtime)
    except Exception:
        return None


# -----------------------------
# Row label helpers
# -----------------------------
def extract_four_digits(relpath: str) -> str:
    """
    User wants the XXXX from names like _DSCXXXX_4 (keep XXXX).
    We use the last 4-digit group in the stem as a robust default.
    """
    stem = Path(relpath).stem
    m = re.findall(r"\d{4}", stem)
    return m[-1] if m else "----"


def row_label(relpath: str, dt: datetime | None) -> str:
    digits = extract_four_digits(relpath)
    if dt is None:
        return f"????-??-??  {digits}"
    return f"{dt.date().isoformat()}  {digits}"


# -----------------------------
# Sorting
# -----------------------------
def sort_images_within_year(relpaths: list[str]) -> list[str]:
    """
    Sort by capture datetime (EXIF/mtime fallback), then by relpath.
    """
    items = []
    for rp in relpaths:
        img_path = RAW_DATA_ROOT / rp
        dt = get_capture_datetime(img_path)
        # Put unknown last by using a far-future sentinel if dt is None (shouldn't happen due to mtime fallback)
        sentinel = datetime(9999, 12, 31)
        items.append((dt or sentinel, rp))
    items.sort(key=lambda x: (x[0], x[1]))
    return [rp for _, rp in items]


# -----------------------------
# Plotting
# -----------------------------
def plot_year_sheet(year: int, palettes_for_year: dict[str, list[dict]]) -> None:
    relpaths = list(palettes_for_year.keys())
    relpaths_sorted = sort_images_within_year(relpaths)
    n_imgs = len(relpaths_sorted)

    fig_h = TOP_MARGIN_IN + BOTTOM_MARGIN_IN + ROW_H_IN * n_imgs
    fig_w = 12.5

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_axis_off()

    # Title: bold, left-aligned, year only, no subtitle
    ax.text(
        0.04,
        1 - (0.22 / fig_h),
        f"{year}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        color=(0.10, 0.10, 0.10),
    )

    # Layout in axes coords
    top_y = 1 - (0.42 / fig_h)
    bottom_y = 0.04

    total_h = top_y - bottom_y
    row_h_ax = total_h / max(n_imgs, 1)

    x_label = 0.04
    x_bar0 = x_label + LEFT_LABEL_W
    x_bar1 = 0.98
    bar_w_ax = x_bar1 - x_bar0

    for idx, relpath in enumerate(relpaths_sorted):
        rows = palettes_for_year[relpath]
        if not rows:
            continue

        rows = sorted(rows, key=lambda d: d["cluster_rank"])
        K = len(rows)

        weights = np.array([r["weight"] for r in rows], dtype=float)
        if weights.sum() > 0:
            weights = weights / weights.sum()

        labs = [(r["L_star"], r["a_star"], r["b_star"]) for r in rows]
        rgbs = [lab_to_rgb01(*lab) for lab in labs]

        y_top = top_y - idx * row_h_ax
        y0 = y_top - row_h_ax * 0.80
        h = row_h_ax * 0.62

        dt = get_capture_datetime(RAW_DATA_ROOT / relpath)
        label = row_label(relpath, dt)

        # Left label: date + 4-digit id only
        ax.text(
            x_label,
            y0 + h * 0.55,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=LABEL_SIZE,
            color=(0.18, 0.18, 0.18),
        )

        # Swatches (no borders), percent label left-edge with padding
        x = x_bar0
        for k in range(K):
            w = float(weights[k])
            sw_w = bar_w_ax * w

            rect = Rectangle(
                (x, y0),
                sw_w,
                h,
                transform=ax.transAxes,
                facecolor=rgbs[k],
                edgecolor="none",
                linewidth=0.0,
            )
            ax.add_patch(rect)

            pct = 100 * w
            tc = text_color_for_rgb(rgbs[k])

            ax.text(
                x + PCT_PAD_AX,
                y0 + h / 2,
                f"{pct:.0f}%",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=PCT_SIZE,
                fontweight="bold",
                color=tc,
            )

            x += sw_w

        # very subtle separator
        ax.plot(
            [x_label, 0.98],
            [y0 - row_h_ax * 0.18, y0 - row_h_ax * 0.18],
            transform=ax.transAxes,
            linewidth=0.6,
            color=(0, 0, 0, 0.06),
        )

    out_path = OUT_DIR / f"palettes_{year}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    csvs = find_palette_csvs()
    if not csvs:
        print("No palette CSVs found.")
        return

    by_year = load_palettes(csvs)

    for year in sorted(by_year.keys()):
        plot_year_sheet(year, by_year[year])


if __name__ == "__main__":
    main()