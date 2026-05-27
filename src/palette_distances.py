from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.optimize import linear_sum_assignment, linprog
from scipy.spatial.distance import squareform
from skimage import color


DEFAULT_PALETTES_CSV = Path("data/processed/seed_0/palettes_medclip/palettes_all_medclip_seed_0.csv")
DEFAULT_OUT_ROOT = Path("data/processed/inter_palette")
RAW_DATA_ROOT = Path("data")


@dataclass(frozen=True)
class Palette:
    year: int
    relpath: str
    filename: str
    weights: np.ndarray
    labs: np.ndarray


def extract_four_digits(filename: str) -> str:
    matches = re.findall(r"\d{4}", Path(filename).stem)
    return matches[-1] if matches else Path(filename).stem[-4:]


def load_palettes(path: Path) -> list[Palette]:
    df = pd.read_csv(path)
    required = {"year", "relpath", "filename", "cluster_rank", "weight", "L_star", "a_star", "b_star"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    palettes: list[Palette] = []
    for relpath, group in df.groupby("relpath", sort=False):
        group = group.sort_values("cluster_rank")
        weights = group["weight"].to_numpy(dtype=float)
        if weights.sum() > 0:
            weights = weights / weights.sum()

        labs = group[["L_star", "a_star", "b_star"]].to_numpy(dtype=float)
        first = group.iloc[0]
        palettes.append(
            Palette(
                year=int(first["year"]),
                relpath=str(relpath),
                filename=str(first["filename"]),
                weights=weights,
                labs=labs,
            )
        )

    return sorted(palettes, key=lambda p: (p.year, p.relpath))


def palette_cost_matrix(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    diff = lab_a[:, None, :] - lab_b[None, :, :]
    return np.linalg.norm(diff, axis=2)


def lab_to_rgb01(L: float, a: float, b: float) -> np.ndarray:
    lab = np.array([[[L, a, b]]], dtype=np.float32)
    rgb = color.lab2rgb(lab)
    return np.clip(rgb[0, 0, :], 0.0, 1.0)


def matched_metrics(a: Palette, b: Palette) -> dict:
    cost = palette_cost_matrix(a.labs, b.labs)
    row_idx, col_idx = linear_sum_assignment(cost)
    delta_e = cost[row_idx, col_idx]
    match_weights = (a.weights[row_idx] + b.weights[col_idx]) / 2.0

    return {
        "matched_unweighted_delta_e": float(delta_e.mean()),
        "matched_weighted_delta_e": float(np.average(delta_e, weights=match_weights)),
        "matched_rms_delta_e": float(np.sqrt(np.mean(delta_e**2))),
        "matched_weighted_rms_delta_e": float(np.sqrt(np.average(delta_e**2, weights=match_weights))),
        "matched_max_delta_e": float(delta_e.max()),
        "matched_weight_l1": float(np.abs(a.weights[row_idx] - b.weights[col_idx]).sum()),
    }


def transport_distance(a: Palette, b: Palette, cost: np.ndarray | None = None) -> float:
    if cost is None:
        cost = palette_cost_matrix(a.labs, b.labs)

    n_a, n_b = cost.shape
    objective = cost.reshape(-1)

    a_eq = []
    b_eq = []
    for i in range(n_a):
        row = np.zeros(n_a * n_b)
        row[i * n_b : (i + 1) * n_b] = 1.0
        a_eq.append(row)
        b_eq.append(a.weights[i])

    for j in range(n_b):
        row = np.zeros(n_a * n_b)
        row[j::n_b] = 1.0
        a_eq.append(row)
        b_eq.append(b.weights[j])

    result = linprog(
        c=objective,
        A_eq=np.vstack(a_eq),
        b_eq=np.asarray(b_eq),
        bounds=(0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Optimal transport failed: {result.message}")

    return float(result.fun)


def pairwise_rows(palettes: list[Palette], include_transport: bool) -> list[dict]:
    rows = []
    n = len(palettes)

    for i, left in enumerate(palettes):
        for j in range(i + 1, n):
            right = palettes[j]
            row = {
                "image_a": left.relpath,
                "filename_a": left.filename,
                "id_a": extract_four_digits(left.filename),
                "year_a": left.year,
                "image_b": right.relpath,
                "filename_b": right.filename,
                "id_b": extract_four_digits(right.filename),
                "year_b": right.year,
                "same_year": int(left.year == right.year),
            }
            row.update(matched_metrics(left, right))
            if include_transport:
                row["transport_delta_e"] = transport_distance(left, right)
            rows.append(row)

        if (i + 1) % 10 == 0 or i == n - 2:
            print(f"computed pairs for {i + 1}/{n} palettes", flush=True)

    return rows


def write_pairwise(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_nearest_neighbors(pairwise: pd.DataFrame, out_path: Path, metric: str, top_n: int) -> None:
    directed_rows = []
    for _, row in pairwise.iterrows():
        directed_rows.append(
            {
                "query_image": row["image_a"],
                "query_filename": row["filename_a"],
                "query_id": row["id_a"],
                "query_year": int(row["year_a"]),
                "neighbor_image": row["image_b"],
                "neighbor_filename": row["filename_b"],
                "neighbor_id": row["id_b"],
                "neighbor_year": int(row["year_b"]),
                metric: float(row[metric]),
            }
        )
        directed_rows.append(
            {
                "query_image": row["image_b"],
                "query_filename": row["filename_b"],
                "query_id": row["id_b"],
                "query_year": int(row["year_b"]),
                "neighbor_image": row["image_a"],
                "neighbor_filename": row["filename_a"],
                "neighbor_id": row["id_a"],
                "neighbor_year": int(row["year_a"]),
                metric: float(row[metric]),
            }
        )

    nearest = pd.DataFrame(directed_rows)
    nearest = nearest.sort_values(["query_image", metric, "neighbor_image"])
    nearest["neighbor_rank"] = nearest.groupby("query_image").cumcount() + 1
    nearest = nearest[nearest["neighbor_rank"] <= top_n]
    nearest.to_csv(out_path, index=False)


def year_pair_label(year_a: int, year_b: int) -> tuple[int, int]:
    return (year_a, year_b) if year_a <= year_b else (year_b, year_a)


def write_year_summaries(pairwise: pd.DataFrame, out_root: Path, metrics: list[str]) -> None:
    rows = []
    for _, row in pairwise.iterrows():
        y0, y1 = year_pair_label(int(row["year_a"]), int(row["year_b"]))
        for metric in metrics:
            rows.append({"year_a": y0, "year_b": y1, "metric": metric, "distance": float(row[metric])})

    long_df = pd.DataFrame(rows)
    summary = (
        long_df.groupby(["metric", "year_a", "year_b"])["distance"]
        .agg(
            n_pairs="size",
            mean="mean",
            median="median",
            std="std",
            p25=lambda x: x.quantile(0.25),
            p75=lambda x: x.quantile(0.75),
            p95=lambda x: x.quantile(0.95),
            min="min",
            max="max",
        )
        .reset_index()
    )
    summary.to_csv(out_root / "year_pair_distance_summary.csv", index=False)

    within = summary[summary["year_a"] == summary["year_b"]].copy()
    within.to_csv(out_root / "within_year_distance_summary.csv", index=False)

    for metric in metrics:
        metric_summary = summary[summary["metric"] == metric]
        matrix = metric_summary.pivot(index="year_a", columns="year_b", values="mean")
        years = sorted(set(matrix.index).union(matrix.columns))
        matrix = matrix.reindex(index=years, columns=years)
        for y0 in years:
            for y1 in years:
                if pd.isna(matrix.loc[y0, y1]) and not pd.isna(matrix.loc[y1, y0]):
                    matrix.loc[y0, y1] = matrix.loc[y1, y0]
        matrix.to_csv(out_root / f"year_distance_matrix_{metric}.csv")
        plot_year_heatmap(matrix, out_root / f"year_distance_heatmap_{metric}.png", metric)


def write_year_medoids(pairwise: pd.DataFrame, palettes: list[Palette], out_path: Path, metrics: list[str]) -> None:
    palette_df = pd.DataFrame(
        [{"image": p.relpath, "filename": p.filename, "id": extract_four_digits(p.filename), "year": p.year} for p in palettes]
    )

    rows = []
    for metric in metrics:
        directed = []
        for _, row in pairwise[pairwise["same_year"] == 1].iterrows():
            directed.append({"image": row["image_a"], "metric": metric, "distance": float(row[metric])})
            directed.append({"image": row["image_b"], "metric": metric, "distance": float(row[metric])})

        directed_df = pd.DataFrame(directed)
        sums = directed_df.groupby(["metric", "image"])["distance"].sum().reset_index()
        scored = palette_df.merge(sums, on="image", how="left")
        scored["distance"] = scored["distance"].fillna(0.0)

        for year, group in scored.groupby("year"):
            best = group.sort_values(["distance", "image"]).iloc[0]
            rows.append(
                {
                    "metric": metric,
                    "year": int(year),
                    "medoid_image": best["image"],
                    "medoid_filename": best["filename"],
                    "medoid_id": best["id"],
                    "within_year_distance_sum": float(best["distance"]),
                }
            )

    pd.DataFrame(rows).to_csv(out_path, index=False)


def plot_year_heatmap(matrix: pd.DataFrame, out_path: Path, metric: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=180)
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis")
    years = [str(y) for y in matrix.index]
    ax.set_xticks(np.arange(len(years)))
    ax.set_yticks(np.arange(len(years)))
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.set_yticklabels(years)
    ax.set_title(metric.replace("_", " "))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8, color="white")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def draw_palette_bar(ax, palette: Palette, x0: float, x1: float, y: float, h: float) -> None:
    x = x0
    width = x1 - x0
    for weight, lab in zip(palette.weights, palette.labs):
        swatch_w = width * float(weight)
        rgb = lab_to_rgb01(float(lab[0]), float(lab[1]), float(lab[2]))
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                swatch_w,
                h,
                facecolor=rgb,
                edgecolor="none",
            )
        )
        x += swatch_w


def plot_pair_gallery(
    pairwise: pd.DataFrame,
    palettes: list[Palette],
    out_path: Path,
    metric: str,
    n_pairs: int,
    largest: bool,
) -> None:
    palette_by_path = {palette.relpath: palette for palette in palettes}
    pairs = pairwise.sort_values(metric, ascending=not largest).head(n_pairs)
    if pairs.empty:
        return

    fig_h = max(4.0, 0.5 * len(pairs) + 1.0)
    fig, ax = plt.subplots(figsize=(12.5, fig_h), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(pairs))
    ax.axis("off")
    title_prefix = "Largest" if largest else "Smallest"
    ax.set_title(f"{title_prefix} inter-palette distances ({metric})", pad=12)

    left_x0, left_x1 = 0.18, 0.55
    right_x0, right_x1 = 0.60, 0.97
    bar_h = 0.58

    for idx, (_, row) in enumerate(pairs.iterrows()):
        y = len(pairs) - idx - 0.78
        left = palette_by_path[row["image_a"]]
        right = palette_by_path[row["image_b"]]

        ax.text(
            0.02,
            y + bar_h / 2,
            f"{int(row['year_a'])} {row['id_a']}",
            ha="left",
            va="center",
            fontsize=8,
            color=(0.15, 0.15, 0.15),
        )
        ax.text(
            0.565,
            y + bar_h / 2,
            f"{float(row[metric]):.1f}",
            ha="center",
            va="center",
            fontsize=8,
            color=(0.15, 0.15, 0.15),
        )
        ax.text(
            0.985,
            y + bar_h / 2,
            f"{int(row['year_b'])} {row['id_b']}",
            ha="right",
            va="center",
            fontsize=8,
            color=(0.15, 0.15, 0.15),
        )
        draw_palette_bar(ax, left, left_x0, left_x1, y, bar_h)
        draw_palette_bar(ax, right, right_x0, right_x1, y, bar_h)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_photo_pair_gallery(
    pairwise: pd.DataFrame,
    palettes: list[Palette],
    out_path: Path,
    metric: str,
    n_pairs: int,
    largest: bool,
) -> None:
    palette_by_path = {palette.relpath: palette for palette in palettes}
    pairs = pairwise.sort_values(metric, ascending=not largest).head(n_pairs)
    if pairs.empty:
        return

    fig_h = max(4.0, 1.45 * len(pairs) + 0.8)
    fig, ax = plt.subplots(figsize=(12.5, fig_h), dpi=170)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(pairs))
    ax.axis("off")
    title_prefix = "Largest" if largest else "Smallest"
    ax.set_title(f"{title_prefix} image-palette distances ({metric})", pad=12)

    for idx, (_, row) in enumerate(pairs.iterrows()):
        y = len(pairs) - idx - 0.95
        left = palette_by_path[row["image_a"]]
        right = palette_by_path[row["image_b"]]

        left_img_ax = ax.inset_axes([0.02, y / len(pairs), 0.11, 0.95 / len(pairs)])
        right_img_ax = ax.inset_axes([0.87, y / len(pairs), 0.11, 0.95 / len(pairs)])
        for image_ax, relpath in [(left_img_ax, left.relpath), (right_img_ax, right.relpath)]:
            image_ax.axis("off")
            try:
                with Image.open(RAW_DATA_ROOT / relpath) as img:
                    img.thumbnail((260, 180), Image.Resampling.LANCZOS)
                    image_ax.imshow(img.convert("RGB"))
            except Exception:
                image_ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=7)

        ax.text(0.14, y + 0.72, f"{left.year} {extract_four_digits(left.filename)}", fontsize=8, va="center")
        ax.text(0.86, y + 0.72, f"{right.year} {extract_four_digits(right.filename)}", fontsize=8, va="center", ha="right")
        ax.text(0.50, y + 0.72, f"{float(row[metric]):.1f}", fontsize=8, va="center", ha="center")

        draw_palette_bar(ax, left, 0.14, 0.47, y + 0.24, 0.22)
        draw_palette_bar(ax, right, 0.53, 0.86, y + 0.24, 0.22)
        ax.plot([0.02, 0.98], [y - 0.10, y - 0.10], color=(0, 0, 0, 0.08), linewidth=0.7)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def distance_matrix_from_pairwise(
    pairwise: pd.DataFrame,
    palettes: list[Palette],
    metric: str,
) -> tuple[np.ndarray, list[Palette]]:
    ordered = sorted(palettes, key=lambda p: (p.year, p.relpath))
    index = {palette.relpath: i for i, palette in enumerate(ordered)}
    matrix = np.zeros((len(ordered), len(ordered)), dtype=float)

    for _, row in pairwise.iterrows():
        i = index[row["image_a"]]
        j = index[row["image_b"]]
        matrix[i, j] = float(row[metric])
        matrix[j, i] = float(row[metric])

    return matrix, ordered


def classical_mds(distance_matrix: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    n = distance_matrix.shape[0]
    squared = distance_matrix**2
    center = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * center @ squared @ center
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    positive = np.maximum(eigvals[:n_components], 0)
    coords = eigvecs[:, :n_components] * np.sqrt(positive)
    return coords, eigvals


def plot_mds_map(
    pairwise: pd.DataFrame,
    palettes: list[Palette],
    out_root: Path,
    metric: str,
    label_points: bool,
) -> None:
    matrix, ordered = distance_matrix_from_pairwise(pairwise, palettes, metric)
    coords, eigvals = classical_mds(matrix)
    total_positive = eigvals[eigvals > 0].sum()
    explained = eigvals[:2].sum() / total_positive if total_positive > 0 else np.nan

    rows = []
    for palette, coord in zip(ordered, coords):
        rows.append(
            {
                "image": palette.relpath,
                "filename": palette.filename,
                "id": extract_four_digits(palette.filename),
                "year": palette.year,
                "x": float(coord[0]),
                "y": float(coord[1]),
            }
        )
    pd.DataFrame(rows).to_csv(out_root / f"mds_embedding_{metric}.csv", index=False)

    years = sorted({palette.year for palette in ordered})
    cmap = plt.get_cmap("tab10")
    color_by_year = {year: cmap(i % 10) for i, year in enumerate(years)}

    fig, ax = plt.subplots(figsize=(9.5, 8.0), dpi=180)
    for year in years:
        idx = [i for i, palette in enumerate(ordered) if palette.year == year]
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=34,
            color=color_by_year[year],
            label=str(year),
            alpha=0.86,
            edgecolor="white",
            linewidth=0.5,
        )

    if label_points:
        for palette, coord in zip(ordered, coords):
            ax.text(
                coord[0],
                coord[1],
                extract_four_digits(palette.filename),
                fontsize=5.5,
                ha="center",
                va="center",
                color=(0.08, 0.08, 0.08, 0.72),
            )

    ax.axhline(0, color=(0, 0, 0, 0.12), linewidth=0.8)
    ax.axvline(0, color=(0, 0, 0, 0.12), linewidth=0.8)
    ax.set_title(f"Classical MDS of image palettes\n{metric}, 2D variance proxy {explained:.1%}")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.legend(title="Year", frameon=False, ncol=2)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    suffix = "labeled" if label_points else "points"
    fig.savefig(out_root / f"mds_map_{metric}_{suffix}.png")
    plt.close(fig)


def mds_pair_distance_lookup(
    pairwise: pd.DataFrame,
    palettes: list[Palette],
    metric: str,
) -> dict[tuple[str, str], float]:
    matrix, ordered = distance_matrix_from_pairwise(pairwise, palettes, metric)
    coords, _ = classical_mds(matrix)
    coord_by_path = {palette.relpath: coord for palette, coord in zip(ordered, coords)}
    distances: dict[tuple[str, str], float] = {}
    for _, row in pairwise.iterrows():
        left = row["image_a"]
        right = row["image_b"]
        d = float(np.linalg.norm(coord_by_path[left] - coord_by_path[right]))
        distances[(left, right)] = d
        distances[(right, left)] = d
    return distances


def plot_metric_comparison_gallery(
    pairwise: pd.DataFrame,
    palettes: list[Palette],
    out_path: Path,
    n_pairs: int,
) -> None:
    metric_a = "matched_weighted_delta_e"
    metric_b = "transport_delta_e"
    if metric_a not in pairwise.columns or metric_b not in pairwise.columns:
        return

    ranked = pairwise.copy()
    ranked["rank_a"] = ranked[metric_a].rank(method="first", ascending=True)
    ranked["rank_b"] = ranked[metric_b].rank(method="first", ascending=True)
    ranked["rank_gap"] = (ranked["rank_a"] - ranked["rank_b"]).abs()

    n_group = max(4, n_pairs // 3)
    selected = pd.concat(
        [
            ranked.sort_values(metric_a).head(n_group).assign(selection="closest matched-weighted"),
            ranked.sort_values(metric_b).head(n_group).assign(selection="closest transport"),
            ranked.sort_values("rank_gap", ascending=False).head(n_group).assign(selection="largest rank disagreement"),
        ]
    )
    selected = selected.drop_duplicates(subset=["image_a", "image_b"]).head(n_pairs)

    mds_a = mds_pair_distance_lookup(pairwise, palettes, metric_a)
    mds_b = mds_pair_distance_lookup(pairwise, palettes, metric_b)
    palette_by_path = {palette.relpath: palette for palette in palettes}

    fig_h = max(4.0, 1.45 * len(selected) + 0.9)
    fig, ax = plt.subplots(figsize=(13.8, fig_h), dpi=170)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(selected))
    ax.axis("off")
    ax.set_title("Image-pair metric comparison: full palette distance vs 2D MDS distance", pad=12)

    for idx, (_, row) in enumerate(selected.iterrows()):
        y = len(selected) - idx - 0.95
        left = palette_by_path[row["image_a"]]
        right = palette_by_path[row["image_b"]]

        left_img_ax = ax.inset_axes([0.02, y / len(selected), 0.10, 0.95 / len(selected)])
        right_img_ax = ax.inset_axes([0.88, y / len(selected), 0.10, 0.95 / len(selected)])
        for image_ax, relpath in [(left_img_ax, left.relpath), (right_img_ax, right.relpath)]:
            image_ax.axis("off")
            try:
                with Image.open(RAW_DATA_ROOT / relpath) as img:
                    img.thumbnail((250, 180), Image.Resampling.LANCZOS)
                    image_ax.imshow(img.convert("RGB"))
            except Exception:
                image_ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=7)

        ax.text(0.13, y + 0.75, f"{left.year} {extract_four_digits(left.filename)}", fontsize=8, va="center")
        ax.text(0.87, y + 0.75, f"{right.year} {extract_four_digits(right.filename)}", fontsize=8, va="center", ha="right")
        ax.text(0.50, y + 0.78, str(row["selection"]), fontsize=7.5, va="center", ha="center", color=(0.18, 0.18, 0.18))

        draw_palette_bar(ax, left, 0.13, 0.41, y + 0.25, 0.22)
        draw_palette_bar(ax, right, 0.59, 0.87, y + 0.25, 0.22)

        key = (row["image_a"], row["image_b"])
        ax.text(
            0.50,
            y + 0.46,
            f"matched {float(row[metric_a]):.1f} | transport {float(row[metric_b]):.1f}",
            fontsize=7.5,
            ha="center",
            va="center",
        )
        ax.text(
            0.50,
            y + 0.22,
            f"MDS matched {mds_a[key]:.1f} | MDS transport {mds_b[key]:.1f}",
            fontsize=7.0,
            ha="center",
            va="center",
            color=(0.35, 0.35, 0.35),
        )
        ax.plot([0.02, 0.98], [y - 0.10, y - 0.10], color=(0, 0, 0, 0.08), linewidth=0.7)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_clustered_distance_heatmap(
    pairwise: pd.DataFrame,
    palettes: list[Palette],
    out_root: Path,
    metric: str,
) -> None:
    matrix, ordered = distance_matrix_from_pairwise(pairwise, palettes, metric)
    condensed = squareform(matrix, checks=False)
    tree = linkage(condensed, method="average")
    order = leaves_list(tree)
    clustered = matrix[np.ix_(order, order)]
    ordered_palettes = [ordered[i] for i in order]
    labels = [f"{p.year} {extract_four_digits(p.filename)}" for p in ordered_palettes]

    pd.DataFrame(
        {
            "order": np.arange(len(ordered_palettes)),
            "image": [p.relpath for p in ordered_palettes],
            "filename": [p.filename for p in ordered_palettes],
            "id": [extract_four_digits(p.filename) for p in ordered_palettes],
            "year": [p.year for p in ordered_palettes],
        }
    ).to_csv(out_root / f"cluster_order_{metric}.csv", index=False)

    fig = plt.figure(figsize=(11.5, 10.5), dpi=180)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.16, 0.84], wspace=0.02)
    dend_ax = fig.add_subplot(gs[0, 0])
    heat_ax = fig.add_subplot(gs[0, 1])

    dendrogram(tree, orientation="left", no_labels=True, ax=dend_ax, color_threshold=None)
    dend_ax.invert_yaxis()
    dend_ax.axis("off")

    im = heat_ax.imshow(clustered, cmap="magma_r", aspect="auto")
    heat_ax.set_title(f"Clustered image-to-image palette distances\n{metric}")
    heat_ax.set_xticks([])
    heat_ax.set_yticks(np.arange(len(labels)))
    heat_ax.set_yticklabels(labels, fontsize=4.5)
    heat_ax.yaxis.tick_right()
    heat_ax.tick_params(axis="y", length=0, pad=2)

    fig.colorbar(im, ax=heat_ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_root / f"clustered_distance_heatmap_{metric}.png")
    plt.close(fig)


def plot_nearest_neighbor_graph(
    pairwise: pd.DataFrame,
    palettes: list[Palette],
    out_root: Path,
    metric: str,
    k: int,
) -> None:
    matrix, ordered = distance_matrix_from_pairwise(pairwise, palettes, metric)
    coords, _ = classical_mds(matrix)

    directed_edges = []
    for i, source in enumerate(ordered):
        neighbors = np.argsort(matrix[i])
        neighbors = [j for j in neighbors if j != i][:k]
        for j in neighbors:
            target = ordered[j]
            directed_edges.append(
                {
                    "source": source.relpath,
                    "source_id": extract_four_digits(source.filename),
                    "source_year": source.year,
                    "target": target.relpath,
                    "target_id": extract_four_digits(target.filename),
                    "target_year": target.year,
                    metric: float(matrix[i, j]),
                }
            )

    edge_df = pd.DataFrame(directed_edges)
    edge_df.to_csv(out_root / f"nearest_neighbor_edges_k{k}_{metric}.csv", index=False)

    years = sorted({palette.year for palette in ordered})
    cmap = plt.get_cmap("tab10")
    color_by_year = {year: cmap(i % 10) for i, year in enumerate(years)}

    fig, ax = plt.subplots(figsize=(10.5, 9.0), dpi=180)
    for edge in directed_edges:
        i = next(idx for idx, p in enumerate(ordered) if p.relpath == edge["source"])
        j = next(idx for idx, p in enumerate(ordered) if p.relpath == edge["target"])
        ax.plot(
            [coords[i, 0], coords[j, 0]],
            [coords[i, 1], coords[j, 1]],
            color=(0, 0, 0, 0.10),
            linewidth=0.6,
            zorder=1,
        )

    for year in years:
        idx = [i for i, palette in enumerate(ordered) if palette.year == year]
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=34,
            color=color_by_year[year],
            label=str(year),
            alpha=0.9,
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )

    for palette, coord in zip(ordered, coords):
        ax.text(
            coord[0],
            coord[1],
            extract_four_digits(palette.filename),
            fontsize=5.2,
            ha="center",
            va="center",
            color=(0.05, 0.05, 0.05, 0.72),
            zorder=3,
        )

    ax.axhline(0, color=(0, 0, 0, 0.10), linewidth=0.8)
    ax.axvline(0, color=(0, 0, 0, 0.10), linewidth=0.8)
    ax.set_title(f"Nearest-neighbor graph on 2D palette map\n{k} outgoing neighbors per image, {metric}")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.legend(title="Year", frameon=False, ncol=2)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_root / f"nearest_neighbor_graph_k{k}_{metric}.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute image-to-image palette dissimilarities.")
    parser.add_argument("--palettes-csv", type=Path, default=DEFAULT_PALETTES_CSV)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--gallery-n", type=int, default=30)
    parser.add_argument("--comparison-n", type=int, default=18)
    parser.add_argument("--skip-year-summaries", action="store_true")
    parser.add_argument("--skip-mds", action="store_true")
    parser.add_argument("--skip-cluster-heatmap", action="store_true")
    parser.add_argument("--skip-neighbor-graph", action="store_true")
    parser.add_argument("--label-mds", action="store_true")
    parser.add_argument("--neighbor-k", type=int, default=3)
    parser.add_argument("--include-transport", action="store_true")
    parser.add_argument(
        "--primary-metric",
        default="matched_weighted_delta_e",
        choices=[
            "matched_weighted_delta_e",
            "matched_unweighted_delta_e",
            "matched_rms_delta_e",
            "matched_weighted_rms_delta_e",
            "matched_max_delta_e",
            "transport_delta_e",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    palettes = load_palettes(args.palettes_csv)
    if len(palettes) < 2:
        return

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = pairwise_rows(palettes, include_transport=args.include_transport)

    pairwise_path = args.out_root / "pairwise_palette_distances.csv"
    write_pairwise(rows, pairwise_path)

    pairwise = pd.DataFrame(rows)
    metrics = [
        "matched_weighted_delta_e",
        "matched_unweighted_delta_e",
        "matched_rms_delta_e",
        "matched_weighted_rms_delta_e",
        "matched_max_delta_e",
        "matched_weight_l1",
    ]
    if args.include_transport:
        metrics.append("transport_delta_e")

    write_nearest_neighbors(
        pairwise=pairwise,
        out_path=args.out_root / f"nearest_neighbors_{args.primary_metric}.csv",
        metric=args.primary_metric,
        top_n=args.top_n,
    )
    if not args.skip_year_summaries:
        write_year_summaries(pairwise, args.out_root, metrics)
        write_year_medoids(pairwise, palettes, args.out_root / "year_medoids.csv", metrics)
    plot_pair_gallery(
        pairwise=pairwise,
        palettes=palettes,
        out_path=args.out_root / f"closest_pairs_{args.primary_metric}.png",
        metric=args.primary_metric,
        n_pairs=args.gallery_n,
        largest=False,
    )
    plot_pair_gallery(
        pairwise=pairwise,
        palettes=palettes,
        out_path=args.out_root / f"farthest_pairs_{args.primary_metric}.png",
        metric=args.primary_metric,
        n_pairs=args.gallery_n,
        largest=True,
    )
    plot_photo_pair_gallery(
        pairwise=pairwise,
        palettes=palettes,
        out_path=args.out_root / f"closest_photo_pairs_{args.primary_metric}.png",
        metric=args.primary_metric,
        n_pairs=min(args.gallery_n, 18),
        largest=False,
    )
    plot_photo_pair_gallery(
        pairwise=pairwise,
        palettes=palettes,
        out_path=args.out_root / f"farthest_photo_pairs_{args.primary_metric}.png",
        metric=args.primary_metric,
        n_pairs=min(args.gallery_n, 18),
        largest=True,
    )
    if not args.skip_mds:
        plot_mds_map(
            pairwise=pairwise,
            palettes=palettes,
            out_root=args.out_root,
            metric=args.primary_metric,
            label_points=args.label_mds,
        )
    plot_metric_comparison_gallery(
        pairwise=pairwise,
        palettes=palettes,
        out_path=args.out_root / "metric_comparison_photo_pairs.png",
        n_pairs=args.comparison_n,
    )
    if not args.skip_cluster_heatmap:
        plot_clustered_distance_heatmap(
            pairwise=pairwise,
            palettes=palettes,
            out_root=args.out_root,
            metric=args.primary_metric,
        )
    if not args.skip_neighbor_graph:
        plot_nearest_neighbor_graph(
            pairwise=pairwise,
            palettes=palettes,
            out_root=args.out_root,
            metric=args.primary_metric,
            k=args.neighbor_k,
        )


if __name__ == "__main__":
    main()
