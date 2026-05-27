from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage import color


DEFAULT_PALETTES_CSV = Path("data/processed/seed_0/palettes_k5/palettes_all_k5_seed_0.csv")
DEFAULT_PAIRWISE_CSV = Path("data/processed/inter_palette/seed_0_palettes_k5/pairwise_palette_distances.csv")
DEFAULT_MDS_DIR = Path("data/processed/inter_palette/seed_0_palettes_k5")
DEFAULT_OUT = Path("figures/inter_palette/interactive_palette_constellation.html")

METRICS = {
    "transport_delta_e": "Transport",
    "matched_weighted_delta_e": "Matched weighted",
}


def lab_to_hex(L: float, a: float, b: float) -> str:
    lab = np.array([[[L, a, b]]], dtype=np.float32)
    rgb = np.clip(color.lab2rgb(lab)[0, 0, :], 0.0, 1.0)
    vals = (rgb * 255).round().astype(int)
    return "#{:02x}{:02x}{:02x}".format(*vals)


def extract_four_digits(filename: str) -> str:
    stem = Path(filename).stem
    groups = re.findall(r"(\d{4})", stem)
    return groups[-1] if groups else stem[-4:]


def describe_lab(L: float, a: float, b: float) -> str:
    if L < 24:
        light = "deep shadow"
    elif L < 48:
        light = "low light"
    elif L < 72:
        light = "balanced midtone"
    elif L < 88:
        light = "soft highlight"
    else:
        light = "near white"

    if a < -12:
        red_green = "green pull"
    elif a > 18:
        red_green = "red warmth"
    elif a > 6:
        red_green = "skin warmth"
    else:
        red_green = "neutral axis"

    if b < -14:
        blue_yellow = "blue cast"
    elif b > 22:
        blue_yellow = "amber cast"
    elif b > 8:
        blue_yellow = "cream cast"
    else:
        blue_yellow = "quiet balance"

    return f"{light} · {red_green} · {blue_yellow}"


def image_dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).size


def build_nodes(palettes_csv: Path, mds_dir: Path, html_out: Path) -> list[dict]:
    palettes = pd.read_csv(palettes_csv)
    matched_mds = pd.read_csv(mds_dir / "mds_embedding_matched_weighted_delta_e.csv")
    transport_mds = pd.read_csv(mds_dir / "mds_embedding_transport_delta_e.csv")
    matched_by_image = matched_mds.set_index("image")[["x", "y"]].to_dict("index")
    transport_by_image = transport_mds.set_index("image")[["x", "y"]].to_dict("index")

    nodes = []
    for relpath, group in palettes.groupby("relpath", sort=False):
        group = group.sort_values("cluster_rank")
        first = group.iloc[0]
        swatches = []
        for _, row in group.iterrows():
            L = float(row["L_star"])
            a = float(row["a_star"])
            b = float(row["b_star"])
            swatches.append(
                {
                    "weight": float(row["weight"]),
                    "hex": lab_to_hex(L, a, b),
                    "lab": [round(L, 2), round(a, 2), round(b, 2)],
                    "language": describe_lab(L, a, b),
                }
            )

        image_path = Path("data") / relpath
        image_src = (Path("../../") / image_path).as_posix()
        width, height = image_dimensions(image_path)
        aspect = width / height
        nodes.append(
            {
                "image": relpath,
                "filename": str(first["filename"]),
                "id": extract_four_digits(str(first["filename"])),
                "year": int(first["year"]),
                "src": image_src,
                "width": width,
                "height": height,
                "aspect": round(aspect, 4),
                "orientation": "landscape" if aspect > 1.12 else "portrait" if aspect < 0.88 else "square",
                "palette": swatches,
                "coords": {
                    "matched_weighted_delta_e": matched_by_image[relpath],
                    "transport_delta_e": transport_by_image[relpath],
                },
            }
        )

    return sorted(nodes, key=lambda node: (node["year"], node["image"]))


def build_neighbors(pairwise_csv: Path, top_n: int) -> dict:
    pairwise = pd.read_csv(pairwise_csv)
    neighbors: dict[str, dict[str, list[dict]]] = {}
    for metric in METRICS:
        directed = []
        for _, row in pairwise.iterrows():
            directed.append({"source": row["image_a"], "target": row["image_b"], "distance": float(row[metric])})
            directed.append({"source": row["image_b"], "target": row["image_a"], "distance": float(row[metric])})

        df = pd.DataFrame(directed).sort_values(["source", "distance", "target"])
        df["rank"] = df.groupby("source").cumcount() + 1
        df = df[df["rank"] <= top_n]
        for source, group in df.groupby("source"):
            neighbors.setdefault(source, {})[metric] = [
                {"target": row["target"], "distance": round(float(row["distance"]), 3), "rank": int(row["rank"])}
                for _, row in group.iterrows()
            ]

    return neighbors


def html_template(data: dict) -> str:
    data_json = json.dumps(data, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Palette Constellation</title>
  <style>
    @import url("https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Source+Sans+3:wght@400;500;600;700&display=swap");
    :root {{
      color-scheme: dark;
      --bg: #0c0b0a;
      --ink: #f1ece3;
      --muted: rgba(241, 236, 227, 0.62);
      --faint: rgba(241, 236, 227, 0.16);
      --line: rgba(241, 236, 227, 0.18);
      --panel: rgba(18, 17, 15, 0.84);
      --panel-solid: #15130f;
      --accent: #e6b35c;
      --accent-soft: rgba(230, 179, 92, 0.32);
      --shadow: rgba(0, 0, 0, 0.56);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      overflow: hidden;
      background:
        radial-gradient(circle at 18% 20%, rgba(166, 105, 57, 0.22), transparent 30%),
        radial-gradient(circle at 78% 68%, rgba(100, 83, 58, 0.20), transparent 32%),
        var(--bg);
      color: var(--ink);
      font-family: "Avenir Next", Inter, "Helvetica Neue", Arial, sans-serif;
      letter-spacing: 0;
    }}
    button, input, select {{
      font: inherit;
    }}
    button {{
      border: 0;
      color: inherit;
      cursor: pointer;
    }}
    .app {{
      position: relative;
      height: 100vh;
      min-height: 720px;
      isolation: isolate;
    }}
    .topbar {{
      position: absolute;
      z-index: 60;
      left: 20px;
      top: 16px;
      width: min(430px, calc(100vw - 40px));
      display: grid;
      gap: 14px;
      padding: 18px;
      border: 1px solid rgba(241, 236, 227, 0.14);
      background: linear-gradient(180deg, rgba(20, 18, 15, 0.82), rgba(13, 12, 10, 0.72));
      backdrop-filter: blur(18px);
      border-radius: 8px;
      box-shadow: 0 18px 55px rgba(0, 0, 0, 0.28);
      transition: opacity 140ms ease, transform 140ms ease, visibility 0s linear 0s;
    }}
    .app.is-focused .topbar {{
      display: none;
      opacity: 0;
      visibility: hidden;
      pointer-events: none;
      transform: translateY(-14px);
      transition-delay: 0s, 0s, 140ms;
    }}
    .brand {{
      min-width: 0;
    }}
    h1 {{
      margin: 0;
      font-size: 21px;
      line-height: 1.15;
      font-family: Poppins, "Avenir Next", "Helvetica Neue", Arial, sans-serif;
      font-style: normal;
      font-weight: 800;
      text-transform: none;
      letter-spacing: 0.035em;
      text-align: left;
    }}
    .brand p {{
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 12.5px;
      line-height: 1.5;
    }}
    .read-note {{
      color: rgba(241, 236, 227, 0.68);
      font-size: 12.5px;
      line-height: 1.48;
    }}
    .read-note strong {{
      color: var(--ink);
      font-family: Poppins, "Avenir Next", "Helvetica Neue", Arial, sans-serif;
      font-style: normal;
      font-weight: 700;
      text-transform: none;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      align-items: stretch;
    }}
    select, input[type="search"], .ghost-button {{
      width: 100%;
      height: 38px;
      min-height: 38px;
      border-radius: 7px;
      border: 1px solid rgba(241, 236, 227, 0.18);
      background:
        linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.025)),
        rgba(16, 14, 12, 0.38);
      color: var(--ink);
      padding: 0 13px;
      outline: none;
      font-size: 12px;
      font-family: Poppins, "Avenir Next", "Helvetica Neue", Arial, sans-serif;
      font-weight: 650;
      text-align: left;
      white-space: nowrap;
      box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.08),
        0 8px 20px rgba(0,0,0,0.12);
      backdrop-filter: blur(8px) saturate(1.08);
      -webkit-backdrop-filter: blur(8px) saturate(1.08);
    }}
    select {{
      appearance: none;
      padding-right: 30px;
      background-image:
        linear-gradient(45deg, transparent 50%, rgba(241, 236, 227, 0.9) 50%),
        linear-gradient(135deg, rgba(241, 236, 227, 0.9) 50%, transparent 50%);
      background-position:
        calc(100% - 17px) 50%,
        calc(100% - 10px) 50%;
      background-size: 7px 7px, 7px 7px;
      background-repeat: no-repeat;
    }}
    select:focus, input[type="search"]:focus {{
      border-color: rgba(230, 179, 92, 0.72);
      box-shadow: 0 0 0 3px rgba(230, 179, 92, 0.16);
    }}
    .ghost-button {{
      font-family: Poppins, "Avenir Next", "Helvetica Neue", Arial, sans-serif;
      font-style: normal;
      font-weight: 650;
      text-transform: uppercase;
      text-align: left;
      letter-spacing: 0.03em;
      padding-left: 13px;
      border-color: rgba(241, 236, 227, 0.15);
      transition: background 160ms ease, border-color 160ms ease, transform 160ms ease;
    }}
    .ghost-button:hover {{
      border-color: rgba(241, 236, 227, 0.22);
      background:
        linear-gradient(135deg, rgba(255,255,255,0.13), rgba(255,255,255,0.04)),
        rgba(18, 16, 14, 0.54);
      transform: translateY(-1px);
    }}
    .stage {{
      position: absolute;
      inset: 0;
      overflow: hidden;
    }}
    #constellation {{
      width: 100%;
      height: 100%;
      display: block;
      transition: opacity 100ms ease;
      transform: translate(var(--zoom-pan-x, 0px), var(--zoom-pan-y, 0px)) scale(1);
      transform-origin: 50% 50%;
      will-change: transform;
    }}
    #constellation.is-zoomed {{
      transform: translate(var(--zoom-pan-x, 0px), var(--zoom-pan-y, 0px)) scale(2.35);
    }}
    .app.is-focused #constellation {{
      opacity: 0.30;
      filter: saturate(0.88);
    }}
    .ambient-edge {{
      stroke: rgba(241, 236, 227, 0.020);
      stroke-width: 0.32;
      vector-effect: non-scaling-stroke;
      pointer-events: none;
      opacity: 1;
    }}
    #constellation.has-active .ambient-edge {{
      opacity: 0;
    }}
    .glow-edge {{
      stroke-width: 0.74;
      stroke-linecap: round;
      opacity: 0;
      pointer-events: none;
      vector-effect: non-scaling-stroke;
      transition: opacity 220ms ease;
    }}
    .glow-edge.visible {{
      opacity: 1;
    }}
    .selection-ring {{
      fill: rgba(230, 179, 92, 0.018);
      stroke: rgba(230, 179, 92, 0.10);
      stroke-width: 0.62;
      vector-effect: non-scaling-stroke;
      pointer-events: none;
    }}
    .node {{
      pointer-events: none;
      opacity: 0.46;
      transition: opacity 180ms ease, r 180ms ease, filter 180ms ease;
    }}
    .hit-node {{
      cursor: pointer;
      fill: transparent;
      stroke: transparent;
      pointer-events: none;
    }}
    .node:hover, .node.active, .node.neighbor {{
      opacity: 1;
    }}
    .node.active {{
      stroke: rgba(255, 255, 255, 0.96);
      stroke-width: 1.55;
      vector-effect: non-scaling-stroke;
    }}
    .node.neighbor {{
      stroke: rgba(230, 179, 92, 0.74);
      stroke-width: 1.1;
      vector-effect: non-scaling-stroke;
    }}
    .id-label {{
      fill: rgba(241, 236, 227, 0.72);
      font-size: 6px;
      pointer-events: none;
      text-anchor: middle;
      opacity: 0;
      transition: opacity 180ms ease;
    }}
    .id-label.visible {{
      opacity: 1;
    }}
    .hover-card {{
      position: fixed;
      z-index: 40;
      width: auto;
      border: 1px solid rgba(241, 236, 227, 0.20);
      border-radius: 8px;
      overflow: hidden;
      background: rgba(14, 13, 11, 0.88);
      box-shadow: 0 14px 36px rgba(0, 0, 0, 0.28);
      backdrop-filter: blur(8px);
      opacity: 0;
      transform: scale(0.95);
      transform-origin: top left;
      pointer-events: none;
      transition: opacity 140ms ease, transform 140ms ease;
    }}
    .hover-card.visible {{
      opacity: 1;
      transform: scale(1);
    }}
    .hover-card img {{
      display: block;
      width: 100%;
      height: auto;
      max-height: none;
      object-fit: cover;
      background: transparent;
      filter: saturate(1.04);
    }}
    .hover-meta {{
      padding: 8px;
      font-size: 12px;
      color: var(--muted);
    }}
    .hover-meta strong {{
      display: block;
      color: var(--ink);
      font-family: Poppins, "Avenir Next", "Helvetica Neue", Arial, sans-serif;
      font-style: normal;
      font-weight: 700;
      text-transform: none;
      font-size: 13px;
      margin-bottom: 4px;
    }}
    .focus-layer {{
      position: absolute;
      inset: 24px 18px 18px;
      z-index: 30;
      display: grid;
      place-items: center;
      opacity: 0;
      visibility: hidden;
      pointer-events: none;
      transition: opacity 115ms ease, visibility 0s linear 115ms;
    }}
    .app.is-focused .focus-layer {{
      opacity: 1;
      visibility: visible;
      pointer-events: auto;
      transition-delay: 0s;
    }}
    .focus-shell {{
      position: relative;
      width: min(1240px, 100%);
      height: min(700px, calc(100vh - 48px));
      display: flex;
      align-items: center;
      justify-content: center;
      gap: clamp(18px, 2.2vw, 34px);
    }}
    .selected-card {{
      position: relative;
      z-index: 4;
      flex: 0 0 auto;
      width: auto;
      min-width: 0;
      transform: translateY(14px) scale(0.96);
      opacity: 0;
      transition: transform 280ms cubic-bezier(.2,.8,.2,1), opacity 220ms ease;
    }}
    .app.is-focused .selected-card {{
      opacity: 1;
      transform: translateY(0) scale(1);
    }}
    .image-frame {{
      border: 1px solid rgba(241, 236, 227, 0.20);
      border-radius: 8px;
      background: rgba(15, 14, 12, 0.84);
      box-shadow: 0 18px 42px rgba(0, 0, 0, 0.28);
      overflow: hidden;
    }}
    .selected-card img {{
      display: block;
      width: 100%;
      height: var(--media-h, auto);
      object-fit: cover;
      object-position: center center;
      background: #050403;
    }}
    .card-body {{
      padding: 14px 16px 16px;
    }}
    .card-title {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 7px;
    }}
    .card-title strong {{
      font-size: 28px;
      font-family: Poppins, "Avenir Next", "Helvetica Neue", Arial, sans-serif;
      font-style: normal;
      font-weight: 800;
      text-transform: none;
    }}
    .card-title span, .stat {{
      color: var(--muted);
      font-size: 12px;
    }}
    .palette {{
      display: flex;
      height: clamp(16px, 3.2vw, 24px);
      overflow: hidden;
      border-radius: 5px;
      border: 1px solid rgba(241, 236, 227, 0.13);
    }}
    .mini-palette {{
      display: flex;
      height: 14px;
      overflow: hidden;
      border-radius: 4px;
      border: 1px solid rgba(241, 236, 227, 0.10);
    }}
    .swatch {{
      height: 100%;
    }}
    .lab-table {{
      display: grid;
      gap: 10px;
      margin-top: 14px;
      font-size: 11.5px;
      color: var(--muted);
    }}
    .lab-row {{
      display: grid;
      grid-template-columns: 64px minmax(0, 1fr);
      gap: 11px;
      align-items: center;
      line-height: 1.12;
    }}
    .lab-chip {{
      height: 30px;
      min-height: 0;
      border-radius: 4px;
    }}
    .lab-row strong {{
      display: block;
      color: var(--ink);
      font-size: 11px;
      letter-spacing: 0.01em;
      font-family: Poppins, "Avenir Next", "Helvetica Neue", Arial, sans-serif;
      font-style: normal;
      font-weight: 700;
      text-transform: none;
    }}
    .lab-row span {{
      display: block;
      text-transform: none;
      font-style: normal;
      font-weight: 500;
      font-size: 10.5px;
      letter-spacing: 0;
      max-width: 100%;
    }}
    .orbit-card {{
      position: absolute;
      z-index: 3;
      left: 0;
      top: 0;
      width: auto;
      opacity: 0;
      transition: opacity 165ms ease;
      will-change: transform;
    }}
    .app.is-focused .orbit-card {{
      opacity: 1;
    }}
    .neighbor-carousel {{
      position: relative;
      z-index: 3;
      flex: 0 0 min(700px, 50vw);
      width: min(700px, 50vw);
      height: min(500px, calc(100vh - 150px));
      min-height: 350px;
      overflow: hidden;
      border-radius: 12px;
      mask-image: linear-gradient(90deg, transparent 0%, #000 4%, #000 96%, transparent 100%);
      -webkit-mask-image: linear-gradient(90deg, transparent 0%, #000 4%, #000 96%, transparent 100%);
    }}
    .neighbor-carousel::before,
    .neighbor-carousel::after {{
      content: "";
      position: absolute;
      top: 0;
      bottom: 0;
      width: 72px;
      z-index: 6;
      pointer-events: none;
    }}
    .neighbor-carousel::before {{
      left: 0;
      background: linear-gradient(90deg, rgba(5,4,3,0.10), rgba(5,4,3,0));
    }}
    .neighbor-carousel::after {{
      right: 0;
      background: linear-gradient(270deg, rgba(5,4,3,0.16), rgba(5,4,3,0));
    }}
    .focus-shell.landscape-focus .neighbor-carousel {{
      flex-basis: min(660px, 48vw);
      width: min(660px, 48vw);
    }}
    .focus-shell.is-switching .selected-card,
    .focus-shell.is-switching .orbit-card {{
      opacity: 0;
      transition-duration: 220ms;
    }}
    .neighbor-panel {{
      overflow: hidden;
      border-radius: 8px;
      border: 1px solid rgba(241, 236, 227, 0.18);
      background: rgba(20, 18, 15, 0.78);
      box-shadow: 0 10px 22px rgba(0, 0, 0, 0.16);
      backdrop-filter: blur(8px);
      cursor: pointer;
    }}
    .neighbor-panel img {{
      display: block;
      width: 100%;
      height: var(--media-h, auto);
      object-fit: cover;
      background: #050403;
    }}
    .neighbor-panel .card-body {{
      display: flex;
      flex-direction: column;
      align-items: stretch;
      gap: 6px;
      padding: 12px 14px 13px;
    }}
    .neighbor-panel strong {{
      display: block;
      font-family: Poppins, "Avenir Next", "Helvetica Neue", Arial, sans-serif;
      font-style: normal;
      font-weight: 700;
      text-transform: none;
      font-size: 13px;
      line-height: 1.15;
      white-space: nowrap;
    }}
    .neighbor-panel span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.15;
    }}
    .close-focus {{
      position: fixed;
      z-index: 8;
      right: clamp(16px, 2vw, 30px);
      top: clamp(16px, 2vw, 30px);
      min-width: 42px;
      height: 24px;
      padding: 0 10px;
      border-radius: 999px;
      border: 1px solid rgba(241, 236, 227, 0.13);
      background:
        linear-gradient(135deg, rgba(255,255,255,0.11), rgba(255,255,255,0.035)),
        rgba(10, 9, 8, 0.34);
      backdrop-filter: blur(10px) saturate(1.12);
      -webkit-backdrop-filter: blur(10px) saturate(1.12);
      box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.10),
        0 7px 16px rgba(0,0,0,0.14);
      font-family: Poppins, "Avenir Next", "Helvetica Neue", Arial, sans-serif;
      font-style: normal;
      font-weight: 700;
      text-transform: none;
      font-size: 9.5px;
      letter-spacing: 0.055em;
      display: grid;
      place-items: center;
      transition: border-color 160ms ease, background 160ms ease, transform 160ms ease;
    }}
    .close-focus:hover {{
      background:
        linear-gradient(135deg, rgba(255,255,255,0.18), rgba(255,255,255,0.055)),
        rgba(14, 12, 10, 0.36);
      transform: scale(1.03);
    }}
    .legend {{
      display: flex;
      gap: 9px;
      flex-wrap: nowrap;
      align-items: center;
      justify-content: center;
      overflow-x: auto;
      padding-bottom: 2px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 5px;
      padding: 5px 6px;
      border-radius: 999px;
      background: rgba(15, 14, 12, 0.54);
      color: var(--muted);
      font-size: 11px;
      white-space: nowrap;
    }}
    .legend::before,
    .legend::after {{
      content: "";
      flex: 0 0 0;
    }}
    .legend-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }}
    @media (max-width: 900px) {{
      body {{ overflow: auto; }}
      .app {{
        min-height: 960px;
        height: auto;
      }}
      .topbar {{
        position: relative;
        left: auto;
        right: auto;
        top: auto;
        width: auto;
        margin: 12px;
      }}
      .controls {{
        width: 100%;
        align-items: stretch;
      }}
      .controls label, .controls button {{
        flex: 1;
      }}
      .stage {{
        position: relative;
        height: 660px;
      }}
      .focus-layer {{
        position: fixed;
        inset: 12px;
      }}
      .focus-shell {{
        width: calc(100vw - 24px);
        height: calc(100vh - 24px);
        flex-direction: column;
        justify-content: flex-start;
        padding-top: 42px;
      }}
      .selected-card {{
        left: auto;
        top: auto;
      }}
      .focus-shell.landscape-focus .selected-card {{
        left: auto;
      }}
      .neighbor-carousel {{
        left: auto;
        right: auto;
        top: auto;
        flex: 0 0 auto;
        width: min(340px, 84vw);
        height: 180px;
        min-height: 180px;
        transform: none;
      }}
    }}
    @media (max-width: 760px) {{
      .app {{
        min-height: 100dvh;
      }}
      .topbar {{
        margin: 10px;
        padding: 14px;
      }}
      .stage {{
        height: 720px;
      }}
      .focus-layer {{
        position: fixed;
        inset: 0;
        display: block;
        overflow: auto;
        padding: 54px 12px 18px;
      }}
      .focus-shell {{
        width: 100%;
        height: 860px;
        flex-direction: column;
        justify-content: flex-start;
        padding-top: 42px;
      }}
      .selected-card {{
        left: auto;
        top: auto;
      }}
      .neighbor-carousel {{
        left: auto;
        right: auto;
        top: auto;
        flex: 0 0 auto;
        width: min(430px, 94vw);
        height: 430px;
        transform: none;
      }}
      .close-focus {{
        right: 14px;
        top: 14px;
      }}
      .lab-row {{
        grid-template-columns: 58px 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="app" id="app">
    <header class="topbar">
      <div class="brand">
        <h1>PALETTE CONSTELLATION</h1>
        <p>Hover for a glimpse. Click a point to bring its nearest palette neighbors into focus.</p>
      </div>
      <div class="read-note">
        <strong>Interpretation note.</strong>
        The projected network is an approximate spatial view. Highlighted links and listed neighbors are computed from the original palette comparison table under the selected metric.
      </div>
      <div class="controls">
        <select id="metric" aria-label="Metric">
          <option value="transport_delta_e">Transport</option>
          <option value="matched_weighted_delta_e">Matched weighted</option>
        </select>
        <input id="search" type="search" aria-label="Find ID" placeholder="Find ID, e.g. 5546">
        <button class="ghost-button" id="shuffle" type="button">RANDOM POINT</button>
        <button class="ghost-button" id="zoomToggle" type="button">ZOOM OUT</button>
      </div>
      <div class="legend" id="legend"></div>
    </header>

    <main class="stage">
      <svg id="constellation" role="img" aria-label="Blurred palette similarity constellation"></svg>
      <div class="hover-card" id="hoverCard"></div>
      <div class="focus-layer" id="focusLayer" aria-live="polite">
        <div class="focus-shell" id="focusShell">
          <button class="close-focus" id="closeFocus" type="button" aria-label="Close focus view">ESC</button>
          <div class="selected-card" id="selectedCard"></div>
          <div class="neighbor-carousel" id="neighborCarousel">
            <div class="orbit-card" id="orbit0"></div>
            <div class="orbit-card" id="orbit1"></div>
            <div class="orbit-card" id="orbit2"></div>
            <div class="orbit-card" id="orbit3"></div>
            <div class="orbit-card" id="orbit4"></div>
            <div class="orbit-card" id="orbit5"></div>
            <div class="orbit-card" id="orbit6"></div>
            <div class="orbit-card" id="orbit7"></div>
            <div class="orbit-card" id="orbit8"></div>
          </div>
        </div>
      </div>
    </main>

  </div>

  <script>
    const DATA = {data_json};
    const app = document.getElementById("app");
    const stage = document.querySelector(".stage");
    const svg = document.getElementById("constellation");
    const metricEl = document.getElementById("metric");
    const searchEl = document.getElementById("search");
    const hoverCard = document.getElementById("hoverCard");
    const selectedCard = document.getElementById("selectedCard");
    const closeFocus = document.getElementById("closeFocus");
    const shuffle = document.getElementById("shuffle");
    const zoomToggle = document.getElementById("zoomToggle");
    const focusLayer = document.getElementById("focusLayer");
    const focusShell = document.getElementById("focusShell");
    const neighborCarousel = document.getElementById("neighborCarousel");
    const orbitEls = Array.from(document.querySelectorAll(".orbit-card"));
    const legendEl = document.getElementById("legend");
    const nodes = DATA.nodes;
    const nodeByImage = new Map(nodes.map(d => [d.image, d]));
    const yearPalette = ["#b88a52", "#d0b174", "#7e8f68", "#a05a49", "#8b7890", "#c87455"];
    const yearColors = new Map([...new Set(nodes.map(d => d.year))].sort().map((year, i) => [year, yearPalette[i % yearPalette.length]]));
    let selectedImage = nodes[0].image;
    let hoverImage = null;
    let pointPositions = new Map();
    let nodeElements = new Map();
    let labelElements = new Map();
    let defsLayer = null;
    let glowLayer = null;
    let focusOpen = false;
    let viewZoomed = true;
    let orbitAnimation = null;
    let focusRenderToken = 0;
    let zoomPanAnimation = null;
    const zoomPan = {{x: 0, y: 0, targetX: 0, targetY: 0, vx: 0, vy: 0, limitX: 0, limitY: 0, active: false}};

    function metricLabel(metric) {{
      return DATA.metrics[metric] || metric;
    }}

    function getNeighbors(image, metric, count=10) {{
      return ((DATA.neighbors[image] || {{}})[metric] || []).slice(0, count);
    }}

    function paletteHTML(palette, cls="palette") {{
      return `<div class="${{cls}}">` + palette.map(s =>
        `<div class="swatch" style="width:${{Math.max(2, s.weight * 100)}}%; background:${{s.hex}}"></div>`
      ).join("") + "</div>";
    }}

    function ordinal(rank) {{
      return ["first", "second", "third"][rank - 1] || `${{rank}}th`;
    }}

    function labHTML(palette) {{
      return `<div class="lab-table">` + palette.map((s, i) =>
        `<div class="lab-row">
          <div class="lab-chip" style="background:${{s.hex}}"></div>
          <div><strong>${{s.hex}}</strong><span>${{s.language}}</span></div>
        </div>`
      ).join("") + "</div>";
    }}

    function mediaStyle(d, mode) {{
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      if (mode === "main") {{
        const compact = vw < 1220 || vh < 760;
        const frameAspect = d.aspect;
        const maxW = d.orientation === "landscape"
          ? Math.min(vw * (compact ? 0.36 : 0.40), compact ? 500 : 560)
          : d.orientation === "square"
            ? Math.min(vw * (compact ? 0.34 : 0.38), compact ? 380 : 430)
            : Math.min(vw * (compact ? 0.34 : 0.38), compact ? 360 : 420);
        const maxH = d.orientation === "landscape"
          ? Math.min(vh * (compact ? 0.31 : 0.36), compact ? 300 : 350)
          : d.orientation === "square"
            ? Math.min(vh * (compact ? 0.42 : 0.46), compact ? 340 : 420)
            : Math.min(vh * (compact ? 0.48 : 0.52), compact ? 360 : 455);
        let width = maxW;
        let height = width / frameAspect;
        if (height > maxH) {{
          height = maxH;
          width = height * frameAspect;
        }}
        width = Math.round(Math.max(compact ? 220 : 260, width));
        height = Math.round(width / frameAspect);
        if (height > maxH) {{
          height = Math.round(maxH);
          width = Math.round(height * frameAspect);
        }}
        return `width:${{width}}px; --media-h:${{height}}px`;
      }}
      const mainMaxH = d.orientation === "portrait" ? Math.min(vh * 0.46, 455) : Math.min(vh * 0.36, 360);
      const mainMaxW = d.orientation === "landscape" ? Math.min(vw * 0.40, 560) : Math.min(vw * 0.58, 820);
      const neighborMaxW = d.orientation === "landscape" ? Math.min(vw * 0.25, 335) : Math.min(vw * 0.21, 285);
      const maxW = mode === "main" ? mainMaxW : mode === "preview" ? Math.min(vw * 0.18, 240) : neighborMaxW;
      const maxH = mode === "main" ? mainMaxH : mode === "preview" ? Math.min(vh * 0.36, 300) : Math.min(vh * 0.42, 360);
      if (mode === "neighbor") {{
        const compact = vw < 1180 || vh < 720;
        const width = Math.round(Math.min(compact ? 272 : 292, Math.max(compact ? 260 : 280, vw * 0.22)));
        const height = Math.round(Math.min(compact ? 330 : 350, Math.max(compact ? 300 : 320, vh * 0.43)));
        return `width:${{width}}px; --media-h:${{height}}px`;
      }}
      let width = maxW;
      let height = width / d.aspect;
      if (height > maxH) {{
        height = maxH;
        width = height * d.aspect;
      }}
      const minW = mode === "main" ? 300 : mode === "preview" ? 118 : 210;
      width = Math.max(minW, Math.round(width));
      return `width:${{width}}px`;
    }}

    function extent(values) {{
      let min = Math.min(...values);
      let max = Math.max(...values);
      if (min === max) {{
        min -= 1;
        max += 1;
      }}
      return [min, max];
    }}

    function scaledPositions(metric) {{
      const rect = svg.getBoundingClientRect();
      const w = Math.max(700, rect.width);
      const h = Math.max(620, rect.height);
      const xs = nodes.map(d => d.coords[metric].x);
      const ys = nodes.map(d => d.coords[metric].y);
      const [minX, maxX] = extent(xs);
      const [minY, maxY] = extent(ys);
      const pad = Math.max(58, Math.min(w, h) * 0.09);
      const sx = x => pad + (x - minX) / (maxX - minX) * (w - 2 * pad);
      const sy = y => h - pad - (y - minY) / (maxY - minY) * (h - 2 * pad);
      return {{w, h, get: d => [sx(d.coords[metric].x), sy(d.coords[metric].y)]}};
    }}

    function clearSvg() {{
      while (svg.firstChild) svg.removeChild(svg.firstChild);
    }}

    function svgEl(name, attrs={{}}) {{
      const el = document.createElementNS("http://www.w3.org/2000/svg", name);
      Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
      return el;
    }}

    function draw() {{
      const metric = metricEl.value;
      const layout = scaledPositions(metric);
      pointPositions = new Map();
      nodeElements = new Map();
      labelElements = new Map();
      clearSvg();
      svg.setAttribute("viewBox", `0 0 ${{layout.w}} ${{layout.h}}`);
      svg.classList.toggle("has-active", Boolean(focusOpen ? selectedImage : null));

      const defs = svgEl("defs");
      defs.innerHTML = `
        <filter id="softBlur"><feGaussianBlur stdDeviation="1.15"></feGaussianBlur></filter>
        <filter id="nodeGlow"><feGaussianBlur stdDeviation="3.4" result="coloredBlur"></feGaussianBlur><feMerge><feMergeNode in="coloredBlur"></feMergeNode><feMergeNode in="SourceGraphic"></feMergeNode></feMerge></filter>
      `;
      svg.appendChild(defs);
      defsLayer = defs;

      const ambient = svgEl("g");
      const glow = svgEl("g");
      const points = svgEl("g");
      const labels = svgEl("g");
      svg.append(ambient, glow, points, labels);
      glowLayer = glow;

      nodes.forEach(d => {{
        const [x, y] = layout.get(d);
        pointPositions.set(d.image, {{x, y}});
      }});

      nodes.forEach(source => {{
        getNeighbors(source.image, metric, 1).forEach(edge => {{
          const a = pointPositions.get(source.image);
          const b = pointPositions.get(edge.target);
          ambient.appendChild(svgEl("line", {{
            class: "ambient-edge",
            x1: a.x,
            y1: a.y,
            x2: b.x,
            y2: b.y,
          }}));
        }});
      }});

      const active = focusOpen ? selectedImage : null;
      if (active) {{
        const a = pointPositions.get(active);
        applyViewportMode(true);
        glow.appendChild(svgEl("circle", {{
          class: "selection-ring",
          cx: a.x,
          cy: a.y,
          r: 54,
        }}));
        getNeighbors(active, metric, 3).forEach((edge, index) => {{
          const b = pointPositions.get(edge.target);
          const gradientId = `grad-${{index}}`;
          const grad = svgEl("linearGradient", {{
            id: gradientId,
            gradientUnits: "userSpaceOnUse",
            x1: a.x,
            y1: a.y,
            x2: b.x,
            y2: b.y,
          }});
          grad.innerHTML = `
            <stop offset="0%" stop-color="rgba(255,232,180,0.95)"></stop>
          <stop offset="36%" stop-color="rgba(230,179,92,0.18)"></stop>
          <stop offset="100%" stop-color="rgba(230,179,92,0.004)"></stop>
          `;
          defs.appendChild(grad);
          glow.appendChild(svgEl("line", {{
            class: "glow-edge visible",
            x1: a.x,
            y1: a.y,
            x2: b.x,
            y2: b.y,
            stroke: `url(#${{gradientId}})`,
          }}));
        }});
      }} else {{
        applyViewportMode(true);
      }}

      const activeNeighbors = active ? new Set(getNeighbors(active, metric, 3).map(n => n.target)) : new Set();
      nodes.forEach(d => {{
        const p = pointPositions.get(d.image);
        const isActive = d.image === active;
        const isNeighbor = activeNeighbors.has(d.image);
        const hit = svgEl("circle", {{
          class: "hit-node",
          cx: p.x,
          cy: p.y,
          r: 7,
          "data-image": d.image,
        }});
        const circle = svgEl("circle", {{
          class: `node${{isActive ? " active" : ""}}${{isNeighbor ? " neighbor" : ""}}`,
          cx: p.x,
          cy: p.y,
          r: isActive ? 6.7 : isNeighbor ? 5.3 : 3.6,
          fill: yearColors.get(d.year),
          "data-image": d.image,
        }});
        points.append(hit, circle);
        nodeElements.set(d.image, circle);

        const label = svgEl("text", {{
          class: `id-label${{isActive || isNeighbor ? " visible" : ""}}`,
          x: p.x,
          y: p.y - 11,
        }});
        label.textContent = d.id;
        labels.appendChild(label);
        labelElements.set(d.image, label);
      }});
    }}

    function setZoomPan(x, y) {{
      svg.style.setProperty("--zoom-pan-x", `${{x.toFixed(2)}}px`);
      svg.style.setProperty("--zoom-pan-y", `${{y.toFixed(2)}}px`);
    }}

    function applyViewportMode(immediate=false) {{
      svg.classList.toggle("is-zoomed", viewZoomed);
      if (zoomToggle) zoomToggle.textContent = viewZoomed ? "ZOOM OUT" : "ZOOM IN";
      if (!viewZoomed) resetZoomPan(immediate);
    }}

    function animateZoomPan() {{
      if (zoomPanAnimation) return;
      const tick = () => {{
        zoomPan.x += (zoomPan.targetX - zoomPan.x) * 0.145;
        zoomPan.y += (zoomPan.targetY - zoomPan.y) * 0.145;
        setZoomPan(zoomPan.x, zoomPan.y);
        if (zoomPan.active || Math.abs(zoomPan.targetX - zoomPan.x) > 0.3 || Math.abs(zoomPan.targetY - zoomPan.y) > 0.3) {{
          zoomPanAnimation = requestAnimationFrame(tick);
        }} else {{
          zoomPan.x = 0;
          zoomPan.y = 0;
          setZoomPan(0, 0);
          zoomPanAnimation = null;
        }}
      }};
      zoomPanAnimation = requestAnimationFrame(tick);
    }}

    function resetZoomPan(immediate=false) {{
      zoomPan.active = false;
      zoomPan.vx = 0;
      zoomPan.vy = 0;
      zoomPan.targetX = 0;
      zoomPan.targetY = 0;
      if (immediate) {{
        if (zoomPanAnimation) cancelAnimationFrame(zoomPanAnimation);
        zoomPanAnimation = null;
        zoomPan.x = 0;
        zoomPan.y = 0;
        setZoomPan(0, 0);
        return;
      }}
      animateZoomPan();
    }}

    function panSignal(value) {{
      const centered = (value - 0.5) / 0.5;
      const deadZone = 0.075;
      const magnitude = Math.max(0, Math.abs(centered) - deadZone) / (1 - deadZone);
      return Math.sign(centered) * Math.pow(magnitude, 1.12);
    }}

    function updateZoomPan(event) {{
      if (focusOpen || !viewZoomed) return;
      const rect = stage.getBoundingClientRect();
      const nx = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
      const ny = Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height));
      zoomPan.limitX = Math.min(980, rect.width * 0.58);
      zoomPan.limitY = Math.min(660, rect.height * 0.50);
      zoomPan.active = true;
      zoomPan.targetX = -panSignal(nx) * zoomPan.limitX;
      zoomPan.targetY = -panSignal(ny) * zoomPan.limitY;
      animateZoomPan();
    }}

    function screenPointFor(image) {{
      const p = pointPositions.get(image);
      if (!p) return null;
      const matrix = svg.getScreenCTM && svg.getScreenCTM();
      if (matrix) {{
        const pt = svg.createSVGPoint();
        pt.x = p.x;
        pt.y = p.y;
        const screen = pt.matrixTransform(matrix);
        return {{x: screen.x, y: screen.y}};
      }}
      const rect = svg.getBoundingClientRect();
      const viewBox = svg.viewBox.baseVal;
      return {{
        x: rect.left + ((p.x - viewBox.x) / viewBox.width) * rect.width,
        y: rect.top + ((p.y - viewBox.y) / viewBox.height) * rect.height,
      }};
    }}

    function nearestImageFromEvent(event) {{
      if (!pointPositions.size) return null;
      let closest = null;
      let closestDist = Infinity;
      nodes.forEach(d => {{
        const screen = screenPointFor(d.image);
        if (!screen) return;
        const dx = screen.x - event.clientX;
        const dy = screen.y - event.clientY;
        const dist = Math.hypot(dx, dy);
        if (dist < closestDist) {{
          closest = d.image;
          closestDist = dist;
        }}
      }});
      const threshold = viewZoomed ? 19 : 13;
      return closestDist <= threshold ? closest : null;
    }}

    function handleStagePointer(event) {{
      if (focusOpen) return;
      const image = nearestImageFromEvent(event);
      updateZoomPan(event);
      if (image) {{
        showHover(image, event);
      }} else if (hoverImage) {{
        hideHover(false);
      }}
    }}

    function freezeZoomPan() {{
      zoomPan.active = false;
      zoomPan.vx = 0;
      zoomPan.vy = 0;
      zoomPan.targetX = zoomPan.x;
      zoomPan.targetY = zoomPan.y;
      if (zoomPanAnimation) cancelAnimationFrame(zoomPanAnimation);
      zoomPanAnimation = null;
      setZoomPan(zoomPan.x, zoomPan.y);
    }}

    function renderActiveOverlay(image) {{
      if (!glowLayer || !defsLayer) return;
      glowLayer.replaceChildren();
      defsLayer.querySelectorAll(".dynamic-gradient").forEach(el => el.remove());
      svg.classList.toggle("has-active", Boolean(image));
      nodeElements.forEach(el => {{
        el.classList.remove("active", "neighbor");
        el.setAttribute("r", "3.6");
      }});
      labelElements.forEach(el => el.classList.remove("visible"));

      if (!image) {{
        applyViewportMode(false);
        return;
      }}

      const metric = metricEl.value;
      const position = pointPositions.get(image);
      const neighbors = getNeighbors(image, metric, 3);
      const neighborSet = new Set(neighbors.map(n => n.target));

      applyViewportMode(false);

      glowLayer.appendChild(svgEl("circle", {{
        class: "selection-ring",
        cx: position.x,
        cy: position.y,
        r: 50,
      }}));

      neighbors.forEach((edge, index) => {{
        const target = pointPositions.get(edge.target);
        const gradientId = `hover-grad-${{index}}`;
        const grad = svgEl("linearGradient", {{
          id: gradientId,
          class: "dynamic-gradient",
          gradientUnits: "userSpaceOnUse",
          x1: position.x,
          y1: position.y,
          x2: target.x,
          y2: target.y,
        }});
        grad.innerHTML = `
          <stop offset="0%" stop-color="rgba(255,232,180,0.86)"></stop>
          <stop offset="42%" stop-color="rgba(230,179,92,0.18)"></stop>
          <stop offset="100%" stop-color="rgba(230,179,92,0.004)"></stop>
        `;
        defsLayer.appendChild(grad);
        glowLayer.appendChild(svgEl("line", {{
          class: "glow-edge visible",
          x1: position.x,
          y1: position.y,
          x2: target.x,
          y2: target.y,
          stroke: `url(#${{gradientId}})`,
        }}));
      }});

      nodeElements.forEach((el, nodeImage) => {{
        const active = nodeImage === image;
        const neighbor = neighborSet.has(nodeImage);
        el.classList.toggle("active", active);
        el.classList.toggle("neighbor", neighbor);
        el.setAttribute("r", active ? "6.7" : neighbor ? "5.3" : "3.6");
      }});
      labelElements.forEach((el, nodeImage) => {{
        el.classList.toggle("visible", nodeImage === image || neighborSet.has(nodeImage));
      }});
    }}

    function showHover(image, event) {{
      if (focusOpen) return;
      const sameImage = hoverImage === image;
      hoverImage = image;
      const d = nodeByImage.get(image);
      if (!sameImage) {{
        hoverCard.style.cssText = mediaStyle(d, "preview");
        hoverCard.innerHTML = `
          <img src="${{d.src}}" alt="${{d.filename}}">
          <div class="hover-meta">
            <strong>${{d.year}}</strong>
            ${{paletteHTML(d.palette, "mini-palette")}}
          </div>`;
        const loadedImage = hoverCard.querySelector("img");
        const pointer = {{clientX: event.clientX, clientY: event.clientY}};
        if (loadedImage) loadedImage.addEventListener("load", () => moveHover(pointer), {{once: true}});
        window.requestAnimationFrame(() => moveHover(pointer));
        renderActiveOverlay(image);
      }}
      moveHover(event);
      hoverCard.classList.add("visible");
    }}

    function moveHover(event) {{
      const margin = 22;
      const cardW = hoverCard.offsetWidth || 190;
      const cardH = hoverCard.offsetHeight || 230;
      const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
      let x = event.clientX + 16;
      let y = event.clientY - cardH - 18;
      if (y < margin) y = event.clientY + 18;
      x = clamp(x, margin, window.innerWidth - cardW - margin);
      y = clamp(y, margin, window.innerHeight - cardH - margin);
      const tooCloseLeft = x <= margin + 2;
      const tooCloseRight = x + cardW >= window.innerWidth - margin - 2;
      if (tooCloseLeft && event.clientX + cardW + 34 <= window.innerWidth - margin) {{
        x = event.clientX + 24;
      }} else if (tooCloseRight && event.clientX - cardW - 24 >= margin) {{
        x = event.clientX - cardW - 24;
      }}
      y = clamp(y, margin, window.innerHeight - cardH - margin);

      const topbar = document.querySelector(".topbar");
      if (topbar && !app.classList.contains("is-focused")) {{
        const block = topbar.getBoundingClientRect();
        const pad = 12;
        const overlaps =
          x < block.right + pad &&
          x + cardW > block.left - pad &&
          y < block.bottom + pad &&
          y + cardH > block.top - pad;
        if (overlaps) {{
          const rightX = block.right + pad;
          const belowY = block.bottom + pad;
          const canMoveRight = rightX + cardW <= window.innerWidth - margin;
          const canMoveBelow = belowY + cardH <= window.innerHeight - margin;
          if (canMoveRight) {{
            x = rightX;
          }} else if (canMoveBelow) {{
            y = belowY;
          }} else {{
            x = clamp(block.right + pad, margin, window.innerWidth - cardW - margin);
            y = clamp(block.bottom + pad, margin, window.innerHeight - cardH - margin);
          }}
        }}
      }}
      hoverCard.style.left = `${{x}}px`;
      hoverCard.style.top = `${{y}}px`;
      updateZoomPan(event);
    }}

    function hideHover(resetPanOnHide=true) {{
      hoverImage = null;
      hoverCard.classList.remove("visible");
      renderActiveOverlay(focusOpen ? selectedImage : null);
      if (resetPanOnHide) resetZoomPan();
    }}

    function selectedHTML(d) {{
      return `
        <div class="image-frame" style="${{mediaStyle(d, "main")}}">
          <img src="${{d.src}}" alt="${{d.filename}}" width="${{d.width}}" height="${{d.height}}" decoding="async" fetchpriority="high" style="aspect-ratio:${{d.aspect}}">
          <div class="card-body">
            <div class="card-title">
              <strong>${{d.year}}</strong>
            </div>
            ${{paletteHTML(d.palette)}}
            ${{labHTML(d.palette)}}
          </div>
        </div>`;
    }}

    function orbitHTML(edge) {{
      const d = nodeByImage.get(edge.target);
      return `
        <div class="neighbor-panel ${{d.orientation}}" data-image="${{d.image}}" style="${{mediaStyle(d, "neighbor")}}">
          <img src="${{d.src}}" alt="${{d.filename}}" width="${{d.width}}" height="${{d.height}}" decoding="async" style="aspect-ratio:${{d.aspect}}">
          <div class="card-body">
            <strong>${{ordinal(edge.rank)}} neighbor · ${{d.year}}</strong>
            <span>transport distance: ${{edge.distance.toFixed(2)}}</span>
            ${{paletteHTML(d.palette, "mini-palette")}}
          </div>
        </div>`;
    }}

    function openFocus(image) {{
      const switching = focusOpen && image !== selectedImage;
      selectedImage = image;
      focusOpen = true;
      freezeZoomPan();
      hoverCard.classList.remove("visible");
      hoverImage = null;
      if (switching) {{
        focusShell.classList.add("is-switching");
        window.setTimeout(() => {{
          renderFocusContent();
          focusShell.classList.remove("is-switching");
          renderActiveOverlay(selectedImage);
        }}, 260);
        return;
      }}
      renderFocusContent();
      app.classList.add("is-focused");
      renderActiveOverlay(selectedImage);
    }}

    function renderFocusContent() {{
      const renderToken = ++focusRenderToken;
      const selected = nodeByImage.get(selectedImage);
      stopOrbit();
      orbitEls.forEach(el => {{
        el.innerHTML = "";
        el.style.opacity = "0";
        el.style.pointerEvents = "none";
        el.style.transform = "translate3d(-9999px, 0, 0)";
      }});
      selectedCard.className = `selected-card ${{selected.orientation}}`;
      focusShell.classList.toggle("landscape-focus", selected.orientation === "landscape");
      selectedCard.innerHTML = selectedHTML(selected);
      const neighbors = getNeighbors(selectedImage, metricEl.value, 3);
      let queuedOrbit = false;
      const queueOrbit = () => {{
        if (queuedOrbit) return;
        queuedOrbit = true;
        window.setTimeout(() => {{
          if (renderToken !== focusRenderToken || !focusOpen) return;
          orbitEls.forEach((el, i) => {{
            const neighbor = neighbors.length ? neighbors[i % neighbors.length] : null;
            el.innerHTML = neighbor ? orbitHTML(neighbor) : "";
            const panel = el.querySelector(".neighbor-panel");
            if (panel) panel.addEventListener("click", () => openFocus(panel.dataset.image));
          }});
          startOrbit();
        }}, 120);
      }};
      const selectedImg = selectedCard.querySelector("img");
      if (selectedImg && !(selectedImg.complete && selectedImg.naturalWidth > 0)) {{
        selectedImg.addEventListener("load", queueOrbit, {{once: true}});
        window.setTimeout(queueOrbit, 450);
      }} else {{
        queueOrbit();
      }}
    }}

    function closeFocusView() {{
      focusOpen = false;
      focusRenderToken += 1;
      app.classList.remove("is-focused");
      stopOrbit();
      renderActiveOverlay(null);
    }}

    function startOrbit() {{
      stopOrbit();
      const rect = neighborCarousel.getBoundingClientRect();
      const visibleEls = orbitEls.filter(el => el.querySelector(".neighbor-panel"));
      if (!visibleEls.length) return;
      const baseCount = Math.min(3, visibleEls.length);
      const gap = 16;
      const speed = Math.max(22, rect.width * 0.028);
      const cardSizes = orbitEls.map(el => ({{
        w: el.offsetWidth || 180,
        h: el.offsetHeight || 240,
      }}));
      const baseSizes = cardSizes.slice(0, baseCount);
      const baseOffsets = [];
      const loopWidth = baseSizes.reduce((x, size, index) => {{
        baseOffsets[index] = x;
        return x + size.w + gap;
      }}, 0);
      const visibleGroupWidth = Math.max(0, loopWidth - gap);
      const startX = Math.max(0, (rect.width - visibleGroupWidth) / 2);
      let groupX = 0;
      let last = performance.now();
      const introStart = last;
      const introDuration = 3000;
      const introDelay = 260;
      const introSlide = 180;
      const easeOut = t => 1 - Math.pow(1 - t, 3);
      const run = now => {{
        const dt = now - last;
        last = now;
        if (!neighborCarousel.matches(":hover")) {{
          groupX += (dt / 1000) * speed;
          if (groupX >= loopWidth) groupX -= loopWidth;
        }}
        orbitEls.forEach((el, i) => {{
          if (!el.querySelector(".neighbor-panel")) return;
          const baseIndex = i % baseCount;
          const setIndex = Math.floor(i / baseCount) - 1;
          const cardH = cardSizes[i].h;
          const x = startX + baseOffsets[baseIndex] + setIndex * loopWidth - groupX;
          const y = rect.height / 2 - cardH / 2;
          el.style.zIndex = String(8 - baseIndex);
          if (baseCount === 3 && now - introStart < introDuration) {{
            const introSlot = [3, 4, 5].indexOf(i);
            if (introSlot < 0) {{
              el.style.opacity = "0";
              el.style.pointerEvents = "none";
              el.style.transform = `translate3d(${{x}}px, ${{y}}px, 0)`;
              return;
            }}
            const raw = Math.min(1, Math.max(0, ((now - introStart) - introSlot * introDelay) / 1250));
            const eased = easeOut(raw);
            el.style.opacity = String(eased);
            el.style.pointerEvents = eased > 0.98 ? "" : "none";
            el.style.transform = `translate3d(${{x + introSlide * (1 - eased)}}px, ${{y}}px, 0)`;
            return;
          }}
          el.style.opacity = "";
          el.style.pointerEvents = "";
          el.style.transform = `translate3d(${{x}}px, ${{y}}px, 0)`;
        }});
        orbitAnimation = requestAnimationFrame(run);
      }};
      orbitAnimation = requestAnimationFrame(run);
    }}

    function stopOrbit() {{
      if (orbitAnimation) cancelAnimationFrame(orbitAnimation);
      orbitAnimation = null;
    }}

    function findAndOpen() {{
      const q = searchEl.value.trim().toLowerCase();
      if (!q) return;
      const hit = nodes.find(d => d.id.includes(q) || d.filename.toLowerCase().includes(q));
      if (hit) openFocus(hit.image);
    }}

    function renderLegend() {{
      legendEl.innerHTML = [...yearColors.entries()].map(([year, color]) =>
        `<div class="legend-item"><span class="legend-dot" style="background:${{color}}"></span>${{year}}</div>`
      ).join("");
    }}

    metricEl.addEventListener("change", () => {{
      draw();
      if (focusOpen) {{
        renderFocusContent();
      }}
    }});
    searchEl.addEventListener("change", findAndOpen);
    searchEl.addEventListener("keydown", event => {{
      if (event.key === "Enter") findAndOpen();
    }});
    closeFocus.addEventListener("click", closeFocusView);
    focusLayer.addEventListener("pointerdown", event => {{
      if (!focusOpen) return;
      event.stopPropagation();
      if (event.target === focusLayer || event.target === focusShell) closeFocusView();
    }});
    selectedCard.addEventListener("pointerdown", event => event.stopPropagation());
    neighborCarousel.addEventListener("pointerdown", event => event.stopPropagation());
    closeFocus.addEventListener("pointerdown", event => event.stopPropagation());
    shuffle.addEventListener("click", () => openFocus(nodes[Math.floor(Math.random() * nodes.length)].image));
    zoomToggle.addEventListener("click", () => {{
      viewZoomed = !viewZoomed;
      applyViewportMode(true);
    }});
    stage.addEventListener("pointermove", handleStagePointer);
    stage.addEventListener("pointerdown", event => {{
      if (focusOpen) return;
      const image = nearestImageFromEvent(event);
      if (image) openFocus(image);
    }});
    stage.addEventListener("mouseleave", hideHover);
    window.addEventListener("keydown", event => {{
      if (event.key === "Escape" && focusOpen) closeFocusView();
    }});
    window.addEventListener("resize", () => {{
      draw();
      if (focusOpen) {{
        renderFocusContent();
        startOrbit();
      }}
    }});

    renderLegend();
    draw();
    applyViewportMode(true);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the cinematic palette constellation HTML view.")
    parser.add_argument("--palettes-csv", type=Path, default=DEFAULT_PALETTES_CSV)
    parser.add_argument("--pairwise-csv", type=Path, default=DEFAULT_PAIRWISE_CSV)
    parser.add_argument("--mds-dir", type=Path, default=DEFAULT_MDS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metrics": METRICS,
        "nodes": build_nodes(args.palettes_csv, args.mds_dir, args.out),
        "neighbors": build_neighbors(args.pairwise_csv, args.top_n),
    }
    args.out.write_text(html_template(data), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
