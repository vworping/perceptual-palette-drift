from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import color


DEFAULT_PALETTES_CSV = Path("data/processed/seed_0/palettes_k5/palettes_all_k5_seed_0.csv")
DEFAULT_PAIRWISE_CSV = Path("data/processed/inter_palette/seed_0_palettes_k5/pairwise_palette_distances.csv")
DEFAULT_MDS_DIR = Path("data/processed/inter_palette/seed_0_palettes_k5")
DEFAULT_OUT = Path("figures/inter_palette/interactive_palette_neighbors.html")

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
    groups = ["".join(chunk) for chunk in __import__("re").findall(r"(\d)(\d)(\d)(\d)", stem)]
    return groups[-1] if groups else stem[-4:]


def build_nodes(palettes_csv: Path, mds_dir: Path, html_out: Path) -> list[dict]:
    palettes = pd.read_csv(palettes_csv)
    matched_mds = pd.read_csv(mds_dir / "mds_embedding_matched_weighted_delta_e.csv")
    transport_mds = pd.read_csv(mds_dir / "mds_embedding_transport_delta_e.csv")
    matched_by_image = matched_mds.set_index("image")[["x", "y"]].to_dict("index")
    transport_by_image = transport_mds.set_index("image")[["x", "y"]].to_dict("index")

    html_dir = html_out.parent
    nodes = []
    for relpath, group in palettes.groupby("relpath", sort=False):
        group = group.sort_values("cluster_rank")
        first = group.iloc[0]
        swatches = []
        for _, row in group.iterrows():
            swatches.append(
                {
                    "weight": float(row["weight"]),
                    "hex": lab_to_hex(float(row["L_star"]), float(row["a_star"]), float(row["b_star"])),
                    "lab": [float(row["L_star"]), float(row["a_star"]), float(row["b_star"])],
                }
            )

        image_path = Path("data") / relpath
        image_src = Path("../../") / image_path
        nodes.append(
            {
                "image": relpath,
                "filename": str(first["filename"]),
                "id": extract_four_digits(str(first["filename"])),
                "year": int(first["year"]),
                "src": image_src.as_posix(),
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
            directed.append(
                {
                    "source": row["image_a"],
                    "target": row["image_b"],
                    "distance": float(row[metric]),
                }
            )
            directed.append(
                {
                    "source": row["image_b"],
                    "target": row["image_a"],
                    "distance": float(row[metric]),
                }
            )

        df = pd.DataFrame(directed).sort_values(["source", "distance", "target"])
        df["rank"] = df.groupby("source").cumcount() + 1
        df = df[df["rank"] <= top_n]
        for source, group in df.groupby("source"):
            neighbors.setdefault(source, {})[metric] = [
                {"target": row["target"], "distance": float(row["distance"]), "rank": int(row["rank"])}
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
  <title>Interactive Palette Neighbors</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #1f2428;
      --muted: #70757a;
      --line: rgba(0, 0, 0, 0.12);
      --accent: #2e6f95;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.35;
    }}
    .app {{
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(620px, 1fr) 390px;
    }}
    main {{
      padding: 18px 18px 22px;
      min-width: 0;
    }}
    aside {{
      background: var(--panel);
      border-left: 1px solid var(--line);
      padding: 18px;
      overflow: auto;
      max-height: 100vh;
      position: sticky;
      top: 0;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 22px;
      letter-spacing: 0;
    }}
    .toolbar {{
      display: flex;
      gap: 12px;
      align-items: end;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    label {{
      display: grid;
      gap: 4px;
      font-size: 12px;
      color: var(--muted);
    }}
    select, input[type="search"], input[type="range"] {{
      font: inherit;
    }}
    select, input[type="search"] {{
      border: 1px solid var(--line);
      border-radius: 7px;
      background: white;
      padding: 7px 8px;
      min-height: 34px;
    }}
    .check {{
      display: flex;
      align-items: center;
      gap: 6px;
      padding-bottom: 7px;
      color: var(--ink);
    }}
    #map {{
      width: 100%;
      height: calc(100vh - 104px);
      min-height: 620px;
      background: #fffefb;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .edge {{
      stroke: rgba(31, 36, 40, 0.14);
      stroke-width: 1.1;
      pointer-events: none;
    }}
    .edge.selected {{
      stroke: rgba(46, 111, 149, 0.50);
      stroke-width: 1.8;
    }}
    .node {{
      cursor: pointer;
      stroke: white;
      stroke-width: 1.1;
    }}
    .node.selected {{
      stroke: #111;
      stroke-width: 2.2;
    }}
    .node.dim {{
      opacity: 0.32;
    }}
    .point-label {{
      font-size: 9px;
      fill: rgba(31, 36, 40, 0.72);
      pointer-events: none;
      text-anchor: middle;
      dominant-baseline: central;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 116px 1fr;
      gap: 12px;
      align-items: start;
      margin-bottom: 14px;
    }}
    .hero img {{
      width: 116px;
      height: 150px;
      object-fit: cover;
      border-radius: 7px;
      background: #eee;
    }}
    .meta h2 {{
      margin: 0 0 4px;
      font-size: 18px;
    }}
    .meta p {{
      margin: 2px 0;
      color: var(--muted);
      font-size: 13px;
      word-break: break-word;
    }}
    .palette {{
      display: flex;
      width: 100%;
      height: 24px;
      overflow: hidden;
      border-radius: 5px;
      border: 1px solid rgba(0,0,0,0.08);
      margin: 9px 0 12px;
    }}
    .swatch {{
      height: 100%;
    }}
    .neighbors {{
      display: grid;
      gap: 9px;
    }}
    .neighbor {{
      display: grid;
      grid-template-columns: 58px 1fr;
      gap: 9px;
      align-items: center;
      padding: 7px;
      border: 1px solid var(--line);
      border-radius: 7px;
      cursor: pointer;
      background: white;
    }}
    .neighbor:hover {{
      border-color: rgba(46,111,149,0.55);
    }}
    .neighbor img {{
      width: 58px;
      height: 72px;
      object-fit: cover;
      border-radius: 5px;
      background: #eee;
    }}
    .neighbor strong {{
      display: block;
      font-size: 13px;
    }}
    .neighbor span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
    }}
    .small-palette {{
      display: flex;
      height: 12px;
      overflow: hidden;
      border-radius: 3px;
      margin-top: 5px;
    }}
    .status {{
      font-size: 12px;
      color: var(--muted);
      margin-left: auto;
      padding-bottom: 8px;
    }}
    @media (max-width: 980px) {{
      .app {{
        grid-template-columns: 1fr;
      }}
      aside {{
        position: static;
        max-height: none;
        border-left: 0;
        border-top: 1px solid var(--line);
      }}
      #map {{
        height: 70vh;
      }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <main>
      <h1>Interactive Palette Neighbor Map</h1>
      <div class="toolbar">
        <label>Metric
          <select id="metric">
            <option value="transport_delta_e">Transport</option>
            <option value="matched_weighted_delta_e">Matched weighted</option>
          </select>
        </label>
        <label>Neighbors per image: <span id="kText">3</span>
          <input id="k" type="range" min="1" max="10" value="3">
        </label>
        <label>Find ID
          <input id="search" type="search" placeholder="e.g. 5546">
        </label>
        <label class="check"><input id="labels" type="checkbox" checked> Show IDs</label>
        <div class="status" id="status"></div>
      </div>
      <svg id="map" role="img" aria-label="Palette neighbor map"></svg>
    </main>
    <aside>
      <div id="details"></div>
    </aside>
  </div>
  <script>
    const DATA = {data_json};
    const nodes = DATA.nodes;
    const nodeByImage = new Map(nodes.map(d => [d.image, d]));
    const yearColors = new Map();
    [...new Set(nodes.map(d => d.year))].sort().forEach((year, i) => {{
      const colors = ["#2f80b9", "#f28e2b", "#4daf4a", "#d84343", "#9067c6", "#8c6257"];
      yearColors.set(year, colors[i % colors.length]);
    }});

    const svg = document.getElementById("map");
    const metricEl = document.getElementById("metric");
    const kEl = document.getElementById("k");
    const kText = document.getElementById("kText");
    const searchEl = document.getElementById("search");
    const labelsEl = document.getElementById("labels");
    const detailsEl = document.getElementById("details");
    const statusEl = document.getElementById("status");
    let selected = nodes[0].image;

    function paletteHTML(palette, small=false) {{
      const cls = small ? "small-palette" : "palette";
      return `<div class="${{cls}}">` + palette.map(s =>
        `<div class="swatch" style="width:${{Math.max(1, s.weight * 100)}}%; background:${{s.hex}}"></div>`
      ).join("") + `</div>`;
    }}

    function metricLabel(metric) {{
      return DATA.metrics[metric] || metric;
    }}

    function getNeighbors(image, metric, k) {{
      return ((DATA.neighbors[image] || {{}})[metric] || []).slice(0, k);
    }}

    function extent(values) {{
      let min = Math.min(...values), max = Math.max(...values);
      if (min === max) {{ min -= 1; max += 1; }}
      return [min, max];
    }}

    function draw() {{
      const metric = metricEl.value;
      const k = Number(kEl.value);
      kText.textContent = k;
      const rect = svg.getBoundingClientRect();
      const w = Math.max(620, rect.width);
      const h = Math.max(560, rect.height);
      svg.setAttribute("viewBox", `0 0 ${{w}} ${{h}}`);
      svg.innerHTML = "";

      const xs = nodes.map(d => d.coords[metric].x);
      const ys = nodes.map(d => d.coords[metric].y);
      const [minX, maxX] = extent(xs);
      const [minY, maxY] = extent(ys);
      const pad = 44;
      const sx = x => pad + (x - minX) / (maxX - minX) * (w - 2 * pad);
      const sy = y => h - pad - (y - minY) / (maxY - minY) * (h - 2 * pad);

      const selectedNeighbors = new Set(getNeighbors(selected, metric, k).map(n => n.target));
      const search = searchEl.value.trim().toLowerCase();
      const searchHits = new Set(nodes.filter(d => search && (d.id.includes(search) || d.filename.toLowerCase().includes(search))).map(d => d.image));

      const edgeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
      svg.appendChild(edgeGroup);
      nodes.forEach(source => {{
        getNeighbors(source.image, metric, k).forEach(edge => {{
          const target = nodeByImage.get(edge.target);
          const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
          line.setAttribute("x1", sx(source.coords[metric].x));
          line.setAttribute("y1", sy(source.coords[metric].y));
          line.setAttribute("x2", sx(target.coords[metric].x));
          line.setAttribute("y2", sy(target.coords[metric].y));
          line.setAttribute("class", source.image === selected || target.image === selected ? "edge selected" : "edge");
          edgeGroup.appendChild(line);
        }});
      }});

      const nodeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
      svg.appendChild(nodeGroup);
      nodes.forEach(d => {{
        const isFocus = d.image === selected || selectedNeighbors.has(d.image) || searchHits.has(d.image);
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", sx(d.coords[metric].x));
        circle.setAttribute("cy", sy(d.coords[metric].y));
        circle.setAttribute("r", d.image === selected ? 7.2 : searchHits.has(d.image) ? 6.8 : 5.3);
        circle.setAttribute("fill", yearColors.get(d.year));
        circle.setAttribute("class", "node" + (d.image === selected ? " selected" : "") + (search && !isFocus ? " dim" : ""));
        circle.addEventListener("click", () => {{ selected = d.image; render(); }});
        nodeGroup.appendChild(circle);

        if (labelsEl.checked || d.image === selected || selectedNeighbors.has(d.image) || searchHits.has(d.image)) {{
          const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
          label.setAttribute("x", sx(d.coords[metric].x));
          label.setAttribute("y", sy(d.coords[metric].y) - 11);
          label.setAttribute("class", "point-label");
          label.textContent = d.id;
          nodeGroup.appendChild(label);
        }}
      }});

      statusEl.textContent = `${{nodes.length}} images | ${{metricLabel(metric)}} | selected ${{nodeByImage.get(selected).year}} ${{nodeByImage.get(selected).id}}`;
    }}

    function renderDetails() {{
      const metric = metricEl.value;
      const k = Number(kEl.value);
      const d = nodeByImage.get(selected);
      const neighbors = getNeighbors(selected, metric, k);
      detailsEl.innerHTML = `
        <div class="hero">
          <img src="${{d.src}}" alt="${{d.filename}}">
          <div class="meta">
            <h2>${{d.year}} ${{d.id}}</h2>
            <p>${{d.filename}}</p>
            <p>${{metricLabel(metric)}} neighbors</p>
            ${{paletteHTML(d.palette)}}
          </div>
        </div>
        <div class="neighbors">
          ${{neighbors.map(n => {{
            const target = nodeByImage.get(n.target);
            return `<div class="neighbor" data-image="${{target.image}}">
              <img src="${{target.src}}" alt="${{target.filename}}">
              <div>
                <strong>#${{n.rank}} · ${{target.year}} ${{target.id}}</strong>
                <span>${{n.distance.toFixed(2)}} · ${{target.filename}}</span>
                ${{paletteHTML(target.palette, true)}}
              </div>
            </div>`;
          }}).join("")}}
        </div>
      `;
      detailsEl.querySelectorAll(".neighbor").forEach(el => {{
        el.addEventListener("click", () => {{ selected = el.dataset.image; render(); }});
      }});
    }}

    function render() {{
      draw();
      renderDetails();
    }}

    metricEl.addEventListener("change", render);
    kEl.addEventListener("input", render);
    searchEl.addEventListener("input", () => {{
      const q = searchEl.value.trim().toLowerCase();
      const hit = nodes.find(d => q && (d.id.includes(q) || d.filename.toLowerCase().includes(q)));
      if (hit) selected = hit.image;
      render();
    }});
    labelsEl.addEventListener("change", render);
    window.addEventListener("resize", render);
    render();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a standalone interactive palette neighbor HTML view.")
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
