"""
visualize.py
============
Produces two sets of visualisations for the HMLP project:

1. Interactive HTML graph (pyvis)
   - Full heterogeneous KG with colour-coded node types
   - Anime-only projection subgraph filtered to top-N nodes
   - Highlights known relation-type edges (SEQUEL / PREQUEL / etc.)

2. Static analysis charts (matplotlib)
   - Bar chart: Precision@K by algorithm and K value
   - Bar chart: MAP and AUC-ROC comparison
   - Bar chart: Relation type distribution
   - Heatmap: Per-relation-type Precision@25 per algorithm
   - Degree distribution histogram of projection graph

Usage
-----
    python visualize.py                       # loads results/eval_results.json
    python visualize.py --no-pyvis            # skip interactive graph
    python visualize.py --top-n 80            # show top-80 anime nodes
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

try:
    import networkx as nx  # type: ignore[no-redef]
    HAS_NX = True
except ImportError:
    HAS_NX = False

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import networkx as nx
    from pyvis.network import Network

# Matplotlib (required)
try:
    import matplotlib
    matplotlib.use("Agg")   # headless backend
    import matplotlib.pyplot as plt  # type: ignore[no-redef]
    import matplotlib.patches as mpatches  # type: ignore[no-redef]
    import numpy as np  # type: ignore[no-redef]
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[visualize] WARNING: matplotlib not found – static charts skipped.")

# pyvis (optional)
try:
    from pyvis.network import Network  # type: ignore[no-redef]
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False
    print("[visualize] WARNING: pyvis not found – interactive graph skipped.")

# Project imports
sys.path.insert(0, os.path.dirname(__file__))
from graph_builder import load_and_build, EVAL_RELATION_TYPES, RELATION_GROUPS
from link_prediction import build_all_attribute_sets, build_attr_frequency

# Colour palette
NODE_COLORS = {
    "anime":  "#4C9BE8",   # blue
    "genre":  "#F4A261",   # orange
    "studio": "#2A9D8F",   # teal
    "source": "#E76F51",   # red-orange
}
RELATION_COLORS = {
    "SEQUEL":      "#E63946",
    "PREQUEL":     "#457B9D",
    "SIDE_STORY":  "#A8DADC",
    "SPIN_OFF":    "#F4A261",
    "ALTERNATIVE": "#8338EC",
}
ALG_COLORS = ["#4C9BE8", "#F4A261", "#2A9D8F"]
ALG_NAMES  = ["jaccard", "adamic_adar", "preferential_attachment"]
ALG_LABELS = ["Jaccard", "Adamic-Adar", "Pref. Attach."]

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")


# 1. Interactive pyvis graph

def build_pyvis_hetero(
    anime_list: list[dict],
    meta: dict,
    typed_triples: list[dict],
    top_n: int = 60,
    out_path: Optional[str] = None,
):
    """
    Heterogeneous KG layout:
      - Genre nodes packed as a tight cluster at the center (small random offsets)
      - Franchise clusters of anime nodes arranged around that genre cluster
      - Studio nodes on an outer ring encircling the anime clusters
    Physics disabled. Positions are pre-computed. No rotation.
    """
    if not HAS_PYVIS:
        return

    anime_by_attrs = sorted(
        anime_list,
        key=lambda a: len(a["genres"]) + len(a["studios"]),
        reverse=True,
    )[:top_n]
    top_ids     = {a["id"] for a in anime_by_attrs}
    id_to_anime = {a["id"]: a for a in anime_list}

    # Unique genres / studios in order of first appearance
    genre_set:  list[str] = []
    studio_set: list[str] = []
    seen_g: set[str] = set()
    seen_s: set[str] = set()
    for a in anime_by_attrs:
        for g in a["genres"]:
            if g not in seen_g:
                seen_g.add(g); genre_set.append(g)
        for s in a["studios"]:
            if s not in seen_s:
                seen_s.add(s); studio_set.append(s)

    # Build franchise groups via SEQUEL/PREQUEL connected components
    if HAS_NX:
        G_rel = nx.Graph()
        for a in anime_by_attrs:
            G_rel.add_node(a["id"])
        for tri in typed_triples:
            h = int(tri["head"].replace("anime_", ""))
            t = int(tri["tail"].replace("anime_", ""))
            if h in top_ids and t in top_ids and tri["relation"] in {"SEQUEL", "PREQUEL"}:
                G_rel.add_edge(h, t)
        components = sorted(nx.connected_components(G_rel), key=len, reverse=True)
        franchise_groups: list[list[int]] = [
            sorted(comp, key=lambda i: id_to_anime[i].get("title", ""))
            for comp in components
        ]
        placed = {aid for g in franchise_groups for aid in g}
        singles = [[a["id"]] for a in anime_by_attrs if a["id"] not in placed]
        franchise_groups += singles
    else:
        franchise_groups = [[a["id"]] for a in anime_by_attrs]

    # Layout constants
    GENRE_SPREAD  = 120   # radius of genre cluster (tight dot in center)
    CLUSTER_GAP   = 80    # min distance between franchise cluster centers
    NODE_RADIUS   = 20    # half-size of a single node (vis.js units)

    def polar(r: float, angle: float) -> tuple[float, float]:
        return round(r * math.cos(angle), 1), round(r * math.sin(angle), 1)

    # Place franchise clusters on concentric rings around the genre cluster.
    # Each ring fits as many clusters as possible without overlap, then
    # spills to the next ring outward. Cluster radius scales with group size.
    def cluster_radius(n: int) -> float:
        """Radius of a franchise's internal sub-circle based on member count."""
        if n <= 1:
            return 0.0
        return max(35.0, n * 22.0 / (2 * math.pi))

    # Compute how many clusters fit on each ring
    cluster_radii = [cluster_radius(len(g)) for g in franchise_groups]
    ring_inner = GENRE_SPREAD + CLUSTER_GAP
    cluster_positions: list[tuple[float, float]] = []
    fi = 0
    n_f = len(franchise_groups)
    ring_r = ring_inner
    while fi < n_f:
        cr = cluster_radii[fi]
        # circumference available / space needed per cluster
        circumference = 2 * math.pi * ring_r
        slot = max(cr * 2 + CLUSTER_GAP, 70.0)
        fits = max(1, int(circumference / slot))
        batch = franchise_groups[fi: fi + fits]
        for bi, _ in enumerate(batch):
            angle = 2 * math.pi * bi / max(len(batch), 1) - math.pi / 2
            cluster_positions.append(polar(ring_r, angle))
        fi += len(batch)
        ring_r += max(cr * 2 + CLUSTER_GAP, 80.0)

    # Studio ring sits just outside the outermost anime ring
    # Use a fixed padding of 150px beyond the last anime ring, ignoring
    # studio count multiplier which was pushing the ring too far out
    outermost_anime_r = ring_r - CLUSTER_GAP
    STUDIO_RADIUS = outermost_anime_r + 150

    import random
    rng = random.Random(42)

    net = Network(height="100vh", width="100%", bgcolor="#1a1a2e",
                  notebook=False, directed=False)
    net.set_options(json.dumps({
        "physics": {"enabled": False},
        "nodes":   {"font": {"color": "#e0e0e0"}},
        "interaction": {"zoomView": True, "dragView": True, "dragNodes": True}
    }))

    def anime_tooltip(a: dict) -> str:
        genres  = ", ".join(a.get("genres",  [])) or "None"
        studios = ", ".join(a.get("studios", [])) or "None"
        rels    = ", ".join(
            r["type"] + "\u2192" + str(r["id"]) for r in a.get("relations", [])[:5]
        ) or "None"
        return (
            a.get("title", "Unknown") + "\n"
            + "Year: "      + str(a.get("seasonYear") or "Unknown") + "\n"
            + "Genres: "    + genres    + "\n"
            + "Studios: "   + studios   + "\n"
            + "Source: "    + a.get("source", "Unknown") + "\n"
            + "Relations: " + rels
        )

    # Inner cluster - Genre nodes packed in a tight circle
    genre_nid: dict[str, str] = {}
    n_g = len(genre_set)
    for i, g in enumerate(genre_set):
        nid = f"genre_{g.replace(' ', '_')}"
        genre_nid[g] = nid
        # Tight packing: small circle so genres form a visible cluster
        angle = 2 * math.pi * i / max(n_g, 1)
        r     = GENRE_SPREAD * math.sqrt(i / max(n_g, 1))
        x, y  = polar(r, angle)
        deg   = sum(1 for a in anime_by_attrs if g in a["genres"])
        net.add_node(nid, label=g, color=NODE_COLORS["genre"],
                     size=12, x=x, y=y,
                     title="Genre: " + g + "\nConnections: " + str(deg))

    # Middle - Anime nodes in franchise clusters
    anime_nid: dict[int, str] = {}
    for fi, group in enumerate(franchise_groups):
        cx, cy = cluster_positions[fi]
        cr     = cluster_radii[fi]
        n_in   = len(group)
        for ji, aid in enumerate(group):
            a = id_to_anime.get(aid)
            if a is None:
                continue
            nid = f"anime_{aid}"
            anime_nid[aid] = nid
            if n_in == 1:
                x, y = cx, cy
            else:
                spread_angle = 2 * math.pi * ji / n_in
                x = round(cx + cr * math.cos(spread_angle), 1)
                y = round(cy + cr * math.sin(spread_angle), 1)
            net.add_node(nid, label=a["title"][:28], color=NODE_COLORS["anime"],
                         size=18, x=x, y=y, title=anime_tooltip(a))

    # Outer ring - Studio nodes
    studio_nid: dict[str, str] = {}
    n_s = len(studio_set)
    for i, s in enumerate(studio_set):
        nid = f"studio_{s.replace(' ', '_')}"
        studio_nid[s] = nid
        angle = 2 * math.pi * i / max(n_s, 1) - math.pi / 2
        x, y  = polar(STUDIO_RADIUS, angle)
        deg   = sum(1 for a in anime_by_attrs if s in a["studios"])
        net.add_node(nid, label=s[:22], color=NODE_COLORS["studio"],
                     size=14, x=x, y=y,
                     title="Studio: " + s + "\nConnections: " + str(deg))

    # Edges: anime - genre and anime - studio
    for a in anime_by_attrs:
        an = anime_nid.get(a["id"])
        if an is None:
            continue
        for g in a["genres"]:
            if g in genre_nid:
                net.add_edge(an, genre_nid[g], color="#334466", width=1)
        for s in a["studios"]:
            if s in studio_nid:
                net.add_edge(an, studio_nid[s], color="#224433", width=1)

    # Coloured relation edges
    for tri in typed_triples:
        h = int(tri["head"].replace("anime_", ""))
        t = int(tri["tail"].replace("anime_", ""))
        if h not in top_ids or t not in top_ids:
            continue
        net.add_edge(
            tri["head"], tri["tail"],
            color=RELATION_COLORS.get(tri["relation"], "#ffffff"),
            width=3, title=tri["relation"],
            dashes=(tri["relation"] in {"ALTERNATIVE", "SIDE_STORY"}),
        )

    if out_path is None:
        out_path = os.path.join(OUT_DIR, "graph_hetero.html")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    net.save_graph(out_path)

    # Post-process: legend + collision JS
    with open(out_path, encoding="utf-8") as f:
        html = f.read()

    node_rows = "".join(
        f"<div style=\'display:flex;align-items:center;gap:6px;margin:2px 0\'>"
        f"<div style=\'width:11px;height:11px;border-radius:50%;background:{col};flex-shrink:0\'></div>"
        f"<span style=\'font-size:11px;color:#e0e0e0\'>{lbl.capitalize()}</span></div>"
        for lbl, col in NODE_COLORS.items()
    )
    rel_rows = "".join(
        f"<div style=\'display:flex;align-items:center;gap:6px;margin:2px 0\'>"
        f"<div style=\'width:14px;height:3px;background:{col};flex-shrink:0;border-radius:2px\'></div>"
        f"<span style=\'font-size:11px;color:#e0e0e0\'>{lbl}</span></div>"
        for lbl, col in RELATION_COLORS.items()
    )
    legend = (
        "<div style=\'position:fixed;top:14px;right:14px;"
        "background:rgba(10,22,40,0.93);border:1px solid #1B6CA8;"
        "border-radius:8px;padding:10px 14px;z-index:9999;"
        "pointer-events:none;min-width:130px;\'>"
        "<div style=\'font-size:11px;font-weight:bold;color:#0D9E8F;margin-bottom:5px;\'>Node Types</div>"
        + node_rows
        + "<div style=\'font-size:11px;font-weight:bold;color:#0D9E8F;margin:7px 0 5px;\'>Relations</div>"
        + rel_rows
        + "</div>"
    )

    collision_js = """
<script>
(function () {
    var D = 55; // min center-to-center distance

    function resolveCollisions(pos, ids) {
        for (var it = 0; it < 150; it++) {
            var moved = false;
            for (var i = 0; i < ids.length; i++) {
                for (var j = i + 1; j < ids.length; j++) {
                    var a = pos[ids[i]], b = pos[ids[j]];
                    var dx = b.x - a.x, dy = b.y - a.y;
                    var d  = Math.sqrt(dx*dx + dy*dy);
                    if (d < D && d > 0.01) {
                        var push = (D - d) / 2 + 1;
                        var nx2 = dx/d*push, ny2 = dy/d*push;
                        pos[ids[i]].x -= nx2; pos[ids[i]].y -= ny2;
                        pos[ids[j]].x += nx2; pos[ids[j]].y += ny2;
                        moved = true;
                    }
                }
            }
            if (!moved) break;
        }
        ids.forEach(function (id) { network.moveNode(id, pos[id].x, pos[id].y); });
    }

    // Resolve collisions only for the dragged node and its nearby neighbours.
    // This runs on every drag frame so it must be fast - limit to nodes
    // within 2*D of the dragged node rather than the full graph.
    function resolveNear(draggedId) {
        var all = network.getPositions();
        var dragged = all[draggedId];
        if (!dragged) return;
        var nearby = [draggedId];
        var ids = Object.keys(all);
        for (var k = 0; k < ids.length; k++) {
            if (ids[k] === draggedId) continue;
            var dx = all[ids[k]].x - dragged.x;
            var dy = all[ids[k]].y - dragged.y;
            if (Math.sqrt(dx*dx + dy*dy) < D * 3) nearby.push(ids[k]);
        }
        if (nearby.length < 2) return;
        var sub = {};
        for (var n = 0; n < nearby.length; n++) sub[nearby[n]] = { x: all[nearby[n]].x, y: all[nearby[n]].y };
        resolveCollisions(sub, nearby);
    }

    document.addEventListener("DOMContentLoaded", function () {
        setTimeout(function () {
            if (typeof network === "undefined") return;

            // Initial pass on load
            var pos = network.getPositions();
            var ids = Object.keys(pos);
            resolveCollisions(pos, ids);
            network.fit();

            // Real-time collision during drag (throttled to every 30ms)
            var lastDrag = 0;
            network.on("dragging", function (params) {
                if (!params.nodes || params.nodes.length === 0) return;
                var now = Date.now();
                if (now - lastDrag < 30) return;
                lastDrag = now;
                resolveNear(String(params.nodes[0]));
            });

            // Final full pass on release to clean up anything missed
            network.on("dragEnd", function (params) {
                if (!params.nodes || params.nodes.length === 0) return;
                var pos2 = network.getPositions();
                var ids2 = Object.keys(pos2);
                resolveCollisions(pos2, ids2);
            });
        }, 300);
    });
})();
</script>"""

    html = html.replace("</body>", legend + collision_js + "</body>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[visualize] Saved interactive KG  -> {out_path}")
    return out_path


def build_pyvis_projection(
    anime_list: list[dict],
    proj_adj: dict[int, dict[int, int]],
    relation_triples: list[dict],
    top_n: int = 80,
    out_path: Optional[str] = None,
):
    """
    Interactive anime-only projection graph.
    Node size = degree.  Relation edges drawn in colour on top.
    """
    if not HAS_PYVIS:
        return

    # Pick top_n by degree in projection
    degrees = {aid: len(nbrs) for aid, nbrs in proj_adj.items()}
    top_ids = set(
        sorted(degrees, key=lambda x: degrees[x], reverse=True)[:top_n]
    )

    id_to_title = {a["id"]: a["title"] for a in anime_list}

    net = Network(
        height="100vh", width="100%",
        bgcolor="#1a1a2e", font_color="#e0e0e0",  # type: ignore[arg-type]
        notebook=False,
    )
    net.set_options(json.dumps({
        "physics": {
            "solver": "repulsion",
            "repulsion": {
                "nodeDistance": 120,
                "centralGravity": 0.1,
                "springLength": 150,
                "springConstant": 0.05,
                "damping": 0.5
            },
            "stabilization": {
                "enabled": True,
                "iterations": 1500,
                "updateInterval": 100,
                "fit": True
            },
            "minVelocity": 1.0,
            "maxVelocity": 50
        }
    }))

    for aid in top_ids:
        deg   = degrees.get(aid, 0)
        title = id_to_title.get(aid, str(aid))
        net.add_node(
            aid, label=title[:25],
            color=NODE_COLORS["anime"],
            size=8 + min(deg / 5, 20),
            title=f"{title}\nDegree: {deg}",
        )

    # Attribute-similarity edges
    drawn: set[frozenset] = set()
    for aid in top_ids:
        for nbr, weight in proj_adj.get(aid, {}).items():
            if nbr not in top_ids:
                continue
            pair = frozenset([aid, nbr])
            if pair in drawn:
                continue
            drawn.add(pair)
            net.add_edge(aid, nbr, color="#334455",
                         width=max(1, weight / 3),
                         title=f"Shared attributes: {weight}")

    # Overlay relation edges in colour
    rel_pairs: set[frozenset] = set()
    for triple in relation_triples:
        h, t = triple["head"], triple["tail"]
        if h not in top_ids or t not in top_ids:
            continue
        pair = frozenset([h, t])
        if pair in rel_pairs:
            continue
        rel_pairs.add(pair)
        color = RELATION_COLORS.get(triple["relation"], "#ffffff")
        net.add_edge(h, t, color=color, width=4,
                     title=triple["relation"])

    if out_path is None:
        out_path = os.path.join(OUT_DIR, "graph_projection.html")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    net.save_graph(out_path)

    with open(out_path, encoding="utf-8") as f:
        html = f.read()

    # No legend overlay for projection graph
    collision_js = """
<script>
(function () {
    var D = 55; // min center-to-center distance

    function resolveCollisions(pos, ids) {
        for (var it = 0; it < 150; it++) {
            var moved = false;
            for (var i = 0; i < ids.length; i++) {
                for (var j = i + 1; j < ids.length; j++) {
                    var a = pos[ids[i]], b = pos[ids[j]];
                    var dx = b.x - a.x, dy = b.y - a.y;
                    var d  = Math.sqrt(dx*dx + dy*dy);
                    if (d < D && d > 0.01) {
                        var push = (D - d) / 2 + 1;
                        var nx2 = dx/d*push, ny2 = dy/d*push;
                        pos[ids[i]].x -= nx2; pos[ids[i]].y -= ny2;
                        pos[ids[j]].x += nx2; pos[ids[j]].y += ny2;
                        moved = true;
                    }
                }
            }
            if (!moved) break;
        }
        ids.forEach(function (id) { network.moveNode(id, pos[id].x, pos[id].y); });
    }

    // Resolve collisions only for the dragged node and its nearby neighbours.
    // This runs on every drag frame so it must be fast - limit to nodes
    // within 2*D of the dragged node rather than the full graph.
    function resolveNear(draggedId) {
        var all = network.getPositions();
        var dragged = all[draggedId];
        if (!dragged) return;
        var nearby = [draggedId];
        var ids = Object.keys(all);
        for (var k = 0; k < ids.length; k++) {
            if (ids[k] === draggedId) continue;
            var dx = all[ids[k]].x - dragged.x;
            var dy = all[ids[k]].y - dragged.y;
            if (Math.sqrt(dx*dx + dy*dy) < D * 3) nearby.push(ids[k]);
        }
        if (nearby.length < 2) return;
        var sub = {};
        for (var n = 0; n < nearby.length; n++) sub[nearby[n]] = { x: all[nearby[n]].x, y: all[nearby[n]].y };
        resolveCollisions(sub, nearby);
    }

    document.addEventListener("DOMContentLoaded", function () {
        setTimeout(function () {
            if (typeof network === "undefined") return;

            // Initial pass on load
            var pos = network.getPositions();
            var ids = Object.keys(pos);
            resolveCollisions(pos, ids);
            network.fit();

            // Real-time collision during drag (throttled to every 30ms)
            var lastDrag = 0;
            network.on("dragging", function (params) {
                if (!params.nodes || params.nodes.length === 0) return;
                var now = Date.now();
                if (now - lastDrag < 30) return;
                lastDrag = now;
                resolveNear(String(params.nodes[0]));
            });

            // Final full pass on release to clean up anything missed
            network.on("dragEnd", function (params) {
                if (!params.nodes || params.nodes.length === 0) return;
                var pos2 = network.getPositions();
                var ids2 = Object.keys(pos2);
                resolveCollisions(pos2, ids2);
            });
        }, 300);
    });
})();
</script>"""
    html = html.replace("</body>", collision_js + "</body>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[visualize] Saved projection graph -> {out_path}")
    return out_path


# 2. Static matplotlib analysis charts
def _save(fig, name: str):
    path = os.path.join(OUT_DIR, name)
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[visualize] Saved chart -> {path}")
    return path


def chart_precision_at_k(metrics: dict) -> str:
    """Grouped bar chart: Precision@K for each algorithm."""
    if not HAS_MPL:
        return ""

    k_keys = [k for k in next(iter(metrics.values())) if k.startswith("precision@")]
    k_labels = [k.replace("precision@", "K=") for k in k_keys]
    x = np.arange(len(k_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    for i, (alg, color, label) in enumerate(zip(ALG_NAMES, ALG_COLORS, ALG_LABELS)):
        vals = [metrics[alg].get(k, 0) for k in k_keys]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8, color="white")

    ax.set_xlabel("K value", color="white")
    ax.set_ylabel("Precision@K", color="white")
    ax.set_title("Precision@K by Algorithm", color="white", fontsize=13, pad=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(k_labels, color="white")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#555555")
    ax.spines["left"].set_color("#555555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#2a2a3e", labelcolor="white", framealpha=0.8)
    ax.set_ylim(0, min(1.0, max(v for m in metrics.values()
                                for v in m.values() if isinstance(v, float)) * 1.3 + 0.05))

    return _save(fig, "chart_precision_at_k.png")


def chart_map_auc(metrics: dict) -> str:
    """Bar chart: MAP and AUC-ROC per algorithm."""
    if not HAS_MPL:
        return ""

    alg_labels_short = ["Jaccard", "Adamic-Adar", "Pref. Attach."]
    map_vals = [metrics[a].get("MAP", 0)     for a in ALG_NAMES]
    auc_vals = [metrics[a].get("AUC-ROC", 0) for a in ALG_NAMES]
    x = np.arange(len(ALG_NAMES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    b1 = ax.bar(x - width / 2, map_vals, width, label="MAP",     color="#4C9BE8", alpha=0.88)
    b2 = ax.bar(x + width / 2, auc_vals, width, label="AUC-ROC", color="#F4A261", alpha=0.88)

    for bar, v in list(zip(b1, map_vals)) + list(zip(b2, auc_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, color="white")

    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8, label="Chance (AUC=0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(alg_labels_short, color="white")
    ax.tick_params(colors="white")
    ax.set_ylabel("Score", color="white")
    ax.set_title("MAP and AUC-ROC Comparison", color="white", fontsize=13, pad=12)
    ax.spines["bottom"].set_color("#555555")
    ax.spines["left"].set_color("#555555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#2a2a3e", labelcolor="white", framealpha=0.8)
    ax.set_ylim(0, 1.05)

    return _save(fig, "chart_map_auc.png")


def chart_relation_distribution(relation_summary: dict) -> str:
    """Horizontal bar chart of relation type counts."""
    if not HAS_MPL:
        return ""

    type_counts = relation_summary.get("type_counts", {})
    if not type_counts:
        return ""

    labels = list(type_counts.keys())
    values = list(type_counts.values())
    colors = [RELATION_COLORS.get(l, "#aaaaaa") for l in labels]

    sorted_pairs = sorted(zip(values, labels, colors), reverse=True)
    values, labels, colors = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    bars = ax.barh(labels, values, color=colors, alpha=0.88)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", color="white", fontsize=9)

    ax.set_xlabel("Count", color="white")
    ax.set_title("Relation Type Distribution in Dataset", color="white", fontsize=13, pad=12)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#555555")
    ax.spines["left"].set_color("#555555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    return _save(fig, "chart_relation_distribution.png")


def chart_per_relation_heatmap(per_relation_type: dict, metric: str = "precision@25") -> str:
    """Heatmap: per-relation-type metric score across algorithms."""
    if not HAS_MPL:
        return ""

    rel_types = sorted(per_relation_type.keys())
    if not rel_types:
        return ""

    data = np.zeros((len(rel_types), len(ALG_NAMES)))
    for i, rtype in enumerate(rel_types):
        for j, alg in enumerate(ALG_NAMES):
            data[i, j] = per_relation_type[rtype].get(alg, {}).get(metric, 0)

    fig, ax = plt.subplots(
        figsize=(8, max(3, len(rel_types) * 0.8)), facecolor="#1a1a2e"
    )
    ax.set_facecolor("#1a1a2e")

    im = ax.imshow(data, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors="white")
    cbar.set_label(metric, color="white")

    ax.set_xticks(range(len(ALG_NAMES)))
    ax.set_xticklabels(ALG_LABELS, color="white", rotation=20, ha="right")
    ax.set_yticks(range(len(rel_types)))
    ax.set_yticklabels(rel_types, color="white")
    ax.tick_params(colors="white")

    for i in range(len(rel_types)):
        for j in range(len(ALG_NAMES)):
            val = data[i, j]
            text_color = "white" if val > 0.35 else "#111111"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    ax.set_title(f"Per-Relation {metric} by Algorithm", color="white", fontsize=13, pad=12)
    fig.tight_layout()

    return _save(fig, "chart_per_relation_heatmap.png")


def chart_degree_distribution(proj_adj: dict, anime_list: list[dict]) -> str:
    """Histogram of degree distribution in the projection graph."""
    if not HAS_MPL:
        return ""

    degrees = [len(nbrs) for nbrs in proj_adj.values()]
    if not degrees:
        return ""

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    ax.hist(degrees, bins=30, color="#4C9BE8", edgecolor="#1a1a2e", alpha=0.85)
    ax.axvline(sum(degrees) / len(degrees), color="#F4A261",
               linestyle="--", linewidth=1.5,
               label=f"Mean = {sum(degrees)/len(degrees):.1f}")

    ax.set_xlabel("Degree (number of similar anime)", color="white")
    ax.set_ylabel("Count", color="white")
    ax.set_title("Degree Distribution - Projection Graph", color="white", fontsize=13, pad=12)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#555555")
    ax.spines["left"].set_color("#555555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#2a2a3e", labelcolor="white", framealpha=0.8)

    return _save(fig, "chart_degree_distribution.png")


def chart_recall_at_k(metrics: dict) -> str:
    """Grouped bar chart: Recall@K for each algorithm."""
    if not HAS_MPL:
        return ""

    k_keys   = [k for k in next(iter(metrics.values())) if k.startswith("recall@")]
    k_labels = [k.replace("recall@", "K=") for k in k_keys]
    x = np.arange(len(k_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    for i, (alg, color, label) in enumerate(zip(ALG_NAMES, ALG_COLORS, ALG_LABELS)):
        vals = [metrics[alg].get(k, 0) for k in k_keys]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8, color="white")

    ax.set_xlabel("K value", color="white")
    ax.set_ylabel("Recall@K", color="white")
    ax.set_title("Recall@K by Algorithm", color="white", fontsize=13, pad=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(k_labels, color="white")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#555555")
    ax.spines["left"].set_color("#555555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#2a2a3e", labelcolor="white", framealpha=0.8)
    ax.set_ylim(0, 1.05)

    return _save(fig, "chart_recall_at_k.png")


# 3. Entry point

def main():
    parser = argparse.ArgumentParser(description="HMLP Visualiser")
    parser.add_argument("--cache",       default=None)
    parser.add_argument("--min-weight",  type=int,   default=4)
    parser.add_argument("--top-n",       type=int,   default=70,
                        help="Max anime nodes to show in interactive graph")
    parser.add_argument("--no-pyvis",    action="store_true",
                        help="Skip interactive HTML graphs")
    parser.add_argument("--results",     default=None,
                        help="Path to eval_results.json from evaluate.py")
    args = parser.parse_args()

    # Load graphs
    (G_hetero, _, meta, typed_triples,
     _, proj_adj, relation_triples,
     anime_list, id_to_title) = load_and_build(
         cache_file=args.cache, min_weight=args.min_weight
     )

    # Load evaluation results if available
    results_path = args.results or os.path.join(OUT_DIR, "eval_results.json")
    eval_data = {}
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        print(f"[visualize] Loaded evaluation results from {results_path}")
    else:
        print(f"[visualize] No eval_results.json found at {results_path}. "
              "Run evaluate.py first for metric charts.")

    # Interactive graphs
    if not args.no_pyvis:
        build_pyvis_hetero(
            anime_list, meta, typed_triples,
            top_n=args.top_n,
        )
        build_pyvis_projection(
            anime_list, proj_adj, relation_triples,
            top_n=args.top_n,
        )

    # Static charts
    if HAS_MPL:
        # Degree distribution (no eval results needed)
        chart_degree_distribution(proj_adj, anime_list)

        # Relation distribution (no eval results needed)
        rel_counts: dict[str, int] = defaultdict(int)
        for t in relation_triples:
            rel_counts[t["relation"]] += 1
        rel_summary = {
            "type_counts": dict(rel_counts),
            "total_eval_triples": len(relation_triples),
        }
        chart_relation_distribution(rel_summary)

        # Metric charts (require eval_results.json)
        if eval_data:
            metrics = eval_data.get("metrics", {})
            if metrics:
                chart_precision_at_k(metrics)
                chart_recall_at_k(metrics)
                chart_map_auc(metrics)

            per_rel = eval_data.get("per_relation_type", {})
            if per_rel:
                chart_per_relation_heatmap(per_rel)

    print(f"\n[visualize] All outputs saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()