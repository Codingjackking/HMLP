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
import os
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

# TYPE_CHECKING block gives Pylance the real types for autocomplete/checking
# without actually importing at runtime when the packages may be missing.
if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
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
    Interactive heterogeneous KG limited to top_n anime by degree
    (most connected) to keep the visualisation readable.
    """
    if not HAS_PYVIS:
        return

    # Pick top_n most-connected anime by attribute count
    anime_by_attrs = sorted(
        anime_list,
        key=lambda a: len(a["genres"]) + len(a["studios"]),
        reverse=True,
    )[:top_n]
    top_ids = {a["id"] for a in anime_by_attrs}

    net = Network(
        height="750px", width="100%",
        bgcolor="#1a1a2e",
        notebook=False, directed=False,
    )
    net.force_atlas_2based(gravity=-50, central_gravity=0.01,
                           spring_length=120, spring_strength=0.05)

    added_nodes: set = set()

    def add_node(node_id: str, label: str, ntype: str, size: int = 15):
        if node_id not in added_nodes:
            net.add_node(
                node_id, label=label,
                color=NODE_COLORS.get(ntype, "#aaaaaa"),
                size=size,
                title=f"Type: {ntype}\n{label}",
            )
            added_nodes.add(node_id)

    for anime in anime_by_attrs:
        a_node = f"anime_{anime['id']}"
        add_node(a_node, anime["title"][:30], "anime", size=20)

        for g in anime["genres"]:
            g_node = f"genre_{g.replace(' ', '_')}"
            add_node(g_node, g, "genre", size=12)
            net.add_edge(a_node, g_node, color="#555555", width=1)

        for s in anime["studios"]:
            s_node = f"studio_{s.replace(' ', '_')}"
            add_node(s_node, s[:20], "studio", size=14)
            net.add_edge(a_node, s_node, color="#555555", width=1)

    # Draw typed relation edges between top_n anime
    for triple in typed_triples:
        head_id = int(triple["head"].replace("anime_", ""))
        tail_id = int(triple["tail"].replace("anime_", ""))
        if head_id not in top_ids or tail_id not in top_ids:
            continue
        color = RELATION_COLORS.get(triple["relation"], "#ffffff")
        net.add_edge(
            triple["head"], triple["tail"],
            color=color, width=3,
            title=triple["relation"],
            dashes=(triple["relation"] in {"ALTERNATIVE", "SIDE_STORY"}),
        )

    # Legend as isolated nodes in a corner
    legend_items = list(NODE_COLORS.items()) + list(RELATION_COLORS.items())
    for i, (label, color) in enumerate(legend_items):
        lid = f"__legend_{i}"
        net.add_node(lid, label=label, color=color, size=8,
                     x=-900, y=-400 + i * 40, physics=False,
                     font={"size": 10})

    if out_path is None:
        out_path = os.path.join(OUT_DIR, "graph_hetero.html")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    net.save_graph(out_path)
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
        height="750px", width="100%",
        bgcolor="#1a1a2e", font_color="#e0e0e0",  # type: ignore[arg-type]
        notebook=False,
    )
    net.force_atlas_2based(gravity=-30, central_gravity=0.005,
                           spring_length=100, spring_strength=0.04)

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
            # Dark blue cells (high values) get white text
            # Light blue / near-zero cells get dark text
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