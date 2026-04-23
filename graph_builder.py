"""
graph_builder.py
================
Constructs two graphs from the collected anime dataset:

1. Heterogeneous Knowledge Graph
   Nodes : Anime, Genre, Studio, Source
   Edges : Anime -[HAS_GENRE]-> Genre
           Anime -[PRODUCED_BY]-> Studio
           Anime -[ADAPTED_FROM]-> Source
           Anime -[SEQUEL/PREQUEL/SPIN_OFF/SIDE_STORY/ALTERNATIVE]-> Anime

   Each Anime->Anime edge now carries a typed 'relation' attribute so the
   graph is a proper KG with named triples <head, relation, tail>.

2. Anime-only projection
   Two anime are connected if they share >= min_weight attributes (genre,
   studio, source). Official relation edges are excluded from the projection
   so they remain a clean held-out evaluation target.

Both graphs are represented as NetworkX graphs AND plain adjacency dicts
for algorithm use without a NetworkX dependency.
"""

import json
import os
from collections import defaultdict
from typing import Optional

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("[graph_builder] WARNING: NetworkX not found – adjacency dict only.")

# Node ID helpers
def anime_id(aid: int)   -> str: return f"anime_{aid}"
def genre_id(g: str)     -> str: return f"genre_{g.replace(' ', '_')}"
def studio_id(s: str)    -> str: return f"studio_{s.replace(' ', '_')}"
def source_id(src: str)  -> str: return f"source_{src}"

# Relation types we treat as evaluation targets (Anime->Anime edges).
# Grouped by semantic class for the relation-type prediction task.
RELATION_GROUPS = {
    "SEQUEL":      "continuation",   # direct chronological next
    "PREQUEL":     "continuation",   # direct chronological prior
    "SIDE_STORY":  "side",           # set in same universe, non-main
    "SPIN_OFF":    "side",           # derived series, new cast
    "ALTERNATIVE": "alternative",    # same story, different adaptation
    "PARENT":      "alternative",    # parent story / franchise root
    "SUMMARY":     "other",
    "COMPILATION": "other",
    "ADAPTATION":  "other",
    "SOURCE":      "other",
}
EVAL_RELATION_TYPES = {"SEQUEL", "PREQUEL", "SIDE_STORY", "SPIN_OFF", "ALTERNATIVE"}


# Heterogeneous KG
def build_graph(anime_list: list[dict]):
    """
    Build the full heterogeneous knowledge graph.

    Returns
    -------
    G        : nx.DiGraph with node_type and (for A->A edges) relation attrs,
               or None if NetworkX is unavailable.
    adj_dict : dict[str, set[str]]  – undirected adjacency for heuristics.
    meta     : dict  node-type -> set of node IDs.
    typed_relation_edges : list[dict]  each entry is
                           {"head": anime_id, "relation": str, "tail": anime_id}
                           Only edges whose relation is in EVAL_RELATION_TYPES
                           AND whose tail is within the dataset are included.
    """
    adj: dict[str, set[str]]  = defaultdict(set)
    node_types: dict[str, str] = {}
    typed_relation_edges: list[dict] = []

    known_ids = {a["id"] for a in anime_list}

    def add_edge(u: str, v: str):
        adj[u].add(v)
        adj[v].add(u)

    def register(node: str, ntype: str):
        node_types[node] = ntype

    for anime in anime_list:
        a_node = anime_id(anime["id"])
        register(a_node, "anime")

        for g in anime["genres"]:
            g_node = genre_id(g)
            register(g_node, "genre")
            add_edge(a_node, g_node)

        for s in anime["studios"]:
            s_node = studio_id(s)
            register(s_node, "studio")
            add_edge(a_node, s_node)

        src_node = source_id(anime["source"])
        register(src_node, "source")
        add_edge(a_node, src_node)

        # Typed Anime->Anime relation edges
        for rel in anime["relations"]:
            if rel["id"] not in known_ids:
                continue
            neighbor = anime_id(rel["id"])
            register(neighbor, "anime")
            add_edge(a_node, neighbor)   # undirected copy for heuristics

            if rel["type"] in EVAL_RELATION_TYPES:
                typed_relation_edges.append({
                    "head":     a_node,
                    "relation": rel["type"],
                    "tail":     neighbor,
                    "group":    RELATION_GROUPS.get(rel["type"], "other"),
                })

    G = None
    if HAS_NX:
        G = nx.DiGraph()
        for node, ntype in node_types.items():
            G.add_node(node, node_type=ntype)
        # Attribute edges (undirected, added both ways)
        for anime in anime_list:
            a_node = anime_id(anime["id"])
            for g in anime["genres"]:
                G.add_edge(a_node, genre_id(g),   relation="HAS_GENRE")
                G.add_edge(genre_id(g), a_node,   relation="HAS_GENRE")
            for s in anime["studios"]:
                G.add_edge(a_node, studio_id(s),  relation="PRODUCED_BY")
                G.add_edge(studio_id(s), a_node,  relation="PRODUCED_BY")
            G.add_edge(a_node, source_id(anime["source"]), relation="ADAPTED_FROM")
            G.add_edge(source_id(anime["source"]), a_node, relation="ADAPTED_FROM")
        # Typed Anime->Anime edges (directed)
        for triple in typed_relation_edges:
            G.add_edge(triple["head"], triple["tail"],
                       relation=triple["relation"],
                       group=triple["group"])

    meta = {
        "anime":  {n for n, t in node_types.items() if t == "anime"},
        "genre":  {n for n, t in node_types.items() if t == "genre"},
        "studio": {n for n, t in node_types.items() if t == "studio"},
        "source": {n for n, t in node_types.items() if t == "source"},
    }

    # Count distinct relation types in the dataset
    rel_counts: dict[str, int] = defaultdict(int)
    for t in typed_relation_edges:
        rel_counts[t["relation"]] += 1

    print(
        f"[graph_builder] Heterogeneous KG - "
        f"{len(node_types)} nodes, {sum(len(v) for v in adj.values()) // 2} edges"
    )
    print(f"[graph_builder] Typed relation edge counts: {dict(rel_counts)}")
    print(f"[graph_builder] Distinct relation types   : {len(rel_counts)}")

    return G, dict(adj), meta, typed_relation_edges


# Anime-only projection
def build_projection_graph(
    anime_list: list[dict],
    min_weight: int = 4,
):
    """
    Project the KG onto anime nodes only using shared genres and studios.

    Source medium (MANGA / LIGHT_NOVEL / ORIGINAL …) is intentionally
    excluded from the projection weight because it is too coarse: ~42% of
    all anime share "MANGA", which would add +1 to nearly half of all pairs
    and inflate the density far above what is useful for link prediction.
    Source is still a node/edge type in the full heterogeneous KG; it is
    simply not used as a projection similarity signal.

    Official Anime->Anime relation edges are NOT included in the projection
    so they remain a clean held-out set for evaluation.

    Edge weight = number of shared genre nodes + shared studio nodes.
    Default min_weight=4 yields density ~0.04-0.07 for a 500-anime dataset.

    Returns
    -------
    G_proj           : nx.Graph with 'weight' edge attribute (or None).
    proj_adj         : dict[int, dict[int, int]]  node -> {neighbor: weight}
    relation_triples : list[dict]  all typed triples within the dataset.
    """
    genre_to_anime:  dict[str, set[int]] = defaultdict(set)
    studio_to_anime: dict[str, set[int]] = defaultdict(set)
    # source is kept in the heterogeneous KG but NOT used in the projection
    # to avoid inflating density (see docstring above).

    known_ids = {a["id"] for a in anime_list}

    for anime in anime_list:
        for g in anime["genres"]:
            genre_to_anime[g].add(anime["id"])
        for s in anime["studios"]:
            studio_to_anime[s].add(anime["id"])

    edge_weights: dict[tuple[int, int], int] = defaultdict(int)

    def bump(a: int, b: int):
        key = (min(a, b), max(a, b))
        edge_weights[key] += 1

    for group in genre_to_anime.values():
        ids = list(group)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                bump(ids[i], ids[j])

    for group in studio_to_anime.values():
        ids = list(group)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                bump(ids[i], ids[j])

    # Source intentionally omitted here (see docstring).

    # Collect typed relation triples (evaluation targets, NOT added to projection)
    relation_triples: list[dict] = []
    for anime in anime_list:
        for rel in anime["relations"]:
            if rel["id"] in known_ids and rel["type"] in EVAL_RELATION_TYPES:
                relation_triples.append({
                    "head":     anime["id"],
                    "relation": rel["type"],
                    "tail":     rel["id"],
                    "group":    RELATION_GROUPS.get(rel["type"], "other"),
                })

    # Build adjacency
    proj_adj: dict[int, dict[int, int]] = defaultdict(dict)
    for (a, b), w in edge_weights.items():
        if w >= min_weight:
            proj_adj[a][b] = w
            proj_adj[b][a] = w

    for anime in anime_list:
        if anime["id"] not in proj_adj:
            proj_adj[anime["id"]] = {}

    G_proj = None
    if HAS_NX:
        G_proj = nx.Graph()
        id_to_title = {a["id"]: a["title"] for a in anime_list}
        for anime in anime_list:
            G_proj.add_node(anime["id"], title=anime["title"])
        for (a, b), w in edge_weights.items():
            if w >= min_weight:
                G_proj.add_edge(a, b, weight=w)

    n_edges = sum(1 for w in edge_weights.values() if w >= min_weight)
    print(
        f"[graph_builder] Projection graph (min_weight={min_weight}) - "
        f"{len(proj_adj)} nodes, {n_edges} edges"
    )
    print(f"[graph_builder] Evaluation triples available: {len(relation_triples)}")

    return G_proj, dict(proj_adj), relation_triples


# Convenience loader
def load_and_build(cache_file: Optional[str] = None, min_weight: int = 4):
    """
    Load anime JSON and build both graphs in one call.

    Returns
    -------
    G_hetero, hetero_adj, meta, typed_triples,
    G_proj, proj_adj, relation_triples, anime_list, id_to_title
    """
    if cache_file is None:
        cache_file = os.path.join(os.path.dirname(__file__), "data", "anime_data.json")

    with open(cache_file, "r", encoding="utf-8") as f:
        anime_list = json.load(f)

    print(f"[graph_builder] Loaded {len(anime_list)} anime.")

    id_to_title = {a["id"]: a["title"] for a in anime_list}

    G_hetero, hetero_adj, meta, typed_triples = build_graph(anime_list)
    G_proj, proj_adj, relation_triples = build_projection_graph(
        anime_list, min_weight=min_weight
    )

    return (
        G_hetero, hetero_adj, meta, typed_triples,
        G_proj, proj_adj, relation_triples,
        anime_list, id_to_title,
    )


def graph_stats(proj_adj: dict, relation_triples: list[dict]) -> dict:
    """Compute basic statistics on the projection graph and relation triples."""
    degrees    = [len(nbrs) for nbrs in proj_adj.values()]
    num_nodes  = len(degrees)
    num_edges  = sum(degrees) // 2
    avg_degree = sum(degrees) / num_nodes if num_nodes else 0
    density    = (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0

    rel_counts: dict[str, int] = defaultdict(int)
    for t in relation_triples:
        rel_counts[t["relation"]] += 1

    return {
        "num_nodes":        num_nodes,
        "num_edges":        num_edges,
        "avg_degree":       round(avg_degree, 2),
        "max_degree":       max(degrees) if degrees else 0,
        "min_degree":       min(degrees) if degrees else 0,
        "density":          round(density, 4),
        "eval_triples":     len(relation_triples),
        "relation_counts":  dict(rel_counts),
    }


def save_graph(
    anime_list: list[dict],
    proj_adj: dict[int, dict[int, int]],
    relation_triples: list[dict],
    out_path: Optional[str] = None,
) -> str:
    """
    Serialise the projection graph and relation triples to JSON so that
    evaluate.py can load them directly without rebuilding.

    Saved keys
    ----------
    anime_list        : original anime dicts (needed by the classifier)
    proj_adj          : {str(id): {str(id): weight}} (JSON keys must be str)
    relation_triples  : list of typed triple dicts
    id_to_title       : {str(id): title} convenience lookup
    """
    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), "data", "graph_data.json")

    payload = {
        "anime_list":       anime_list,
        "proj_adj":         {str(k): {str(n): w for n, w in v.items()}
                             for k, v in proj_adj.items()},
        "relation_triples": relation_triples,
        "id_to_title":      {str(a["id"]): a["title"] for a in anime_list},
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[graph_builder] Graph data saved to {out_path}")
    return out_path


def load_graph(path: Optional[str] = None) -> tuple:
    """
    Load the pre-built graph data saved by save_graph().

    Returns
    -------
    anime_list, proj_adj, relation_triples, id_to_title
    proj_adj keys are restored to int.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "data", "graph_data.json")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    anime_list       = payload["anime_list"]
    relation_triples = payload["relation_triples"]
    id_to_title      = {int(k): v for k, v in payload["id_to_title"].items()}
    proj_adj         = {
        int(k): {int(n): w for n, w in nbrs.items()}
        for k, nbrs in payload["proj_adj"].items()
    }

    print(f"[graph_builder] Loaded graph data from {path}")
    print(f"[graph_builder] {len(anime_list)} anime, "
          f"{sum(len(v) for v in proj_adj.values()) // 2} projection edges, "
          f"{len(relation_triples)} relation triples")

    return anime_list, proj_adj, relation_triples, id_to_title


if __name__ == "__main__":
    (_, _, _, typed_triples,
     _, proj_adj, relation_triples,
     anime_list, _) = load_and_build()

    stats = graph_stats(proj_adj, relation_triples)
    print("\n--- Projection Graph Statistics ---")
    for k, v in stats.items():
        print(f"  {k:<22} {v}")

    save_graph(anime_list, proj_adj, relation_triples)
