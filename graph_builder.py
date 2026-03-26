"""
Constructs two graphs from the collected anime dataset:

1. Heterogeneous graph  – nodes: Anime, Genre, Studio, Source
                          edges: Anime→Genre, Anime→Studio,
                                 Anime→Source, Anime→Anime (relations)

2. Anime-only projection – two anime connected if they share ≥1 attribute;
                           edge weight = number of shared attributes.

Both graphs are represented as NetworkX graphs AND as plain adjacency dicts
(for algorithm use without a NetworkX dependency if needed).
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

# e.g.  anime_123   genre_Action   studio_Madhouse   source_MANGA
def anime_id(aid: int)     -> str: return f"anime_{aid}"
def genre_id(g: str)       -> str: return f"genre_{g.replace(' ', '_')}"
def studio_id(s: str)      -> str: return f"studio_{s.replace(' ', '_')}"
def source_id(src: str)    -> str: return f"source_{src}"


def build_graph(anime_list: list[dict]):
    """
    Build a heterogeneous graph with four node types.

    Returns
    -------
    G        : nx.Graph (or None if NetworkX unavailable)
    adj_dict : dict[str, set[str]] – plain adjacency list
    meta     : dict with node-type → set of node IDs
    """
    adj: dict[str, set[str]] = defaultdict(set)
    node_types: dict[str, str] = {}   # node_id → type label

    known_ids = {a["id"] for a in anime_list}

    def add_edge(u: str, v: str):
        adj[u].add(v)
        adj[v].add(u)

    def register(node: str, ntype: str):
        node_types[node] = ntype

    for anime in anime_list:
        a_node = anime_id(anime["id"])
        register(a_node, "anime")

        # Anime → Genre edges
        for g in anime["genres"]:
            g_node = genre_id(g)
            register(g_node, "genre")
            add_edge(a_node, g_node)

        # Anime → Studio edges
        for s in anime["studios"]:
            s_node = studio_id(s)
            register(s_node, "studio")
            add_edge(a_node, s_node)

        # Anime → Source edge
        src_node = source_id(anime["source"])
        register(src_node, "source")
        add_edge(a_node, src_node)

        # Anime → Anime relation edges (only within our dataset)
        for rel in anime["relations"]:
            if rel["id"] in known_ids:
                neighbor = anime_id(rel["id"])
                register(neighbor, "anime")
                add_edge(a_node, neighbor)

    G = None
    if HAS_NX:
        G = nx.Graph()
        for node, ntype in node_types.items():
            G.add_node(node, node_type=ntype)
        for u, neighbors in adj.items():
            for v in neighbors:
                G.add_edge(u, v)

    meta = {
        "anime":  {n for n, t in node_types.items() if t == "anime"},
        "genre":  {n for n, t in node_types.items() if t == "genre"},
        "studio": {n for n, t in node_types.items() if t == "studio"},
        "source": {n for n, t in node_types.items() if t == "source"},
    }

    print(
        f"[graph_builder] Heterogeneous graph — "
        f"{len(node_types)} nodes, {sum(len(v) for v in adj.values()) // 2} edges"
    )
    return G, dict(adj), meta


def build_projection_graph(anime_list: list[dict], include_official_relations: bool = True):
    """
    Project the heterogeneous graph onto anime nodes only.

    Two anime are connected if they share at least one of:
      - a genre
      - a studio
      - a source type
      - an official relation (optional, toggle with flag)

    Edge weight = count of shared attributes.

    Returns
    -------
    G_proj       : nx.Graph with 'weight' edge attribute (or None)
    proj_adj     : dict[str, dict[str, int]]  node → {neighbor: weight}
    relation_edges : set of frozensets – the true anime↔anime relation edges
                     (used later for the link-prediction evaluation task)
    """
    # Index lookup structures
    genre_to_anime:  dict[str, set[int]] = defaultdict(set)
    studio_to_anime: dict[str, set[int]] = defaultdict(set)
    source_to_anime: dict[str, set[int]] = defaultdict(set)

    known_ids = {a["id"] for a in anime_list}

    for anime in anime_list:
        for g in anime["genres"]:
            genre_to_anime[g].add(anime["id"])
        for s in anime["studios"]:
            studio_to_anime[s].add(anime["id"])
        source_to_anime[anime["source"]].add(anime["id"])

    # weight accumulator: (id_a, id_b) → weight
    edge_weights: dict[tuple[int, int], int] = defaultdict(int)

    def bump(a: int, b: int):
        key = (min(a, b), max(a, b))
        edge_weights[key] += 1

    # Shared genres
    for group in genre_to_anime.values():
        ids = list(group)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                bump(ids[i], ids[j])

    # Shared studios
    for group in studio_to_anime.values():
        ids = list(group)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                bump(ids[i], ids[j])

    # Shared source
    for group in source_to_anime.values():
        ids = list(group)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                bump(ids[i], ids[j])

    # Collect official relation edges (before optionally adding them to proj)
    relation_edges: set[frozenset] = set()
    for anime in anime_list:
        for rel in anime["relations"]:
            if rel["id"] in known_ids:
                relation_edges.add(frozenset([anime["id"], rel["id"]]))

    if include_official_relations:
        for fs in relation_edges:
            a, b = list(fs)
            bump(a, b)

    # Build adjacency dict  {anime_id → {neighbor_id → weight}}
    proj_adj: dict[int, dict[int, int]] = defaultdict(dict)
    for (a, b), w in edge_weights.items():
        proj_adj[a][b] = w
        proj_adj[b][a] = w

    # Ensure all anime appear (even isolated ones)
    for anime in anime_list:
        if anime["id"] not in proj_adj:
            proj_adj[anime["id"]] = {}

    G_proj = None
    if HAS_NX:
        G_proj = nx.Graph()
        for anime in anime_list:
            G_proj.add_node(anime["id"], title=anime["title"])
        for (a, b), w in edge_weights.items():
            G_proj.add_edge(a, b, weight=w)

    print(
        f"[graph_builder] Projection graph — "
        f"{len(proj_adj)} nodes, {len(edge_weights)} edges, "
        f"{len(relation_edges)} official-relation edges"
    )
    return G_proj, dict(proj_adj), relation_edges


def load_and_build(cache_file: Optional[str] = None):
    """
    Load anime data from JSON cache and build both graphs.
    Returns (G_hetero, hetero_adj, meta, G_proj, proj_adj, relation_edges, anime_list)
    """
    if cache_file is None:
        cache_file = os.path.join(os.path.dirname(__file__), "data", "anime_data.json")

    with open(cache_file, "r", encoding="utf-8") as f:
        anime_list = json.load(f)

    print(f"[graph_builder] Loaded {len(anime_list)} anime.")

    G_hetero, hetero_adj, meta = build_graph(anime_list)
    G_proj,   proj_adj,   relation_edges = build_projection_graph(anime_list)

    return G_hetero, hetero_adj, meta, G_proj, proj_adj, relation_edges, anime_list


def graph_stats(proj_adj: dict) -> dict:
    """Compute basic statistics on the projection graph."""
    degrees     = [len(nbrs) for nbrs in proj_adj.values()]
    num_nodes   = len(degrees)
    num_edges   = sum(degrees) // 2
    avg_degree  = sum(degrees) / num_nodes if num_nodes else 0
    max_degree  = max(degrees) if degrees else 0
    min_degree  = min(degrees) if degrees else 0
    density     = (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0

    return {
        "num_nodes":  num_nodes,
        "num_edges":  num_edges,
        "avg_degree": round(avg_degree, 2),
        "max_degree": max_degree,
        "min_degree": min_degree,
        "density":    round(density, 4),
    }


if __name__ == "__main__":
    _, _, _, _, proj_adj, rel_edges, _ = load_and_build()
    stats = graph_stats(proj_adj)
    print("\n--- Projection Graph Statistics ---")
    for k, v in stats.items():
        print(f"  {k:<20} {v}")
    print(f"  {'official_rel_edges':<20} {len(rel_edges)}")