from __future__ import annotations
from typing import Any, Dict, List, Tuple
import random
import networkx as nx
import numpy as np
from .graph_store import get_store
import community as community_louvain


def _G() -> nx.Graph:
    return get_store().load()


# CENTRALITIES
def centralities_top_k(
    k: int = 10,
    betweenness_k: int = 2000,
    seed: int = 42,
    **kwargs,
) -> Dict[str, List[Dict[str, Any]]]:
    G = _G()
    n = G.number_of_nodes()
    if n == 0:
        return {"degree": [], "closeness": [], "betweenness": [], "pagerank": []}

    deg_cent = nx.degree_centrality(G)
    top_deg = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:k]

    lcc_nodes = max(nx.connected_components(G), key=len)
    H = G.subgraph(lcc_nodes).copy()

    clos = nx.closeness_centrality(H)
    top_clos = sorted(clos.items(), key=lambda x: x[1], reverse=True)[:k]

    btw = nx.betweenness_centrality(H, k=min(betweenness_k, H.number_of_nodes()), seed=seed)
    top_btw = sorted(btw.items(), key=lambda x: x[1], reverse=True)[:k]

    pr = nx.pagerank(G, alpha=0.85)
    top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:k]

    return {
        "degree": [{"node": int(u), "score": float(s)} for u, s in top_deg],
        "closeness(lcc)": [{"node": int(u), "score": float(s)} for u, s in top_clos],
        "betweenness(lcc_approx)": [{"node": int(u), "score": float(s)} for u, s in top_btw],
        "pagerank": [{"node": int(u), "score": float(s)} for u, s in top_pr],
    }


# CLUSTERING
def clustering_stats(samples: int = 5000, seed: int = 42, **kwargs) -> Dict[str, Any]:
    G = _G()
    n = G.number_of_nodes()
    if n == 0:
        return {"avg_clustering_est": 0.0, "transitivity": 0.0, "samples": 0}

    rng = random.Random(seed)
    sample_n = min(samples, n)
    nodes = rng.sample(list(G.nodes()), k=sample_n)
    local = nx.clustering(G, nodes=nodes)
    vals = np.array(list(local.values()), dtype=float)

    return {
        "avg_clustering_est": float(vals.mean()) if len(vals) else 0.0,
        "avg_clustering_std": float(vals.std()) if len(vals) else 0.0,
        "transitivity": float(nx.transitivity(G)),
        "samples": sample_n,
    }


# K-CORE
def k_core_summary(k: int = 10, **kwargs) -> Dict[str, Any]:
    G = _G()
    if G.number_of_nodes() == 0:
        return {"k": k, "k_core_size": 0, "max_core_number": 0}

    core_num = nx.core_number(G)
    max_core = max(core_num.values()) if core_num else 0
    H = nx.k_core(G, k=k) if k <= max_core else nx.Graph()

    return {
        "k": k,
        "k_core_size": H.number_of_nodes(),
        "k_core_edges": H.number_of_edges(),
        "max_core_number": int(max_core),
    }


# COMMUNITIES
def louvain_communities(**kwargs) -> Dict[str, Any]:
    G = _G()
    if G.number_of_nodes() == 0:
        return {"communities": 0, "modularity": 0.0, "top_sizes": []}

    partition = community_louvain.best_partition(G)
    mod = community_louvain.modularity(partition, G)

    sizes: Dict[int, int] = {}
    for _, cid in partition.items():
        sizes[cid] = sizes.get(cid, 0) + 1

    top_sizes = sorted(sizes.items(), key=lambda x: x[1], reverse=True)[:15]

    return {
        "communities": len(sizes),
        "modularity": float(mod),
        "top_sizes": [{"community": int(c), "size": int(sz)} for c, sz in top_sizes],
    }


# ASSORTATIVITY
def degree_assortativity(**kwargs) -> Dict[str, Any]:
    G = _G()
    if G.number_of_nodes() == 0:
        return {"assortativity": 0.0}
    return {"assortativity": float(nx.degree_assortativity_coefficient(G))}


# ARTICULATION POINTS & BRIDGES
import time

_BRIDGE_CACHE: Dict[str, Any] = {
    "art_points": None,
    "bridges": None,
    "n": None,
    "m": None,
    "computed_secs": None,
}


def _ensure_bridge_cache(G: nx.Graph) -> None:
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if (
        _BRIDGE_CACHE["art_points"] is not None
        and _BRIDGE_CACHE["bridges"] is not None
        and _BRIDGE_CACHE["n"] == n
        and _BRIDGE_CACHE["m"] == m
    ):
        return

    t0 = time.time()

    # articulation points
    art = sorted(int(x) for x in nx.articulation_points(G))

    # bridges (normalize order u < v)
    br = []
    for u, v in nx.bridges(G):
        u_i, v_i = int(u), int(v)
        br.append((u_i, v_i) if u_i < v_i else (v_i, u_i))
    br.sort()

    _BRIDGE_CACHE["art_points"] = art
    _BRIDGE_CACHE["bridges"] = br
    _BRIDGE_CACHE["n"] = n
    _BRIDGE_CACHE["m"] = m
    _BRIDGE_CACHE["computed_secs"] = time.time() - t0


def articulation_points_top_k(k: int = 10, **kwargs) -> Dict[str, Any]:
    G = _G()
    if G.number_of_nodes() == 0:
        return {"count": 0, "k": max(1, int(k)), "top": [], "cached_compute_secs": 0.0}

    k = max(1, int(k))
    _ensure_bridge_cache(G)

    art: List[int] = _BRIDGE_CACHE["art_points"] or []
    top = art[:k]

    return {
        "count": len(art),
        "k": k,
        "returned": len(top),
        "top": [{"node": int(u)} for u in top],
        "graph": {"nodes": int(_BRIDGE_CACHE["n"]), "edges": int(_BRIDGE_CACHE["m"])},
        "cached_compute_secs": float(_BRIDGE_CACHE["computed_secs"] or 0.0),
        "note": "Articulation points are nodes whose removal increases the number of connected components.",
    }


def bridges_top_k(k: int = 10, **kwargs) -> Dict[str, Any]:
    G = _G()
    if G.number_of_nodes() == 0:
        return {"count": 0, "k": max(1, int(k)), "top": [], "cached_compute_secs": 0.0}

    k = max(1, int(k))
    _ensure_bridge_cache(G)

    br: List[Tuple[int, int]] = _BRIDGE_CACHE["bridges"] or []
    top = br[:k]

    return {
        "count": len(br),
        "k": k,
        "returned": len(top),
        "top": [{"u": int(u), "v": int(v)} for u, v in top],
        "graph": {"nodes": int(_BRIDGE_CACHE["n"]), "edges": int(_BRIDGE_CACHE["m"])},
        "cached_compute_secs": float(_BRIDGE_CACHE["computed_secs"] or 0.0),
        "note": "Bridges are edges whose removal increases the number of connected components.",
    }


def bridge_summary(**kwargs) -> Dict[str, Any]:
    G = _G()
    if G.number_of_nodes() == 0:
        return {
            "graph": {"nodes": 0, "edges": 0},
            "articulation_points": 0,
            "bridges": 0,
            "cached_compute_secs": 0.0,
        }

    _ensure_bridge_cache(G)

    return {
        "graph": {"nodes": int(_BRIDGE_CACHE["n"]), "edges": int(_BRIDGE_CACHE["m"])},
        "articulation_points": int(len(_BRIDGE_CACHE["art_points"] or [])),
        "bridges": int(len(_BRIDGE_CACHE["bridges"] or [])),
        "cached_compute_secs": float(_BRIDGE_CACHE["computed_secs"] or 0.0),
        "note": "First call computes and caches results; subsequent calls reuse cache (unless the graph changes).",
    }


# GRAPH OVERVIEW
def graph_overview(**kwargs) -> Dict[str, Any]:
    G = _G()
    n = G.number_of_nodes()
    m = G.number_of_edges()
    comps = list(nx.connected_components(G))
    comps_sizes = sorted([len(c) for c in comps], reverse=True)

    return {
        "nodes": n,
        "edges": m,
        "density": nx.density(G),
        "connected_components": len(comps_sizes),
        "largest_component_size": comps_sizes[0] if comps_sizes else 0,
        "largest_component_fraction": (comps_sizes[0] / n) if n else 0.0,
        "avg_degree": (2 * m / n) if n else 0.0,
        "is_connected": nx.is_connected(G) if n > 0 else False,
    }


# EGO NETWORK
def ego_network(u: int, radius: int = 1, **kwargs) -> Dict[str, Any]:
    G = _G()
    if u not in G:
        return {"error": "Node not in graph.", "node": u}

    H = nx.ego_graph(G, u, radius=radius)
    return {
        "node": u,
        "radius": radius,
        "ego_nodes": H.number_of_nodes(),
        "ego_edges": H.number_of_edges(),
        "avg_clustering_in_ego": nx.average_clustering(H) if H.number_of_nodes() > 1 else 0.0,
    }


# NEIGHBORS
def get_node_neighbors(u: Any, limit: int = 20, show_all: bool = False, **kwargs) -> Dict[str, Any]:
    G = _G()

    try:
        node_id = int(u)
    except (ValueError, TypeError):
        return {"error": f"Invalid node ID format: {u}", "node": u}

    if node_id not in G:
        return {"error": f"Node {node_id} not found in the graph.", "node": node_id}

    neighbors_list = list(G.neighbors(node_id))
    total_neighbors = len(neighbors_list)

    if show_all:
        returned_neighbors = neighbors_list
    else:
        returned_neighbors = neighbors_list[:limit]

    return {
        "node": node_id,
        "total_neighbors": total_neighbors,
        "returned_count": len(returned_neighbors),
        "neighbors": [int(n) for n in returned_neighbors],
        "has_more": total_neighbors > len(returned_neighbors)
    }


# FRIEND RECOMMENDATION
def recommend_friends(u: int, k: int = 5, **kwargs) -> Dict[str, Any]:
    G = _G()
    node_id = int(u)

    if node_id not in G:
        return {"error": f"Node {node_id} not found.", "node": node_id}

    neighbors = set(G.neighbors(node_id))
    candidates = [v for v in G.nodes() if v != node_id and v not in neighbors]

    preds = nx.adamic_adar_index(G, [(node_id, v) for v in candidates])

    scored = []
    for u_idx, v_idx, score in preds:
        if score > 0:
            scored.append({"node": int(v_idx), "score": float(score)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    top_recommendations = scored[:k]

    for rec in top_recommendations:
        common = list(nx.common_neighbors(G, node_id, rec["node"]))
        rec["common_neighbors_count"] = len(common)
        rec["common_neighbors_sample"] = common[:5]

    return {
        "target_node": node_id,
        "recommendations": top_recommendations,
        "total_candidates_found": len(scored)
    }


# DIAMETER
def diameter_estimate(samples: int = 50, seed: int = 42, **kwargs) -> Dict[str, Any]:
    G = _G()
    if G.number_of_nodes() == 0:
        return {"diameter_est": 0, "lcc_nodes": 0}

    lcc_nodes = max(nx.connected_components(G), key=len)
    H = G.subgraph(lcc_nodes).copy()
    nodes = list(H.nodes())
    rng = random.Random(seed)

    def bfs_ecc(start: int) -> int:
        lengths = nx.single_source_shortest_path_length(H, start)
        return max(lengths.values())

    diam = 0
    for _ in range(min(samples, len(nodes))):
        s = rng.choice(nodes)
        diam = max(diam, bfs_ecc(s))

    return {
        "diameter_est": int(diam),
        "lcc_nodes": H.number_of_nodes(),
        "samples": min(samples, len(nodes)),
    }


# CONNECTED COMPONENTS
def component_summary(top: int = 10, **kwargs) -> Dict[str, Any]:
    G = _G()
    comps = sorted([len(c) for c in nx.connected_components(G)], reverse=True)
    return {"components": len(comps), "top_sizes": comps[:top]}


# DEGREE
def top_k_by_degree(k: int = 3, **kwargs) -> List[Dict[str, Any]]:
    G = _G()
    deg = sorted(G.degree(), key=lambda x: x[1], reverse=True)[: max(1, k)]
    return [{"node": int(u), "degree": int(d)} for u, d in deg]


# SHORTEST PATH
def shortest_path(u: int, v: int, **kwargs) -> Dict[str, Any]:
    G = _G()
    if u not in G or v not in G:
        return {"error": "One or both nodes not in graph.", "u": u, "v": v}

    try:
        path = nx.shortest_path(G, u, v)
        return {"u": u, "v": v, "length": len(path) - 1, "path": path}
    except nx.NetworkXNoPath:
        return {"u": u, "v": v, "length": None, "path": None}

