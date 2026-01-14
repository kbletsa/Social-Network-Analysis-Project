def homophily_index(G):
    same = 0
    total = G.number_of_edges()

    for u, v in G.edges():
        if G.nodes[u]["opinion"] == G.nodes[v]["opinion"]:
            same += 1

    return same / total if total > 0 else 0


def diffusion_reach(G):
    return sum(
        1 for n in G.nodes()
        if G.nodes[n]["informed"]
    ) / G.number_of_nodes()
