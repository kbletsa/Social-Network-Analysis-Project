import random


def agent_decision_to_adopt(G, node, p_base=0.2):
    """
    Απόφαση agent αν θα υιοθετήσει πληροφορία
    επηρεάζεται από ομοφιλία
    """
    neighbors = G.neighbors(node)
    informed_neighbors = [
        n for n in neighbors if G.nodes[n]["informed"]
    ]

    if not informed_neighbors:
        return False

    same_opinion = sum(
        1 for n in informed_neighbors
        if G.nodes[n]["opinion"] == G.nodes[node]["opinion"]
    )

    homophily_factor = same_opinion / len(informed_neighbors)

    probability = p_base + 0.5 * homophily_factor
    return random.random() < probability
