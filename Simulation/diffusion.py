from agents import agent_decision_to_adopt


def diffusion_step(G):
    """
    Ένα βήμα διάδοσης πληροφορίας
    """
    new_informed = []

    for node in G.nodes():
        if not G.nodes[node]["informed"]:
            if agent_decision_to_adopt(G, node):
                new_informed.append(node)

    for node in new_informed:
        G.nodes[node]["informed"] = True
