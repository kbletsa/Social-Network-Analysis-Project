import networkx as nx
import random


def create_network(n_agents, k, p):
    """
    Δημιουργεί small-world κοινωνικό δίκτυο
    με παραμέτρους που ορίζει ο χρήστης
    """
    G = nx.watts_strogatz_graph(
        n=n_agents,
        k=k,
        p=p
    )

    for node in G.nodes():
        G.nodes[node]["opinion"] = random.choice([0, 1])
        G.nodes[node]["informed"] = False

    return G
