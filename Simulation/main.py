import random
import matplotlib.pyplot as plt
import networkx as nx

from network import create_network
from diffusion import diffusion_step
from metrics import homophily_index, diffusion_reach


def run_simulation():
    print("=== Social Network Diffusion Simulation ===")

    n_agents = 60

    # User input
    k = int(input("Give k (number of neighbours, e.g. 4, 6, 8 even number): "))
    p = float(input("Give p (randomness, e.g. 0.0 – 1.0): "))

    if k >= n_agents:
        raise ValueError("Τhe k must be smaller than the number of agents")
    if k % 2 != 0:
        raise ValueError("The k must be even number")

    steps = int(input("How many steps of diffusion for the simulation? (e.g. 10): "))

    # Network
    G = create_network(n_agents=n_agents, k=k, p=p)

    seed = random.choice(list(G.nodes()))
    G.nodes[seed]["informed"] = True

    print(f"\nSeed agent: {seed}")
    print(f"Initial homophily: {homophily_index(G):.3f}\n")

    reach_over_time = []

    # Simulation
    for t in range(steps):
        diffusion_step(G)
        reach = diffusion_reach(G)
        reach_over_time.append(reach)
        print(f"Step {t+1}: diffusion reach = {reach:.2f}")


    # Plot diffusion over time
    plt.figure()
    plt.plot(range(1, steps + 1), reach_over_time, marker='o')
    plt.xlabel("Time step")
    plt.ylabel("Diffusion reach")
    plt.title(f"Diffusion (k={k}, p={p})")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show(block=False)

    # Plot network graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    colors = [
        "red" if G.nodes[n]["informed"] else "lightgray"
        for n in G.nodes()
    ]

    nx.draw(G, pos, node_color=colors, node_size=120, edge_color="gray")
    plt.title("Final network state (red = informed)")
    plt.show()


if __name__ == "__main__":
    run_simulation()

