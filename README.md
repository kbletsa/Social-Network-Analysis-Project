# Social Network Analysis & Information Diffusion

This project explores Social Network Analysis (SNA) through two core components: an **Agentic AI Chatbot** for analyzing social graphs using natural language, and an **Agent-Based Simulation** for studying information diffusion within a network.

## Part 1: Agentic Graph Analysis Chatbot

An intelligent system capable of interacting with a social graph (Facebook dataset) via natural language queries. It utilizes a Large Language Model (LLM) to dynamically select and execute Python-based graph algorithms .

### Architecture
* **`agent.py`**: Manages the agent and LLM integration .
* **`tools_graph.py`**: Contains the implementation of analysis algorithms .
* **`graph_store.py`**: Handles efficient graph loading and memory management .
* **`callback.py`**: Logs session data .

### Key Capabilities
The agent can answer questions regarding:
* **Structure:** Node/edge counts, density, components, and diameter .
* **Centrality:** Degree, Closeness, Betweenness, and PageRank measures .
* **Community Detection:** Implements the **Louvain algorithm** and k-core analysis .
* **Connectivity:** Shortest paths, articulation points, and bridge edges .
* **Friend Recommendation:** Suggests connections using the **Adamic Adar** algorithm .
* **Local Analysis:** Ego networks and clustering coefficients .

---

## Part 2: Information Diffusion Simulation

An agent-based simulation that models how information spreads through a social network based on structure and social influence .

### Simulation Logic
* **Network Model:** Generates a **Watts-Strogatz small-world network** .
* **Propagation Rule:** Uses **homophily**, where agents adopt information if a sufficient number of neighbors share similar opinions .

### Parameters
Users can configure the simulation with:
* **`k` (Neighbors):** Determines the density of the network (must be even) .
* **`p` (Randomness):** The probability of edge rewiring (0.0 - 1.0). Higher values increase the "small-world" effect .
* **Iterations:** The number of time steps for the simulation .

### Output
* **Diffusion Reach:** Tracks the percentage of informed agents over time .
* **Visualization:** Displays the spread curve and the final network state, coloring nodes based on their informed status .
