from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import networkx as nx


@dataclass
class GraphStore:
    data_path: Path
    _G: nx.Graph | None = None

    def load(self) -> nx.Graph:
        if self._G is not None:
            return self._G

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Graph file not found at: {self.data_path}. "
                f"Put facebook_combined.txt under /data."
            )

        G = nx.Graph()
        # facebook_combined.txt.gz: κάθε γραμμή "u v"
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                a, b = line.split()
                G.add_edge(int(a), int(b))

        self._G = G
        return self._G


_STORE: GraphStore | None = None


def get_store() -> GraphStore:
    global _STORE
    if _STORE is None:
        root = Path(__file__).resolve().parent.parent  # sna_project2_agentic/
        _STORE = GraphStore(data_path=root / "data" / "facebook_combined.txt")
    return _STORE
