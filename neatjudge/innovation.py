"""Innovation tracking -- the historical spine of NEAT.

Every structural change (a new pathway, or a new agent that splits a pathway) is
stamped with a globally-unique, monotonically-increasing *innovation number*. The
SAME structural change occurring in different genomes receives the SAME number,
because the tracker memoizes by structure. This is what makes gene-by-gene
alignment -- and therefore safe crossover and honest topological distance --
possible.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# Reserved node ids for the universal I/O of every judge graph.
INPUT_NODE_ID = 0    # ingests the raw (prompt, response) item under evaluation
OUTPUT_NODE_ID = 1   # emits the final aggregated verdict


class InnovationTracker:
    """Central registry that assigns and *memoizes* innovation numbers.

    Memoization keys:
      * edges       -> keyed by (in_node, out_node)
      * split-nodes -> keyed by the innovation number of the edge being split, so
                       splitting the same historical edge always yields the same
                       new node id and the same two child edges.
    """

    def __init__(self, first_node_id: int = 2):
        self._edge_innovations: Dict[Tuple[int, int], int] = {}
        self._split_nodes: Dict[int, int] = {}      # split-edge innov -> new node id
        self._node_archetype: Dict[int, str] = {}   # node id -> immutable personality
        self._next_innovation = 0
        self._first_node_id = first_node_id
        self._next_node_id = first_node_id

    # ---- edges ----------------------------------------------------------------------

    def get_edge_innovation(self, in_node: int, out_node: int) -> int:
        """Return a stable innovation number for the pathway (in_node -> out_node)."""
        key = (in_node, out_node)
        if key not in self._edge_innovations:
            self._edge_innovations[key] = self._next_innovation
            self._next_innovation += 1
        return self._edge_innovations[key]

    # ---- split nodes ----------------------------------------------------------------

    def get_split_node(self, split_edge_innovation: int, archetype_pool: List[str]) -> int:
        """Return the stable node id created by splitting a given edge.

        The archetype assigned to that node is fixed on first creation and keyed to
        the node id, so the same historical split always produces the same kind of
        specialist -- keeping crossover semantically coherent.
        """
        if split_edge_innovation not in self._split_nodes:
            node_id = self._next_node_id
            self._next_node_id += 1
            self._split_nodes[split_edge_innovation] = node_id
            # Deterministic, structure-derived archetype choice (no RNG dependence,
            # so parallel evolution of the same split agrees on the specialist).
            # Offset by the first mintable id so the earliest split yields the first
            # specialist in the pool (the Safety Arbitrator), regardless of config.
            idx = (node_id - self._first_node_id) % len(archetype_pool)
            self._node_archetype[node_id] = archetype_pool[idx]
        return self._split_nodes[split_edge_innovation]

    def archetype_of(self, node_id: int) -> Optional[str]:
        return self._node_archetype.get(node_id)
