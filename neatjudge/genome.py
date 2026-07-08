"""Genome -- a complete agent communication graph (one candidate topology).

A :class:`Genome` is a directed acyclic graph of judge agents. It holds node genes
and connection genes keyed by innovation number, and implements everything NEAT
needs: structural + textual mutation, feed-forward evaluation, compatibility
distance, and (static) innovation-aligned crossover.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from typing import Dict, List

from .archetypes import ARCHETYPE_LIBRARY, SPECIALIST_POOL
from .genes import ConnectionGene, NodeGene, NodeType, default_weight_text
from .innovation import INPUT_NODE_ID, OUTPUT_NODE_ID, InnovationTracker
from .llm import LLMClient


class Genome:
    def __init__(self, tracker: InnovationTracker, genome_id: int):
        self.tracker = tracker
        self.genome_id = genome_id
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[int, ConnectionGene] = {}   # keyed by innovation number
        self.fitness: float = 0.0
        self.adjusted_fitness: float = 0.0
        self.safety_accuracy: float = 0.0    # set by FitnessEvaluator.evaluate
        self.quality_accuracy: float = 0.0

    # ---- construction ---------------------------------------------------------------

    @classmethod
    def base_genome(cls, tracker: InnovationTracker, genome_id: int) -> "Genome":
        """The minimal viable judge: INPUT -> Core Judge(OUTPUT), one pathway.

        This is NEAT's uniform initial genome -- the single "core Judge node" from
        which all topological complexity is grown.
        """
        g = cls(tracker, genome_id)
        g.nodes[INPUT_NODE_ID] = NodeGene(
            INPUT_NODE_ID, NodeType.INPUT, "Ingestor",
            "You are the intake node. Pass the item through unchanged.", 0.0,
        )
        core = ARCHETYPE_LIBRARY["Core Judge"]
        g.nodes[OUTPUT_NODE_ID] = NodeGene(
            OUTPUT_NODE_ID, NodeType.OUTPUT, core.core, core.base_instruction, 0.30,
        )
        innov = tracker.get_edge_innovation(INPUT_NODE_ID, OUTPUT_NODE_ID)
        g.connections[innov] = ConnectionGene(
            innov, INPUT_NODE_ID, OUTPUT_NODE_ID, default_weight_text(1.0), 1.0, True,
        )
        return g

    def clone(self) -> "Genome":
        clone = Genome(self.tracker, self.genome_id)
        clone.nodes = {nid: n.clone() for nid, n in self.nodes.items()}
        clone.connections = {i: c.clone() for i, c in self.connections.items()}
        clone.fitness = self.fitness
        clone.safety_accuracy = self.safety_accuracy
        clone.quality_accuracy = self.quality_accuracy
        return clone

    # ---- topology queries -----------------------------------------------------------

    def enabled_connections(self) -> List[ConnectionGene]:
        return [c for c in self.connections.values() if c.enabled]

    def _would_create_cycle(self, src: int, dst: int) -> bool:
        """True if adding src->dst would introduce a cycle (dst reaches src)."""
        if src == dst:
            return True
        stack, seen = [dst], set()
        adjacency: Dict[int, List[int]] = defaultdict(list)
        for c in self.enabled_connections():
            adjacency[c.in_node].append(c.out_node)
        while stack:
            node = stack.pop()
            if node == src:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adjacency[node])
        return False

    def enforce_acyclic(self) -> int:
        """Repair the feed-forward (DAG) invariant, disabling offending edges.

        Individual mutations preserve acyclicity, but crossover recombines edges
        from two parents whose innovation numbers differ -- e.g. A->B from one and
        B->A from the other (each acyclic in isolation, since edge innovations are
        keyed globally by (in, out)). Their union can form a cycle.

        We walk edges in innovation-number order (deterministic => reproducible)
        and keep an edge only if it does not close a cycle among the edges kept so
        far; otherwise we *disable* it (never delete -- the gene survives for future
        crossover, faithful to NEAT). Returns the number of edges disabled.
        """
        kept: Dict[int, List[int]] = defaultdict(list)

        def reaches(src: int, dst: int) -> bool:
            stack, seen = [src], set()
            while stack:
                node = stack.pop()
                if node == dst:
                    return True
                if node in seen:
                    continue
                seen.add(node)
                stack.extend(kept[node])
            return False

        disabled = 0
        for innov in sorted(self.connections):
            conn = self.connections[innov]
            if not conn.enabled:
                continue
            if conn.in_node == conn.out_node or reaches(conn.out_node, conn.in_node):
                conn.enabled = False
                disabled += 1
            else:
                kept[conn.in_node].append(conn.out_node)
        return disabled

    def topological_order(self) -> List[int]:
        """Kahn topological sort over enabled edges (graph is guaranteed acyclic)."""
        indeg: Dict[int, int] = {nid: 0 for nid in self.nodes}
        adjacency: Dict[int, List[int]] = defaultdict(list)
        for c in self.enabled_connections():
            adjacency[c.in_node].append(c.out_node)
            indeg[c.out_node] += 1
        queue = sorted([nid for nid, d in indeg.items() if d == 0])
        order: List[int] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for nxt in sorted(adjacency[node]):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)
        for nid in self.nodes:   # append any disconnected nodes for completeness
            if nid not in order:
                order.append(nid)
        return order

    def incoming(self, node_id: int) -> List[ConnectionGene]:
        return [c for c in self.enabled_connections() if c.out_node == node_id]

    # ---- structural mutation --------------------------------------------------------

    def mutate_add_node(self, rng: random.Random) -> bool:
        """Split an existing enabled edge by inserting a new specialist sub-judge.

        A->B becomes A->C->B; the original A->B is *disabled* (kept for history).
        The new node C's id and archetype are drawn from the InnovationTracker, so
        the same historical split is identical across genomes.
        """
        candidates = self.enabled_connections()
        if not candidates:
            return False
        old = rng.choice(candidates)
        old.enabled = False

        new_node_id = self.tracker.get_split_node(old.innovation, SPECIALIST_POOL)
        core = self.tracker.archetype_of(new_node_id)
        spec = ARCHETYPE_LIBRARY[core]
        if new_node_id not in self.nodes:
            self.nodes[new_node_id] = NodeGene(
                new_node_id, NodeType.HIDDEN, spec.core, spec.base_instruction, 0.30,
            )

        innov_in = self.tracker.get_edge_innovation(old.in_node, new_node_id)
        self.connections[innov_in] = ConnectionGene(
            innov_in, old.in_node, new_node_id, default_weight_text(1.0), 1.0, True,
        )
        innov_out = self.tracker.get_edge_innovation(new_node_id, old.out_node)
        self.connections[innov_out] = ConnectionGene(
            innov_out, new_node_id, old.out_node, old.weight_text, old.priority, True,
        )
        return True

    def mutate_model(self, rng: random.Random, model_pool: List[str]) -> bool:
        """Model-mutating gene: reassign one agent's model from the allowed pool.

        Picks a non-input node and sets its ``model`` gene to a (different) member
        of ``model_pool``. The choice is heritable through crossover, so evolution
        searches over which model runs each agent alongside topology and prompts.
        """
        if not model_pool:
            return False
        candidates = [n for n in self.nodes.values() if n.node_type != NodeType.INPUT]
        if not candidates:
            return False
        node = rng.choice(candidates)
        choices = [m for m in model_pool if m != node.model] or list(model_pool)
        node.model = rng.choice(choices)
        return True

    def mutate_add_edge(self, rng: random.Random, max_tries: int = 20) -> bool:
        """Add a new context pathway between two existing agents (feed-forward only)."""
        node_ids = list(self.nodes.keys())
        for _ in range(max_tries):
            src = rng.choice(node_ids)
            dst = rng.choice(node_ids)
            if dst == INPUT_NODE_ID or src == OUTPUT_NODE_ID:
                continue
            innov = self.tracker.get_edge_innovation(src, dst)
            if innov in self.connections:
                continue
            if self._would_create_cycle(src, dst):
                continue
            priority = round(rng.uniform(0.3, 1.0), 2)
            self.connections[innov] = ConnectionGene(
                innov, src, dst, default_weight_text(priority), priority, True,
            )
            return True
        return False

    # ---- textual (prompt) mutation --------------------------------------------------

    # Which evaluation axis each archetype is primarily responsible for. Used by
    # reflective prompt mutation to score a node on the axis it owns.
    AXIS_OWNER = {
        "Safety Arbitrator": ("safety",),
        "Fact-Checker": ("quality",),
        "Tone Judge": ("quality",),
        "Coherence Judge": ("quality",),
        "Relevance Judge": ("quality",),
        "Core Judge": ("safety", "quality"),
    }

    def mutate_prompt(self, rng: random.Random, critic: LLMClient, evaluator,
                      reflective: bool = False, batch_size: int = 6) -> bool:
        """Reflective mutation of one node's system instruction.

        Two modes:

        * ``reflective=False`` (offline mock): bump the node's calibration marker
          and append a generic refinement note. The mock judge's competence is
          driven by calibration, so this is the improvement lever offline.

        * ``reflective=True`` (real LLM, GEPA-style): run the node *solo* on a
          sample of the train split, collect the items it judges wrong on the axis
          it owns, and ask the critic to rewrite the system instruction so it would
          get those right. The rewritten instruction is what a real model reads, so
          this is the lever that actually improves a real judge's behavior.
          Returns False (no change) when the node already judges the batch
          correctly or the critic returns nothing usable.
        """
        mutable = [n for n in self.nodes.values() if n.node_type != NodeType.INPUT]
        if not mutable:
            return False
        node = rng.choice(mutable)

        if not reflective:
            _ = critic.complete(
                "You are a prompt-optimization critic.",
                f"Refine the system prompt of a '{node.personality_core}':\n"
                f"{node.system_instruction}",
            )
            node.calibration = min(0.90, round(node.calibration + 0.15, 2))
            node.system_instruction = (
                f"{node.system_instruction.split(' [refined')[0]} "
                f"[refined pass {int(node.calibration / 0.15)}: sharpen focus on "
                f"decisive cues and reduce hedging.]"
            )
            return True

        return self._reflective_rewrite(node, rng, critic, evaluator, batch_size)

    def _run_node_solo(self, node, item: dict, client: LLMClient) -> dict:
        """Judge one item with a single node in isolation (no upstream context)."""
        item_text = f"PROMPT: {item['prompt']}\nRESPONSE: {item['response']}"
        raw = client.complete(node.rendered_system_instruction(), item_text)
        return self._to_verdict(self._safe_parse(raw))

    def _reflective_rewrite(self, node, rng, critic, evaluator, batch_size: int) -> bool:
        axes = self.AXIS_OWNER.get(node.personality_core, ("safety", "quality"))
        client = self._resolve_client(evaluator.llm, node)
        batch = evaluator.sample_train(rng, batch_size)

        mistakes = []
        for item in batch:
            verdict = self._run_node_solo(node, item, client)
            truth = item["truth"]
            for axis in axes:
                if str(verdict.get(axis)) != str(truth.get(axis)):
                    mistakes.append((item, axis, verdict.get(axis), truth.get(axis)))
        if not mistakes:
            return False   # already correct on this batch -- leave a working prompt

        lines = []
        for item, axis, got, want in mistakes[:6]:
            lines.append(
                f"- PROMPT: {item['prompt']}\n  RESPONSE: {item['response']}\n"
                f"  Your {axis} verdict was '{got}' but the correct {axis} is '{want}'."
            )
        reflection = (
            f"Current system instruction for a '{node.personality_core}':\n"
            f"\"\"\"{node.system_instruction}\"\"\"\n\n"
            f"On these cases the judge was WRONG on the '{'/'.join(axes)}' axis:\n"
            + "\n".join(lines)
            + "\n\nRewrite the system instruction so this judge would get these right. "
            "Diagnose the general failure mode (do not memorize these specific "
            "examples), state the decision rule crisply, and keep the same role. "
            "Output ONLY the revised instruction text, under 120 words."
        )
        critic_system = (
            "You are an expert prompt engineer improving an LLM-as-a-judge system "
            "instruction. Output only the improved instruction: no preamble, no "
            "quotes, no markdown, no JSON."
        )
        revised = critic.complete(critic_system, reflection).strip()
        revised = revised.strip('"').strip()
        if len(revised) < 20:
            return False
        revised = revised[:800]
        # Keep the role keyword present so downstream routing and mock detection
        # still identify the archetype.
        if node.personality_core.lower() not in revised.lower():
            revised = f"You are a {node.personality_core}. {revised}"
        node.system_instruction = revised
        return True

    # ---- feed-forward evaluation ----------------------------------------------------

    @staticmethod
    def _resolve_client(llm, node):
        """Pick the client for a node: a ModelRouter routes by the node's model
        gene; a plain LLMClient is used for every node."""
        if hasattr(llm, "client_for"):
            return llm.client_for(node)
        return llm

    def evaluate_item(self, item: dict, llm) -> dict:
        """Run one dataset item through the agent graph and return the OUTPUT verdict.

        Nodes fire in topological order. Each non-input node is handed the item plus
        the JSON verdicts of its upstream neighbors (formatted per edge weight) and
        is executed on the client resolved for its model gene. The verdict emitted
        by the OUTPUT node is the graph's answer.

        ``llm`` may be a single :class:`~neatjudge.llm.LLMClient` (used for every
        node) or a :class:`~neatjudge.llm.ModelRouter` (routes per node's model gene).
        """
        emitted: Dict[int, dict] = {}
        item_text = f"PROMPT: {item['prompt']}\nRESPONSE: {item['response']}"

        for node_id in self.topological_order():
            node = self.nodes[node_id]
            if node.node_type == NodeType.INPUT:
                emitted[node_id] = {"passthrough": item_text}
                continue

            incoming = sorted(self.incoming(node_id), key=lambda c: -c.priority)
            context_lines = [item_text]
            for conn in incoming:
                up = emitted.get(conn.in_node)
                if isinstance(up, dict) and "per_axis" in up:
                    context_lines.append(conn.weight_text)
                    context_lines.append(f"UPSTREAM_JSON::{json.dumps(up)}")

            user_content = "\n".join(context_lines)
            client = self._resolve_client(llm, node)
            raw = client.complete(node.rendered_system_instruction(), user_content)
            emitted[node_id] = self._safe_parse(raw)

        return self._to_verdict(emitted.get(OUTPUT_NODE_ID, {}))

    @staticmethod
    def _safe_parse(raw: str) -> dict:
        """Parse an LLM response into a dict, tolerating malformed / fenced output."""
        if not isinstance(raw, str):
            return {}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        # Real models sometimes wrap JSON in prose or ```json fences; extract the
        # outermost {...} span and retry.
        start, end = raw.find("{"), raw.rfind("}")
        if 0 <= start < end:
            try:
                parsed = json.loads(raw[start:end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _to_verdict(node_output: dict) -> dict:
        """Coerce a node's raw JSON into a normalized {safety, quality} verdict.

        Tolerant of a real judge returning axis cells as bare labels/scalars rather
        than the {"value": ..., "confidence": ...} objects the mock emits.
        """
        per_axis = node_output.get("per_axis", {}) if isinstance(node_output, dict) else {}
        if not isinstance(per_axis, dict):
            per_axis = {}

        def axis_value(axis: str, default):
            cell = per_axis.get(axis, {})
            if isinstance(cell, dict):
                return cell.get("value", default)
            return cell if cell is not None else default

        return {"safety": axis_value("safety", "safe"),
                "quality": axis_value("quality", 3)}

    # ---- compatibility distance (speciation) ----------------------------------------

    def compatibility_distance(self, other: "Genome", c1: float, c2: float, c3: float) -> float:
        """NEAT compatibility distance  delta = c1*E/N + c2*D/N + c3*W_bar.

        E = excess genes, D = disjoint genes, W_bar = mean priority difference of
        matching genes. Alignment is purely by innovation number -- i.e. by
        structural history -- so topologically similar graphs stay close.
        """
        innov_a = set(self.connections)
        innov_b = set(other.connections)
        if not innov_a and not innov_b:
            return 0.0

        max_a = max(innov_a) if innov_a else -1
        max_b = max(innov_b) if innov_b else -1
        split_point = min(max_a, max_b)

        matching = innov_a & innov_b
        only = innov_a ^ innov_b
        excess = {i for i in only if i > split_point}
        disjoint = only - excess

        weight_diff = 0.0
        for i in matching:
            weight_diff += abs(self.connections[i].priority - other.connections[i].priority)
        w_bar = (weight_diff / len(matching)) if matching else 0.0

        n = max(len(innov_a), len(innov_b))
        n = n if n >= 1 else 1

        return c1 * len(excess) / n + c2 * len(disjoint) / n + c3 * w_bar

    # ---- crossover ------------------------------------------------------------------

    @staticmethod
    def crossover(parent_a: "Genome", parent_b: "Genome", tracker: InnovationTracker,
                  child_id: int, rng: random.Random) -> "Genome":
        """Innovation-aligned crossover, faithful to NEAT.

        Matching genes (same innovation number) are inherited at random from either
        parent. Disjoint/excess genes are inherited from the *fitter* parent (from
        both when fitness is tied). A gene disabled in either parent has a chance to
        stay disabled in the child. Genes are only ever combined when their
        innovation numbers -- their identities -- match, strictly honoring history.
        """
        if parent_b.fitness > parent_a.fitness:
            parent_a, parent_b = parent_b, parent_a
        equal_fitness = math.isclose(parent_a.fitness, parent_b.fitness)

        child = Genome(tracker, child_id)
        innov_a = set(parent_a.connections)
        innov_b = set(parent_b.connections)

        # Sorted iteration => the rng draw sequence is a deterministic function of
        # the seed alone (independent of set-hashing internals).
        for innov in sorted(innov_a | innov_b):
            in_a, in_b = innov in innov_a, innov in innov_b
            if in_a and in_b:
                source = rng.choice([parent_a, parent_b])
                gene = source.connections[innov].clone()
                if (not parent_a.connections[innov].enabled
                        or not parent_b.connections[innov].enabled):
                    gene.enabled = rng.random() >= 0.75
            elif in_a:
                gene = parent_a.connections[innov].clone()
            else:
                if not equal_fitness:
                    continue
                gene = parent_b.connections[innov].clone()
            child.connections[gene.innovation] = gene

        needed = {INPUT_NODE_ID, OUTPUT_NODE_ID}
        for gene in child.connections.values():
            needed.add(gene.in_node)
            needed.add(gene.out_node)
        for nid in needed:
            na, nb = parent_a.nodes.get(nid), parent_b.nodes.get(nid)
            if na and nb:
                child.nodes[nid] = (na if na.calibration >= nb.calibration else nb).clone()
            elif na:
                child.nodes[nid] = na.clone()
            elif nb:
                child.nodes[nid] = nb.clone()

        child.enforce_acyclic()   # a recombined union can be cyclic; repair to a DAG
        return child

    # ---- introspection --------------------------------------------------------------

    def describe(self) -> str:
        hidden = [n.personality_core for n in self.nodes.values()
                  if n.node_type == NodeType.HIDDEN]
        edges = len(self.enabled_connections())
        specialists = ", ".join(sorted(hidden)) if hidden else "none"
        return (f"genome#{self.genome_id} | nodes={len(self.nodes)} "
                f"edges={edges} | specialists=[{specialists}] | fit={self.fitness:.2f}")
