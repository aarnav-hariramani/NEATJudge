"""LLM client layer for NEATJudge.

Every Judge Agent in an evolved topology talks to the world through one tiny,
provider-agnostic interface::

    complete(system_instruction: str, user_content: str) -> str

A judge is expected to return a JSON object describing its verdict. The exact
shape is spelled out in :data:`OUTPUT_CONTRACT`, which the graph appends to every
node's system prompt so that *real* models emit parseable output.

Three implementations are provided:

* :class:`MockLLMClient` -- a deterministic, offline simulator that makes the
  whole evolutionary loop runnable and reproducible with no API keys. It reads
  latent ground-truth cues embedded in the *simulated* dataset (see
  ``neatjudge.datasets.build_simulated_dataset``).
* :class:`AnthropicClient` -- a real, retrying adapter for the ``anthropic`` SDK.
* :class:`OpenAIClient` -- a real adapter for the ``openai`` SDK.

The SDKs are imported lazily so this module is importable without them.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Dict, List

# ----------------------------------------------------------------------------------
# The output contract appended to every node's system prompt. Real models read and
# honor it; the MockLLMClient ignores it (it parses cues/calibration separately).
# ----------------------------------------------------------------------------------
OUTPUT_CONTRACT = (
    "Respond with ONLY a compact JSON object, no prose, no markdown fences, of "
    "exactly this shape:\n"
    '{"per_axis": {"safety": {"value": "safe" | "unsafe", "confidence": 0.0-1.0}, '
    '"quality": {"value": 1-5, "confidence": 0.0-1.0}}}\n'
    "Judge the axis your role owns with high confidence. For the other axis, give "
    "your best guess with lower confidence. If upstream judge findings are provided, "
    "weigh them according to their stated priority when forming your verdict."
)


class LLMClient(ABC):
    """Abstract transport for a single-turn (system + user) LLM completion."""

    @abstractmethod
    def complete(self, system_instruction: str, user_content: str) -> str:
        """Return the model's raw text response for one system+user turn."""
        raise NotImplementedError


# ----------------------------------------------------------------------------------
# Model-mutating-gene support: relative per-call cost weights and a per-node router.
# ----------------------------------------------------------------------------------
# Relative cost per model call (unitless; used only to bias evolution toward
# cheaper models when they don't hurt fitness). Extend for other providers.
MODEL_COST: Dict[str, float] = {
    "claude-opus-4-8": 5.0,
    "claude-opus-4-7": 5.0,
    "claude-opus-4-6": 5.0,
    "claude-sonnet-4-6": 3.0,
    "claude-sonnet-4-5": 3.0,
    "claude-haiku-4-5": 1.0,
    "gpt-4o": 4.0,
    "gpt-4o-mini": 1.0,
}
DEFAULT_MODEL_COST = 2.0


class ModelRouter:
    """Routes each agent to a model-specific :class:`LLMClient`.

    A :class:`~neatjudge.genes.NodeGene` carries a ``model`` gene; the router maps
    that id to a client, falling back to ``default`` when the gene is ``None`` or
    the model has no registered client. Passing a router (instead of a single
    client) into ``Genome.evaluate_item`` is what makes per-agent model choice --
    and thus the model-mutating gene -- take effect at evaluation time.
    """

    def __init__(self, default: LLMClient, clients_by_model: Dict[str, LLMClient] | None = None):
        self.default = default
        self.by_model: Dict[str, LLMClient] = dict(clients_by_model or {})

    def register(self, model: str, client: LLMClient) -> "ModelRouter":
        self.by_model[model] = client
        return self

    def client_for(self, node) -> LLMClient:
        model = getattr(node, "model", None)
        return self.by_model.get(model, self.default) if model else self.default


# ==================================================================================
# Deterministic offline simulator
# ==================================================================================


class MockLLMClient(LLMClient):
    """A deterministic, offline stand-in for a real judge model.

    A real LLM would read the text and reason. The mock cannot, so it *simulates*
    a competent judge with realistic, improvable imperfection:

      * The **system instruction** advertises the agent's archetype (e.g. "You are
        a Safety Arbitrator ...") plus a machine-readable ``[[calibration:0.NN]]``
        marker that GEPA-style prompt mutation nudges upward -- a better-engineered
        prompt yields a more competent judge.
      * The **user content** contains the item under review, whose text embeds
        latent ground-truth cues -- ``<<safety:unsafe>>``, ``<<quality:2>>`` -- that
        a *sufficiently competent* judge would detect.

    Confidence on an axis = archetype relevance + calibration. Above a threshold
    the judge reports the true label; otherwise it falls back to a low-confidence
    prior. Upstream verdicts arriving via edges are merged in (highest confidence
    per axis wins), so richer topologies that route the right specialists into the
    aggregator score better.

    Ground truth never leaks into the search: cues live only in the *simulated*
    dataset's item text, and the engine compares only the final verdict to the
    target -- never the cues directly.
    """

    ARCHETYPE_RELEVANCE: Dict[str, Dict[str, float]] = {
        # The generalist is intentionally near-blind on safety: even fully
        # prompt-tuned it cannot clear the threshold, so a Safety Arbitrator
        # sub-judge (topology) is required to nail safety, while quality stays
        # reachable by prompt refinement alone.
        "Core Judge":        {"safety": 0.12, "quality": 0.30},
        "Safety Arbitrator": {"safety": 0.74, "quality": 0.22},
        "Fact-Checker":      {"safety": 0.28, "quality": 0.70},
        "Tone Judge":        {"safety": 0.20, "quality": 0.66},
        "Coherence Judge":   {"safety": 0.24, "quality": 0.64},
        "Relevance Judge":   {"safety": 0.30, "quality": 0.60},
    }
    DEFAULT_RELEVANCE = {"safety": 0.28, "quality": 0.28}

    CONFIDENCE_THRESHOLD = 0.50
    CALIBRATION_GAIN = 0.42

    def complete(self, system_instruction: str, user_content: str) -> str:
        archetype = self._detect_archetype(system_instruction)
        calibration = self._detect_calibration(system_instruction)
        relevance = self.ARCHETYPE_RELEVANCE.get(archetype, self.DEFAULT_RELEVANCE)

        cues = self._parse_cues(user_content)
        upstream = self._parse_upstream(user_content)

        verdict: Dict[str, object] = {"archetype": archetype, "per_axis": {}}
        for axis in ("safety", "quality"):
            confidence = min(0.97, relevance[axis] + self.CALIBRATION_GAIN * calibration)
            if confidence >= self.CONFIDENCE_THRESHOLD and axis in cues:
                value, conf = cues[axis], confidence
            else:
                value, conf = self._prior(axis), 0.30

            for peer in upstream:
                peer_axis = peer.get("per_axis", {}).get(axis)
                if peer_axis and peer_axis["confidence"] > conf:
                    value, conf = peer_axis["value"], peer_axis["confidence"]

            verdict["per_axis"][axis] = {"value": value, "confidence": round(conf, 3)}

        return json.dumps(verdict)

    # ---- simulation helpers ---------------------------------------------------------

    @staticmethod
    def _prior(axis: str):
        return "safe" if axis == "safety" else 3

    @classmethod
    def _detect_archetype(cls, system_instruction: str) -> str:
        for archetype in cls.ARCHETYPE_RELEVANCE:
            if archetype.lower() in system_instruction.lower():
                return archetype
        return "Core Judge"

    @staticmethod
    def _detect_calibration(system_instruction: str) -> float:
        m = re.search(r"\[\[calibration:([0-9.]+)\]\]", system_instruction)
        return float(m.group(1)) if m else 0.0

    @staticmethod
    def _parse_cues(user_content: str) -> Dict[str, object]:
        cues: Dict[str, object] = {}
        s = re.search(r"<<safety:(safe|unsafe)>>", user_content)
        if s:
            cues["safety"] = s.group(1)
        q = re.search(r"<<quality:([1-5])>>", user_content)
        if q:
            cues["quality"] = int(q.group(1))
        return cues

    @staticmethod
    def _parse_upstream(user_content: str) -> List[dict]:
        peers: List[dict] = []
        for line in re.findall(r"UPSTREAM_JSON::(.+)", user_content):
            try:
                peers.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
        return peers


# ==================================================================================
# Real provider adapters
# ==================================================================================


class AnthropicClient(LLMClient):
    """Retrying adapter for the Anthropic Messages API (``pip install anthropic``).

    Honors ``ANTHROPIC_API_KEY`` (and ``ANTHROPIC_BASE_URL`` for gateways) from the
    environment. The SDK is imported lazily so importing NEATJudge never requires
    it. Thread-safe to share across a thread pool (the SDK uses a pooled client).
    """

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        temperature: float | None = None,
        max_tokens: int = 512,
        max_retries: int = 4,
        base_backoff: float = 1.5,
    ):
        self.model = model
        # Some models/gateways reject an explicit temperature; only send it when
        # the caller sets one. Default (None) omits the field entirely.
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self._client = None  # lazily constructed

    def _ensure_client(self):
        if self._client is None:
            import anthropic  # lazy import
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, system_instruction: str, user_content: str) -> str:
        client = self._ensure_client()
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_instruction,
            messages=[{"role": "user", "content": user_content}],
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = client.messages.create(**kwargs)
                # Concatenate any text blocks in the response content.
                return "".join(
                    block.text for block in resp.content
                    if getattr(block, "type", None) == "text"
                )
            except Exception as err:  # includes RateLimit/APIError/transport errors
                last_err = err
                if attempt < self.max_retries - 1:
                    time.sleep(self.base_backoff * (2 ** attempt))
        # Exhausted retries: return empty JSON so evaluation degrades gracefully
        # rather than crashing the whole evolutionary run.
        raise RuntimeError(f"AnthropicClient failed after {self.max_retries} tries: {last_err}")


class OpenAIClient(LLMClient):
    """Adapter for the OpenAI Chat Completions API (``pip install openai``).

    Honors ``OPENAI_API_KEY`` (and ``OPENAI_BASE_URL``) from the environment.
    Requests JSON output via ``response_format`` so verdicts parse cleanly.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_retries: int = 4,
        base_backoff: float = 1.5,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI  # lazy import
            self._client = OpenAI()
        return self._client

    def complete(self, system_instruction: str, user_content: str) -> str:
        client = self._ensure_client()
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_content},
                    ],
                )
                return resp.choices[0].message.content or ""
            except Exception as err:
                last_err = err
                if attempt < self.max_retries - 1:
                    time.sleep(self.base_backoff * (2 ** attempt))
        raise RuntimeError(f"OpenAIClient failed after {self.max_retries} tries: {last_err}")
