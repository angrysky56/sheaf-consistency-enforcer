"""
Session state management for the Sheaf Consistency Enforcer.

Holds per-session agent states, dual variables, restriction maps,
primal residual history, and closure status.
State persists to JSON between server restarts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


class ClosureStatus(str, Enum):
    KERNEL1 = "KERNEL1"          # Full causal + computational closure
    WEAK = "WEAK"                # Weak lumpability - distribution-dependent
    WARNING = "WARNING"          # Early warning signals active
    TIMEOUT = "TIMEOUT"          # Coherence timeout - Lambda > tau_{T,m}
    KERNEL2 = "KERNEL2"          # Collapsed to irreversible hardware


@dataclass
class EdgeState:
    """State for one directed edge between two MCP agents."""
    from_agent: str
    to_agent: str
    dual_variable: float = 0.0
    primal_residuals: list[float] = field(default_factory=list)
    dual_residuals: list[float] = field(default_factory=list)
    last_coboundary: float = 0.0
    iteration: int = 0

    @property
    def edge_id(self) -> str:
        return f"{self.from_agent}\u2192{self.to_agent}"

    @property
    def pressure(self) -> float:
        """Dual variable magnitude = buffering pressure on this edge."""
        return abs(self.dual_variable)

    @property
    def converging(self) -> bool:
        """True if primal residuals are decreasing."""
        if len(self.primal_residuals) < 2:
            return True
        return self.primal_residuals[-1] <= self.primal_residuals[-2]

    @property
    def stalled(self) -> bool:
        """True if residuals stopped improving (early timeout signal)."""
        if len(self.primal_residuals) < 5:
            return False
        window = self.primal_residuals[-5:]
        return max(window) - min(window) < 1e-4


@dataclass
class SessionState:
    """Full enforcer session state."""
    # Agent states: agent_id -> {key: value} snapshot of reported state
    agent_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    agent_last_seen: dict[str, float] = field(default_factory=dict)

    # Restriction maps: "from->to" -> list of key mappings
    restriction_maps: dict[str, list[dict]] = field(default_factory=dict)

    # Edge states keyed by "from->to"
    edges: dict[str, EdgeState] = field(default_factory=dict)

    # Global cycle counter and closure status
    admm_iterations: int = 0
    closure_status: ClosureStatus = ClosureStatus.KERNEL1
    h1_obstruction: bool = False
    last_cycle_time: float = 0.0

    # Thresholds
    coherence_window_s: float = 30.0
    epsilon_primal: float = 0.15
    dual_warning_threshold: float = 2.0
    max_stall_cycles: int = 10

    def get_or_create_edge(self, from_agent: str, to_agent: str) -> EdgeState:
        eid = f"{from_agent}\u2192{to_agent}"
        if eid not in self.edges:
            self.edges[eid] = EdgeState(from_agent=from_agent, to_agent=to_agent)
        return self.edges[eid]

    def get_restriction_map(self, from_agent: str, to_agent: str) -> list[dict]:
        eid = f"{from_agent}\u2192{to_agent}"
        return self.restriction_maps.get(eid, [])

    def all_agents(self) -> list[str]:
        return list(self.agent_states.keys())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["closure_status"] = self.closure_status.value
        d["edges"] = {k: asdict(v) for k, v in self.edges.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SessionState":
        edges = {k: EdgeState(**v) for k, v in d.pop("edges", {}).items()}
        status = ClosureStatus(d.pop("closure_status", "KERNEL1"))
        obj = cls(**d)
        obj.edges = edges
        obj.closure_status = status
        return obj


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

_STATE_PATH = Path(__file__).parent.parent.parent / "data" / "session_state.json"
_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

_session: SessionState | None = None


def get_state() -> SessionState:
    global _session
    if _session is None:
        _session = load_state()
    return _session


def load_state() -> SessionState:
    if _STATE_PATH.exists():
        try:
            with open(_STATE_PATH) as f:
                return SessionState.from_dict(json.load(f))
        except Exception:
            pass
    state = SessionState()
    _seed_default_restriction_maps(state)
    return state


def save_state() -> None:
    if _session is not None:
        with open(_STATE_PATH, "w") as f:
            json.dump(_session.to_dict(), f, indent=2, default=str)


def reset_state() -> None:
    global _session
    _session = SessionState()
    _seed_default_restriction_maps(_session)
    save_state()


# ---------------------------------------------------------------------------
# Default restriction maps for the EFH MCP stack
# ---------------------------------------------------------------------------

def _seed_default_restriction_maps(state: SessionState) -> None:
    """
    Pre-wire restriction maps for the four known MCP agents.

    CRITICAL DESIGN RULE: for any bidirectional edge {A, B}, both directions
    A->B and B->A MUST map onto the SAME shared edge-space key names so that
    coboundary_norm() can compare them (intersection of projected keys).

    Shared edge space for all bidirectional pairs:
      edge_claim       - the proposition under scrutiny (string -> stable hash)
      edge_confidence  - degree of certainty (float 0-1)
      edge_inconsistent - contradiction flag (bool -> 0.0/1.0)
    """
    arrow = "\u2192"
    maps: dict[str, list[dict]] = {
        # hipai-montague <-> mcp-logic
        f"hipai-montague{arrow}mcp-logic": [
            {"from_key": "last_assertion",     "to_key": "edge_claim",        "weight": 1.0},
            {"from_key": "belief_score",       "to_key": "edge_confidence",   "weight": 1.0},
            {"from_key": "inconsistency_flag", "to_key": "edge_inconsistent", "weight": 1.0},
        ],
        f"mcp-logic{arrow}hipai-montague": [
            {"from_key": "last_proof_result",    "to_key": "edge_claim",        "weight": 1.0},
            {"from_key": "proof_confidence",     "to_key": "edge_confidence",   "weight": 1.0},
            {"from_key": "contradictions_found", "to_key": "edge_inconsistent", "weight": 1.0},
        ],
        # mcp-logic <-> advanced-reasoning
        f"mcp-logic{arrow}advanced-reasoning": [
            {"from_key": "last_proof_result",    "to_key": "edge_claim",        "weight": 1.0},
            {"from_key": "proof_confidence",     "to_key": "edge_confidence",   "weight": 1.0},
            {"from_key": "contradictions_found", "to_key": "edge_inconsistent", "weight": 1.0},
        ],
        f"advanced-reasoning{arrow}mcp-logic": [
            {"from_key": "current_hypothesis", "to_key": "edge_claim",        "weight": 1.0},
            {"from_key": "confidence_score",   "to_key": "edge_confidence",   "weight": 1.0},
            {"from_key": "halt_flag",          "to_key": "edge_inconsistent", "weight": 1.0},
        ],
        # advanced-reasoning <-> hipai-montague
        f"advanced-reasoning{arrow}hipai-montague": [
            {"from_key": "verified_claim",   "to_key": "edge_claim",        "weight": 1.0},
            {"from_key": "confidence_score", "to_key": "edge_confidence",   "weight": 1.0},
            {"from_key": "halt_flag",        "to_key": "edge_inconsistent", "weight": 1.0},
        ],
        f"hipai-montague{arrow}advanced-reasoning": [
            {"from_key": "last_assertion",     "to_key": "edge_claim",        "weight": 1.0},
            {"from_key": "belief_score",       "to_key": "edge_confidence",   "weight": 1.0},
            {"from_key": "inconsistency_flag", "to_key": "edge_inconsistent", "weight": 1.0},
        ],
        # verifier-graph -> * (hub, one-way provenance)
        f"verifier-graph{arrow}hipai-montague": [
            {"from_key": "last_verified_claim", "to_key": "edge_claim",      "weight": 1.0},
            {"from_key": "chain_length",        "to_key": "edge_confidence", "weight": 0.5},
        ],
        f"verifier-graph{arrow}mcp-logic": [
            {"from_key": "last_verified_claim", "to_key": "edge_claim",      "weight": 1.0},
            {"from_key": "chain_length",        "to_key": "edge_confidence", "weight": 0.5},
        ],
        f"verifier-graph{arrow}advanced-reasoning": [
            {"from_key": "last_verified_claim", "to_key": "edge_claim",      "weight": 1.0},
            {"from_key": "chain_length",        "to_key": "edge_confidence", "weight": 0.5},
        ],
    }
    state.restriction_maps.update(maps)
