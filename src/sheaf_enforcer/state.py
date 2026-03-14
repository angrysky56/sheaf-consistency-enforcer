"""
Session state management for the Sheaf Consistency Enforcer.

Holds per-session agent states, dual variables, restriction maps,
primal residual history, and closure status.
State persists to JSON.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ClosureStatus(str, Enum):
    KERNEL1 = "KERNEL1"  # Full causal + computational closure
    WEAK = "WEAK"  # Weak lumpability - distribution-dependent
    WARNING = "WARNING"  # Early warning signals active
    TIMEOUT = "TIMEOUT"  # Coherence timeout - Lambda > tau_{T,m}
    KERNEL2 = "KERNEL2"  # Collapsed to irreversible hardware


@dataclass
class AgentState:
    """Represents the internal consistency pressure of a single agent."""

    pressure_p: float = 0.0  # Primal pressure
    pressure_q: float = 0.0  # Dual pressure (rate of change)
    last_action: str = "IDLE"
    last_update: float = 0.0


@dataclass
class EdgeState:
    """Represents the consistency state of a single sheaf edge (dual variables, residuals)."""

    from_agent: str
    to_agent: str
    dual_claim: float = 0.0
    dual_variable: float = 0.0
    primal_residuals: list[float] = field(default_factory=list)
    dual_residuals: list[float] = field(default_factory=list)
    last_coboundary: float = 0.0
    iteration: int = 0

    @property
    def edge_id(self) -> str:
        return f"{self.from_agent}\u2192{self.to_agent}"

    def record_iteration(self, primal_res: float) -> float:
        """
        Record a new ADMM iteration for this edge.
        Returns the dual residual (change in coboundary).
        """
        self.primal_residuals.append(primal_res)
        if len(self.primal_residuals) > 50:
            self.primal_residuals.pop(0)

        dual_res = abs(primal_res - self.last_coboundary)
        self.dual_residuals.append(dual_res)
        if len(self.dual_residuals) > 50:
            self.dual_residuals.pop(0)

        # Record this as the baseline for next iteration's dual residual
        self.last_coboundary = primal_res
        self.dual_variable += primal_res
        self.iteration += 1
        return dual_res

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

    # Agent states: agent_id -> AgentState tracking pressure
    agents: dict[str, AgentState] = field(default_factory=dict)
    agent_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    agent_last_seen: dict[str, float] = field(default_factory=dict)

    # Decay configuration
    decay_rate: float = 0.95

    def get_agent_state(self, agent_id: str) -> AgentState:
        """Get or create agent state."""
        if agent_id not in self.agents:
            self.agents[agent_id] = AgentState()
        return self.agents[agent_id]

    def decay_agent_pressures(self, rate: float | None = None):
        """Standard dissipative pressure decay."""
        decay = rate if rate is not None else self.decay_rate
        for agent in self.agents.values():
            agent.pressure_p *= decay
            agent.pressure_q *= decay

    # Restriction maps: "from->to" -> list of key mappings
    restriction_maps: dict[str, list[dict]] = field(default_factory=dict)

    # Global cycle counter and closure status
    admm_iterations: int = 0
    closure_status: ClosureStatus = ClosureStatus.KERNEL1
    h1_obstruction: bool = False
    last_cycle_time: float = 0.0

    # Thresholds
    coherence_window_s: float = 30.0
    epsilon_primal: float = 0.15
    dual_decay_rate: float = 0.15  # Higher decay to resolve pressure faster
    dual_pressure_per_agent: dict[str, float] = field(default_factory=dict)
    dual_warning_threshold: float = 5.0
    max_stall_cycles: int = 10

    # Edge states: "from->to" -> EdgeState
    edges: dict[str, EdgeState] = field(default_factory=dict)

    def get_or_create_edge(self, from_agent: str, to_agent: str) -> EdgeState:
        """Retrieve or create the EdgeState for a directed pair."""
        eid = f"{from_agent}\u2192{to_agent}"
        if eid not in self.edges:
            self.edges[eid] = EdgeState(from_agent=from_agent, to_agent=to_agent)
        return self.edges[eid]

    def get_edge(self, edge_id: str) -> EdgeState:
        """Retrieve the EdgeState for a given edge ID, initializing if missing."""
        if edge_id not in self.edges:
            from_agent, to_agent = edge_id.split("\u2192")
            self.edges[edge_id] = EdgeState(from_agent=from_agent, to_agent=to_agent)
        return self.edges[edge_id]

    def reset_admm(self) -> None:
        """Reset ADMM iterations and dual variables to clear inconsistency memory."""
        self.admm_iterations = 0
        for edge in self.edges.values():
            edge.dual_variable = 0.0
            edge.primal_residuals = []
            edge.dual_residuals = []
            edge.iteration = 0
        self.closure_status = ClosureStatus.KERNEL1

    def remove_agent(self, agent_id: str) -> bool:
        """Remove agent and all its associated edges/maps."""
        if agent_id not in self.agent_states:
            return False
        del self.agent_states[agent_id]
        if agent_id in self.agent_last_seen:
            del self.agent_last_seen[agent_id]

        # Remove edges
        to_del = [eid for eid in self.edges if agent_id in (eid.split("\u2192"))]
        for eid in to_del:
            del self.edges[eid]

        # Remove restriction maps
        to_del_maps = [
            eid for eid in self.restriction_maps if agent_id in (eid.split("\u2192"))
        ]
        for eid in to_del_maps:
            del self.restriction_maps[eid]

        # Recompute closure status
        if not self.agent_states:
            self.closure_status = ClosureStatus.KERNEL1
        return True

    def update_status(
        self, new_status: ClosureStatus | str, message: str | None = None
    ) -> None:
        """Type-safe status updates with higher priority overrides."""
        if not isinstance(new_status, ClosureStatus):
            # Handle cases where the library or other components pass a string message
            logging.info("STATUS MESSAGE: %s", new_status)
            if message:
                logging.info("STATUS DETAIL: %s", message)
            return

        # Simple priority: KERNEL1 (least) -> WEAK -> WARNING -> TIMEOUT -> KERNEL2 (most)
        # We only escalate status levels, never de-escalate without explicit reset
        priority = {
            ClosureStatus.KERNEL1: 0,
            ClosureStatus.WEAK: 1,
            ClosureStatus.WARNING: 2,
            ClosureStatus.TIMEOUT: 3,
            ClosureStatus.KERNEL2: 4,
        }

        target_prio = priority.get(new_status, 0)
        current_prio = priority.get(self.closure_status, 0)

        if target_prio > current_prio:
            old_status = self.closure_status
            self.closure_status = new_status
            log_msg = "STATUS ESCALATION: %s -> %s"
            args = [old_status.name, new_status.name]
            if message:
                log_msg += " - %s"
                args.append(message)
            logging.info(log_msg, *args)
        elif target_prio < current_prio:
            # Optionally log that we are ignoring a lower-priority state
            pass
        elif new_status != self.closure_status:
            # Same priority level but different status (unlikely with current list)
            self.closure_status = new_status

    def get_restriction_map(self, from_agent: str, to_agent: str) -> list[dict]:
        """Get the restriction map for a given directed edge."""
        eid = f"{from_agent}\u2192{to_agent}"
        return self.restriction_maps.get(eid, [])

    def all_agents(self) -> list[str]:
        """Return a list of all known agent IDs."""
        return list(self.agent_states.keys())

    def to_dict(self) -> dict:
        """Serialize state to a dictionary."""
        d = asdict(self)
        d["closure_status"] = self.closure_status.value
        d["edges"] = {k: asdict(v) for k, v in self.edges.items()}
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "SessionState":
        """Deserialize session state from a dictionary."""
        # Convert edge nested dicts to EdgeState objects
        edges = {k: EdgeState(**v) for k, v in data.pop("edges", {}).items()}
        status = ClosureStatus(data.pop("closure_status", "KERNEL1"))
        obj = cls(**data)
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
    """Get the singleton session state, loading from disk if necessary."""
    global _session  # pylint: disable=global-statement
    if _session is None:
        _session = load_state()
    return _session


def load_state() -> SessionState:
    """Load session state from JSON file."""
    _state = SessionState()  # Initialize with default
    if _STATE_PATH.exists():
        try:
            with open(_STATE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                _state = SessionState.from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.warning(
                "Could not load session state from %s: %s. Initializing new state.",
                _STATE_PATH,
                e,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error(
                "Unexpected error loading session state from %s: %s. Initializing new state.",
                _STATE_PATH,
                e,
            )
    _seed_default_restriction_maps(_state)
    return _state


def save_state() -> None:
    """Persist session state to JSON file."""
    state = get_state()
    try:
        with open(_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Could not save session state to %s: %s", _STATE_PATH, e)


def reset_state() -> None:
    """Reset the singleton session state to defaults and save."""
    global _session  # pylint: disable=global-statement
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
            {"from_key": "last_assertion", "to_key": "edge_claim", "weight": 1.0},
            {"from_key": "belief_score", "to_key": "edge_confidence", "weight": 1.0},
            {
                "from_key": "inconsistency_flag",
                "to_key": "edge_inconsistent",
                "weight": 1.0,
            },
        ],
        f"mcp-logic{arrow}hipai-montague": [
            {"from_key": "last_proof_result", "to_key": "edge_claim", "weight": 1.0},
            {
                "from_key": "proof_confidence",
                "to_key": "edge_confidence",
                "weight": 1.0,
            },
            {
                "from_key": "contradictions_found",
                "to_key": "edge_inconsistent",
                "weight": 1.0,
            },
        ],
        # mcp-logic <-> advanced-reasoning
        f"mcp-logic{arrow}advanced-reasoning": [
            {"from_key": "last_proof_result", "to_key": "edge_claim", "weight": 1.0},
            {
                "from_key": "proof_confidence",
                "to_key": "edge_confidence",
                "weight": 1.0,
            },
            {
                "from_key": "contradictions_found",
                "to_key": "edge_inconsistent",
                "weight": 1.0,
            },
        ],
        f"advanced-reasoning{arrow}mcp-logic": [
            {"from_key": "current_hypothesis", "to_key": "edge_claim", "weight": 1.0},
            {
                "from_key": "confidence_score",
                "to_key": "edge_confidence",
                "weight": 1.0,
            },
            {"from_key": "halt_flag", "to_key": "edge_inconsistent", "weight": 1.0},
        ],
        # advanced-reasoning <-> hipai-montague
        f"advanced-reasoning{arrow}hipai-montague": [
            {"from_key": "verified_claim", "to_key": "edge_claim", "weight": 1.0},
            {
                "from_key": "confidence_score",
                "to_key": "edge_confidence",
                "weight": 1.0,
            },
            {"from_key": "halt_flag", "to_key": "edge_inconsistent", "weight": 1.0},
        ],
        f"hipai-montague{arrow}advanced-reasoning": [
            {"from_key": "last_assertion", "to_key": "edge_claim", "weight": 1.0},
            {"from_key": "belief_score", "to_key": "edge_confidence", "weight": 1.0},
            {
                "from_key": "inconsistency_flag",
                "to_key": "edge_inconsistent",
                "weight": 1.0,
            },
        ],
        # verifier-graph -> * (hub, one-way provenance)
        # Only compares edge_claim — the hub confirms claims are verified.
        # chain_length is NOT mapped to edge_confidence: chain_length is an integer
        # (e.g. 5) and would project as 2.5 (weight 0.5) vs confidence 0.95,
        # producing a false coboundary spike of ~1.55 on every hub edge.
        f"verifier-graph{arrow}hipai-montague": [
            {"from_key": "last_verified_claim", "to_key": "edge_claim", "weight": 1.0},
        ],
        f"verifier-graph{arrow}mcp-logic": [
            {"from_key": "last_verified_claim", "to_key": "edge_claim", "weight": 1.0},
        ],
        f"verifier-graph{arrow}advanced-reasoning": [
            {"from_key": "last_verified_claim", "to_key": "edge_claim", "weight": 1.0},
        ],
    }
    state.restriction_maps.update(maps)
