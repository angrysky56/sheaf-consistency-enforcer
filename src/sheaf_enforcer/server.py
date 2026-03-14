"""
Sheaf Consistency Enforcer — FastMCP Server

Implements Sheaf Laplacian-based consistency enforcement for Kernel 1 persistence
across a multi-agent MCP tool stack.

Tools:
  register_agent_state   - Report current state for an MCP agent
  run_admm_cycle         - Execute one full ADMM consistency cycle
  get_closure_status     - Get current Kernel 1 / warning status
  set_restriction_map    - Define or update restriction map for an edge
  trigger_recovery       - Execute a recovery strategy
  get_edge_report        - Detailed report for a specific agent pair
  configure_thresholds   - Update epsilon_primal, dual threshold, coherence window
  reset_session          - Clear all state and restart from defaults
"""

from __future__ import annotations

import time
from typing import Any

from fastmcp import FastMCP

from .admm import run_full_cycle
from .state import ClosureStatus, get_state, reset_state, save_state

mcp = FastMCP(
    "sheaf-consistency-enforcer",
    instructions=(
        "Sheaf Laplacian consistency enforcer for Kernel 1 persistence. "
        "After each MCP tool call: register_agent_state -> run_admm_cycle -> get_closure_status. "
        "Agents: hipai-montague, mcp-logic, advanced-reasoning, verifier-graph."
    ),
)


@mcp.tool
def register_agent_state(agent_id: str, state: dict[str, Any]) -> dict[str, Any]:
    """
    Register or update the current state snapshot for an MCP agent.

    Call after each interaction with an MCP tool. Recommended state keys:
      last_assertion / verified_claim / current_hypothesis (str)
      confidence_score / belief_score / proof_confidence (float 0-1)
      node_count / edge_count / kb_size (int)
      contradictions_found / inconsistency_flag / halt_flag (bool)
      reasoning_depth / chain_length (int)

    Args:
        agent_id: hipai-montague | mcp-logic | advanced-reasoning | verifier-graph
        state: Dict of current state values for this agent.
    """
    s = get_state()
    s.agent_states[agent_id] = state
    s.agent_last_seen[agent_id] = time.time()
    save_state()
    return {
        "registered": agent_id,
        "agent_count": len(s.agent_states),
        "active_agents": s.all_agents(),
        "closure_status": s.closure_status.value,
        "keys_recorded": list(state.keys()),
    }


@mcp.tool
def run_admm_cycle() -> dict[str, Any]:
    """
    Execute one full ADMM cycle across all active agent edges.

    Runs the three-step Sheaf Laplacian loop:
      1. Primal Update: measure coboundary ||delta_Fx|| per edge
      2. Sheaf Diffusion: compare projected states via restriction maps
      3. Dual Update: accumulate inconsistency memory (dual variable = T analog)

    Closure status returned:
      KERNEL1  - full causal closure, no action needed
      WEAK     - weak lumpability, distribution-dependent coherence
      WARNING  - residuals elevated + dual pressure building
      TIMEOUT  - H1 obstruction or ADMM stall — Kernel 1 failing
      KERNEL2  - collapsed to irreversible substrate

    Call every 2-3 tool interactions to maintain closure monitoring.
    """
    s = get_state()
    if len(s.agent_states) < 2:
        return {
            "error": "Need at least 2 registered agents to run consistency check",
            "active_agents": s.all_agents(),
            "hint": "Call register_agent_state for each MCP tool you are using",
        }
    report = run_full_cycle(s)
    save_state()
    return report


@mcp.tool
def get_closure_status() -> dict[str, Any]:
    """
    Get current Kernel 1 closure status and key metrics.

    Returns closure status, active agents, highest-pressure edges,
    H1 obstruction flag, ADMM iteration count, and interpretation.
    """
    s = get_state()
    now = time.time()

    edge_summaries = sorted(
        [
            {
                "edge": eid,
                "coboundary": round(e.last_coboundary, 4),
                "dual_pressure": round(e.pressure, 4),
                "converging": e.converging,
                "stalled": e.stalled,
            }
            for eid, e in s.edges.items()
        ],
        key=lambda x: x["dual_pressure"],
        reverse=True,
    )

    return {
        "closure_status": s.closure_status.value,
        "needs_action": s.closure_status not in (ClosureStatus.KERNEL1,),
        "h1_obstruction": s.h1_obstruction,
        "admm_iterations": s.admm_iterations,
        "active_agents": s.all_agents(),
        "agent_last_seen": {a: round(now - t, 1) for a, t in s.agent_last_seen.items()},
        "edge_health": edge_summaries[:6],
        "thresholds": {
            "epsilon_primal": s.epsilon_primal,
            "dual_warning": s.dual_warning_threshold,
            "coherence_window_s": s.coherence_window_s,
        },
        "interpretation": _interpret_status(s),
    }

def _interpret_status(s) -> str:
    match s.closure_status:
        case ClosureStatus.KERNEL1:
            return "System in Kernel 1: full causal closure. Continue normal operation."
        case ClosureStatus.WEAK:
            return "Weak lumpability: coherence is distribution-dependent. Increase verification frequency."
        case ClosureStatus.WARNING:
            return "Warning: dual pressure building. Run additional mcp-logic passes. Tighten restriction maps."
        case ClosureStatus.TIMEOUT:
            return "Coherence TIMEOUT. Execute recovery immediately. Do NOT commit new claims."
        case ClosureStatus.KERNEL2:
            return "KERNEL 2: macro shielding failed. Full kernel retreat and re-partition required."
        case _:
            return "Unknown status."


@mcp.tool
def set_restriction_map(
    from_agent: str,
    to_agent: str,
    mappings: list[dict[str, Any]],
    replace: bool = False,
) -> dict[str, Any]:
    """
    Define or update the restriction map for an agent edge.

    Args:
        from_agent: Source agent ID.
        to_agent: Target agent ID.
        mappings: List of dicts with keys: from_key, to_key, weight (float, default 1.0).
                  Both directions of a bidirectional edge MUST use the same to_key names.
        replace: If True, replace existing map. If False, merge.

    Example:
        [{"from_key": "last_assertion", "to_key": "edge_claim", "weight": 1.0}]
    """
    s = get_state()
    arrow = "\u2192"
    eid = f"{from_agent}{arrow}{to_agent}"
    if replace or eid not in s.restriction_maps:
        s.restriction_maps[eid] = mappings
    else:
        existing = {(m["from_key"], m["to_key"]) for m in s.restriction_maps[eid]}
        for m in mappings:
            if (m["from_key"], m["to_key"]) not in existing:
                s.restriction_maps[eid].append(m)
    save_state()
    return {"edge": eid, "map_entries": len(s.restriction_maps[eid]), "mappings": s.restriction_maps[eid]}


@mcp.tool
def get_edge_report(from_agent: str, to_agent: str) -> dict[str, Any]:
    """
    Get detailed consistency report for a specific agent pair.

    Returns edge state: dual variable, primal/dual residuals,
    restriction map, projected states, and convergence status.
    """
    s = get_state()
    arrow = "\u2192"
    eid = f"{from_agent}{arrow}{to_agent}"
    edge = s.edges.get(eid)
    if edge is None:
        return {"error": f"No edge data for {eid}. Run run_admm_cycle first.", "known_edges": list(s.edges.keys())}

    from .admm import apply_restriction_map
    rmap = s.get_restriction_map(from_agent, to_agent)
    from_proj = apply_restriction_map(s.agent_states.get(from_agent, {}), rmap)
    to_proj = apply_restriction_map(s.agent_states.get(to_agent, {}), s.get_restriction_map(to_agent, from_agent))

    return {
        "edge": eid,
        "iteration": edge.iteration,
        "last_coboundary_norm": round(edge.last_coboundary, 4),
        "dual_variable": round(edge.dual_variable, 4),
        "converging": edge.converging,
        "stalled": edge.stalled,
        "primal_residuals_last10": [round(r, 4) for r in edge.primal_residuals[-10:]],
        "dual_residuals_last10": [round(r, 4) for r in edge.dual_residuals[-10:]],
        "restriction_map": rmap,
        "from_projected": from_proj,
        "to_projected": to_proj,
        "interpretation": "CONSISTENT" if edge.last_coboundary < s.epsilon_primal else "INCONSISTENT",
    }


@mcp.tool
def trigger_recovery(strategy: str, target_agent: str | None = None) -> dict[str, Any]:
    """
    Execute a recovery strategy to restore Kernel 1 persistence.

    Strategies:
      kernel_retreat - Remove highest-pressure agent. Use on H1 obstruction.
      re_partition   - Reset target agent state to force re-partitioning.
      admm_reset     - Reset all dual variables to zero.
      soft_relax     - Accept approximate solution; continue ADMM.
      fusion         - Prompt to re-register fragmented sub-kernels.

    Args:
        strategy: kernel_retreat | re_partition | admm_reset | soft_relax | fusion
        target_agent: Required for re_partition; optional for kernel_retreat.
    """
    s = get_state()
    result: dict[str, Any] = {"strategy": strategy, "pre_recovery_status": s.closure_status.value}

    if strategy == "kernel_retreat":
        if target_agent and target_agent in s.agent_states:
            s.agent_states.pop(target_agent)
            s.agent_last_seen.pop(target_agent, None)
            for k in [k for k in s.edges if target_agent in k]:
                del s.edges[k]
            result["action"] = f"Removed {target_agent} from coherence kernel"
            result["remaining_agents"] = s.all_agents()
        elif s.edges:
            pressures: dict[str, float] = {}
            for edge in s.edges.values():
                pressures[edge.from_agent] = pressures.get(edge.from_agent, 0) + edge.pressure
                pressures[edge.to_agent] = pressures.get(edge.to_agent, 0) + edge.pressure
            worst = max(pressures, key=lambda a: pressures[a])
            s.agent_states.pop(worst, None)
            s.agent_last_seen.pop(worst, None)
            for k in [k for k in s.edges if worst in k]:
                del s.edges[k]
            result["action"] = f"Auto-retreated: removed {worst} (highest pressure {pressures[worst]:.3f})"
            result["remaining_agents"] = s.all_agents()
        else:
            result["action"] = "No edges to retreat from"

    elif strategy == "re_partition":
        if target_agent and target_agent in s.agent_states:
            s.agent_states[target_agent] = {}
            for edge in s.edges.values():
                if target_agent in (edge.from_agent, edge.to_agent):
                    edge.dual_variable = 0.0
                    edge.primal_residuals.clear()
                    edge.dual_residuals.clear()
                    edge.last_coboundary = 0.0
            result["action"] = f"Reset {target_agent} state for re-partitioning"
        else:
            result["action"] = "Specify target_agent for re_partition"
            result["available_agents"] = s.all_agents()

    elif strategy == "admm_reset":
        for edge in s.edges.values():
            edge.dual_variable = 0.0
        s.h1_obstruction = False
        result["action"] = f"Reset dual variables on {len(s.edges)} edges"

    elif strategy == "soft_relax":
        result["action"] = "Soft-relax mode: ADMM will converge to approximate solution."

    elif strategy == "fusion":
        result["action"] = "Re-register all fragment agents via register_agent_state, then run run_admm_cycle."

    else:
        return {"error": f"Unknown strategy: {strategy}. Use: kernel_retreat, re_partition, admm_reset, soft_relax, fusion"}

    if len(s.agent_states) >= 2:
        report = run_full_cycle(s)
        result["post_recovery_status"] = report["closure_status"]
        result["post_recovery_mean_coboundary"] = report["mean_coboundary_norm"]
    else:
        s.closure_status = ClosureStatus.WEAK
        result["post_recovery_status"] = s.closure_status.value

    save_state()
    return result


@mcp.tool
def configure_thresholds(
    epsilon_primal: float | None = None,
    dual_warning_threshold: float | None = None,
    coherence_window_s: float | None = None,
    max_stall_cycles: int | None = None,
) -> dict[str, Any]:
    """
    Update consistency enforcement thresholds.

    Args:
        epsilon_primal: Max acceptable coboundary norm before warning (default 0.15).
        dual_warning_threshold: Dual accumulation before WARNING state (default 2.0).
        coherence_window_s: Max latency for reciprocal signaling in seconds (default 30).
        max_stall_cycles: ADMM cycles without improvement before stall (default 10).
    """
    s = get_state()
    updated = {}
    if epsilon_primal is not None:
        s.epsilon_primal = epsilon_primal
        updated["epsilon_primal"] = epsilon_primal
    if dual_warning_threshold is not None:
        s.dual_warning_threshold = dual_warning_threshold
        updated["dual_warning_threshold"] = dual_warning_threshold
    if coherence_window_s is not None:
        s.coherence_window_s = coherence_window_s
        updated["coherence_window_s"] = coherence_window_s
    if max_stall_cycles is not None:
        s.max_stall_cycles = max_stall_cycles
        updated["max_stall_cycles"] = max_stall_cycles
    save_state()
    return {
        "updated": updated,
        "current_thresholds": {
            "epsilon_primal": s.epsilon_primal,
            "dual_warning_threshold": s.dual_warning_threshold,
            "coherence_window_s": s.coherence_window_s,
            "max_stall_cycles": s.max_stall_cycles,
        },
    }


@mcp.tool
def reset_session(confirm: bool = False) -> dict[str, Any]:
    """
    Clear all session state and restart with default restriction maps.

    Args:
        confirm: Must be True to execute. Prevents accidental resets.
    """
    if not confirm:
        return {"error": "Pass confirm=True to reset. This clears all agent states, dual variables, and ADMM history."}
    reset_state()
    return {"status": "Session reset", "default_restriction_maps_loaded": True, "closure_status": ClosureStatus.KERNEL1.value}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
