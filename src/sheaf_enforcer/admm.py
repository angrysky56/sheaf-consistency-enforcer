"""
ADMM-based Sheaf Laplacian consistency enforcement.

Three-step cycle:
  1. Local Primal Update  - measure coboundary ||delta_Fx|| per edge
  2. Sheaf Diffusion      - compare projected states via restriction maps
  3. Dual Update          - accumulate inconsistency memory (dual variable)

Early-warning signals:
  - Primal residual > epsilon_primal       -> lumpability shift (WEAK)
  - Dual variable > dual_warning_threshold -> Q approaching T (WARNING)
  - H1 obstruction: cyclic sum > threshold -> coherence timeout (TIMEOUT)
  - ADMM stall: residuals not improving    -> timeout approaching
"""

from __future__ import annotations

import math
import time
from typing import Any

from .state import ClosureStatus, EdgeState, SessionState


def apply_restriction_map(
    agent_state: dict[str, Any],
    restriction_map: list[dict],
) -> dict[str, float]:
    """
    Project agent_state onto the edge space using the restriction map.

    String values: identical strings -> same hash -> coboundary = 0.
    Bool: True->1.0, False->0.0. Missing keys contribute nothing (soft).
    """
    projected: dict[str, float] = {}
    for mapping in restriction_map:
        from_key = mapping["from_key"]
        to_key = mapping["to_key"]
        weight = float(mapping.get("weight", 1.0))
        val = agent_state.get(from_key)
        if val is None:
            continue
        if isinstance(val, bool):
            numeric = float(val)
        elif isinstance(val, (int, float)):
            numeric = float(val)
        elif isinstance(val, str):
            # Stable hash: identical strings -> identical projection -> coboundary = 0
            numeric = (hash(val) % 100_000) / 100_000.0
        else:
            continue
        projected[to_key] = projected.get(to_key, 0.0) + numeric * weight
    return projected

def coboundary_norm(
    from_projected: dict[str, float],
    to_projected: dict[str, float],
) -> float:
    """
    Compute ||delta_Fx|| — coboundary norm measuring inconsistency on this edge.

    Only keys present in BOTH projected states contribute. If no shared keys,
    returns 0.0 (agents don't constrain each other on this edge — correct
    soft-constraint interpretation, not a false alarm).
    """
    shared_keys = set(from_projected) & set(to_projected)
    if not shared_keys:
        return 0.0
    sq_sum = sum((from_projected[k] - to_projected[k]) ** 2 for k in shared_keys)
    return math.sqrt(sq_sum / len(shared_keys))


def admm_step(state: SessionState, from_agent: str, to_agent: str) -> dict[str, Any]:
    """Run one ADMM step on a single edge. Returns per-edge report."""
    edge = state.get_or_create_edge(from_agent, to_agent)

    from_proj = apply_restriction_map(
        state.agent_states.get(from_agent, {}),
        state.get_restriction_map(from_agent, to_agent),
    )
    to_proj = apply_restriction_map(
        state.agent_states.get(to_agent, {}),
        state.get_restriction_map(to_agent, from_agent),
    )

    # Step 1 + 2: Primal update + Sheaf diffusion
    cb_norm = coboundary_norm(from_proj, to_proj)
    prev = edge.last_coboundary

    edge.primal_residuals.append(cb_norm)
    if len(edge.primal_residuals) > 50:
        edge.primal_residuals = edge.primal_residuals[-50:]

    dual_res = abs(cb_norm - prev)
    edge.dual_residuals.append(dual_res)
    if len(edge.dual_residuals) > 50:
        edge.dual_residuals = edge.dual_residuals[-50:]

    # Step 3: Dual update — accumulate inconsistency memory
    edge.dual_variable = (edge.dual_variable * (1.0 - state.dual_decay_rate)) + cb_norm
    edge.last_coboundary = cb_norm
    edge.iteration += 1

    return {
        "edge_id": edge.edge_id,
        "coboundary_norm": round(cb_norm, 4),
        "dual_variable": round(edge.dual_variable, 4),
        "primal_residual": round(cb_norm, 4),
        "dual_residual": round(dual_res, 4),
        "converging": edge.converging,
        "stalled": edge.stalled,
        "pressure": round(edge.pressure, 4),
    }


def detect_h1_obstruction(state: SessionState) -> tuple[bool, str]:
    """
    Detect H1(G; F) != 0 — topological obstruction to global consistency.

    Looks for 3-agent directed cycles where accumulated dual variables sum
    above threshold. This indicates unresolvable cyclic dependencies.
    Only fires after >= 3 ADMM iterations to avoid warmup false positives.
    """
    agents = state.all_agents()
    if len(agents) < 3 or state.admm_iterations < 3:
        return False, "Insufficient agents or iterations for cycle detection"

    threshold = state.dual_warning_threshold * 3
    max_cycle = 0.0
    worst: list[str] = []

    for a in agents:
        for b in agents:
            if b == a:
                continue
            for c in agents:
                if c == a or c == b:
                    continue
                e1 = state.edges.get(f"{a}\u2192{b}")
                e2 = state.edges.get(f"{b}\u2192{c}")
                e3 = state.edges.get(f"{c}\u2192{a}")
                if e1 and e2 and e3:
                    total = e1.dual_variable + e2.dual_variable + e3.dual_variable
                    if total > max_cycle:
                        max_cycle = total
                        worst = [a, b, c]

    if max_cycle > threshold and worst:
        return True, (
            f"Cyclic obstruction {worst[0]}->{worst[1]}->{worst[2]}->{worst[0]}: "
            f"accumulated inconsistency {max_cycle:.3f} > threshold {threshold:.1f}"
        )
    return False, "No cyclic obstruction detected"


def run_full_cycle(state: SessionState) -> dict[str, Any]:
    """Run one complete ADMM cycle over all registered agent pairs."""
    t0 = time.time()
    state.admm_iterations += 1

    agents = state.all_agents()
    edge_reports = []

    for eid in state.restriction_maps:
        if "\u2192" not in eid:
            continue
        from_a, to_a = eid.split("\u2192", 1)
        if from_a in agents and to_a in agents:
            edge_reports.append(admm_step(state, from_a, to_a))

    if edge_reports:
        mean_cb = sum(r["coboundary_norm"] for r in edge_reports) / len(edge_reports)
        max_dual = max(r["dual_variable"] for r in edge_reports)
        any_stalled = any(r["stalled"] for r in edge_reports)
        any_diverging = any(not r["converging"] for r in edge_reports)

        # Aggregate per-agent pressure (max pressure from incident edges)
        state.dual_pressure_per_agent = {}
        for r in edge_reports:
            e = state.get_edge(r["edge_id"])
            if e:
                for agent in [e.from_agent, e.to_agent]:
                    state.dual_pressure_per_agent[agent] = max(
                        state.dual_pressure_per_agent.get(agent, 0.0),
                        r["dual_variable"]
                    )
    else:
        mean_cb = max_dual = 0.0
        any_stalled = any_diverging = False
        state.dual_pressure_per_agent = {}

    h1_found, h1_msg = detect_h1_obstruction(state)
    state.h1_obstruction = h1_found

    prev_status = state.closure_status
    warnings: list[str] = []

    if h1_found:
        state.update_status(ClosureStatus.TIMEOUT)
        warnings.append(f"H1 OBSTRUCTION: {h1_msg}")
    elif any_stalled and max_dual > state.dual_warning_threshold * 2:
        state.update_status(ClosureStatus.TIMEOUT)
        warnings.append("ADMM stalled + high dual pressure -> coherence timeout")
    elif mean_cb > state.epsilon_primal or any_diverging:
        if max_dual > state.dual_warning_threshold:
            state.update_status(ClosureStatus.WARNING)
            warnings.append(
                f"Elevated residuals ({mean_cb:.3f} > {state.epsilon_primal}) "
                f"and dual pressure {max_dual:.3f} > {state.dual_warning_threshold}"
            )
        else:
            state.update_status(ClosureStatus.WEAK)
            warnings.append(f"Elevated residuals ({mean_cb:.3f}) -> weak lumpability")
    elif len(agents) >= 2:
        state.update_status(ClosureStatus.KERNEL1)

    state.last_cycle_time = time.time()
    recovery = _recommend_recovery(state, mean_cb, max_dual, h1_found, any_stalled)

    return {
        "iteration": state.admm_iterations,
        "closure_status": state.closure_status.value,
        "previous_status": prev_status.value,
        "status_changed": state.closure_status != prev_status,
        "agents_active": agents,
        "edges_evaluated": len(edge_reports),
        "mean_coboundary_norm": round(mean_cb, 4),
        "max_dual_variable": round(max_dual, 4),
        "dual_pressure_per_agent": {k: round(v, 4) for k, v in state.dual_pressure_per_agent.items()},
        "h1_obstruction": h1_found,
        "h1_detail": h1_msg,
        "warnings": warnings,
        "recovery_recommendation": recovery,
        "edge_reports": edge_reports,
        "cycle_duration_ms": round((time.time() - t0) * 1000, 2),
    }

def _recommend_recovery(
    state: SessionState,
    mean_cb: float,
    max_dual: float,
    h1: bool,
    stalled: bool,
) -> dict[str, Any]:
    if state.closure_status == ClosureStatus.KERNEL1:
        return {"strategy": "none", "reason": "System in Kernel 1 — no action needed"}
    if h1:
        return {
            "strategy": "kernel_retreat",
            "reason": "H1 obstruction: global consistency topologically impossible. Remove highest-pressure agent.",
            "action": "Call trigger_recovery('kernel_retreat')",
        }
    if stalled and max_dual > state.dual_warning_threshold * 2:
        return {
            "strategy": "re_partition",
            "reason": "ADMM stalled with high dual pressure. Current macro-state partition no longer strongly lumpable.",
            "action": "Call trigger_recovery('re_partition', target_agent='<agent_id>')",
        }
    if mean_cb > state.epsilon_primal * 2:
        return {
            "strategy": "admm_reset",
            "reason": "Coboundary norms persistently high. Reset dual variables and retry.",
            "action": "Call trigger_recovery('admm_reset')",
        }
    return {
        "strategy": "soft_relax",
        "reason": "Early warning state. Continue ADMM; increase mcp-logic verification frequency.",
        "action": "Call trigger_recovery('soft_relax') or continue monitoring",
    }
