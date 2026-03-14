# Sheaf Consistency Enforcer

Sheaf Laplacian-based consistency enforcer for **Kernel 1 persistence** across the EFH MCP tool stack.

Closes the "missing piece" identified in `docs/session-findings/2026-03-13-ai-application.md`:
the automatic feedback loop that detects and recovers from lumpability failures before they cascade.

## Theory Mapping

| EFH / Persistence Theory | Enforcer implementation |
|---|---|
| Reciprocal Coherence Kernel K_T | Active agent set with all edges defined |
| Λ(K_T) < τ_{T,m} | Agents updated within coherence_window_s |
| Restriction maps F_{i→e} | `restriction_maps` dict per edge |
| Coboundary δFx | `coboundary_norm()` — intersection of projected keys |
| Dual variable (buffering capacity T) | `edge.dual_variable` accumulated per cycle |
| H¹(G;F) ≠ 0 | `detect_h1_obstruction()` 3-cycle sum check |
| Strong lumpability | KERNEL1 status + mean_coboundary < ε_primal |
| Weak lumpability | WEAK status — residuals elevated |
| Kernel retreat | `trigger_recovery("kernel_retreat")` |
| Re-partition | `trigger_recovery("re_partition")` |

## Installation

```bash
cd sheaf-consistency-enforcer
uv sync
```

## MCP Config

```json
{
  "mcpServers": {
    "sheaf-consistency-enforcer": {
      "command": "uv",
      "args": ["--directory", "/your/path/to/sheaf-consistency-enforcer", "run", "sheaf-enforcer"]
    }
  }
}
```
## Workflow

After each MCP tool call, report its output as agent state:

```
register_agent_state("hipai-montague", {
    "last_assertion": "Causal closure implies computational closure",
    "belief_score": 0.95,
    "inconsistency_flag": False
})

register_agent_state("mcp-logic", {
    "last_proof_result": "Causal closure implies computational closure",
    "proof_confidence": 0.98,
    "contradictions_found": False
})

run_admm_cycle()     # returns closure_status + warnings
get_closure_status() # concise summary + interpretation
```

## Recovery

```
# H1 obstruction detected:
trigger_recovery("kernel_retreat")

# ADMM stalled — macro states no longer lumpable:
trigger_recovery("re_partition", target_agent="hipai-montague")

# High coboundary norms:
trigger_recovery("admm_reset")
```

## Tools

| Tool | Purpose |
|---|---|
| `register_agent_state` | Report current state for an MCP agent |
| `run_admm_cycle` | Execute one full ADMM consistency cycle |
| `get_closure_status` | Closure status + top-pressure edges |
| `set_restriction_map` | Define/update restriction map for an edge |
| `get_edge_report` | Detailed report for a specific agent pair |
| `trigger_recovery` | Execute recovery strategy |
| `configure_thresholds` | Tune ε_primal, dual threshold, coherence window |
| `reset_session` | Clear all state, reload defaults |

## Default Restriction Maps

Pre-wired for the EFH stack. All bidirectional edges share three edge-space keys:
- `edge_claim` — the proposition under scrutiny
- `edge_confidence` — degree of certainty (0–1)
- `edge_inconsistent` — contradiction flag

Edges: hipai-montague ↔ mcp-logic ↔ advanced-reasoning ↔ hipai-montague + verifier-graph hub.
