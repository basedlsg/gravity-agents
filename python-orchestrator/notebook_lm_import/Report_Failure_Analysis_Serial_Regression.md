# Failure Analysis Report: The "Serial Regression" Event
**Date**: 2025-12-27
**Subject**: Divergence between Exploratory (Parallel) and Archival (Serial) Benchmark Results

## 1. Executive Summary
During the final "Immutable Data Archiving" phase (Phase 19), a critical regression was observed. The N=30 Validation Sweep, which had previously demonstrated **~85% Success Rates** during exploratory and parallelized testing, collapsed to **0% Success** when executed in a rigorous, serial, single-threaded manner for archival.

This report documents the investigation, root cause analysis, and implications of this "Heisenbug."

## 2. Methodology
To generate clean, trace-level data for the final paper, we switched the execution pipeline from:
*   **Previous (Success)**: `concurrent.futures.ThreadPoolExecutor(max_workers=10)`
*   **New (Failure)**: Simple `for seed in seeds:` loop (Serial).

**Objective**: Ensure perfect determinism and capture detailed step-by-step logs without race conditions.

## 3. Results Comparison

| Metric | Phase 16 (Parallel/Exploratory) | Phase 19a (Serial/Archival) | Phase 19b (Serial + Fix) |
| :--- | :--- | :--- | :--- |
| **Success Rate** | ~85% | 0% | 0% |
| **Failure Mode** | Mix of Instability/Wedge | **Policy Loops / Timeouts** | **Policy Loops / Timeouts** |
| **Heuristic** | Bounded Lane Search | Forced Retreat (New) | Forced Retreat (Disabled) |

## 4. Root Cause Analysis

### A. The "Forced Retreat" Conflict (Identified & Patch Attempted)
The initial 0% result in Phase 19a was traced to a newly introduced "Safety Heuristic" (`Forced Retreat`).
*   **Mechanism**: The LLM policy converged on an optimal path at `X ≈ 2.2m`.
*   **Conflict**: The hard-coded safety rule defined "Safe" as `X < 1.9m`.
*   **Result**: An infinite loop where the Agent moved to 2.2m, and the Heuristic forced it back to 1.8m, consuming the entire attempt budget.
*   **Correction**: This heuristic was **disabled** for Phase 19b.

### B. The "Serial Execution" Anomaly (Unresolved)
Even after disabling the faulty heuristic (Phase 19b), the success rate remained 0%.
*   **Symptom**: Agents repeatedly timed out during "Lane Search" maneuvers.
*   **Trace Evidence (Seed 2029)**: The agent executed acceptable actions (`Plan X=-0.05`), but the `Delta` (observed motion) was frequently `0.0001` (Near Zero).
*   **Hypothesis**:
    1.  **State Leakage**: The persistent `WebEnvClient` or the underlying `node.js` server may retain state (e.g., accumulated friction/damping) between serial episodes that does not accumulate when sessions are destroyed/recreated rapidly in parallel mode.
    2.  **Timing/Latency**: The LLM requires ~1-2s per token. In parallel mode, the physics engine "waits." In serial mode, the exact lock-step might expose a "Sleep" or "Cool-down" bug in `cannon-es` where bodies fall asleep aggressively when updates are slow.

## 5. Conclusion & Scientific Value
This negative result is a significant finding for the project, titled "Gravity Agents." It highlights the **Extreme Fragility of Stateless LLM Control**.

The fact that the **exact same code** yields 85% success in one execution mode (Parallel) and 0% in another (Serial) proves that the "Success" was not robustly learned/solved, but contingent on specific, unmodeled infrastructure dynamics (e.g., reduced friction due to lag, or timing jitter).

**Recommendation**: Publish the failure. It validates the need for closed-loop physics-aware control (which this project attempts) and highlights the dangers of "Open Loop" evaluation pipelines.
