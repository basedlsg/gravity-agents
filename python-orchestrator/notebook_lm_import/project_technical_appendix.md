# Gravity Agents: Technical Appendix & Methods

## A. Environment and Physics

### A1. Gravity Definition
*   **Constant**: `-9.81 m/s²`.
*   **Implementation**: Global World Setting.
*   **Source**: `web-env/src/physics/PhysicsWorld.js` (Constructor).
    ```javascript
    this.gravity = options.gravity ?? 9.81;
    this.world.gravity.set(0, -this.gravity, 0);
    ```
*   **0.5g Usage**: This was an exploratory phase parameter (Phase 11-12). In the final benchmark (N=30), gravity is **fixed at 1g**. The code supports variable gravity via the `/reset` payload, but the final script uses default `9.81`.

### A2. Physics Parameters (Step Length Factors)
*   **Engine**: `cannon-es` (^0.20.0).
*   **Timestep**: Fixed **60Hz** (`1/60`s).
*   **Solver Iterations**: **10**.
*   **Material Properties** (Global Default):
    *   `friction`: **0.01** (Very Low).
    *   `restitution`: **0.1** (Low Bounce).
    *   **Source**: `PhysicsWorld.js` (Lines 26-27).
*   **Damping**: Defaults are used (Linear: 0.01, Angular: 0.01). Not explicitly overridden.

### A3. Instability (Teleport) Mechanism
*   **Cause**: "Tunneling" due to Discrete Collision Detection (No CCD).
*   **Symptom**: When high velocity (Launch) coincides with a collider boundary, the solver ejects the body to the nearest legal position, which can be >0.8m away in a single tick.
*   **Detection**: Wrapper Logic (`total_delta > 0.8`).
*   **Resets**: No mid-episode resets occur. The agent continues from the teleported location.

---

## B. Action Interface

### B1. Movement Command Definition
*   **Implementation**: Velocity Set.
*   **Duration**: **10 Physics Ticks** (~166ms).
*   **Application**: Velocity is **re-applied every tick** inside the server loop (`server.js`, lines 107-110).
    *   This overrides collision friction/restitution impulses if they are weaker than the 3.0 m/s drive.
*   **Zeroing**: No explicit zeroing at end of step. The next step command (or "idle") handles velocity decay (0.9 damping in "idle").

### B2. Step Length Math Consistency (SOLVED)
*   **The "Mismatch" Explained**:
    *   Configured Granularity: **"fine"** (Index 0.25).
    *   Base Velocity: **3.0 m/s**.
    *   Duration: **0.166s** (10 ticks).
    *   **Theoretical Gain**: `3.0 * 0.25 * 0.1666 = 0.125 m`.
    *   **Measured Gain**: **~0.11 m**.
    *   **Conclusion**: There is **no massive physics failure**. The gain tracks the "fine" granularity setting almost perfectly (90% efficiency), with small losses to friction/drag.

### B3. Axis Isotropy
*   **Assumption**: X-Gain = Z-Gain.
*   **Justification**: The `Move` primitives (`forward` vs `left`) use identical velocity magnitudes (`3.0 * scale`) and the floor friction is uniform (`0.01`).

---

## C. Observation Design

### C1. State vs Statelessness
*   **LLM**: **Stateless**. Prompt contains only `CURRENT` + `GOAL` + `TELEMETRY`.
*   **Controller State (Python Wrapper)**:
    *   `mode` (APPROACH/REROUTE/DIAGNOSTIC).
    *   `lane_index` (Search memory).
    *   `stuck_counter` (Oscillation damper).
    *   `last_stuck` (Shimmy detector).
    *   `probe_step_count` (Multi-step action memory).

### C2. Rounding
*   **Format**: `:.2f`.
*   **Precision**: 0.01m (1cm).
*   **Impact**:
    *   Typical Step: 0.11m (Visible).
    *   Stuck Threshold: 0.02m (Visible).
    *   Rounding does not hide meaningful movement, only micro-jitter.

---

## D. Controller Definitions

### D1. Attempt vs Env Steps
*   **Attempt**: 1 LLM Call.
*   **Env Steps**: `steps = move / gain`.
    *   Example: Prompt `move_x=0.5` -> Gain 0.11 -> **5 Env Actions**.
*   **Budget**: MAX 100 **Attempts** (LLM Calls).

### D2. Stuck Detection
*   **Criterion**: `total_delta < 0.02` (Euclidean).
    *   Code: `np.sqrt(dx**2 + dz**2) < EPS_MOTION`.
*   **Condition**: Only triggers if `current_x < obstacle_plane_x` (Behind Wall).
*   **Shimmy**: Implicitly handled by `total_delta`. If you move Z, total_delta > 0.02, so NOT stuck. (Reroute is triggered by Logic Rule 2: "If stuck repeatedly").

### D3. Lane Search & Probe
*   **Probe**: "2 Steps Forward".
    *   Action: `move_x=0.5`. (Execute ~5 env steps).
    *   Wait, code says: `if mode == "REROUTE" ... move_x = 0.5 ... probe_step_count += 1`.
    *   This is **1 Attempt** of size 0.5m? No, `run_planner` is skipped.
    *   It forces `move_x=0.5`.
    *   The loop executes the move.
    *   So 1 Probe Step = 1 Attempt of 0.5m.
    *   "Probe Definition": 2 Attempts.
*   **Success**: `cumulative_dx > 0.10`.

### D4. Forced Retreat
*   **Boundary**: `safe_x = obstacle_plane_x - 1.2m`.
*   **Logic**: If `x > safe_x`, force `x = -0.5`.
*   **Behavior**: Does not drop lane. Just moves back to clear the "Boxed In" zone.

### D5. Gain Update
*   **Rule**: EMA `0.3`.
*   **Gating**: Updates strictly on `executed_steps > 0`.
*   **Calibration**: Phase 0 (10 fwd/back) sets initial prior.

---

## E. Failure Taxonomy

### E1. Precedence Order
1.  **UNSAT_WEDGED**: Diagnostic confirms 4-way block. (Terminates immediately).
2.  **FAIL_INSTABILITY**: Teleport > 0.8m detected. (Labeled at end if success not achieved).
3.  **SUCCESS**: Goal reached.
4.  **FAIL_POLICY**: Budget exhausted (100 attempts).

### E2. UNSAT_WEDGED Validity
*   **Trigger**: `stuck_counter >= 5` in `OFFSET` phase.
*   **Probe**: 4 directions (Fwd, Back, Left, Right). 1.0m command each.
*   **Threshold**: `delta < 0.05m`.

### E3. FAIL_POLICY Evidence
*   **Seeds**: 2000, 2003 (N=30 Baseline).
*   **Solvability**: Humans/Oracle can solve. The failure is typically "Looping" or "Oscillation" in the heuristic state machine.

---

## F. Results Accounting (Pending N=30 Run)

*   **Dataset**: `final_classified_results_N30_commit_c006ddb.json`.
*   **Status**: Regenerating immutable ground truth.

---

## G. Gravity Linkage

*   **Final Benchmark**: **1g Only**.
*   **Disclaimer**: "0.5g Generalization" was demonstrated in exploring phases but is not part of the final rigorous N=30 benchmark table.

---

## H. N=100 Failure Analysis

*   **Mode**: Parallel Execution (`max_workers=10`).
*   **Root Cause**: Node.js Single-Threaded Event Loop Starvation.
*   **Symptom**: `dx=0` returns for all actions (Timeouts/Lag), masquerading as `UNSAT_WEDGED`.
*   **Verdict**: Infrastructure Failure, not Agent Failure.
