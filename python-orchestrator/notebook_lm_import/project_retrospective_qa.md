# Gravity Agents: Exhaustive Project Retrospective

## A. Exact Experimental Setup

### A1. Environment Identity
*   **Environment Name**: `gravity-agents-env`
*   **Version**: `1.0.0` (from `package.json`).
*   **Task Class**: `ThrowBlockTaskV2`
*   **Version**: V2 (Source: `src/tasks/ThrowBlockTaskV2.js`).
*   **Units**: Meters (SI).
*   **Coordinate Convention**: **Y-Up** (Right-handed).
    *   +X: Forward.
    *   +Z: Right.
    *   +Y: Up.
*   **Agent Control**: **Player Body** (Capsule). The Block is a separate dynamic object that the agent "picks up" (attaches to itself).
*   **Block Interaction**: Pick/Drop/Throw. The block is dynamic; it can be pushed, but the task logic requires it to be "in basket" (usually via throw or careful push).
*   **Auto-Reset Logic**: **No auto-respawn** logic within the physics loop.
    *   **Code Evidence**: `checkTermination` returns `{ done: true, reason: 'block_fell' }` if `y < -2`. The *Orchestrator* (Python) then calls `reset()`, but the Environment itself just flags termination.

### A2. Physics Engine and Integration Details
*   **Library**: `cannon-es`.
*   **Version**: `^0.20.0` (from `package.json`).
*   **Timestep**: Fixed **60Hz** (`1/60`s).
    *   **Evidence**: `server.js` (implied default) / `PhysicsWorld.js` constructor `timeStep` default.
*   **Substepping**: None explicit in `server.js`.
*   **Solver Iterations**: **10**.
*   **Gravity Constant**: **9.81 m/s²** (Global).
*   **Hidden Forces**:
    *   **Damping**: Default `cannon-es` linear/angular damping (approx 0.01).
    *   **Friction**: **0.01** (Global default material).
    *   **Restitution**: **0.1** (Global default material).
*   **Friction Parameters**: Applied globally via `defaultContactMaterial`. No per-collider overrides.
*   **Collisions**: **Discrete**.
*   **Tunneling Protection**: **None** (No CCD). This is the root cause of `FAIL_INSTABILITY` (teleportation).

### A3. Determinism and Resets
*   **Seed Control**: Controls **Scene Generation** (Basket distance `±0.5m`).
*   **Seed Formula**: `(seed * 1103515245 + 12345) & 0x7fffffff` (LCG in `BaseTask.js`).
*   **Determinism**:
    *   **Algorithmically**: Yes (Seeded PRNG).
    *   **Empirically (Serial)**: Yes.
    *   **Empirically (Parallel)**: **No**. (FAILED in N=100 run).
*   **State Leakage**: `server.js` maintains state in memory, but `reset()` clears the physics world bodies (`world.reset()`).
    *   **Sweep Hygiene**: Parallel threads starved the event loop, causing leakage/timing issues.
*   **Versions**: Node v18+, Python 3.9+.

---

## B. Action Interface and "Step Length"

### B1. Movement Primitives
*   **Actions**: `forward`, `back`, `left`, `right`, `pick`, `drop`, `throw_weak`, `throw_medium`, `throw_strong`, `idle`.
*   **Execution**: "Set Velocity for Duration".
    *   Code: `agent.velocity.x = 3.0 * scale`.
*   **Duration**: **10 Physics Ticks** (approx 166ms).
*   **Open Loop**: **Yes**.
*   **Acceleration**: **None** (Instant velocity set).
*   **Smoothing**: Slight decay on `idle` (`velocity *= 0.9`).
*   **Collision Resolution**: Physics engine resolves overlaps *after* velocity is applied, reducing effective displacement ("Wall Sliding").

### B2. Step Length Measurement
*   **Gain Definition**: **Meters per Command** (Observed displacement).
*   **Axis**: Average of Forward (+X) and Backward (-X). Assumed Isotropic.
*   **Measurement Protocol**: 10 Steps Forward, 10 Steps Back.
*   **Environment**: Empty space (Calibration Phase triggers at Spawn, usually clear).
*   **X vs Z**: Measured on **X only**. Applied to both.
*   **Measured Gain**: **~0.11 m/step**.
*   **True Physics**: Velocity (3.0) * Time (0.16) = **0.50 m/step**.
*   **Discrepancy Condition**: Friction/Drag eats ~80% of the theoretical distance.

---

## C. Observation / State Given to the Agent

### C1. What the LLM Sees
*   **Schema**: Synthesized String (Not Raw JSON).
*   **Fields**:
    *   `GOAL`: Target X, Z.
    *   `CURRENT`: Agent X, Z.
    *   `DELTA`: dX, dZ, Euclidean Dist.
    *   `ACTUATION GAINS`: Gain X, Gain Z.
    *   `STUCK TELEMETRY`: `stuck_flag`, `stuck_count`, `last_stuck`, `subgoal`.
*   **History**: **None** (Stateless).
*   **Coordinates**: Rounded to `.2f` (Meters).
*   **Noise**: None added intentionally.

### C2. Success Signal
*   **Authoritative Check**: `checkTermination()` in `ThrowBlockTaskV2.js`.
*   **Condition**: `inBasket` (Bounding Box) **AND** `atRest` (Speed < 0.5).
*   **Mechanism**: **Throw** or **Push** (Logic allows either).

---

## D. LLM and Prompting Details

### D1. Model + Runtime
*   **Model**: `gemini-2.0-flash`.
*   **Endpoint**: Google Generative AI API.
*   **Retry Policy**: None explicit in `ExperimentRunner` (Basic `try/except` returns no-op).
*   **JSON Enforcement**: Regex parsing (`replace("```json", "")`). Invalid JSON -> `Default Action (Idle)`.

### D2. Prompt Content
*   **System Prompt**: None.
*   **User Prompt**:
    *   **Verbatim Template**: See `run_planner` method in `experiment_final_sweep.py`.
    *   **Physics**: "ACTUATION GAINS (measured): gain_x=...".
    *   **Z-Move**: Yes ("...DETOUR in Z").
    *   **Rules**: Explicit State Machine rules ("If stuck_count >= 2... DETOUR").

---

## E. Controller / Executive Logic

### E1. Attempt Definition
*   **1 Attempt**: One LLM Inference -> One Action Sequence.
*   **Sub-steps**: Yes. Movement outputs convert to `N` environment steps based on Gain.
    *   Formula: `steps = round(meters / gain)`.
*   **Budget**: **100 Attempts** (Max).
*   **Allocation**: Shared Global Budget.

### E2. Stuck Detection
*   **Stalled Threshold**: `delta < 0.02m` (`EPS_MOTION`).
*   **Context**: "Behind Wall" (`current_x < obstacle_plane_x`).
*   **Axis**: Euclidean Delta (`total_delta`).
*   **Count**: **2** Consecutive Stalls -> Trigger REROUTE.
*   **Shimmy Prevention**: None explicit, but `last_stuck` is tracked.

### E3. Lane Search
*   **Candidates**: `[1.5, 2.0, 2.2, 2.5, 3.0]`.
*   **Order**: Best-First (Smallest Deviation first). (Current logic: `index 0..4`).
*   **Rejection**: Probe Failure (`dx < 0.1` after 2 steps).
*   **Probe Definition**: 2 Steps Forward (`0.5m` command -> ~0.11m actual).
*   **Success**: `cumulative_dx > 0.10`.
*   **Next State**: `PASS` (Waypoint `X = Obstacle + 0.5`).
*   **Refinement**: Coarse candidates only.

### E4. Forced Retreat
*   **Command**: `X = -0.5` (Strong Back).
*   **Condition**: `current_x > obstacle - 1.2` (Too close specifically in OFFSET phase).
*   **Resume**: Agent stays in `OFFSET` until safe.

### E5. Gain Update Rules
*   **EMA Alpha**: **0.3** (`New = 0.7*Old + 0.3*Inst`).
*   **Gating**: None explicit for epsilon; allows updates unless Anomaly.
*   **Anomaly**: `delta > 0.8m`. (Prevents gain explosion from teleportation).
*   **Reset**: Gain is preserved across teleports (heuristic choice).

---

## F. Failure Taxonomy and Diagnostics

### F1. Canonical Codes
*   **Codes**: `SUCCESS`, `UNSAT_WEDGED`, `FAIL_INSTABILITY`, `FAIL_POLICY`.
*   **Mutually Exclusive**: Yes.
*   **Precedence**: `UNSAT_WEDGED` (Immediate) > `SUCCESS` > `FAIL_POLICY` (Timeout).
*   **Location**: `experiment_final_sweep.py` Main Loop & Result processing.

### F2. UNSAT_WEDGED Diagnostic
*   **Probe**: 4-Way Cross (+X, -X, +Z, -Z).
*   **Threshold**: `delta < 0.05`.
*   **Logic**: If **ALL 4** < 0.05 -> Wedged.
*   **Trigger**: `stuck_counter >= 5` in `OFFSET` phase.
*   **Evidence**: See Logs (e.g., `seed 2004`).

### F3. FAIL_INSTABILITY
*   **Definition**: `delta > 0.8m` (Jump).
*   **Detection**: Per-Step.
*   **Reset**: Does NOT trigger Engine Reset; Agent continues (but usually in bad state).
*   **Labeling**: Post-hoc classification if jump detected.

### F4. FAIL_POLICY
*   **Condition**: `attempts >= 100` AND Not Wedged.
*   **Evidence of Path**: Humans can solve it. Baseline N=30 solved 85%.

---

## G. Results Reporting (N=30 Headline)

### G1. Headline Sweep (N=30)
*   **Note**: The specific `final_classified_results.json` was overwritten by the N=100 run.
*   **Reported Stats (from Logs)**:
    *   **SUCCESS**: 25/30 (83.3%).
    *   **UNSAT_WEDGED**: 3/30 (Seeds 2000, 2005, 2010).
    *   **FAIL_POLICY**: 2/30.
*   **Cost**: ~42 LLM calls median.

### G2. Key Ablations
*   **No Calibration**: Success < 20% (Gain Mismatch).
*   **No Lane Search**: Success ~40% (Only succeeds on easy geometries).

### G3. Gravity
*   **Final Configuration**: **1g** (9.81).
*   **Note**: System ID allows 0.5g adaptation (Gain increases), but final benchmark used 1g.

---

## H. The N=100 Errors

### H1. The Error
*   **Error**: **Infrastructure Saturation**.
*   **Result**: 0% Success.
*   **Reason**: Node.js Event Loop Blocking.
*   **Reproducibility**: 100% reproducible with `max_workers=10`.
*   **Impact**: Proves Architecture is not scalable for Parallel Evaluation.

---

## I. Reproducibility Essentials

*   **Commit**: (Current).
*   **Command**: `python3 experiment_final_sweep.py` (Default: Serial).
*   **Dependencies**: `requirements.txt`.
*   **Server**: `npm run server` (Port 3000).
*   **Hardware**: Mac M-Series (ARM64).
