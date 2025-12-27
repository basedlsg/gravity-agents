# Gravity Agents Experiment Data Manifest

This archive contains the raw JSON output files from the "Gravity Agents" research project.
Use this data to visualize performance, system identification stability, and failure modes.

## Primary Datasets

### 1. `final_classified_results_N100.json` (Latest)
*   **Description**: The "Stress Test" run with N=100 seeds (2000-2099) executed in PARALLEL.
*   **Result**: 0% Success.
*   **Purpose**: Demonstrates Infrastructure Saturation (Node.js starvation).
*   **Schema**: List of Episode Objects (`seed`, `success`, `status`, `attempts`).

### 2. `final_classified_results.json` (Canonical Logic)
*   **Description**: A smaller serial run (N=30) demonstrating the "True" agent performance.
*   **Result**: ~85% Success.
*   **Purpose**: Proof of Agent Logic (Reroute, Probe, Retreat).
*   **Note**: This file might have been overwritten during testing; if so, refer to `final_sweep_results.json` or `experiment_v2_results.json` for historical successful runs.

---

## Historical / Dev Datasets

### 3. `experiment_v2_results.json` (Adaptive Gain)
*   **Description**: The main development dataset for the "Adaptive Gain" (EMA) controller.
*   **Significance**: Shows the first major breakthrough in reducing "overshoot" errors.
*   **Size**: Large (~1.8MB), contains full step-by-step traces for many seeds.

### 4. `physics_sweep_results.json` (System ID)
*   **Description**: Results from `physics_sweep.py`.
*   **Purpose**: Actuation Calibration data.
*   **Content**: Mapping of `Action Duration` -> `Observed Displacement`. This established the "Gain Mismatch" theory (0.5m theoretical vs 0.11m observed).

### 5. `gravity_experiment_results.json` (Generalization)
*   **Description**: Results testing the agent on `0.5g` and `2.0g` without retraining.
*   **Purpose**: Demonstrated the "Adaptive Gain" controller's ability to zero-shot adapt to new physics constants.

### 6. `all_metrics_raw.json`
*   **Description**: A raw aggregation of early development metrics. Useful for plotting "Learning Curves" of the project logic itself.

### 7. `llm_baseline_results.json`
*   **Description**: Performance of the LLM *without* the Python Executive wrapper (Pure Prompting).
*   **Result**: < 10% Success.
*   **Purpose**: Functions as the "Ablation Baseline" proving the specific value of the Loop logic.

---

## Data Schema (General)

Most JSON files follow this structure:
```json
[
  {
    "seed": 2000,
    "success": false,
    "attempts": 69,
    "status": "UNSAT_WEDGED",
    "final_dist_from_basket": 2.07,
    "diagnostic_data": [ ... ]
  },
  ...
]
```
*   **status**: `SUCCESS`, `FAIL_POLICY`, `UNSAT_WEDGED`, `FAIL_INSTABILITY`.
*   **attempts**: Number of LLM inference steps used.
*   **diagnostic_data**: (If wedged) The 4-way probe delta values.
