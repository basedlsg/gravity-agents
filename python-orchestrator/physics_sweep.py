"""
Physics Sweep - Ground truth validation
========================================
Tests which jump_step values actually succeed for each (task, gravity) combination.
No LLM involved - pure scripted execution.
"""

import requests
import json
from config import ENV_SERVER_URL

TASK_VERSION = "v2"

# Task geometries - UPDATED based on physics sweep results
# goalPlatformWidth extends the physical platform to accommodate long jumps at 0.5g
TASK_A = {"land_start": 7.0, "land_end": 20.0, "name": "Task A (invariant, 7-20m)", "goalPlatformWidth": 16.0}
# Task B: Based on Task A results:
# - At 1g: jump@6 lands at ~9.35m
# - At 0.5g: jump@6 lands at ~10.65m
# Zone 8.5-10.0 should accept 1g jump@6 (9.35 in zone) but reject 0.5g jump@6 (10.65 out)
# At 0.5g, jump@3 lands at ~9.67m which IS in 8.5-10.0, so there IS a solution
TASK_B = {"land_start": 8.5, "land_end": 10.0, "name": "Task B (adaptive, 8.5-10m)", "goalPlatformWidth": 16.0}

GRAVITIES = {
    "1g": 9.81,
    "0.5g": 4.905
}


def compile_sequence(jump_step: int, total_steps: int = 30) -> list[str]:
    """Deterministic sequence: (jump_step) forwards, then jump, then forward."""
    return ["forward"] * jump_step + ["jump"] + ["forward"] * (total_steps - jump_step - 1)


def run_sweep_episode(gravity: float, jump_step: int, task_geometry: dict) -> dict:
    """Run single episode with scripted sequence."""

    # Reset
    response = requests.post(
        f"{ENV_SERVER_URL}/reset",
        json={
            "task": "gap",
            "taskVersion": TASK_VERSION,
            "gravity": gravity,
            "landingZoneStart": task_geometry["land_start"],
            "landingZoneEnd": task_geometry["land_end"],
            "goalPlatformWidth": task_geometry.get("goalPlatformWidth", 4.0)
        },
        timeout=10
    )
    response.raise_for_status()
    obs = response.json().get("observation", response.json())

    sequence = compile_sequence(jump_step)

    trajectory = []
    max_x = -999
    jump_x = None
    landed_x = None

    for step, action in enumerate(sequence):
        # Record state
        x = obs["agentPosition"][0]
        y = obs["agentPosition"][1]
        trajectory.append({"step": step, "x": x, "y": y, "action": action})

        if x > max_x:
            max_x = x

        if action == "jump":
            jump_x = x

        # Execute
        response = requests.post(
            f"{ENV_SERVER_URL}/step",
            json={"action": action},
            timeout=10
        )
        result = response.json()
        obs = result.get("observation", {})
        done = result.get("done", False)
        info = result.get("info", {})

        if done:
            landed_x = obs.get("agentPosition", [0, 0, 0])[0]
            return {
                "jump_step": jump_step,
                "success": info.get("success", False),
                "reason": info.get("reason", "unknown"),
                "jump_x": jump_x,
                "landed_x": landed_x,
                "max_x": max_x,
                "steps": step + 1
            }

    # Timeout
    landed_x = obs.get("agentPosition", [0, 0, 0])[0]
    return {
        "jump_step": jump_step,
        "success": False,
        "reason": "timeout",
        "jump_x": jump_x,
        "landed_x": landed_x,
        "max_x": max_x,
        "steps": len(sequence)
    }


def run_full_sweep():
    """Run sweep across all conditions."""

    print("=" * 80)
    print("PHYSICS SWEEP: Ground Truth Validation")
    print("=" * 80)
    print("Testing jump_step = 3, 4, 5, 6, 7, 8 for each (task, gravity)")
    print()

    # Check server
    try:
        response = requests.get(f"{ENV_SERVER_URL}/health", timeout=5)
        print(f"Server: {response.json()}")
    except Exception as e:
        print(f"ERROR: {e}")
        return

    results = {}

    for task_name, task_geo in [("Task_A", TASK_A), ("Task_B", TASK_B)]:
        for grav_name, grav_val in GRAVITIES.items():
            condition = f"{task_name}_{grav_name}"
            print(f"\n{'='*60}")
            print(f"{task_geo['name']} at {grav_name} (g={grav_val})")
            print(f"Landing zone: x={task_geo['land_start']} to x={task_geo['land_end']}")
            print(f"{'='*60}")

            condition_results = []

            for jump_step in [3, 4, 5, 6, 7, 8]:
                # Run 3 trials to check consistency
                trials = []
                for trial in range(3):
                    result = run_sweep_episode(grav_val, jump_step, task_geo)
                    trials.append(result)

                # Aggregate
                successes = sum(1 for t in trials if t["success"])
                avg_landed_x = sum(t["landed_x"] for t in trials if t["landed_x"]) / 3

                status = "✓" if successes == 3 else ("~" if successes > 0 else "✗")
                print(f"  jump@{jump_step}: {status} ({successes}/3) | "
                      f"landed_x={avg_landed_x:.2f}m | "
                      f"reason={trials[0]['reason']}")

                condition_results.append({
                    "jump_step": jump_step,
                    "successes": successes,
                    "avg_landed_x": avg_landed_x,
                    "trials": trials
                })

            results[condition] = condition_results

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Which jump_step values succeed?")
    print("=" * 80)
    print()
    print(f"{'Condition':<25} | " + " | ".join(f"s={s}" for s in [3, 4, 5, 6, 7, 8]))
    print("-" * 80)

    for condition, data in results.items():
        row = f"{condition:<25} | "
        for item in data:
            successes = item["successes"]
            if successes == 3:
                row += "  ✓  | "
            elif successes > 0:
                row += f" {successes}/3 | "
            else:
                row += "  ✗  | "
        print(row)

    # Landing positions table
    print("\n" + "=" * 80)
    print("LANDING POSITIONS (avg x)")
    print("=" * 80)
    print()
    print(f"{'Condition':<25} | " + " | ".join(f"s={s:>5}" for s in [3, 4, 5, 6, 7, 8]))
    print("-" * 80)

    for condition, data in results.items():
        row = f"{condition:<25} | "
        for item in data:
            x = item["avg_landed_x"]
            row += f" {x:>5.1f} | "
        print(row)

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Task A analysis
    task_a_1g = results.get("Task_A_1g", [])
    task_a_05g = results.get("Task_A_0.5g", [])

    print("\nTask A (wide landing: 7-12m):")
    a_1g_success = [d["jump_step"] for d in task_a_1g if d["successes"] == 3]
    a_05g_success = [d["jump_step"] for d in task_a_05g if d["successes"] == 3]
    print(f"  1g success steps: {a_1g_success}")
    print(f"  0.5g success steps: {a_05g_success}")

    common = set(a_1g_success) & set(a_05g_success)
    if common:
        print(f"  INVARIANT? Yes - steps {common} work at both gravities")
    else:
        print(f"  INVARIANT? NO - no common success steps!")

    # Task B analysis
    task_b_1g = results.get("Task_B_1g", [])
    task_b_05g = results.get("Task_B_0.5g", [])

    print("\nTask B (narrow landing: 7-8m):")
    b_1g_success = [d["jump_step"] for d in task_b_1g if d["successes"] == 3]
    b_05g_success = [d["jump_step"] for d in task_b_05g if d["successes"] == 3]
    print(f"  1g success steps: {b_1g_success}")
    print(f"  0.5g success steps: {b_05g_success}")

    if b_1g_success != b_05g_success:
        print(f"  ADAPTIVE? Yes - requires different timing")
    else:
        print(f"  ADAPTIVE? No - same timing works")

    # Save results
    with open("physics_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to physics_sweep_results.json")

    return results


if __name__ == "__main__":
    run_full_sweep()
