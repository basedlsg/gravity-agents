"""Quick test of gravity experiment with step mapping hint."""

import json
import time
import requests
from config import ENV_SERVER_URL
from llm_policy_v3_experiment import ExperimentConfig, GravityExperimentPolicy

TASK_VERSION = "v2"
TRAINING_GRAVITY = 9.81
TEST_GRAVITY = 4.905


def run_quick_test(num_episodes=10):
    """Quick test of NRL-F at training gravity."""
    print("=" * 60)
    print("QUICK TEST: NRL-F with step mapping hint")
    print("=" * 60)

    # Check server
    try:
        response = requests.get(f"{ENV_SERVER_URL}/health", timeout=5)
        print(f"Server: {response.json()}")
    except Exception as e:
        print(f"ERROR: {e}")
        return

    successes = 0
    jump_positions = []

    for ep in range(num_episodes):
        config = ExperimentConfig(
            agent_type="NRL-F",
            gravity_condition="training"
        )
        policy = GravityExperimentPolicy(config)

        # Reset
        response = requests.post(
            f"{ENV_SERVER_URL}/reset",
            json={"task": "gap", "taskVersion": TASK_VERSION, "gravity": TRAINING_GRAVITY},
            timeout=10
        )
        obs = response.json().get("observation", response.json())

        policy.reset()

        # Plan
        start = time.time()
        plan = policy.plan_episode(obs)
        latency = time.time() - start

        sequence = plan.get("sequence", [])
        jump_pos = sequence.index("jump") if "jump" in sequence else -1
        jump_positions.append(jump_pos)

        print(f"\nEpisode {ep+1}:")
        print(f"  Sequence ({len(sequence)} actions): {sequence[:15]}...")
        print(f"  Jump position: {jump_pos}")
        if plan.get("physics_reasoning"):
            print(f"  Physics: {plan['physics_reasoning'][:100]}...")

        # Execute
        step = 0
        while step < 80:
            action, _ = policy.select_action(obs, "training")
            response = requests.post(
                f"{ENV_SERVER_URL}/step",
                json={"action": action},
                timeout=10
            )
            result = response.json()
            obs = result.get("observation", {})
            done = result.get("done", False)
            info = result.get("info", {})
            step += 1

            if done:
                if info.get("success"):
                    successes += 1
                    print(f"  -> SUCCESS in {step} steps")
                else:
                    print(f"  -> FAIL ({info.get('reason')}) in {step} steps")
                break

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Success rate: {successes}/{num_episodes} = {successes/num_episodes*100:.0f}%")
    print(f"Avg jump position: {sum(jump_positions)/len(jump_positions):.1f} (optimal ~6)")
    print(f"Jump positions: {jump_positions}")


if __name__ == "__main__":
    run_quick_test(10)
