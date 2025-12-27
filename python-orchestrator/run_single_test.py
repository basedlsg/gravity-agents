"""
Single episode test - verify v2 experiments work
"""

import os
import json
import time
import requests

from config import ENV_SERVER_URL, GravityConfig
from llm_policy_v2 import PolicyConfigV2, LLMPolicyV2

TASK_VERSION = "v2"


def reset_environment(task: str, gravity: float) -> dict:
    """Reset environment and return initial observation"""
    response = requests.post(
        f"{ENV_SERVER_URL}/reset",
        json={
            "task": task,
            "taskVersion": TASK_VERSION,
            "gravity": gravity
        },
        timeout=10
    )
    response.raise_for_status()
    data = response.json()
    return data.get("observation", data)


def step_environment(action: str) -> dict:
    """Take action and return result"""
    response = requests.post(
        f"{ENV_SERVER_URL}/step",
        json={"action": action},
        timeout=10
    )
    response.raise_for_status()
    return response.json()


def run_single_episode():
    """Run a single episode with verbose output"""
    print("=" * 60)
    print("SINGLE EPISODE TEST")
    print("=" * 60)

    # Check server
    try:
        response = requests.get(f"{ENV_SERVER_URL}/health", timeout=5)
        print(f"Server: {response.json()}")
    except Exception as e:
        print(f"ERROR: Cannot connect to server: {e}")
        return

    # Setup
    gravity_config = GravityConfig()
    task = "gap"
    gravity = gravity_config.training  # 9.81

    print(f"\nTask: {task}")
    print(f"Gravity: {gravity}")

    # Create policy
    config = PolicyConfigV2(agent_type="NRL-F", task=task)
    policy = LLMPolicyV2(config)

    # Reset environment
    print("\n--- Reset Environment ---")
    obs = reset_environment(task, gravity)
    print(f"Initial position: {obs.get('agentPosition', 'N/A')}")
    print(f"Gap: {obs.get('gapStart', 'N/A')} to {obs.get('gapEnd', 'N/A')}")

    # Run up to 250 steps
    max_steps = 250
    policy.reset()

    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")

        # Get action
        start = time.time()
        action, response_data = policy.select_action(obs, "training")
        api_time = time.time() - start

        print(f"  Action: {action} (LLM took {api_time:.2f}s)")

        # Check for physics calculation
        if "physics_calculation" in response_data:
            calc = response_data["physics_calculation"]
            print(f"  Physics: {calc[:80]}..." if len(calc) > 80 else f"  Physics: {calc}")

        # Step environment
        result = step_environment(action)
        obs = result.get("observation", {})
        done = result.get("done", False)
        success = result.get("info", {}).get("success", False)

        print(f"  Position: x={obs.get('agentPosition', [0,0,0])[0]:.2f}")
        print(f"  Done: {done}, Success: {success}")

        if done:
            reason = result.get("info", {}).get("reason", "unknown")
            print(f"\n=== EPISODE ENDED ===")
            print(f"Reason: {reason}")
            print(f"Success: {success}")
            print(f"Steps: {step + 1}")
            return success

    print(f"\n=== EPISODE ENDED (timeout) ===")
    print(f"Steps: {max_steps}")
    return False


if __name__ == "__main__":
    run_single_episode()
