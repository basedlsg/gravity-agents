"""
Quick test of v2 experiments - just 3 episodes per condition
"""

import os
import json
import time
from datetime import datetime
import requests

from config import ENV_SERVER_URL, GravityConfig
from llm_policy_v2 import PolicyConfigV2, LLMPolicyV2, ValueCachedPolicy

TASK_VERSION = "v2"


def reset_environment(task: str, gravity: float, seed: int = None) -> dict:
    """Reset environment and return initial observation"""
    response = requests.post(
        f"{ENV_SERVER_URL}/reset",
        json={
            "task": task,
            "taskVersion": TASK_VERSION,
            "gravity": gravity,
            "seed": seed
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


def run_episode(policy, task: str, gravity: float, gravity_condition: str, max_steps: int = 100):
    """Run a single episode"""
    obs = reset_environment(task, gravity)
    policy.reset()

    actions_taken = []
    total_reward = 0.0
    done = False
    step = 0

    while not done and step < max_steps:
        action, response_data = policy.select_action(obs, gravity_condition)
        actions_taken.append(action)

        result = step_environment(action)
        obs = result.get("observation", {})
        reward = result.get("reward", 0)
        done = result.get("done", False)
        total_reward += reward
        step += 1

        # Early output
        if step == 1:
            print(f"    Step 1: action={action}")
        if done:
            print(f"    Done at step {step}: {result.get('info', {}).get('reason', 'unknown')}")

    success = result.get("info", {}).get("success", False)
    return {
        "success": success,
        "steps": step,
        "reward": total_reward,
        "action_counts": {a: actions_taken.count(a) for a in set(actions_taken)}
    }


def main():
    print("Quick V2 Experiment Test")
    print("=" * 60)

    # Check server
    try:
        response = requests.get(f"{ENV_SERVER_URL}/health", timeout=5)
        print(f"Server: {response.json()}")
    except Exception as e:
        print(f"ERROR: Cannot connect to server: {e}")
        return

    gravity_config = GravityConfig()

    # Test just NRL-F on gap task with 3 episodes per condition
    agent_type = "NRL-F"
    task = "gap"

    print(f"\nTesting {agent_type} on {task}")
    print("-" * 40)

    config = PolicyConfigV2(agent_type=agent_type, task=task)
    policy = LLMPolicyV2(config)

    conditions = {
        "training": gravity_config.training,
        "test_silent": gravity_config.test,
        "test_explained": gravity_config.test
    }

    results = {}
    for condition, gravity in conditions.items():
        print(f"\nCondition: {condition} (g={gravity})")
        successes = 0

        for ep in range(3):  # Just 3 episodes
            print(f"  Episode {ep+1}/3:")
            result = run_episode(policy, task, gravity, condition)
            if result["success"]:
                successes += 1
            print(f"    Result: {'SUCCESS' if result['success'] else 'FAIL'}, actions: {result['action_counts']}")

        results[condition] = successes / 3
        print(f"  Success rate: {results[condition]:.0%}")

    print("\n" + "=" * 60)
    print("QUICK TEST SUMMARY")
    print("=" * 60)
    print(f"Agent: {agent_type}, Task: {task}")
    for cond, rate in results.items():
        print(f"  {cond}: {rate:.0%}")

    adaptation_gap = results.get("test_explained", 0) - results.get("test_silent", 0)
    print(f"\nAdaptation gap (explained - silent): {adaptation_gap:+.0%}")


if __name__ == "__main__":
    main()
