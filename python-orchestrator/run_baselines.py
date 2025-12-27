"""
Baseline Agent Runner
=====================
Runs optimal, random, and heuristic agents to validate task difficulty.

Expected results:
- Optimal: ~100% success
- Heuristic: 40-70% success
- Random: 0-5% success
"""

import json
import time
import requests
from dataclasses import dataclass, asdict
from typing import Callable

from env_config_v2 import (
    ENV_CONFIG,
    get_optimal_action,
    get_random_action,
    get_heuristic_action,
    reset_heuristic_state,
    BaselineResults
)
from config import ENV_SERVER_URL

TASK_VERSION = "v2"


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    reason: str
    final_x: float
    actions: list[str]


def run_episode(
    get_action: Callable,
    gravity: float = ENV_CONFIG.training_gravity,
    max_steps: int = ENV_CONFIG.max_steps
) -> EpisodeResult:
    """Run a single episode with the given action function."""

    # Reset
    response = requests.post(
        f"{ENV_SERVER_URL}/reset",
        json={
            "task": "gap",
            "taskVersion": TASK_VERSION,
            "gravity": gravity
        },
        timeout=10
    )
    response.raise_for_status()
    obs = response.json().get("observation", response.json())

    actions = []
    step = 0

    while step < max_steps:
        # Get action based on agent type
        if get_action.__name__ == 'get_optimal_action':
            action = get_action(step)
        elif get_action.__name__ == 'get_heuristic_action':
            action = get_action(obs)
        else:  # random
            action = get_action()

        actions.append(action)

        # Step environment
        response = requests.post(
            f"{ENV_SERVER_URL}/step",
            json={"action": action},
            timeout=10
        )
        response.raise_for_status()
        result = response.json()

        obs = result.get("observation", {})
        done = result.get("done", False)
        info = result.get("info", {})

        step += 1

        if done:
            return EpisodeResult(
                success=info.get("success", False),
                steps=step,
                reason=info.get("reason", "unknown"),
                final_x=obs.get("agentPosition", [0, 0, 0])[0],
                actions=actions
            )

    # Timeout
    return EpisodeResult(
        success=False,
        steps=step,
        reason="timeout",
        final_x=obs.get("agentPosition", [0, 0, 0])[0],
        actions=actions
    )


def run_baseline(
    agent_name: str,
    get_action: Callable,
    num_episodes: int = 50,
    verbose: bool = True
) -> BaselineResults:
    """Run multiple episodes for a baseline agent."""

    successes = 0
    failures = 0
    success_steps = []
    failure_steps = []
    failure_reasons = {}

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running {agent_name} agent for {num_episodes} episodes")
        print(f"{'='*60}")

    for ep in range(num_episodes):
        # Reset heuristic state between episodes
        if get_action.__name__ == 'get_heuristic_action':
            reset_heuristic_state()
        result = run_episode(get_action)

        if result.success:
            successes += 1
            success_steps.append(result.steps)
        else:
            failures += 1
            failure_steps.append(result.steps)
            failure_reasons[result.reason] = failure_reasons.get(result.reason, 0) + 1

        if verbose and (ep + 1) % 10 == 0:
            rate = successes / (ep + 1) * 100
            print(f"  Episode {ep+1}/{num_episodes}: {successes} successes ({rate:.1f}%)")

    success_rate = successes / num_episodes
    avg_success_steps = sum(success_steps) / len(success_steps) if success_steps else 0
    avg_failure_steps = sum(failure_steps) / len(failure_steps) if failure_steps else 0

    results = BaselineResults(
        agent_name=agent_name,
        num_episodes=num_episodes,
        successes=successes,
        failures=failures,
        success_rate=success_rate,
        avg_steps_to_success=avg_success_steps,
        avg_steps_to_failure=avg_failure_steps,
        failure_reasons=failure_reasons
    )

    if verbose:
        print(f"\n{agent_name} Results:")
        print(f"  Success rate: {success_rate*100:.1f}%")
        print(f"  Avg steps to success: {avg_success_steps:.1f}")
        print(f"  Avg steps to failure: {avg_failure_steps:.1f}")
        print(f"  Failure reasons: {failure_reasons}")

    return results


def main():
    print("=" * 70)
    print("BASELINE AGENT VALIDATION")
    print("=" * 70)
    print(f"Environment: gap task v2")
    print(f"Gravity: {ENV_CONFIG.training_gravity} m/s²")
    print(f"Max steps: {ENV_CONFIG.max_steps}")
    print()

    # Check server
    try:
        response = requests.get(f"{ENV_SERVER_URL}/health", timeout=5)
        print(f"Server status: {response.json()}")
    except Exception as e:
        print(f"ERROR: Cannot connect to server at {ENV_SERVER_URL}")
        print(f"Start the server first: PORT=3002 node server.js")
        return

    # Run baselines
    results = {}

    # 1. Optimal agent
    results['optimal'] = run_baseline(
        "Optimal",
        get_optimal_action,
        num_episodes=50
    )

    # 2. Random agent
    results['random'] = run_baseline(
        "Random",
        get_random_action,
        num_episodes=50
    )

    # 3. Heuristic agent
    results['heuristic'] = run_baseline(
        "Heuristic",
        get_heuristic_action,
        num_episodes=50
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Agent':<15} {'Success Rate':<15} {'Avg Success Steps':<20} {'Avg Failure Steps'}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{r.agent_name:<15} {r.success_rate*100:>6.1f}%        {r.avg_steps_to_success:>10.1f}          {r.avg_steps_to_failure:>10.1f}")

    # Validate difficulty
    print("\n" + "=" * 70)
    print("DIFFICULTY VALIDATION")
    print("=" * 70)

    optimal_ok = results['optimal'].success_rate >= 0.95
    random_ok = results['random'].success_rate <= 0.10
    heuristic_ok = 0.30 <= results['heuristic'].success_rate <= 0.80

    print(f"Optimal ≈ 100%: {'PASS' if optimal_ok else 'FAIL'} ({results['optimal'].success_rate*100:.1f}%)")
    print(f"Random ≈ 0-10%: {'PASS' if random_ok else 'FAIL'} ({results['random'].success_rate*100:.1f}%)")
    print(f"Heuristic ≈ 30-80%: {'PASS' if heuristic_ok else 'FAIL'} ({results['heuristic'].success_rate*100:.1f}%)")

    if optimal_ok and random_ok and heuristic_ok:
        print("\n✓ Task difficulty is in the correct range!")
    else:
        print("\n✗ Task difficulty needs adjustment")

    # Save results
    with open("baseline_results.json", "w") as f:
        json.dump({name: asdict(r) for name, r in results.items()}, f, indent=2)
    print("\nResults saved to baseline_results.json")


if __name__ == "__main__":
    main()
