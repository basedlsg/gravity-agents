"""
LLM Baseline Runner
===================
Runs NRL-F (non-RL, formula-based) agent for baseline measurement.

This measures how well the LLM can control the agent without any RL training,
just using the physics formulas in its prompt.
"""

import json
import time
import requests
from dataclasses import dataclass, asdict
from typing import Optional

from env_config_v2 import ENV_CONFIG
from config import ENV_SERVER_URL
from llm_policy_v2 import PolicyConfigV2, LLMPolicyV2

TASK_VERSION = "v2"


@dataclass
class LLMEpisodeResult:
    episode_id: int
    success: bool
    steps: int
    reason: str
    final_x: float
    actions: list[str]
    llm_latencies: list[float]
    physics_calculations: list[str]
    trajectory: list[dict]


def run_llm_episode(
    policy: LLMPolicyV2,
    episode_id: int,
    gravity: float = ENV_CONFIG.training_gravity,
    max_steps: int = ENV_CONFIG.max_steps,
    verbose: bool = False
) -> LLMEpisodeResult:
    """Run a single episode with the LLM policy."""

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

    policy.reset()

    actions = []
    llm_latencies = []
    physics_calculations = []
    trajectory = []

    step = 0
    while step < max_steps:
        # Record state
        trajectory.append({
            "step": step,
            "x": obs["agentPosition"][0],
            "y": obs["agentPosition"][1],
            "vx": obs["agentVelocity"][0],
            "grounded": obs["isGrounded"]
        })

        # Get action from LLM
        start = time.time()
        action, response_data = policy.select_action(obs, "training")
        latency = time.time() - start

        actions.append(action)
        llm_latencies.append(latency)

        if "physics_calculation" in response_data:
            physics_calculations.append(response_data["physics_calculation"])

        if verbose:
            x = obs["agentPosition"][0]
            print(f"  Step {step+1}: x={x:.2f}, action={action} ({latency:.2f}s)")

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
            return LLMEpisodeResult(
                episode_id=episode_id,
                success=info.get("success", False),
                steps=step,
                reason=info.get("reason", "unknown"),
                final_x=obs.get("agentPosition", [0, 0, 0])[0],
                actions=actions,
                llm_latencies=llm_latencies,
                physics_calculations=physics_calculations,
                trajectory=trajectory
            )

    # Timeout
    return LLMEpisodeResult(
        episode_id=episode_id,
        success=False,
        steps=step,
        reason="timeout",
        final_x=obs.get("agentPosition", [0, 0, 0])[0],
        actions=actions,
        llm_latencies=llm_latencies,
        physics_calculations=physics_calculations,
        trajectory=trajectory
    )


def run_llm_baseline(
    num_episodes: int = 20,
    use_groq: bool = True,  # Groq is faster
    verbose: bool = True
) -> dict:
    """Run the full LLM baseline."""

    print("=" * 70)
    print("LLM BASELINE (NRL-F)")
    print("=" * 70)
    print(f"Agent: NRL-F (Non-RL, Formula-based)")
    print(f"Model: {'Groq llama3' if use_groq else 'Gemini 2.0 Flash'}")
    print(f"Episodes: {num_episodes}")
    print(f"Max steps: {ENV_CONFIG.max_steps}")
    print()

    # Check server
    try:
        response = requests.get(f"{ENV_SERVER_URL}/health", timeout=5)
        print(f"Server: {response.json()}")
    except Exception as e:
        print(f"ERROR: Cannot connect to server: {e}")
        return {}

    # Create policy
    config = PolicyConfigV2(
        agent_type="NRL-F",
        task="gap",
        use_groq=use_groq,
        model="llama3-70b-8192" if use_groq else "gemini-2.0-flash"
    )
    policy = LLMPolicyV2(config)

    results = []
    successes = 0
    total_steps = 0
    total_latency = 0

    print(f"\nRunning {num_episodes} episodes...")
    print("-" * 70)

    for ep in range(num_episodes):
        if verbose:
            print(f"\nEpisode {ep+1}/{num_episodes}")

        result = run_llm_episode(policy, ep, verbose=verbose)
        results.append(result)

        if result.success:
            successes += 1

        total_steps += result.steps
        total_latency += sum(result.llm_latencies)

        if verbose:
            status = "SUCCESS" if result.success else f"FAIL ({result.reason})"
            print(f"  -> {status} in {result.steps} steps")

        # Progress update every 5 episodes
        if (ep + 1) % 5 == 0:
            rate = successes / (ep + 1) * 100
            avg_latency = total_latency / total_steps if total_steps > 0 else 0
            print(f"\n  Progress: {ep+1}/{num_episodes}, Success: {successes} ({rate:.1f}%), Avg latency: {avg_latency:.2f}s")

    # Summary
    success_rate = successes / num_episodes
    avg_steps = total_steps / num_episodes
    avg_latency = total_latency / total_steps if total_steps > 0 else 0

    success_results = [r for r in results if r.success]
    failure_results = [r for r in results if not r.success]

    failure_reasons = {}
    for r in failure_results:
        failure_reasons[r.reason] = failure_reasons.get(r.reason, 0) + 1

    summary = {
        "agent": "NRL-F",
        "model": config.model,
        "num_episodes": num_episodes,
        "successes": successes,
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "avg_latency_per_step": avg_latency,
        "avg_steps_success": sum(r.steps for r in success_results) / len(success_results) if success_results else 0,
        "avg_steps_failure": sum(r.steps for r in failure_results) / len(failure_results) if failure_results else 0,
        "failure_reasons": failure_reasons
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Success rate: {success_rate*100:.1f}%")
    print(f"Avg steps: {avg_steps:.1f}")
    print(f"Avg latency per step: {avg_latency:.2f}s")
    print(f"Failure reasons: {failure_reasons}")

    # Action distribution
    all_actions = []
    for r in results:
        all_actions.extend(r.actions)
    action_counts = {}
    for a in all_actions:
        action_counts[a] = action_counts.get(a, 0) + 1
    print(f"\nAction distribution: {action_counts}")

    # Compare to baselines
    print("\n" + "-" * 70)
    print("COMPARISON TO BASELINES")
    print("-" * 70)
    print(f"Optimal:   100.0%")
    print(f"LLM:       {success_rate*100:>5.1f}%  {'< better than random' if success_rate > 0 else '= same as random'}")
    print(f"Heuristic:  44.0%")
    print(f"Random:      0.0%")

    # Save results
    output = {
        "summary": summary,
        "episodes": [asdict(r) for r in results]
    }
    with open("llm_baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to llm_baseline_results.json")

    return summary


if __name__ == "__main__":
    # Use Gemini since Groq llama3 is deprecated
    run_llm_baseline(num_episodes=20, use_groq=False, verbose=True)
