"""
Gravity Experiment Runner
=========================
Tests the 3 hypotheses about LLM physics adaptation:

H1: Formula vs Story
    - NRL-F: Explicit physics equations in prompt
    - NRL-N: Intuitive story-based physics description

H2: RL vs No-RL (future - not implemented yet)
    - RL agents learn from experience
    - NRL agents use zero-shot planning only

H3: Explained vs Silent gravity change
    - test_explained: Prompt tells agent gravity has changed
    - test_silent: Prompt doesn't mention the change

Experiment Design:
- Training phase: 20 episodes at g=9.81 (normal gravity)
- Test phase: 20 episodes at g=4.905 (half gravity)
- Compare adaptation between conditions
"""

import json
import time
import requests
from dataclasses import dataclass, asdict
from typing import Literal

from env_config_v2 import ENV_CONFIG
from config import ENV_SERVER_URL
from llm_policy_v3_experiment import ExperimentConfig, GravityExperimentPolicy

TASK_VERSION = "v2"

# Gravity values
TRAINING_GRAVITY = 9.81
TEST_GRAVITY = 4.905  # Half gravity


@dataclass
class ExperimentResult:
    episode_id: int
    phase: str  # "training" or "test"
    condition: str  # "NRL-F_explained", "NRL-N_silent", etc.
    gravity: float
    success: bool
    steps: int
    reason: str
    final_x: float
    planned_sequence: list[str]
    jump_position: int  # Index of jump in sequence (-1 if no jump)
    llm_latency: float
    physics_reasoning: str
    strategy: str


def run_single_episode(
    policy: GravityExperimentPolicy,
    episode_id: int,
    phase: str,
    condition: str,
    gravity: float,
    max_steps: int = ENV_CONFIG.max_steps,
    verbose: bool = False
) -> ExperimentResult:
    """Run a single episode of the experiment."""

    # Reset environment with specified gravity
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

    # Get the plan
    start = time.time()
    plan_response = policy.plan_episode(obs)
    llm_latency = time.time() - start

    planned_sequence = plan_response.get("sequence", [])
    physics_reasoning = plan_response.get("physics_reasoning", "")
    strategy = plan_response.get("strategy", "")

    # Find jump position
    jump_position = -1
    if "jump" in planned_sequence:
        jump_position = planned_sequence.index("jump")

    if verbose:
        print(f"  Plan ({len(planned_sequence)} actions): jump at position {jump_position}")
        if physics_reasoning:
            print(f"  Physics: {physics_reasoning[:80]}...")

    # Execute the plan
    step = 0
    while step < max_steps:
        action, _ = policy.select_action(obs, phase)

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
            return ExperimentResult(
                episode_id=episode_id,
                phase=phase,
                condition=condition,
                gravity=gravity,
                success=info.get("success", False),
                steps=step,
                reason=info.get("reason", "unknown"),
                final_x=obs.get("agentPosition", [0, 0, 0])[0],
                planned_sequence=planned_sequence,
                jump_position=jump_position,
                llm_latency=llm_latency,
                physics_reasoning=physics_reasoning,
                strategy=strategy
            )

    return ExperimentResult(
        episode_id=episode_id,
        phase=phase,
        condition=condition,
        gravity=gravity,
        success=False,
        steps=step,
        reason="timeout",
        final_x=obs.get("agentPosition", [0, 0, 0])[0],
        planned_sequence=planned_sequence,
        jump_position=jump_position,
        llm_latency=llm_latency,
        physics_reasoning=physics_reasoning,
        strategy=strategy
    )


def run_condition(
    agent_type: Literal["NRL-F", "NRL-N"],
    test_condition: Literal["test_silent", "test_explained"],
    num_episodes: int = 20,
    verbose: bool = True
) -> dict:
    """Run both training and test phases for a single experimental condition."""

    condition_name = f"{agent_type}_{test_condition.split('_')[1]}"
    print(f"\n{'='*70}")
    print(f"CONDITION: {condition_name}")
    print(f"{'='*70}")
    print(f"Agent type: {agent_type} ({'Formula' if agent_type == 'NRL-F' else 'Story'})")
    print(f"Test mode: {test_condition}")
    print(f"Episodes: {num_episodes} training + {num_episodes} test")
    print()

    results = []

    # Phase 1: Training (normal gravity)
    print(f"TRAINING PHASE (g={TRAINING_GRAVITY})")
    print("-" * 50)

    training_successes = 0
    for ep in range(num_episodes):
        config = ExperimentConfig(
            agent_type=agent_type,
            gravity_condition="training"
        )
        policy = GravityExperimentPolicy(config)

        result = run_single_episode(
            policy, ep, "training", condition_name,
            TRAINING_GRAVITY, verbose=verbose
        )
        results.append(result)

        if result.success:
            training_successes += 1

        if verbose:
            status = "SUCCESS" if result.success else f"FAIL ({result.reason})"
            print(f"  Episode {ep+1}: {status}, jump@{result.jump_position}")

    training_rate = training_successes / num_episodes
    print(f"\nTraining success rate: {training_rate*100:.1f}%")

    # Phase 2: Test (half gravity)
    print(f"\nTEST PHASE (g={TEST_GRAVITY}) - {test_condition}")
    print("-" * 50)

    test_successes = 0
    for ep in range(num_episodes):
        config = ExperimentConfig(
            agent_type=agent_type,
            gravity_condition=test_condition
        )
        policy = GravityExperimentPolicy(config)

        result = run_single_episode(
            policy, ep + num_episodes, "test", condition_name,
            TEST_GRAVITY, verbose=verbose
        )
        results.append(result)

        if result.success:
            test_successes += 1

        if verbose:
            status = "SUCCESS" if result.success else f"FAIL ({result.reason})"
            print(f"  Episode {ep+1}: {status}, jump@{result.jump_position}")

    test_rate = test_successes / num_episodes
    print(f"\nTest success rate: {test_rate*100:.1f}%")

    # Analyze jump timing adaptation
    training_jumps = [r.jump_position for r in results if r.phase == "training" and r.jump_position >= 0]
    test_jumps = [r.jump_position for r in results if r.phase == "test" and r.jump_position >= 0]

    avg_training_jump = sum(training_jumps) / len(training_jumps) if training_jumps else 0
    avg_test_jump = sum(test_jumps) / len(test_jumps) if test_jumps else 0

    summary = {
        "condition": condition_name,
        "agent_type": agent_type,
        "test_mode": test_condition,
        "training": {
            "gravity": TRAINING_GRAVITY,
            "episodes": num_episodes,
            "successes": training_successes,
            "success_rate": training_rate,
            "avg_jump_position": avg_training_jump
        },
        "test": {
            "gravity": TEST_GRAVITY,
            "episodes": num_episodes,
            "successes": test_successes,
            "success_rate": test_rate,
            "avg_jump_position": avg_test_jump
        },
        "adaptation": {
            "jump_shift": avg_test_jump - avg_training_jump,
            "rate_change": test_rate - training_rate
        }
    }

    print(f"\n{'='*70}")
    print(f"SUMMARY: {condition_name}")
    print(f"{'='*70}")
    print(f"Training: {training_rate*100:.1f}% (avg jump @ {avg_training_jump:.1f})")
    print(f"Test:     {test_rate*100:.1f}% (avg jump @ {avg_test_jump:.1f})")
    print(f"Jump shift: {avg_test_jump - avg_training_jump:+.1f} positions")

    return {"summary": summary, "results": [asdict(r) for r in results]}


def run_full_experiment(num_episodes: int = 20, verbose: bool = True):
    """Run all 4 experimental conditions."""

    print("=" * 70)
    print("GRAVITY ADAPTATION EXPERIMENT")
    print("=" * 70)
    print(f"Testing LLM physics understanding across gravity changes")
    print(f"Training gravity: {TRAINING_GRAVITY} m/s²")
    print(f"Test gravity: {TEST_GRAVITY} m/s² (half strength)")
    print()

    # Check server
    try:
        response = requests.get(f"{ENV_SERVER_URL}/health", timeout=5)
        print(f"Server: {response.json()}")
    except Exception as e:
        print(f"ERROR: Cannot connect to server: {e}")
        return

    all_results = {}

    # Run all 4 conditions
    conditions = [
        ("NRL-F", "test_explained"),  # Formula + Explained
        ("NRL-F", "test_silent"),     # Formula + Silent
        ("NRL-N", "test_explained"),  # Story + Explained
        ("NRL-N", "test_silent"),     # Story + Silent
    ]

    for agent_type, test_condition in conditions:
        result = run_condition(agent_type, test_condition, num_episodes, verbose)
        condition_name = f"{agent_type}_{test_condition.split('_')[1]}"
        all_results[condition_name] = result

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL RESULTS COMPARISON")
    print("=" * 70)
    print(f"\n{'Condition':<20} {'Training':>10} {'Test':>10} {'Change':>10} {'Jump Shift':>12}")
    print("-" * 62)

    for name, data in all_results.items():
        s = data["summary"]
        train_rate = s["training"]["success_rate"] * 100
        test_rate = s["test"]["success_rate"] * 100
        change = test_rate - train_rate
        jump_shift = s["adaptation"]["jump_shift"]
        print(f"{name:<20} {train_rate:>9.1f}% {test_rate:>9.1f}% {change:>+9.1f}% {jump_shift:>+11.1f}")

    # Hypothesis testing
    print("\n" + "-" * 70)
    print("HYPOTHESIS ANALYSIS")
    print("-" * 70)

    # H1: Formula vs Story
    f_explained = all_results.get("NRL-F_explained", {}).get("summary", {})
    n_explained = all_results.get("NRL-N_explained", {}).get("summary", {})
    if f_explained and n_explained:
        f_test = f_explained.get("test", {}).get("success_rate", 0)
        n_test = n_explained.get("test", {}).get("success_rate", 0)
        print(f"\nH1 (Formula vs Story with explanation):")
        print(f"  Formula (NRL-F): {f_test*100:.1f}%")
        print(f"  Story (NRL-N):   {n_test*100:.1f}%")
        print(f"  Difference:      {(f_test-n_test)*100:+.1f}%")

    # H3: Explained vs Silent
    f_explained = all_results.get("NRL-F_explained", {}).get("summary", {})
    f_silent = all_results.get("NRL-F_silent", {}).get("summary", {})
    if f_explained and f_silent:
        e_test = f_explained.get("test", {}).get("success_rate", 0)
        s_test = f_silent.get("test", {}).get("success_rate", 0)
        print(f"\nH3 (Explained vs Silent for Formula agent):")
        print(f"  Explained: {e_test*100:.1f}%")
        print(f"  Silent:    {s_test*100:.1f}%")
        print(f"  Difference: {(e_test-s_test)*100:+.1f}%")

    # Save all results
    with open("gravity_experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to gravity_experiment_results.json")


if __name__ == "__main__":
    run_full_experiment(num_episodes=20, verbose=True)
