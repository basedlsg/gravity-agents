#!/usr/bin/env python3
"""
Run all experiments for Gravity Agents study.

Tests 3 agents × 2 tasks × 3 conditions = 18 experiment runs
Uses 10 episodes per condition for quick iteration (increase for final runs)
"""

import json
import time
from datetime import datetime
from pathlib import Path

from atropos_env import GravityAtroposEnv, AtroposEnvConfig
from logger import ExperimentLogger
from config import GravityConfig

# Configuration
EPISODES_PER_CONDITION = 10  # Increase to 100 for final runs
MAX_STEPS_PER_EPISODE = 100  # Reduced for faster iteration

AGENTS = ["RL-F", "RL-N", "NRL-F"]
TASKS = ["gap", "throw"]
CONDITIONS = ["baseline", "silent", "explained"]

def run_condition(agent_type: str, task: str, condition: str,
                  num_episodes: int, logger: ExperimentLogger) -> dict:
    """Run experiments for a single agent/task/condition combination."""

    # Determine gravity and condition mapping
    gravity_config = GravityConfig()
    if condition == "baseline":
        gravity = gravity_config.training
        gravity_condition = "training"
    else:
        gravity = gravity_config.test
        gravity_condition = "test_silent" if condition == "silent" else "test_explained"

    # Create environment
    env_config = AtroposEnvConfig(
        task=task,
        agent_type=agent_type,
        max_steps=MAX_STEPS_PER_EPISODE
    )
    env = GravityAtroposEnv(env_config)
    env.logger = logger

    results = []
    successes = 0

    for ep in range(num_episodes):
        seed = 2000 + ep  # Consistent seeds for reproducibility

        # Start episode logging
        logger.start_episode(
            episode_id=ep,
            gravity=gravity,
            condition=condition,
            seed=seed
        )

        # Run episode
        obs = env.reset(gravity=gravity, seed=seed, condition=gravity_condition)

        done = False
        total_reward = 0.0
        step = 0

        while not done and step < MAX_STEPS_PER_EPISODE:
            obs, reward, done, info = env.step()
            total_reward += reward
            step += 1

        success = info.get("success", False)
        if success:
            successes += 1

        results.append({
            "episode": ep,
            "success": success,
            "steps": step,
            "reward": total_reward,
            "reason": info.get("reason", "unknown")
        })

        # End episode logging
        logger.end_episode(success=success)

        # Progress indicator
        status = "✓" if success else "✗"
        print(f"  Episode {ep+1}/{num_episodes}: {status} ({step} steps)", end="\r")

    success_rate = successes / num_episodes
    print(f"  Episodes complete: {successes}/{num_episodes} succeeded ({success_rate:.1%})    ")

    return {
        "agent": agent_type,
        "task": task,
        "condition": condition,
        "gravity": gravity,
        "num_episodes": num_episodes,
        "successes": successes,
        "success_rate": success_rate,
        "avg_steps": sum(r["steps"] for r in results) / num_episodes,
        "episodes": results
    }


def run_all_experiments():
    """Run the complete experiment matrix."""

    print("=" * 60)
    print("GRAVITY AGENTS EXPERIMENT")
    print("=" * 60)
    print(f"Agents: {AGENTS}")
    print(f"Tasks: {TASKS}")
    print(f"Conditions: {CONDITIONS}")
    print(f"Episodes per condition: {EPISODES_PER_CONDITION}")
    print("=" * 60)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./logs/full_experiment_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for agent in AGENTS:
        for task in TASKS:
            key = f"{agent}_{task}"
            all_results[key] = {}

            # Create logger for this agent/task combo
            exp_name = f"{key}_{timestamp}"
            logger = ExperimentLogger(log_dir="./logs", experiment_name=exp_name)
            logger.start_experiment(
                agent_type=agent,
                task=task,
                config={"episodes_per_condition": EPISODES_PER_CONDITION}
            )

            for condition in CONDITIONS:
                print(f"\n>>> {agent} | {task} | {condition}")

                result = run_condition(
                    agent_type=agent,
                    task=task,
                    condition=condition,
                    num_episodes=EPISODES_PER_CONDITION,
                    logger=logger
                )

                all_results[key][condition] = result

            # Save logger data
            logger.save()

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    # Create summary table
    print(f"\n{'Agent':<8} {'Task':<8} {'Baseline':>10} {'Silent':>10} {'Explained':>10}")
    print("-" * 50)

    for agent in AGENTS:
        for task in TASKS:
            key = f"{agent}_{task}"
            baseline = all_results[key]["baseline"]["success_rate"]
            silent = all_results[key]["silent"]["success_rate"]
            explained = all_results[key]["explained"]["success_rate"]

            print(f"{agent:<8} {task:<8} {baseline:>10.1%} {silent:>10.1%} {explained:>10.1%}")

    # Save full results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    # Hypothesis analysis
    print("\n" + "=" * 60)
    print("HYPOTHESIS ANALYSIS")
    print("=" * 60)

    # H1: Law vs Story (RL-F vs RL-N at 0.5g explained)
    print("\nH1 - Law vs Story (RL-F vs RL-N at explained condition):")
    for task in TASKS:
        rl_f = all_results[f"RL-F_{task}"]["explained"]["success_rate"]
        rl_n = all_results[f"RL-N_{task}"]["explained"]["success_rate"]
        diff = rl_f - rl_n
        print(f"  {task}: RL-F={rl_f:.1%}, RL-N={rl_n:.1%}, diff={diff:+.1%}")

    # H2: RL vs No-RL (RL-F vs NRL-F at 0.5g explained)
    print("\nH2 - RL vs No-RL (RL-F vs NRL-F at explained condition):")
    for task in TASKS:
        rl_f = all_results[f"RL-F_{task}"]["explained"]["success_rate"]
        nrl_f = all_results[f"NRL-F_{task}"]["explained"]["success_rate"]
        diff = rl_f - nrl_f
        print(f"  {task}: RL-F={rl_f:.1%}, NRL-F={nrl_f:.1%}, diff={diff:+.1%}")

    # H3: Information vs Silence (explained vs silent)
    print("\nH3 - Information vs Silence (explained vs silent):")
    for agent in AGENTS:
        for task in TASKS:
            key = f"{agent}_{task}"
            silent = all_results[key]["silent"]["success_rate"]
            explained = all_results[key]["explained"]["success_rate"]
            diff = explained - silent
            print(f"  {agent}/{task}: silent={silent:.1%}, explained={explained:.1%}, diff={diff:+.1%}")

    return all_results


if __name__ == "__main__":
    start_time = time.time()
    results = run_all_experiments()
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f} seconds")
