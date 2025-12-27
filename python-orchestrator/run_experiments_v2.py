"""
V2 Experiment Runner - With actual RL training and structured logging

Key differences from V1:
1. RL agents actually learn (using ValueCachedPolicy)
2. Structured JSON responses are logged and verified
3. Physics calculations are captured for analysis
4. Training phase precedes evaluation
"""

import os
import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Literal
import requests
import pandas as pd

from config import ENV_SERVER_URL, GravityConfig
from llm_policy_v2 import PolicyConfigV2, LLMPolicyV2, ValueCachedPolicy

# Use V2 tasks
TASK_VERSION = "v2"


@dataclass
class EpisodeResult:
    """Single episode result with full data"""
    episode_id: int
    agent_type: str
    task: str
    gravity_condition: str
    actual_gravity: float
    success: bool
    steps: int
    total_reward: float
    termination_reason: str
    action_counts: dict
    physics_calculations: list  # New: captured physics reasoning
    trajectory: list  # Full state-action sequence


@dataclass
class ExperimentResultV2:
    """Full experiment result"""
    name: str
    agent_type: str
    task: str
    training_episodes: int
    eval_episodes: int
    timestamp: str
    training_results: list[EpisodeResult]
    eval_results: dict[str, list[EpisodeResult]]  # condition -> episodes
    learning_stats: dict  # RL learning statistics


def reset_environment(task: str, gravity: float, seed: int = None) -> dict:
    """Reset environment and return initial observation"""
    response = requests.post(
        f"{ENV_SERVER_URL}/reset",
        json={
            "task": task,
            "taskVersion": TASK_VERSION,  # Use V2 tasks
            "gravity": gravity,
            "seed": seed
        },
        timeout=10
    )
    response.raise_for_status()
    data = response.json()
    # Extract observation from response
    return data.get("observation", data)


def step_environment(action: str) -> dict:
    """Take action and return new observation"""
    response = requests.post(
        f"{ENV_SERVER_URL}/step",
        json={"action": action},
        timeout=10
    )
    response.raise_for_status()
    return response.json()


def run_episode(
    policy,
    task: str,
    gravity: float,
    gravity_condition: str,
    max_steps: int = 200,
    episode_id: int = 0,
    agent_type: str = ""
) -> EpisodeResult:
    """Run a single episode with full logging"""

    # Reset
    obs = reset_environment(task, gravity)
    policy.reset()

    action_counts = {}
    physics_calculations = []
    trajectory = []

    total_reward = 0.0
    done = False
    step = 0

    while not done and step < max_steps:
        # Get action with reasoning
        action, response_data = policy.select_action(obs, gravity_condition)

        # Log physics calculation if present
        if "physics_calculation" in response_data:
            physics_calculations.append({
                "step": step,
                "calculation": response_data["physics_calculation"],
                "reasoning": response_data.get("reasoning", ""),
                "action": action
            })

        # Count actions
        action_counts[action] = action_counts.get(action, 0) + 1

        # Store trajectory
        trajectory.append({
            "step": step,
            "observation": {
                "position": obs.get("agentPosition", []),
                "velocity": obs.get("agentVelocity", []),
                "grounded": obs.get("isGrounded", False),
                "holding": obs.get("holdingBlock", False)
            },
            "action": action,
            "response_data": response_data
        })

        # Step environment
        result = step_environment(action)

        obs = result.get("observation", {})
        reward = result.get("reward", 0)
        done = result.get("done", False)
        success = result.get("success", False)
        reason = result.get("reason", "unknown")

        total_reward += reward
        step += 1

    return EpisodeResult(
        episode_id=episode_id,
        agent_type=agent_type,
        task=task,
        gravity_condition=gravity_condition,
        actual_gravity=gravity,
        success=success,
        steps=step,
        total_reward=total_reward,
        termination_reason=reason,
        action_counts=action_counts,
        physics_calculations=physics_calculations,
        trajectory=trajectory
    )


def run_training_phase(
    policy: ValueCachedPolicy,
    task: str,
    gravity: float,
    num_episodes: int = 50,
    agent_type: str = ""
) -> list[EpisodeResult]:
    """Run training phase for RL agents"""
    print(f"  Training {agent_type} on {task} for {num_episodes} episodes...")

    results = []
    for ep in range(num_episodes):
        result = run_episode(
            policy=policy,
            task=task,
            gravity=gravity,
            gravity_condition="training",
            episode_id=ep,
            agent_type=agent_type
        )

        # Update RL policy with episode reward
        policy.update_episode(result.total_reward)

        results.append(result)

        if (ep + 1) % 10 == 0:
            recent_success = sum(1 for r in results[-10:] if r.success) / 10
            print(f"    Episode {ep+1}/{num_episodes}: recent success rate = {recent_success:.1%}")

    return results


def run_eval_phase(
    policy,
    task: str,
    gravity_config: GravityConfig,
    num_episodes: int = 30,
    agent_type: str = ""
) -> dict[str, list[EpisodeResult]]:
    """Run evaluation on all gravity conditions"""
    print(f"  Evaluating {agent_type} on {task}...")

    conditions = {
        "training": gravity_config.training,        # 1g baseline
        "test_silent": gravity_config.test,         # 0.5g without telling
        "test_explained": gravity_config.test       # 0.5g with explanation
    }

    results = {}
    for condition, gravity in conditions.items():
        print(f"    Condition: {condition} (g={gravity})")
        condition_results = []

        for ep in range(num_episodes):
            result = run_episode(
                policy=policy,
                task=task,
                gravity=gravity,
                gravity_condition=condition,
                episode_id=ep,
                agent_type=agent_type
            )
            condition_results.append(result)

        success_rate = sum(1 for r in condition_results if r.success) / len(condition_results)
        print(f"      Success rate: {success_rate:.1%}")

        results[condition] = condition_results

    return results


def create_policy(agent_type: str, task: str) -> LLMPolicyV2:
    """Create appropriate policy for agent type"""
    config = PolicyConfigV2(
        agent_type=agent_type,
        task=task
    )

    if agent_type in ["RL-F", "RL-N"]:
        # RL agents use value-cached policy
        return ValueCachedPolicy(config, epsilon=0.2)
    else:
        # NRL-F uses base LLM policy
        return LLMPolicyV2(config)


def run_full_experiment(
    agent_type: str,
    task: str,
    training_episodes: int = 50,
    eval_episodes: int = 30,
    log_dir: str = "./logs_v2"
) -> ExperimentResultV2:
    """Run complete experiment for one agent-task combination"""

    print(f"\n{'='*60}")
    print(f"Running V2 Experiment: {agent_type} on {task}")
    print(f"{'='*60}")

    gravity_config = GravityConfig()
    policy = create_policy(agent_type, task)

    # Training phase (only for RL agents)
    training_results = []
    if agent_type in ["RL-F", "RL-N"]:
        training_results = run_training_phase(
            policy=policy,
            task=task,
            gravity=gravity_config.training,
            num_episodes=training_episodes,
            agent_type=agent_type
        )

    # Evaluation phase
    eval_results = run_eval_phase(
        policy=policy,
        task=task,
        gravity_config=gravity_config,
        num_episodes=eval_episodes,
        agent_type=agent_type
    )

    # Get learning stats if RL
    learning_stats = {}
    if hasattr(policy, 'get_learning_stats'):
        learning_stats = policy.get_learning_stats()

    # Create result
    result = ExperimentResultV2(
        name=f"{agent_type}_{task}_v2",
        agent_type=agent_type,
        task=task,
        training_episodes=training_episodes if agent_type != "NRL-F" else 0,
        eval_episodes=eval_episodes,
        timestamp=datetime.now().isoformat(),
        training_results=training_results,
        eval_results=eval_results,
        learning_stats=learning_stats
    )

    # Save results
    save_results(result, log_dir)

    return result


def save_results(result: ExperimentResultV2, log_dir: str):
    """Save experiment results to disk"""
    os.makedirs(log_dir, exist_ok=True)
    exp_dir = os.path.join(log_dir, result.name)
    os.makedirs(exp_dir, exist_ok=True)

    # Save experiment metadata
    metadata = {
        "name": result.name,
        "agent_type": result.agent_type,
        "task": result.task,
        "training_episodes": result.training_episodes,
        "eval_episodes": result.eval_episodes,
        "timestamp": result.timestamp,
        "learning_stats": result.learning_stats
    }
    with open(os.path.join(exp_dir, "experiment.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save training results if any
    if result.training_results:
        training_df = pd.DataFrame([
            {
                "episode": r.episode_id,
                "success": r.success,
                "steps": r.steps,
                "reward": r.total_reward,
                "reason": r.termination_reason,
                **{f"action_{k}": v for k, v in r.action_counts.items()}
            }
            for r in result.training_results
        ])
        training_df.to_parquet(os.path.join(exp_dir, "training.parquet"))

    # Save eval results per condition
    for condition, episodes in result.eval_results.items():
        eval_df = pd.DataFrame([
            {
                "episode": r.episode_id,
                "gravity": r.actual_gravity,
                "success": r.success,
                "steps": r.steps,
                "reward": r.total_reward,
                "reason": r.termination_reason,
                **{f"action_{k}": v for k, v in r.action_counts.items()}
            }
            for r in episodes
        ])
        eval_df.to_parquet(os.path.join(exp_dir, f"eval_{condition}.parquet"))

    # Save physics calculations for analysis
    all_physics = []
    for condition, episodes in result.eval_results.items():
        for ep in episodes:
            for calc in ep.physics_calculations:
                all_physics.append({
                    "condition": condition,
                    "episode": ep.episode_id,
                    **calc
                })
    if all_physics:
        physics_df = pd.DataFrame(all_physics)
        physics_df.to_parquet(os.path.join(exp_dir, "physics_calculations.parquet"))

    print(f"  Results saved to {exp_dir}")


def print_summary(results: list[ExperimentResultV2]):
    """Print summary table of all results"""
    print("\n" + "="*80)
    print("V2 EXPERIMENT SUMMARY")
    print("="*80)

    print("\nTraining Performance (RL agents only):")
    print("-" * 60)
    for r in results:
        if r.training_results:
            first_10 = sum(1 for ep in r.training_results[:10] if ep.success) / 10
            last_10 = sum(1 for ep in r.training_results[-10:] if ep.success) / 10
            print(f"  {r.name}: {first_10:.0%} -> {last_10:.0%} (improvement: {last_10-first_10:+.0%})")

    print("\nEvaluation Results:")
    print("-" * 60)
    header = f"{'Agent':<12} {'Task':<8} {'1g Base':<10} {'0.5g Silent':<12} {'0.5g Explained':<14} {'Adapt Gap':<10}"
    print(header)
    print("-" * len(header))

    for r in results:
        base = sum(1 for ep in r.eval_results["training"] if ep.success) / len(r.eval_results["training"])
        silent = sum(1 for ep in r.eval_results["test_silent"] if ep.success) / len(r.eval_results["test_silent"])
        explained = sum(1 for ep in r.eval_results["test_explained"] if ep.success) / len(r.eval_results["test_explained"])
        adapt_gap = explained - silent

        print(f"{r.agent_type:<12} {r.task:<8} {base:<10.0%} {silent:<12.0%} {explained:<14.0%} {adapt_gap:<+10.0%}")

    print("\nLearning Statistics (RL agents):")
    print("-" * 60)
    for r in results:
        if r.learning_stats:
            print(f"  {r.name}:")
            for k, v in r.learning_stats.items():
                print(f"    {k}: {v}")


def main():
    """Run all V2 experiments"""
    print("Starting Gravity Agents V2 Experiments")
    print("="*60)

    # Check environment
    try:
        response = requests.get(f"{ENV_SERVER_URL}/health", timeout=5)
        print(f"Environment server: {response.json()}")
    except Exception as e:
        print(f"ERROR: Cannot connect to environment server at {ENV_SERVER_URL}")
        print(f"Start the server with: cd web-env && PORT=3002 node server.js")
        return

    results = []

    # Run all agent-task combinations
    for agent_type in ["NRL-F", "RL-N", "RL-F"]:
        for task in ["gap", "throw"]:
            try:
                result = run_full_experiment(
                    agent_type=agent_type,
                    task=task,
                    training_episodes=50,  # Fewer for testing
                    eval_episodes=30,
                    log_dir="./logs_v2"
                )
                results.append(result)
            except Exception as e:
                print(f"ERROR in {agent_type}/{task}: {e}")
                import traceback
                traceback.print_exc()

    # Print summary
    print_summary(results)

    print("\nV2 Experiments complete!")


if __name__ == "__main__":
    main()
