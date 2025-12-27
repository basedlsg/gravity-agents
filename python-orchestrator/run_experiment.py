#!/usr/bin/env python3
"""
Main experiment runner for Gravity Agents

Usage:
    python run_experiment.py --agent RL-F --task gap --mode train
    python run_experiment.py --agent RL-F --task gap --mode eval --condition explained
    python run_experiment.py --agent all --task all --mode full
"""

import argparse
from datetime import datetime
from typing import Literal

from config import (
    AgentConfig, TaskConfig, ExperimentConfig, GravityConfig,
    ENV_SERVER_URL
)
from atropos_env import GravityAtroposEnv, AtroposTrainer, AtroposEnvConfig
from logger import ExperimentLogger
from env_client import GravityEnvClient


def run_training(
    agent_type: str,
    task: str,
    num_episodes: int = 200,
    log_dir: str = "./logs"
) -> dict:
    """Run training for a single agent/task combination"""
    print(f"\n{'='*60}")
    print(f"Training {agent_type} on {task} task")
    print(f"{'='*60}\n")

    # Create experiment name
    exp_name = f"{agent_type}_{task}_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Set up logger
    logger = ExperimentLogger(log_dir=log_dir, experiment_name=exp_name)
    logger.start_experiment(
        agent_type=agent_type,
        task=task,
        config={"mode": "training", "num_episodes": num_episodes}
    )

    # Create environment
    env_config = AtroposEnvConfig(
        server_url=ENV_SERVER_URL,
        task=task,
        agent_type=agent_type
    )
    env = GravityAtroposEnv(env_config)

    # Create trainer
    trainer = AtroposTrainer(env=env, logger=logger)

    # Run training
    results = trainer.train(num_episodes=num_episodes)

    # Save and summarize
    logger.save()
    summary = logger.get_summary()

    print(f"\nTraining complete!")
    print(f"Success rate: {summary.get('overall_success_rate', 0):.1%}")

    return summary


def run_evaluation(
    agent_type: str,
    task: str,
    condition: Literal["baseline", "silent", "explained"],
    num_episodes: int = 100,
    log_dir: str = "./logs"
) -> dict:
    """Run evaluation for a specific condition"""
    print(f"\n{'='*60}")
    print(f"Evaluating {agent_type} on {task} task - {condition} condition")
    print(f"{'='*60}\n")

    # Create experiment name
    exp_name = f"{agent_type}_{task}_eval_{condition}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Set up logger
    logger = ExperimentLogger(log_dir=log_dir, experiment_name=exp_name)
    logger.start_experiment(
        agent_type=agent_type,
        task=task,
        config={"mode": "evaluation", "condition": condition}
    )

    # Create environment
    env_config = AtroposEnvConfig(
        server_url=ENV_SERVER_URL,
        task=task,
        agent_type=agent_type
    )
    env = GravityAtroposEnv(env_config)

    # Create trainer (for evaluation)
    trainer = AtroposTrainer(env=env, logger=logger)

    # Run evaluation
    results = trainer.evaluate(
        condition=condition,
        num_episodes=num_episodes
    )

    # Save
    logger.save()

    print(f"\nEvaluation complete!")
    print(f"Condition: {condition}")
    print(f"Success rate: {results['success_rate']:.1%}")
    print(f"Avg steps: {results['avg_steps']:.1f}")

    return results


def run_full_experiment(
    agents: list[str] = None,
    tasks: list[str] = None,
    train_episodes: int = 200,
    eval_episodes: int = 100,
    log_dir: str = "./logs"
) -> dict:
    """Run complete experiment matrix"""
    if agents is None:
        agents = ["RL-F", "RL-N", "NRL-F"]
    if tasks is None:
        tasks = ["gap", "throw"]

    conditions = ["baseline", "silent", "explained"]

    all_results = {}

    for agent in agents:
        for task in tasks:
            key = f"{agent}_{task}"
            all_results[key] = {}

            # Training (skip for NRL-F)
            if agent != "NRL-F":
                train_result = run_training(
                    agent_type=agent,
                    task=task,
                    num_episodes=train_episodes,
                    log_dir=log_dir
                )
                all_results[key]["training"] = train_result

            # Evaluation under each condition
            for condition in conditions:
                eval_result = run_evaluation(
                    agent_type=agent,
                    task=task,
                    condition=condition,
                    num_episodes=eval_episodes,
                    log_dir=log_dir
                )
                all_results[key][condition] = eval_result

    return all_results


def check_server():
    """Check if the environment server is running"""
    client = GravityEnvClient(ENV_SERVER_URL)
    if not client.health_check():
        print(f"ERROR: Environment server not running at {ENV_SERVER_URL}")
        print("Please start the server first:")
        print("  cd web-env && npm install && node server.js")
        return False
    print(f"Server running at {ENV_SERVER_URL}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Gravity Agents Experiment Runner")

    parser.add_argument(
        "--agent",
        type=str,
        default="RL-F",
        choices=["RL-F", "RL-N", "NRL-F", "all"],
        help="Agent type to run"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="gap",
        choices=["gap", "throw", "all"],
        help="Task to run"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="eval",
        choices=["train", "eval", "full"],
        help="Run mode"
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="baseline",
        choices=["baseline", "silent", "explained"],
        help="Evaluation condition (for eval mode)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for logs"
    )
    parser.add_argument(
        "--skip-server-check",
        action="store_true",
        help="Skip server health check"
    )

    args = parser.parse_args()

    # Check server
    if not args.skip_server_check:
        if not check_server():
            return

    # Run based on mode
    if args.mode == "full":
        agents = None if args.agent == "all" else [args.agent]
        tasks = None if args.task == "all" else [args.task]

        results = run_full_experiment(
            agents=agents,
            tasks=tasks,
            train_episodes=args.episodes,
            eval_episodes=args.episodes,
            log_dir=args.log_dir
        )

        print("\n" + "="*60)
        print("FULL EXPERIMENT SUMMARY")
        print("="*60)
        for key, data in results.items():
            print(f"\n{key}:")
            for cond, result in data.items():
                if cond == "training":
                    print(f"  Training success: {result.get('overall_success_rate', 0):.1%}")
                else:
                    print(f"  {cond}: {result.get('success_rate', 0):.1%}")

    elif args.mode == "train":
        if args.agent == "NRL-F":
            print("NRL-F does not require training (no RL)")
            return

        agents = ["RL-F", "RL-N"] if args.agent == "all" else [args.agent]
        tasks = ["gap", "throw"] if args.task == "all" else [args.task]

        for agent in agents:
            for task in tasks:
                run_training(
                    agent_type=agent,
                    task=task,
                    num_episodes=args.episodes,
                    log_dir=args.log_dir
                )

    elif args.mode == "eval":
        agents = ["RL-F", "RL-N", "NRL-F"] if args.agent == "all" else [args.agent]
        tasks = ["gap", "throw"] if args.task == "all" else [args.task]

        for agent in agents:
            for task in tasks:
                run_evaluation(
                    agent_type=agent,
                    task=task,
                    condition=args.condition,
                    num_episodes=args.episodes,
                    log_dir=args.log_dir
                )


if __name__ == "__main__":
    main()
