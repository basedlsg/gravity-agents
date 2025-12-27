"""
Logging pipeline for experiment data
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from dataclasses import dataclass, field, asdict
import pandas as pd


@dataclass
class StepLog:
    """Log entry for a single step"""
    step: int
    observation: dict
    action: str
    reward: float
    done: bool
    info: dict


@dataclass
class EpisodeLog:
    """Log entry for a complete episode"""
    episode_id: int
    agent_type: str
    task: str
    gravity: float
    condition: Literal["baseline", "silent", "explained"]
    seed: int
    success: bool
    total_reward: float
    num_steps: int
    steps: list[StepLog] = field(default_factory=list)

    # Behavioral metrics
    metrics: dict = field(default_factory=dict)


@dataclass
class ExperimentLog:
    """Log for entire experiment run"""
    experiment_name: str
    start_time: str
    agent_type: str
    task: str
    config: dict
    episodes: list[EpisodeLog] = field(default_factory=list)


class ExperimentLogger:
    """Handles all experiment logging"""

    def __init__(self, log_dir: str = "./logs", experiment_name: str = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.exp_dir = self.log_dir / self.experiment_name
        self.exp_dir.mkdir(exist_ok=True)

        self.current_experiment: ExperimentLog = None
        self.current_episode: EpisodeLog = None

    def start_experiment(self, agent_type: str, task: str, config: dict):
        """Start a new experiment"""
        self.current_experiment = ExperimentLog(
            experiment_name=self.experiment_name,
            start_time=datetime.now().isoformat(),
            agent_type=agent_type,
            task=task,
            config=config
        )

    def start_episode(
        self,
        episode_id: int,
        gravity: float,
        condition: Literal["baseline", "silent", "explained"],
        seed: int
    ):
        """Start a new episode"""
        if not self.current_experiment:
            raise RuntimeError("No experiment started")

        self.current_episode = EpisodeLog(
            episode_id=episode_id,
            agent_type=self.current_experiment.agent_type,
            task=self.current_experiment.task,
            gravity=gravity,
            condition=condition,
            seed=seed,
            success=False,
            total_reward=0.0,
            num_steps=0
        )

    def log_step(
        self,
        step: int,
        observation: dict,
        action: str,
        reward: float,
        done: bool,
        info: dict
    ):
        """Log a single step"""
        if not self.current_episode:
            raise RuntimeError("No episode started")

        step_log = StepLog(
            step=step,
            observation=observation,
            action=action,
            reward=reward,
            done=done,
            info=info
        )
        self.current_episode.steps.append(step_log)
        self.current_episode.total_reward += reward
        self.current_episode.num_steps = step + 1

    def end_episode(self, success: bool, metrics: dict = None):
        """End current episode and compute metrics"""
        if not self.current_episode:
            return

        self.current_episode.success = success
        self.current_episode.metrics = metrics or self._compute_metrics()

        self.current_experiment.episodes.append(self.current_episode)
        self.current_episode = None

    def _compute_metrics(self) -> dict:
        """Compute behavioral metrics from episode steps"""
        if not self.current_episode or not self.current_episode.steps:
            return {}

        steps = self.current_episode.steps
        task = self.current_episode.task
        metrics = {}

        # Action counts
        action_counts = {}
        for step in steps:
            action = step.action
            action_counts[action] = action_counts.get(action, 0) + 1
        metrics["action_counts"] = action_counts

        if task == "gap":
            # Gap-specific metrics
            jump_positions = []
            for step in steps:
                if step.action == "jump":
                    pos = step.observation.get("agentPosition", [0, 0, 0])
                    jump_positions.append(pos[0])

            metrics["num_jumps"] = len(jump_positions)
            if jump_positions:
                metrics["first_jump_x"] = jump_positions[0]
                metrics["avg_jump_x"] = sum(jump_positions) / len(jump_positions)

            # Final position
            if steps:
                final_pos = steps[-1].observation.get("agentPosition", [0, 0, 0])
                metrics["final_x"] = final_pos[0]

        elif task == "throw":
            # Throw-specific metrics
            throw_actions = [s for s in steps if "throw" in s.action]
            metrics["num_throws"] = len(throw_actions)

            throw_strengths = {"weak": 0, "medium": 0, "strong": 0}
            for s in throw_actions:
                if "weak" in s.action:
                    throw_strengths["weak"] += 1
                elif "medium" in s.action:
                    throw_strengths["medium"] += 1
                elif "strong" in s.action:
                    throw_strengths["strong"] += 1
            metrics["throw_strengths"] = throw_strengths

            # Pick actions
            metrics["num_picks"] = sum(1 for s in steps if s.action == "pick")

        return metrics

    def save(self):
        """Save experiment data to files"""
        if not self.current_experiment:
            return

        # Save full experiment as JSON
        exp_file = self.exp_dir / "experiment.json"
        with open(exp_file, "w") as f:
            json.dump(asdict(self.current_experiment), f, indent=2, default=str)

        # Save episodes summary as Parquet
        episodes_data = []
        for ep in self.current_experiment.episodes:
            row = {
                "episode_id": ep.episode_id,
                "agent_type": ep.agent_type,
                "task": ep.task,
                "gravity": ep.gravity,
                "condition": ep.condition,
                "seed": ep.seed,
                "success": ep.success,
                "total_reward": ep.total_reward,
                "num_steps": ep.num_steps,
                **{f"metric_{k}": v for k, v in ep.metrics.items() if not isinstance(v, dict)}
            }
            episodes_data.append(row)

        if episodes_data:
            df = pd.DataFrame(episodes_data)
            df.to_parquet(self.exp_dir / "episodes.parquet")

        print(f"Saved experiment to {self.exp_dir}")

    def get_summary(self) -> dict:
        """Get summary statistics"""
        if not self.current_experiment:
            return {}

        episodes = self.current_experiment.episodes
        if not episodes:
            return {}

        # Group by condition
        by_condition = {}
        for ep in episodes:
            cond = ep.condition
            if cond not in by_condition:
                by_condition[cond] = []
            by_condition[cond].append(ep)

        summary = {
            "total_episodes": len(episodes),
            "overall_success_rate": sum(1 for e in episodes if e.success) / len(episodes),
        }

        for cond, eps in by_condition.items():
            summary[f"{cond}_success_rate"] = sum(1 for e in eps if e.success) / len(eps)
            summary[f"{cond}_avg_steps"] = sum(e.num_steps for e in eps) / len(eps)

        return summary


def load_experiment(log_dir: str, experiment_name: str) -> dict:
    """Load saved experiment data"""
    exp_dir = Path(log_dir) / experiment_name

    with open(exp_dir / "experiment.json") as f:
        data = json.load(f)

    return data


def load_episodes_df(log_dir: str, experiment_name: str) -> pd.DataFrame:
    """Load episodes as pandas DataFrame"""
    exp_dir = Path(log_dir) / experiment_name
    return pd.read_parquet(exp_dir / "episodes.parquet")
