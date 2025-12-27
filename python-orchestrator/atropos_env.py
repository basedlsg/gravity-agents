"""
Atropos Environment Wrapper

Wraps the web physics environment for use with Nous Atropos RL framework.
"""

from typing import Any, Literal
from dataclasses import dataclass
import numpy as np

from env_client import GravityEnvClient, StepResult
from llm_policy import LLMPolicy, FewShotPolicy, PolicyConfig
from logger import ExperimentLogger
from config import ACTION_SPACES, GravityConfig


@dataclass
class AtroposEnvConfig:
    """Configuration for Atropos environment wrapper"""
    server_url: str = "http://localhost:3002"
    task: Literal["gap", "throw"] = "gap"
    agent_type: Literal["RL-F", "RL-N", "NRL-F"] = "RL-F"
    model: str = "gemini-2.0-flash"
    use_groq: bool = False
    max_steps: int = 500
    history_length: int = 5


class GravityAtroposEnv:
    """
    Atropos-compatible environment for gravity experiments.

    This wraps the web physics environment and LLM policy to provide
    the interface expected by Atropos for RL training.
    """

    def __init__(self, config: AtroposEnvConfig, session_id: str = None):
        self.config = config
        self.session_id = session_id or f"atropos_{id(self)}"

        # Environment client
        self.client = GravityEnvClient(
            server_url=config.server_url,
            session_id=self.session_id
        )

        # Policy
        policy_config = PolicyConfig(
            agent_type=config.agent_type,
            task=config.task,
            model=config.model,
            use_groq=config.use_groq,
            history_length=config.history_length
        )

        if config.agent_type == "NRL-F":
            self.policy = FewShotPolicy(policy_config)
        else:
            self.policy = LLMPolicy(policy_config)

        # Action space
        self.actions = ACTION_SPACES[config.task]
        self.action_space_size = len(self.actions)

        # State
        self.current_gravity = 9.81
        self.gravity_condition = "training"
        self.step_count = 0
        self.last_observation = None

        # Logger (optional, set externally)
        self.logger: ExperimentLogger = None

    def reset(
        self,
        gravity: float = 9.81,
        seed: int = None,
        condition: Literal["training", "test_silent", "test_explained"] = "training"
    ) -> dict:
        """Reset the environment"""
        self.current_gravity = gravity
        self.gravity_condition = condition
        self.step_count = 0
        self.policy.reset()

        obs = self.client.reset(
            task=self.config.task,
            gravity=gravity,
            seed=seed or np.random.randint(0, 100000),
            max_steps=self.config.max_steps
        )

        self.last_observation = obs
        return obs

    def step(self, action: str = None) -> tuple[dict, float, bool, dict]:
        """
        Take a step in the environment.

        If action is None, uses the LLM policy to select an action.
        """
        # Get action from policy if not provided
        if action is None:
            action = self.policy.select_action(
                self.last_observation,
                self.gravity_condition
            )

        # Step environment
        result = self.client.step(action)

        self.step_count += 1
        self.last_observation = result.observation

        # Log if logger is attached
        if self.logger and self.logger.current_episode:
            self.logger.log_step(
                step=self.step_count - 1,
                observation=result.observation,
                action=action,
                reward=result.reward,
                done=result.done,
                info=result.info
            )

        return result.observation, result.reward, result.done, result.info

    def get_action_from_policy(self) -> str:
        """Get action from LLM policy without stepping"""
        return self.policy.select_action(
            self.last_observation,
            self.gravity_condition
        )

    def render(self):
        """Optional: could connect to WebSocket viewer"""
        pass


class AtroposTrainer:
    """
    Trainer for RL agents using Atropos framework.

    This provides the high-level training loop that integrates with Atropos.
    """

    def __init__(
        self,
        env: GravityAtroposEnv,
        logger: ExperimentLogger = None
    ):
        self.env = env
        self.logger = logger
        self.env.logger = logger

        # Training state
        self.episode_count = 0
        self.total_steps = 0

    def train_episode(self, seed: int = None) -> dict:
        """Run a single training episode"""
        self.episode_count += 1
        gravity = GravityConfig().training

        # Start episode logging
        if self.logger:
            self.logger.start_episode(
                episode_id=self.episode_count,
                gravity=gravity,
                condition="baseline",
                seed=seed or 0
            )

        # Reset environment
        obs = self.env.reset(gravity=gravity, seed=seed, condition="training")

        total_reward = 0.0
        done = False

        while not done:
            obs, reward, done, info = self.env.step()
            total_reward += reward
            self.total_steps += 1

        success = info.get("success", False)

        # End episode logging
        if self.logger:
            self.logger.end_episode(success=success)

        return {
            "episode": self.episode_count,
            "reward": total_reward,
            "success": success,
            "steps": self.env.step_count
        }

    def train(self, num_episodes: int, save_every: int = 50) -> list[dict]:
        """Run multiple training episodes"""
        results = []

        for i in range(num_episodes):
            seed = 1000 + i  # Consistent seeds for reproducibility
            result = self.train_episode(seed=seed)
            results.append(result)

            if (i + 1) % 10 == 0:
                recent_success = sum(1 for r in results[-10:] if r["success"]) / 10
                print(f"Episode {i+1}/{num_episodes} | Recent success rate: {recent_success:.1%}")

            if self.logger and (i + 1) % save_every == 0:
                self.logger.save()

        return results

    def evaluate(
        self,
        condition: Literal["baseline", "silent", "explained"],
        num_episodes: int = 100,
        seeds: list[int] = None
    ) -> dict:
        """Evaluate agent under specific condition"""
        gravity = GravityConfig().test if condition != "baseline" else GravityConfig().training
        gravity_condition = {
            "baseline": "training",
            "silent": "test_silent",
            "explained": "test_explained"
        }[condition]

        if seeds is None:
            seeds = list(range(2000, 2000 + num_episodes))

        results = []
        for i, seed in enumerate(seeds[:num_episodes]):
            # Log episode
            if self.logger:
                self.logger.start_episode(
                    episode_id=i,
                    gravity=gravity,
                    condition=condition,
                    seed=seed
                )

            # Run episode
            obs = self.env.reset(gravity=gravity, seed=seed, condition=gravity_condition)

            total_reward = 0.0
            done = False

            while not done:
                obs, reward, done, info = self.env.step()
                total_reward += reward

            success = info.get("success", False)
            results.append({"success": success, "reward": total_reward, "steps": self.env.step_count})

            if self.logger:
                self.logger.end_episode(success=success)

        success_rate = sum(1 for r in results if r["success"]) / len(results)
        avg_steps = sum(r["steps"] for r in results) / len(results)

        return {
            "condition": condition,
            "gravity": gravity,
            "num_episodes": len(results),
            "success_rate": success_rate,
            "avg_steps": avg_steps
        }
