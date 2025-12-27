"""
Configuration for Gravity Agents experiments
"""

import os
from dataclasses import dataclass, field
from typing import Literal

# API Keys - LOAD FROM ENVIRONMENT
# Google Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Web Env
ENV_SERVER_URL = os.environ.get("ENV_SERVER_URL", "http://localhost:3000")

# Legacy/Unused
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


@dataclass
class GravityConfig:
    """Gravity settings for experiments"""
    training: float = 9.81  # Earth gravity
    test: float = 4.9       # 0.5g (weaker)


@dataclass
class TaskConfig:
    """Task configuration"""
    name: Literal["gap", "throw"]
    max_steps: int = 500
    physics_ticks_per_step: int = 10

    # Task-specific params
    gap_width: float = 3.0
    gap_variance: float = 0.5
    basket_distance: float = 6.0
    basket_distance_variance: float = 1.0
    action_granularity: Literal["coarse", "medium", "fine"] = "coarse"


@dataclass
class AgentConfig:
    """Agent configuration"""
    agent_type: Literal["RL-F", "RL-N", "NRL-F"]
    model: str = "gemini-2.0-flash"  # or groq model
    use_groq: bool = False

    # RL settings
    use_rl: bool = True
    learning_rate: float = 1e-4
    episodes_per_update: int = 10

    # Context settings
    history_length: int = 5  # Number of past steps to include


@dataclass
class ExperimentConfig:
    """Full experiment configuration"""
    name: str
    agent: AgentConfig
    task: TaskConfig
    gravity: GravityConfig = field(default_factory=GravityConfig)

    # Evaluation settings
    num_episodes: int = 100
    eval_seeds: list = field(default_factory=lambda: list(range(1000, 1100)))

    # Logging
    log_dir: str = "./logs"
    save_trajectories: bool = True


# Gravity descriptions for different agent types
GRAVITY_DESCRIPTIONS = {
    "formula": {
        "training": "Gravity in this world follows the law F = m·g with g = 9.81 m/s². Heavier objects feel proportionally more downward force.",
        "test_silent": "Gravity in this world follows the law F = m·g with g = 9.81 m/s². Heavier objects feel proportionally more downward force.",
        "test_explained": "Gravity in this world now follows the law F = m·g with g = 4.9 m/s², which is weaker than before. Objects fall more slowly and stay in the air longer."
    },
    "normal": {
        "training": "This world has normal Earth gravity. Objects fall down at a standard rate, just like on Earth.",
        "test_silent": "This world has normal Earth gravity. Objects fall down at a standard rate, just like on Earth.",
        "test_explained": "Gravity here is now weaker than before. Objects fall more slowly than normal Earth gravity and stay in the air longer."
    }
}


# Task descriptions
TASK_DESCRIPTIONS = {
    "gap": "You are standing on the START platform. Your goal is to reach the GOAL platform across the gap and stand inside the highlighted goal area. The gap is between the two platforms.",
    "throw": "Pick up the block near you and throw it into the basket so that it rests inside. The basket is elevated and at some distance from you."
}


# Action spaces
ACTION_SPACES = {
    "gap": ["forward", "back", "left", "right", "jump", "idle"],
    "throw": ["forward", "back", "left", "right", "pick", "drop", "throw_weak", "throw_medium", "throw_strong", "idle"]
}


def get_agent_gravity_style(agent_type: str) -> str:
    """Get gravity description style for agent type"""
    if agent_type in ["RL-F", "NRL-F"]:
        return "formula"
    return "normal"


def get_experiment_configs() -> list[ExperimentConfig]:
    """Generate all experiment configurations"""
    configs = []

    for agent_type in ["RL-F", "RL-N", "NRL-F"]:
        for task_name in ["gap", "throw"]:
            config = ExperimentConfig(
                name=f"{agent_type}_{task_name}",
                agent=AgentConfig(
                    agent_type=agent_type,
                    use_rl=(agent_type != "NRL-F")
                ),
                task=TaskConfig(name=task_name)
            )
            configs.append(config)

    return configs
