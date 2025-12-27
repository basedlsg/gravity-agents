"""
Environment client - connects to the web physics server
"""

import requests
from typing import Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result from a single environment step"""
    observation: dict
    reward: float
    done: bool
    info: dict


class GravityEnvClient:
    """Client for the web-based physics environment"""

    def __init__(self, server_url: str = "http://localhost:3000", session_id: str = "default"):
        self.server_url = server_url.rstrip("/")
        self.session_id = session_id
        self._last_observation = None

    def reset(
        self,
        task: str = "gap",
        gravity: float = 9.81,
        seed: int = 42,
        max_steps: int = 500,
        **kwargs
    ) -> dict:
        """Reset the environment with given configuration"""
        config = {
            "task": task,
            "gravity": gravity,
            "seed": seed,
            "maxSteps": max_steps,
            **kwargs
        }

        response = requests.post(
            f"{self.server_url}/reset",
            json={"sessionId": self.session_id, "config": config}
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            raise RuntimeError(f"Reset failed: {data.get('error')}")

        self._last_observation = data["observation"]
        return data["observation"]

    def step(self, action: str | int) -> StepResult:
        """Take a step in the environment"""
        response = requests.post(
            f"{self.server_url}/step",
            json={"sessionId": self.session_id, "action": action}
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            raise RuntimeError(f"Step failed: {data.get('error')}")

        self._last_observation = data["observation"]

        return StepResult(
            observation=data["observation"],
            reward=data["reward"],
            done=data["done"],
            info=data["info"]
        )

    def get_info(self) -> dict:
        """Get environment info (available tasks, actions, etc.)"""
        response = requests.get(f"{self.server_url}/info")
        response.raise_for_status()
        return response.json()

    def health_check(self) -> bool:
        """Check if server is running"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    @property
    def last_observation(self) -> dict | None:
        return self._last_observation


class MockGravityEnvClient:
    """Mock client for testing without server"""

    def __init__(self, *args, **kwargs):
        self.step_count = 0
        self.task = "gap"
        self.gravity = 9.81

    def reset(self, task: str = "gap", gravity: float = 9.81, **kwargs) -> dict:
        self.step_count = 0
        self.task = task
        self.gravity = gravity

        if task == "gap":
            return {
                "agentPosition": [-1.25, 1.0, 0],
                "agentVelocity": [0, 0, 0],
                "gapStart": 2.5,
                "gapEnd": 5.5,
                "gapWidth": 3.0,
                "goalZone": {"minX": 6.0, "maxX": 9.0, "minZ": -2.0, "maxZ": 2.0},
                "gravity": gravity,
                "isGrounded": True,
                "actions": ["forward", "back", "left", "right", "jump", "idle"]
            }
        else:
            return {
                "agentPosition": [-2, 1.0, 0],
                "agentVelocity": [0, 0, 0],
                "blockPosition": [-1.5, 0.3, 0],
                "blockVelocity": [0, 0, 0],
                "holdingBlock": False,
                "basketPosition": [6.0, 1.5, 0],
                "basketBounds": {"minX": 5.4, "maxX": 6.6, "minY": 1.5, "maxY": 2.3},
                "gravity": gravity,
                "isGrounded": True,
                "actions": ["forward", "back", "left", "right", "pick", "drop",
                           "throw_weak", "throw_medium", "throw_strong", "idle"]
            }

    def step(self, action: str | int) -> StepResult:
        self.step_count += 1

        # Simple mock behavior
        done = self.step_count >= 50
        success = done and (self.step_count % 10 == 0)

        return StepResult(
            observation=self.reset(self.task, self.gravity),
            reward=1.0 if success else 0.0,
            done=done,
            info={"step": self.step_count, "success": success}
        )

    def health_check(self) -> bool:
        return True
