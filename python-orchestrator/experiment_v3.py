"""
Experiment V3 - The Precision Throw
===================================
Tests multi-variable physics (gravity + distance) in a parabolic throw task.

Factors:
- Representation: formula, story
- Guidance: neutral, guided (stubs)
- Task: throwing (fixed angle, variable strength)
- Gravity: training (9.81), test (4.905)
"""

import json
import time
import random
import requests
from dataclasses import dataclass, asdict, field
from typing import Literal
import google.generativeai as genai

from config import GEMINI_API_KEY, ENV_SERVER_URL
from prompts_v2 import build_throw_planning_prompt

# Constants
TRAINING_GRAVITY = 9.81
TEST_GRAVITY = 4.905
TASK_VERSION = "v2" # Using ThrowBlockTaskV2 in the server

@dataclass
class ExperimentConfig:
    representation: Literal["formula", "story"]
    guidance: Literal["neutral", "guided"]
    gravity_phase: Literal["training", "test"]
    gravity_explained: bool = True
    model: str = "gemini-2.0-flash"
    temperature: float = 0.2

@dataclass
class PlanningResult:
    episode_id: int
    seed: int
    condition: str
    gravity: float
    # LLM outputs
    predicted_move_steps: int
    predicted_throw_strength: str
    physics_reasoning: str
    confidence: str
    llm_latency: float
    # Ground truth (calculated post-hoc)
    optimal_throw_strength: str
    raw_response: str = ""

@dataclass
class ExecutionResult:
    episode_id: int
    success: bool
    steps: int
    reason: str
    final_pos: list
    strength_executed: str
    moved_steps: int

@dataclass
class FullResult:
    planning: PlanningResult
    execution: ExecutionResult

    @property
    def success(self) -> bool:
        return self.execution.success

class ExperimentRunnerV3:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(config.model)
        self.gravity = TRAINING_GRAVITY if config.gravity_phase == "training" else TEST_GRAVITY

    def run_planning(self, episode_id: int, seed: int, obs: dict) -> PlanningResult:
        """Stage 1: Get LLM's planning (steps to move + strength)"""
        prompt = build_throw_planning_prompt(
            representation=self.config.representation,
            guidance=self.config.guidance,
            gravity=self.config.gravity_phase,
            obs=obs,
            gravity_explained=self.config.gravity_explained
        )

        start = time.time()
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=500
                )
            )
            response_text = response.text.strip()
            latency = time.time() - start
            parsed = self._parse_planning_response(response_text)
        except Exception as e:
            latency = time.time() - start
            parsed = {"move_steps": 0, "throw_strength": "medium", "physics_reasoning": str(e), "confidence": "error"}
            response_text = str(e)

        condition = f"{self.config.representation}_{self.config.guidance}_{self.config.gravity_phase}"
        
        return PlanningResult(
            episode_id=episode_id,
            seed=seed,
            condition=condition,
            gravity=self.gravity,
            predicted_move_steps=parsed["move_steps"],
            predicted_throw_strength=parsed["throw_strength"],
            physics_reasoning=parsed.get("physics_reasoning", ""),
            confidence=parsed.get("confidence", "unknown"),
            llm_latency=latency,
            optimal_throw_strength=obs.get("optimalThrowStrength", "unknown"),
            raw_response=response_text[:500]
        )

    def _parse_planning_response(self, text: str) -> dict:
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            data = json.loads(text.strip())
            return {
                "move_steps": int(data.get("move_steps", 0)),
                "throw_strength": data.get("throw_strength", "medium").lower(),
                "physics_reasoning": data.get("physics_reasoning", ""),
                "confidence": data.get("confidence", "unknown")
            }
        except:
            return {"move_steps": 0, "throw_strength": "medium", "confidence": "parse_error"}

    def run_execution(self, episode_id: int, seed: int, plan: PlanningResult) -> ExecutionResult:
        """Stage 2: Execute the plan (Move -> Pick -> Throw)"""
        # Reset env
        response = requests.post(
            f"{ENV_SERVER_URL}/reset",
            json={"task": "throw", "taskVersion": TASK_VERSION, "gravity": self.gravity, "seed": seed},
            timeout=10
        )
        response.raise_for_status()
        
        # 1. Move steps
        steps_taken = 0
        for _ in range(plan.predicted_move_steps):
            requests.post(f"{ENV_SERVER_URL}/step", json={"action": "forward"})
            steps_taken += 1

        # 2. Pick up
        requests.post(f"{ENV_SERVER_URL}/step", json={"action": "pick"})

        # 3. Throw
        action = f"throw_{plan.predicted_throw_strength}"
        response = requests.post(f"{ENV_SERVER_URL}/step", json={"action": action})
        result = response.json()

        # 4. Wait for outcome (settle)
        for _ in range(30):
            response = requests.post(f"{ENV_SERVER_URL}/step", json={"action": "idle"})
            result = response.json()
            if result.get("done"):
                break

        info = result.get("info", {})
        return ExecutionResult(
            episode_id=episode_id,
            success=info.get("success", False),
            steps=steps_taken + 2 + info.get("step", 0),
            reason=info.get("reason", "unknown"),
            final_pos=result.get("observation", {}).get("blockPosition", [0,0,0]),
            strength_executed=plan.predicted_throw_strength,
            moved_steps=steps_taken
        )

    def run_episode(self, episode_id: int) -> FullResult:
        seed = random.randint(0, 2**31 - 1)
        # Initial peek at env to get starting obs for planning
        response = requests.post(
            f"{ENV_SERVER_URL}/reset",
            json={"task": "throw", "taskVersion": TASK_VERSION, "gravity": self.gravity, "seed": seed},
            timeout=10
        )
        obs = response.json().get("observation", {})
        
        planning = self.run_planning(episode_id, seed, obs)
        execution = self.run_execution(episode_id, seed, planning)
        return FullResult(planning=planning, execution=execution)

def run_full_experiment(episodes_per_condition: int = 10):
    conditions = []
    for rep in ["formula", "story"]:
        for guidance in ["neutral"]: # Guided stubs for now
            for phase in ["training", "test"]:
                # Factor in 'invariant' (standard) vs 'adaptive' (Moon-g) logic
                # For v3, Moon-g IS the challenge
                conditions.append(ExperimentConfig(representation=rep, guidance=guidance, gravity_phase=phase))

    all_results = {}
    
    for config in conditions:
        condition_name = f"{config.representation}_{config.guidance}_{config.gravity_phase}"
        print(f"\n{'='*60}\nCONDITION: {condition_name}\n{'='*60}")
        
        runner = ExperimentRunnerV3(config)
        results = []
        successes = 0
        
        for ep in range(episodes_per_condition):
            result = runner.run_episode(ep)
            results.append({
                "planning": asdict(result.planning),
                "execution": asdict(result.execution)
            })
            if result.success:
                successes += 1
            
            status = "✓" if result.success else "✗"
            print(f"  Ep {ep+1}: Moved {result.execution.moved_steps}, Threw {result.execution.strength_executed} → {status} ({result.execution.reason})")
            
        all_results[condition_name] = {
            "summary": {
                "condition": condition_name,
                "episodes": episodes_per_condition,
                "successes": successes,
                "success_rate": successes / episodes_per_condition
            },
            "episodes": results
        }

        # Save partial results
        with open("experiment_v3_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nFinal results saved to experiment_v3_results.json")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        run_quick_test()
    else:
        run_full_experiment(episodes_per_condition=10)
