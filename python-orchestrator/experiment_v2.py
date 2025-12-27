"""
Experiment V2 - Clean Factorial Design
======================================
2×2×2 Design:
- Representation: Formula vs Story
- Guidance: Neutral vs Guided
- Task: Invariant (A) vs Adaptive (B)

Plus gravity conditions:
- Training: g = 9.81
- Test: g = 4.905

Two-stage pipeline:
- Stage 1: Planning (LLM predicts jump_step)
- Stage 2: Execution (deterministic compilation)
"""

import json
import time
import random
import requests
from dataclasses import dataclass, asdict, field
from typing import Literal
import google.generativeai as genai

from config import GEMINI_API_KEY, ENV_SERVER_URL
from prompts_v2 import (
    build_planning_prompt,
    compile_sequence,
    TASK_A_GEOMETRY,
    TASK_B_GEOMETRY
)

# Constants
TRAINING_GRAVITY = 9.81
TEST_GRAVITY = 4.905
TASK_VERSION = "v2"


@dataclass
class ExperimentConfig:
    """Configuration for a single experimental condition"""
    representation: Literal["formula", "story"]
    guidance: Literal["neutral", "guided"]
    task_type: Literal["invariant", "adaptive"]
    gravity_phase: Literal["training", "test"]
    gravity_explained: bool = True  # For story test condition
    model: str = "gemini-2.0-flash"
    temperature: float = 0.2


@dataclass
class PlanningResult:
    """Result from Stage 1: Planning"""
    episode_id: int
    seed: int
    condition: str
    gravity: float
    task_type: str

    # LLM outputs
    predicted_jump_step: int
    physics_reasoning: str
    confidence: str
    llm_latency: float

    # Ground truth
    optimal_jump_step: int
    planning_error: int  # predicted - optimal

    # Raw response
    raw_response: str = ""


@dataclass
class ExecutionResult:
    """Result from Stage 2: Execution"""
    episode_id: int
    success: bool
    steps: int
    reason: str
    final_x: float

    # Trajectory info
    jump_step_executed: int
    landed_x: float
    trajectory: list = field(default_factory=list)


@dataclass
class FullResult:
    """Combined planning + execution result"""
    planning: PlanningResult
    execution: ExecutionResult

    @property
    def success(self) -> bool:
        return self.execution.success


class ExperimentRunner:
    """Runs the V2 experiment with clean factorial design"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(config.model)

        # Select task geometry
        if config.task_type == "invariant":
            self.geometry = TASK_A_GEOMETRY.copy()
        else:
            self.geometry = TASK_B_GEOMETRY.copy()

        # Get optimal jump step for this gravity
        if config.gravity_phase == "training":
            self.gravity = TRAINING_GRAVITY
            self.optimal_jump = self.geometry["optimal_jump_1g"]
        else:
            self.gravity = TEST_GRAVITY
            self.optimal_jump = self.geometry["optimal_jump_0p5g"]

    def run_planning(self, episode_id: int, seed: int) -> PlanningResult:
        """Stage 1: Get LLM's predicted jump step"""

        # Apply seed-based jitter to geometry
        rng = random.Random(seed)
        jittered_geometry = self.geometry.copy()
        jittered_geometry["start_x"] += rng.uniform(-0.1, 0.1)
        jittered_geometry["gap_start"] += rng.uniform(-0.05, 0.05)

        # Build prompt
        prompt = build_planning_prompt(
            representation=self.config.representation,
            guidance=self.config.guidance,
            gravity=self.config.gravity_phase,
            task_geometry=jittered_geometry,
            gravity_explained=self.config.gravity_explained
        )

        # Call LLM
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

            # Parse response
            parsed = self._parse_planning_response(response_text)

        except Exception as e:
            latency = time.time() - start
            parsed = {
                "jump_step": 6,  # Default fallback
                "physics_reasoning": f"Error: {e}",
                "confidence": "low"
            }
            response_text = str(e)

        # Build condition string
        condition = f"{self.config.representation}_{self.config.guidance}"

        return PlanningResult(
            episode_id=episode_id,
            seed=seed,
            condition=condition,
            gravity=self.gravity,
            task_type=self.config.task_type,
            predicted_jump_step=parsed["jump_step"],
            physics_reasoning=parsed.get("physics_reasoning", ""),
            confidence=parsed.get("confidence", "unknown"),
            llm_latency=latency,
            optimal_jump_step=self.optimal_jump,
            planning_error=parsed["jump_step"] - self.optimal_jump,
            raw_response=response_text[:500]
        )

    def run_execution(
        self,
        episode_id: int,
        seed: int,
        jump_step: int,
        max_steps: int = 30
    ) -> ExecutionResult:
        """Stage 2: Execute the planned sequence"""

        # Compile action sequence
        sequence = compile_sequence(jump_step, total_steps=max_steps)

        # Reset environment with seed
        response = requests.post(
            f"{ENV_SERVER_URL}/reset",
            json={
                "task": "gap",
                "taskVersion": TASK_VERSION,
                "gravity": self.gravity,
                "seed": seed,
                # Pass custom geometry for Task B
                "landingZoneStart": self.geometry["land_start"],
                "landingZoneEnd": self.geometry["land_end"]
            },
            timeout=10
        )
        response.raise_for_status()
        obs = response.json().get("observation", response.json())

        trajectory = []
        jump_step_executed = -1

        for step, action in enumerate(sequence):
            # Record state
            trajectory.append({
                "step": step,
                "x": obs["agentPosition"][0],
                "y": obs["agentPosition"][1],
                "vx": obs["agentVelocity"][0],
                "action": action
            })

            if action == "jump":
                jump_step_executed = step

            # Execute action
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

            if done:
                return ExecutionResult(
                    episode_id=episode_id,
                    success=info.get("success", False),
                    steps=step + 1,
                    reason=info.get("reason", "unknown"),
                    final_x=obs.get("agentPosition", [0, 0, 0])[0],
                    jump_step_executed=jump_step_executed,
                    landed_x=obs.get("agentPosition", [0, 0, 0])[0],
                    trajectory=trajectory
                )

        # Timeout
        return ExecutionResult(
            episode_id=episode_id,
            success=False,
            steps=len(sequence),
            reason="timeout",
            final_x=obs.get("agentPosition", [0, 0, 0])[0],
            jump_step_executed=jump_step_executed,
            landed_x=obs.get("agentPosition", [0, 0, 0])[0],
            trajectory=trajectory
        )

    def run_episode(self, episode_id: int) -> FullResult:
        """Run complete episode: planning + execution"""

        # Generate random seed
        seed = random.randint(0, 2**31 - 1)

        # Stage 1: Planning
        planning = self.run_planning(episode_id, seed)

        # Stage 2: Execution (using predicted jump step)
        execution = self.run_execution(
            episode_id, seed,
            jump_step=planning.predicted_jump_step
        )

        return FullResult(planning=planning, execution=execution)

    def _parse_planning_response(self, text: str) -> dict:
        """Parse LLM planning response"""
        try:
            # Remove markdown
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            data = json.loads(text.strip())

            # Validate jump_step
            jump_step = int(data.get("jump_step", 6))
            jump_step = max(1, min(jump_step, 15))  # Clamp to reasonable range

            return {
                "jump_step": jump_step,
                "physics_reasoning": data.get("physics_reasoning", ""),
                "confidence": data.get("confidence", "unknown")
            }

        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback: try to extract number
            import re
            match = re.search(r'"jump_step"\s*:\s*(\d+)', text)
            if match:
                return {"jump_step": int(match.group(1)), "confidence": "low"}
            return {"jump_step": 6, "confidence": "parse_error"}


def run_full_experiment(
    episodes_per_condition: int = 50,
    verbose: bool = True
) -> dict:
    """
    Run the complete 2×2×2×2 factorial experiment.

    Factors:
    - Representation: formula, story
    - Guidance: neutral, guided
    - Task: invariant (A), adaptive (B)
    - Gravity: training, test
    """

    print("=" * 70)
    print("EXPERIMENT V2: Clean Factorial Design")
    print("=" * 70)
    print(f"Episodes per condition: {episodes_per_condition}")
    print(f"Total conditions: 2×2×2×2 = 16")
    print(f"Total episodes: {episodes_per_condition * 16}")
    print()

    # Check server
    try:
        response = requests.get(f"{ENV_SERVER_URL}/health", timeout=5)
        print(f"Server: {response.json()}")
    except Exception as e:
        print(f"ERROR: Cannot connect to server: {e}")
        return {}

    all_results = {}

    # Iterate through all conditions
    for representation in ["formula", "story"]:
        for guidance in ["neutral", "guided"]:
            for task_type in ["invariant", "adaptive"]:
                for gravity_phase in ["training", "test"]:

                    condition_name = f"{representation}_{guidance}_{task_type}_{gravity_phase}"

                    if verbose:
                        print(f"\n{'='*60}")
                        print(f"CONDITION: {condition_name}")
                        print(f"{'='*60}")

                    config = ExperimentConfig(
                        representation=representation,
                        guidance=guidance,
                        task_type=task_type,
                        gravity_phase=gravity_phase,
                        gravity_explained=(gravity_phase == "test")
                    )

                    runner = ExperimentRunner(config)

                    results = []
                    successes = 0
                    planning_errors = []

                    for ep in range(episodes_per_condition):
                        result = runner.run_episode(ep)
                        results.append(result)

                        if result.success:
                            successes += 1

                        planning_errors.append(result.planning.planning_error)

                        if verbose and ep < 3:
                            status = "✓" if result.success else "✗"
                            print(f"  Ep {ep+1}: jump@{result.planning.predicted_jump_step} "
                                  f"(optimal={result.planning.optimal_jump_step}) "
                                  f"→ {status} {result.execution.reason}")

                    # Compute statistics
                    success_rate = successes / episodes_per_condition
                    avg_error = sum(planning_errors) / len(planning_errors)
                    exact_match = sum(1 for e in planning_errors if e == 0) / len(planning_errors)
                    within_1 = sum(1 for e in planning_errors if abs(e) <= 1) / len(planning_errors)

                    summary = {
                        "condition": condition_name,
                        "representation": representation,
                        "guidance": guidance,
                        "task_type": task_type,
                        "gravity_phase": gravity_phase,
                        "episodes": episodes_per_condition,
                        "successes": successes,
                        "success_rate": success_rate,
                        "avg_planning_error": avg_error,
                        "exact_match_rate": exact_match,
                        "within_1_step_rate": within_1,
                        "optimal_jump": runner.optimal_jump
                    }

                    if verbose:
                        print(f"\n  Success: {success_rate*100:.1f}%")
                        print(f"  Planning error: {avg_error:+.2f} steps (avg)")
                        print(f"  Exact match: {exact_match*100:.1f}%")
                        print(f"  Within ±1: {within_1*100:.1f}%")

                    all_results[condition_name] = {
                        "summary": summary,
                        "episodes": [
                            {
                                "planning": asdict(r.planning),
                                "execution": asdict(r.execution)
                            }
                            for r in results
                        ]
                    }

    # Final analysis
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    # Create comparison table
    print(f"\n{'Condition':<45} {'Success':>8} {'AvgErr':>8} {'Exact':>8}")
    print("-" * 70)

    for name, data in sorted(all_results.items()):
        s = data["summary"]
        print(f"{name:<45} {s['success_rate']*100:>7.1f}% {s['avg_planning_error']:>+7.2f} {s['exact_match_rate']*100:>7.1f}%")

    # Save results
    with open("experiment_v2_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to experiment_v2_results.json")

    return all_results


def run_quick_test(episodes: int = 5):
    """Quick test of a single condition"""

    print("Quick test: formula_neutral_invariant_training")

    config = ExperimentConfig(
        representation="formula",
        guidance="neutral",
        task_type="invariant",
        gravity_phase="training"
    )

    runner = ExperimentRunner(config)

    for ep in range(episodes):
        result = runner.run_episode(ep)
        print(f"\nEpisode {ep+1}:")
        print(f"  Planning: jump@{result.planning.predicted_jump_step} "
              f"(optimal={result.planning.optimal_jump_step}, "
              f"error={result.planning.planning_error:+d})")
        print(f"  Reasoning: {result.planning.physics_reasoning[:100]}...")
        print(f"  Execution: {'SUCCESS' if result.success else 'FAIL'} "
              f"({result.execution.reason})")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        run_quick_test()
    else:
        run_full_experiment(episodes_per_condition=20, verbose=True)
