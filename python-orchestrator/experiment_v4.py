import requests
import json
import random
from dataclasses import dataclass, asdict
from typing import Literal
from config import GEMINI_API_KEY, ENV_SERVER_URL
import google.generativeai as genai
from prompts_v2 import (
    build_throw_planning_prompt,
    STAGE2_FEEDBACK_PROMPT,
    get_feedback_hint,
    TASK_THROW_TEMPLATE
)

@dataclass
class ExperimentConfig:
    representation: Literal["formula", "story"]
    guidance: Literal["neutral", "guided"]
    gravity_phase: Literal["training", "test"]
    granularity: Literal["coarse", "medium", "fine"] = "coarse" # New Factor
    gravity_explained: bool = True
    model: str = "gemini-2.0-flash"
    temperature: float = 0.2

@dataclass
class TaskParams:
    taskId: str
    difficulty: str
    gravity: float
    seed: int

class WebEnvClient:
    def reset(self, params: TaskParams) -> dict:
        response = requests.post(
            f"{ENV_SERVER_URL}/reset",
            json={
                "task": "throw", 
                "taskVersion": "v2", 
                "gravity": params.gravity, 
                "seed": params.seed
            },
            timeout=10
        )
        response.raise_for_status()
        self.last_obs = response.json().get("observation", {})
        return self.last_obs

    def step(self, action: str, granularity: str = "coarse") -> dict:
        # Scale logic
        scale = 1.0
        if granularity == "medium": scale = 0.5
        if granularity == "fine": scale = 0.25
        
        req_body = {"action": action}
        if action in ["forward", "back", "left", "right"]:
            req_body["durationScale"] = scale 
            
        response = requests.post(f"{ENV_SERVER_URL}/step", json=req_body)
        result = response.json()
        self.last_obs = result.get("observation", {})
        
        self.last_obs["isTaskComplete"] = result.get("info", {}).get("done", False)
        self.last_obs["info"] = result.get("info", {})
        if result.get("done"):
             self.last_obs["isTaskComplete"] = True
             
        return self.last_obs

    def get_last_obs(self) -> dict:
        return self.last_obs

# Reuse v3 runner but extend for loops
class ExperimentRunnerV4:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(config.model)
        self.env = WebEnvClient()

    def run_planning(self, episode_id: int, seed: int, obs: dict, history: list = None) -> dict:
        """
        Stage 1: Initial Planning OR Re-Planning
        If history is provided, it's a Feedback Loop.
        """
        if not history:
            # First attempt - Standard v3 Prompt
            prompt = build_throw_planning_prompt(
                self.config.representation,
                self.config.guidance,
                self.config.gravity_phase,
                obs,
                gravity_explained=(self.config.gravity_phase == "test")
            )
        else:
            # Retry - Feedback Prompt
            last_attempt = history[-1]
            last_execution = last_attempt["execution"]
            last_plan = last_attempt["planning"]
            
            # semantic analysis
            target = last_execution["basket_x"]
            # Block x is where it landed.
            # IN V3 result, we didn't explicitly track block_landing_x, 
            # but we can infer it from the 'reason' or we need to capture it.
            # OPTIMIZATION: We'll assume the client returns block position in 'final_state'
            
            actual = last_execution.get("block_landing_x", target + 5.0) # Default huge miss if not found
            error = actual - target
            
            error_desc = "OVERSHOT" if error > 0 else "UNDERSHOT"
            
            hint = get_feedback_hint(self.config.representation, error)
            
            prompt = STAGE2_FEEDBACK_PROMPT.format(
                moved_steps=last_execution["moved_steps"],
                throw_strength=last_execution["strength_executed"],
                target_dist=abs(target - obs["agentPosition"][0]), # Approx original dist
                actual_dist=abs(actual - (obs["agentPosition"][0] + last_execution["moved_steps"] * 0.5)), # Approx throw range
                error_desc=error_desc,
                error_val=error,
                physics_hint=hint
            )

        # Call LLM
        response = self.model.generate_content(prompt)
        try:
            text = response.text.replace("```json", "").replace("```", "").strip()
            plan = json.loads(text)
            return plan
        except Exception as e:
            print(f"Planning Error: {e}")
            # Fallback
            return {"move_steps": 0, "throw_strength": "medium", "error": str(e)}

    def run_execution(self, episode_id: int, seed: int, plan: dict) -> dict:
        """
        Execute the plan and return detailed telemetry including landing position.
        """
        # 1. Setup
        task_params = TaskParams(
            taskId="ThrowBlockV2",
            difficulty="hard",
            gravity=9.81 if self.config.gravity_phase == "training" else 4.905,
            seed=seed
        )
        self.env.reset(task_params)
        
        # 2. Compile Actions
        move_steps = int(plan.get("move_steps", 0))
        strength = plan.get("throw_strength", "medium").lower()
        if "weak" in strength: strength = "throw_weak"
        elif "strong" in strength: strength = "throw_strong"
        else: strength = "throw_medium"
        
        actions = ["forward"] * move_steps + ["pick", strength, "idle", "idle", "idle", "idle"] # Idles to let it land
        
        # 3. Step Loop
        block_landing_x = 0.0
        success = False
        basket_x = 0.0
        
        for action in actions:
            obs = self.env.step(action, granularity=self.config.granularity)
            basket_x = obs["basketPosition"][0]
            block_x = obs["blockPosition"][0]
            
            # Simple landing detector: if block is on ground (y ~ 0.15) and not holding
            if not obs["holdingBlock"] and obs["blockPosition"][1] < 0.2:
                 block_landing_x = block_x
                 
            if obs["isTaskComplete"]:
                success = True
                
        # If it fell off world or didn't settle, use last known
        if block_landing_x == 0.0:
            block_landing_x = self.env.get_last_obs()["blockPosition"][0]

        return {
            "success": success,
            "moved_steps": move_steps,
            "strength_executed": strength,
            "basket_x": basket_x,
            "block_landing_x": block_landing_x,
            "reason": "success" if success else "miss"
        }

    def run_closed_loop_episode(self, episode_id: int, max_retries: int = 3):
        seed = episode_id * 1000 # Deterministic seed
        
        # Initial Observation
        task_params = TaskParams(
            taskId="ThrowBlockV2",
            difficulty="hard",
            gravity=9.81 if self.config.gravity_phase == "training" else 4.905,
            seed=seed
        )
        obs = self.env.reset(task_params)
        
        history = []
        
        print(f"\n--- Episode {episode_id} Start (Goal x={obs['basketPosition'][0]:.2f}) ---")
        
        for attempt in range(max_retries):
            # 1. Plan
            plan = self.run_planning(episode_id, seed, obs, history)
            
            # 2. Execute
            result = self.run_execution(episode_id, seed, plan)
            
            # 3. Log
            print(f"Attempt {attempt+1}: Moved {result['moved_steps']}, Threw {result['strength_executed']} -> Landed {result['block_landing_x']:.2f} (Target {result['basket_x']:.2f}) [{'✓' if result['success'] else '✗'}]")
            
            history.append({"planning": plan, "execution": result})
            
            if result["success"]:
                return True, history
                
        print("FAIL: Max retries reached.")
        return False, history

def run_v4_experiment():
    # Only test the problem cases: Moon Gravity
    # Factorial Design: Representation x Granularity
    conditions = []
    for rep in ["formula"]: # Focus on Formula for System ID
        for gran in ["coarse", "medium", "fine"]:
            conditions.append(ExperimentConfig(rep, "neutral", "test", granularity=gran))
    
    episodes = 5
    
    for config in conditions:
        print(f"\n\n{'='*60}\nRUNNING v4: {config.representation} | Granularity: {config.granularity}\n{'='*60}")
        runner = ExperimentRunnerV4(config)
        success_count = 0
        
        for ep in range(episodes):
            success, _ = runner.run_closed_loop_episode(ep, max_retries=3)
            if success: success_count += 1
            
        print(f"\nResult: {success_count}/{episodes} Successes")

if __name__ == "__main__":
    run_v4_experiment()
