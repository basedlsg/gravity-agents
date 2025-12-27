import requests
import json
import random
import numpy as np
from dataclasses import dataclass
from typing import Literal, Tuple
from config import GEMINI_API_KEY, ENV_SERVER_URL
import google.generativeai as genai
from prompts_v2 import get_feedback_hint

# Configure GenAI
genai.configure(api_key=GEMINI_API_KEY)

@dataclass
class ExperimentConfig:
    representation: str = "formula"
    guidance: str = "neutral"
    granularity: str = "fine" 
    model: str = "gemini-2.0-flash"
    temperature: float = 0.2

@dataclass
class TaskParams:
    taskId: str = "ThrowBlockV2"
    difficulty: str = "hard"
    gravity: float = 9.81
    seed: int = 1000

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
        if result.get("done"): self.last_obs["isTaskComplete"] = True
        return self.last_obs
    
    def get_last_obs(self) -> dict:
        return self.last_obs

class CalibrationRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = genai.GenerativeModel(config.model)
        self.env = WebEnvClient()
        self.gain_estimate = 0.12 # Default prior
        
    def run_phase0_calibration(self) -> Tuple[float, float, float]:
        """
        Phase 0: Robust Actuator ID (Multi-step)
        """
        print("  [Phase 0] Calibrating Actuators (Multi-step)...")
        
        # Samples
        deltas = []
        
        # We'll do 1 sequence: Forward 10, Backward 10
        # This gives us a long baseline to average out sliding/jitter
        
        CALIB_STEPS = 10
        
        # 1. Forward Sequence
        start_x = self.env.get_last_obs()["agentPosition"][0]
        for _ in range(CALIB_STEPS):
            self.env.step("forward", granularity=self.config.granularity)
        # Settle
        obs_end = self.env.step("idle", granularity="coarse")
        end_x = obs_end["agentPosition"][0]
        
        delta_fwd = end_x - start_x
        gain_fwd = abs(delta_fwd) / CALIB_STEPS
        deltas.append(gain_fwd)
        print(f"    Forward {CALIB_STEPS} steps. Delta: {delta_fwd:.4f}m. Gain: {gain_fwd:.4f}")
        
        # 2. Backward Sequence
        start_x = end_x
        for _ in range(CALIB_STEPS):
            self.env.step("back", granularity=self.config.granularity)
        # Settle
        obs_end = self.env.step("idle", granularity="coarse")
        end_x = obs_end["agentPosition"][0]
        
        delta_bk = end_x - start_x
        gain_bk = abs(delta_bk) / CALIB_STEPS
        deltas.append(gain_bk)
        print(f"    Back {CALIB_STEPS} steps. Delta: {delta_bk:.4f}m. Gain: {gain_bk:.4f}")

        # Robust Aggregation
        measured_step_len = np.mean(deltas)
        self.gain_estimate = measured_step_len
        
        # True values (Empirically known)
        true_step_len = {
            "coarse": 0.4973,
            "medium": 0.2479,
            "fine": 0.1233
        }[self.config.granularity]
        
        error = abs(measured_step_len - true_step_len)
        rel_error = error / true_step_len if true_step_len > 0 else 0
        
        print(f"  [Calibrated] Mean Gain: {measured_step_len:.4f}m/step (True: {true_step_len:.4f}m, Err: {rel_error:.1%})")
        
        return measured_step_len, true_step_len, rel_error

    def update_gain(self, moved_steps: int, actual_delta: float):
        if moved_steps <= 0: return
        
        inst_gain = abs(actual_delta) / moved_steps
        alpha = 0.3
        
        old_gain = self.gain_estimate
        self.gain_estimate = (1 - alpha) * old_gain + alpha * inst_gain
        # print(f"    [Adaptive] Gain Update: {old_gain:.4f} -> {self.gain_estimate:.4f} (Inst: {inst_gain:.4f})")

    def run_planning_phase1(self, obs, history):
        """
        Phase 1: Planning with Meters
        """
        current_x = obs["agentPosition"][0]
        target_x = obs["basketPosition"][0]
        dist = target_x - current_x
        
        # Calculate retries based on current error and gain
        # This is strictly for the prompt context, actual loop handles limits
        
        prompt = f"""
        You are a physics agent. 
        GOAL: Throw the block into the basket at X={target_x:.2f}.
        CURRENT POSITION: X={current_x:.2f}.
        DISTANCE: {dist:.2f} meters.
        
        [SYSTEM CALIBRATION DATA]
        Current Actuator Gain: {self.gain_estimate:.4f} meters/step.
        
        INSTRUCTIONS:
        1. Decide how many METERS to move.
        2. I will automatically convert meters to steps using the gain.
        
        OUTPUT FORMAT:
        {{
            "thought": "reasoning...",
            "move_meters": 0.5,
            "throw_strength": "medium"
        }}
        """
        
        if history:
             last = history[-1]
             exec_data = last['execution']
             prompt += f"\n\nPREVIOUS ATTEMPT:\nPlanned {last['planning'].get('move_meters')}m. Moved {exec_data['moved_steps']} steps. Landed at {exec_data['block_landing_x']}. Missed by {exec_data['block_landing_x'] - target_x:.2f}."
        
        response = self.model.generate_content(prompt)
        try:
            text = response.text.replace("```json", "").replace("```", "").strip()
            plan = json.loads(text)
            return plan
        except Exception as e:
            print(f"Planning Error: {e}")
            return {"move_meters": 0.0, "throw_strength": "medium"}

    def run_episode(self, seed):
        print(f"\n--- Episode (Seed {seed}) Start ---")
        
        # Reset
        task_params = TaskParams(seed=seed, gravity=9.81)
        obs = self.env.reset(task_params)
        
        # PHASE 0: CALIBRATION
        step_len, true_len, gain_error = self.run_phase0_calibration()
        
        # PHASE 1: LOOP
        # Dynamic Budget: Roughly ceiling(error / step_len)
        initial_error = abs(obs["basketPosition"][0] - obs["agentPosition"][0])
        # A conservative upper bound.
        # But we need a simpler rule: max_retries = clamp(3, error/gain, 30)
        
        max_retries = max(3, min(30, int(initial_error / (2 * step_len))))
        print(f"  Dynamic Retry Budget: {max_retries}")
        
        history = []
        success = False
        
        for attempt in range(max_retries):
            # Plan
            plan = self.run_planning_phase1(obs, history)
            
            # Execute (Adapter)
            meters = float(plan.get("move_meters", 0))
            steps = int(round(meters / self.gain_estimate))
            strength = plan.get("throw_strength", "medium")
            
            # Action Sequence
            start_x = obs["agentPosition"][0]
            
            actions = ["forward"] * steps if steps > 0 else ["back"] * abs(steps)
            if steps == 0: actions = []
            
            actions += ["pick", strength, "idle", "idle", "idle", "idle"]
            
            block_landing_x = 0.0
            
            # Step Loop
            step_count_actual = 0
            for act in actions:
                obs = self.env.step(act, granularity=self.config.granularity)
                if act in ["forward", "back"]: step_count_actual += 1
                
                if not obs["holdingBlock"] and obs["blockPosition"][1] < 0.2:
                     block_landing_x = obs["blockPosition"][0]
                if obs["isTaskComplete"]:
                    success = True
            
            # Adaptive Update
            end_x = obs["agentPosition"][0]
            actual_delta = abs(end_x - start_x)
            if step_count_actual > 0:
                self.update_gain(step_count_actual, actual_delta)
            
            # Fallback landing
            if block_landing_x == 0.0: block_landing_x = self.env.get_last_obs()["blockPosition"][0]
            
            print(f"  Attempt {attempt+1}: Planned {meters:.2f}m -> {steps} steps. Landed {block_landing_x:.2f}. Success: {success}. New Gain: {self.gain_estimate:.4f}")
            
            history.append({"planning": plan, "execution": {"moved_steps": steps, "block_landing_x": block_landing_x}})
            
            if success: break
            
        return {
            "seed": seed,
            "success": success,
            "gain_error": gain_error,
            "calibrated_len": step_len
        }

def run_calibration_experiment():
    seeds = [1000, 1001, 1002, 1003, 1004]
    config = ExperimentConfig(granularity="fine")
    runner = CalibrationRunner(config)
    
    results = []
    for s in seeds:
        res = runner.run_episode(s)
        results.append(res)
        
    print("\n\n=== FINAL RESULTS (Calibration Enabled) ===")
    print(f"{'Seed':<5} | {'Success':<8} | {'Gain Error':<10} | {'Meas Step':<10}")
    for r in results:
        print(f"{r['seed']:<5} | {str(r['success']):<8} | {r['gain_error']:.1%}      | {r['calibrated_len']:.4f}")

if __name__ == "__main__":
    run_calibration_experiment()
