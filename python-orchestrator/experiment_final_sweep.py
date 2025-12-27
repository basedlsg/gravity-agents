import requests
import json
import numpy as np
import time
from dataclasses import dataclass
from typing import Literal, Tuple, List, Optional
from config import GEMINI_API_KEY, ENV_SERVER_URL
import google.generativeai as genai

# Configure GenAI
genai.configure(api_key=GEMINI_API_KEY)

@dataclass
class ExperimentConfig:
    mode: Literal["adaptive", "phase0_only", "ema_only"]
    granularity: str = "fine" 
    model: str = "gemini-2.0-flash"
    num_seeds: int = 50
    start_seed: int = 2000

class WebEnvClient:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id

    def reset(self, seed: int) -> dict:
        payload = {
            "sessionId": self.session_id,
            "task": "throw", 
            "taskVersion": "v2", 
            "gravity": 9.81, 
            "seed": seed
        }
        response = requests.post(f"{ENV_SERVER_URL}/reset", json=payload, timeout=10)
        response.raise_for_status()
        return response.json().get("observation", {})

    def step(self, action: str, granularity: str = "coarse") -> dict:
        scale = {"coarse": 1.0, "medium": 0.5, "fine": 0.25}[granularity]
        req = {
            "sessionId": self.session_id,
            "action": action
        }
        if action in ["forward", "back", "left", "right"]: req["durationScale"] = scale
        
        response = requests.post(f"{ENV_SERVER_URL}/step", json=req)
        response.raise_for_status()
        return response.json().get("observation", {})

    def get_info(self) -> dict:
        response = requests.get(f"{ENV_SERVER_URL}/info")
        response.raise_for_status()
        return response.json()

class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, session_id: str = "default"):
        self.config = config
        self.model = genai.GenerativeModel(config.model)
        self.env = WebEnvClient(session_id=session_id)
        self.results = []

    def run_calibration(self) -> float:
        """Phase 0: 10-step fwd/back"""
        if self.config.mode == "ema_only": return 1.0 # Bad Prior
        
        start_obs = self.env.step("idle", granularity="coarse")
        start_x_0 = start_obs.get("agentPosition", [0])[0]
        
        for _ in range(10): 
            self.env.step("forward", self.config.granularity)
        
        # Settle
        mid_obs = self.env.step("idle", granularity="coarse")
        mid_x = mid_obs.get("agentPosition", [0])[0]
        
        for _ in range(10): 
            self.env.step("back", self.config.granularity)
            
        end_obs = self.env.step("idle", granularity="coarse")
        end_x = end_obs.get("agentPosition", [0])[0]
        
        gain_fwd = abs(mid_x - start_x_0) / 10
        gain_bk = abs(end_x - mid_x) / 10
        avg_gain = (gain_fwd + gain_bk) / 2
        print(f"DEBUG: Calibration Gain: {avg_gain} (Fwd: {gain_fwd}, Bk: {gain_bk})")
        return avg_gain

    def run_planner(self, obs, history, gain, telemetry):
        current_x = obs["agentPosition"][0]
        current_z = obs["agentPosition"][2]
        target_x = obs["basketPosition"][0]
        target_z = obs["basketPosition"][2] # Usually 0.0
        
        dist_x = target_x - current_x
        dist_z = target_z - current_z
        dist = np.sqrt(dist_x**2 + dist_z**2)
        
        obstacle_plane_x = target_x - 0.75
        
        prompt = f"""
        GOAL: Get the block into the basket at (X={target_x:.2f}, Z={target_z:.2f}).
        CURRENT: (X={current_x:.2f}, Z={current_z:.2f}). DELTA: dX={dist_x:.2f}m, dZ={dist_z:.2f}m. DIST={dist:.2f}m.

        ACTUATION GAINS (measured): gain_x={gain:.4f} m/step, gain_z={gain:.4f} m/step.

        STUCK TELEMETRY:
        stuck_flag={telemetry['stuck_flag']}, stuck_count={telemetry['stuck_count']},
        last_stuck={telemetry['last_stuck_str']}, last_escape={telemetry['last_escape_dir']}.
        
        {telemetry.get('subgoal', '')}

        OBSTACLE HINT:
        There may be a wall near X ≈ {obstacle_plane_x:.2f}. If you are repeatedly stuck at the same X, you must DETOUR in Z.

        POLICY:
        1) Normal approach: reduce |dX| while keeping Z near target_z.
        2) If stuck_count >= 2 OR you see you are stuck at similar X repeatedly:
           - DETOUR: move laterally to reach |Z - target_z| >= 2.0.
           - PASS: move forward until X > {obstacle_plane_x:.2f} + 0.30.
           - RETURN: move Z back toward target_z (usually 0.0).
        3) Always output small, safe moves in meters. You are allowed to move in both X and Z in the same step.

        OUTPUT ONLY JSON with:
        {{"move_x_meters": float, "move_z_meters": float, "throw_strength": "weak"|"medium"|"strong"}}
        """
             
        try:
            resp = self.model.generate_content(prompt)
            return json.loads(resp.text.replace("```json", "").replace("```", "").strip())
        except: return {"move_x_meters": 0.0, "move_z_meters": 0.0, "throw_strength": "medium"}

    def run_seed(self, seed: int):
        obs = self.env.reset(seed)
        if not obs: return None
        
        # 1. Calibration
        gain = self.run_calibration()
        initial_gain = gain
        
        initial_dist = abs(obs["basketPosition"][0] - obs["agentPosition"][0])
        max_retries = 100
        
        history = []
        success = False
        attempts = 0
        
        # State Machine Variables
        mode = "APPROACH"
        reroute_stage = "OFFSET" 
        detour_sign = -1 
        
        # Lane Search
        # Candidates: [1.5, 2.0, 2.2, 2.5, 3.0]
        LANE_CANDIDATES = [1.5, 2.0, 2.2, 2.5, 3.0]
        lane_index = 0
        
        # Telemetry State
        stuck_counter = 0
        last_stuck_x = -999.0
        last_stuck_z = -999.0
        last_escape_dir = "none"
        
        prev_last_stuck_x = -999.0 
        
        # Phase Tracking & Budgets
        phases_visited = set()
        
        # Lane-specific tracking
        lane_attempts = 0
        
        # Probe State
        probe_step_count = 0
        probe_accum_dx = 0.0
        
        # Heuristics
        pass_hesitation_counter = 0
        
        for _ in range(max_retries):
            attempts += 1
            lane_attempts += 1
            phases_visited.add(f"{mode}_{reroute_stage}")
            
            # --- EXECUTIVE LOGIC (Harness) ---
            current_x = obs["agentPosition"][0]
            current_z = obs["agentPosition"][2]
            target_x = obs["basketPosition"][0]
            target_z = obs["basketPosition"][2]
            obstacle_plane_x = target_x - 0.75
            
            # Active Lane Target
            current_detour_mag = LANE_CANDIDATES[lane_index % len(LANE_CANDIDATES)]
            z_detour = target_z + detour_sign * current_detour_mag
            x_pass_target = obstacle_plane_x + 0.50 
            
            current_subgoal_x = target_x 
            current_subgoal_z = target_z
            
            FORCE_LANE_SWITCH = False
            
            # 1. Trigger REROUTE if stuck repeatedly
            if mode == "APPROACH":
                current_subgoal_x = target_x
                current_subgoal_z = target_z
                if stuck_counter >= 2:
                    print(f"  [Executive] Stuck x2 at X={current_x:.2f}. Switching to REROUTE.")
                    mode = "REROUTE"
                    reroute_stage = "OFFSET"
                    lane_index = 0 # Start with tightest lane
                    lane_attempts = 0
            
            # 2. REROUTE State Machine (Lane Search)
            subgoal_str = ""
            
            if mode == "REROUTE":
                # Global Lane Timeout (Budget)
                if lane_attempts > 25: 
                    print(f"  [Executive] Lane {lane_index} Timeout (>25 attempts). Switching.")
                    FORCE_LANE_SWITCH = True
                
                # Check Lateral Stuck (Only in OFFSET)
                if reroute_stage == "OFFSET" and stuck_counter >= 5: # Relaxed tolerance (5)
                     print(f"  [Executive] Lane {lane_index} Layout Blocked (Stuck x5). TRIGGERING WEDGE DIAGNOSTIC.")
                     # --- WEDGE DIAGNOSTIC INIT ---
                     mode = "DIAGNOSTIC"
                     diagnostic_queue = [
                         ("Forward (+X)", 1.0, 0.0),
                         ("Back (-X)", -1.0, 0.0),
                         ("Right (+Z)", 0.0, 1.0),
                         ("Left (-Z)", 0.0, -1.0)
                     ]
                     diagnostic_index = 0
                     diagnostic_results = []
                     stuck_counter = 0 
                     continue

                if reroute_stage == "OFFSET":
                    # Goal: Reach Z lane, but retreat if too close to wall to slide
                    safe_offset_x = obstacle_plane_x - 1.2 
                    current_subgoal_x = min(current_x, safe_offset_x)
                    
                    current_subgoal_z = z_detour
                    
                    if current_x > safe_offset_x + 0.1:
                        subgoal_str = f"SUBGOAL: RETREAT to X={safe_offset_x:.2f} (Current X={current_x:.2f})."
                    else:
                        subgoal_str = f"SUBGOAL: MOVE LATERALLY to Z={z_detour:.2f} (Lane {lane_index})."
                    
                    if abs(current_z - z_detour) < 0.2: 
                        print(f"  [Executive] Offset Achieved (Lane {lane_index}). Switching to PROBE.")
                        reroute_stage = "PROBE"
                        probe_step_count = 0
                        probe_accum_dx = 0.0
                        stuck_counter = 0 

                elif reroute_stage == "PROBE":
                    subgoal_str = f"SUBGOAL: PROBE FORWARD. Testing Lane {lane_index}."
                    pass # Evaluated post-step
                    
                elif reroute_stage == "PASS":
                     current_subgoal_x = x_pass_target
                     current_subgoal_z = z_detour
                     subgoal_str = f"SUBGOAL: MOVE FORWARD past X={x_pass_target:.2f} (Lane {lane_index})."
                     
                     if current_x > x_pass_target:
                        print(f"  [Executive] Passed Obstacle. Switching to RETURN.")
                        reroute_stage = "RETURN"
                        lane_attempts = 0 
                     
                     if pass_hesitation_counter >= 8: # Relaxed watchdog
                         print(f"  [Executive] Stuck in PASS (Lane {lane_index}) despite Probe. Switching Lane.")
                         FORCE_LANE_SWITCH = True
                        
                elif reroute_stage == "RETURN":
                     current_subgoal_x = target_x
                     current_subgoal_z = target_z
                     subgoal_str = f"SUBGOAL: RETURN Z to {target_z:.2f}."
                     if abs(current_z - target_z) < 0.3:
                        print(f"  [Executive] Returned to Center. Switching to APPROACH.")
                        mode = "APPROACH"
                        stuck_counter = 0

            # 3. DIAGNOSTIC MODE
            if mode == "DIAGNOSTIC":
                # Evaluate Previous Step
                if diagnostic_index > 0:
                     prev_d = total_delta
                     prev_name = diagnostic_queue[diagnostic_index-1][0]
                     print(f"  [Diagnostic] Result for {prev_name}: Delta={prev_d:.4f}")
                     diagnostic_results.append((prev_name, prev_d))
                
                # Check Completion
                if diagnostic_index >= len(diagnostic_queue):
                     print("\n*** DIAGNOSTIC REPORT ***")
                     wedged_count = 0
                     for name, d in diagnostic_results:
                         is_stuck_move = d < 0.05
                         status = "WEDGED" if is_stuck_move else "FREE"
                         # print(f"  {name}: Delta={d:.4f} [{status}]")
                         if is_stuck_move: wedged_count += 1
                     
                     if wedged_count == 4:
                         print(f"  [Executive] CONCLUSION: UNSAT_WEDGED. Terminating Episode.")
                         return {
                             "seed": seed, "success": False, "attempts": attempts,
                             "status": "UNSAT_WEDGED", "diagnostic_data": diagnostic_results
                         }
                     else:
                         print(f"  [Executive] CONCLUSION: ESCAPABLE (Policy). Switching Lane.")
                         mode = "REROUTE" # Resume REROUTE mode
                         FORCE_LANE_SWITCH = True
                         diagnostic_index = 0 # Reset
                
                else:
                    # Execute Next Step
                    name, dx_cmd, dz_cmd = diagnostic_queue[diagnostic_index]
                    subgoal_str = f"DIAGNOSTIC: {name}"
                    move_x = dx_cmd
                    move_z = dz_cmd
                    strength = "strong" 
                    forced_action = True
                    print(f"  [Diagnostic] Executing {name}...")
                    diagnostic_index += 1

            if FORCE_LANE_SWITCH:
                lane_index += 1
                reroute_stage = "OFFSET"
                lane_attempts = 0
                stuck_counter = 0
                probe_fail_count = 0
                pass_hesitation_counter = 0
                print(f"  [Executive] Switching to Lane {lane_index} ({LANE_CANDIDATES[lane_index % len(LANE_CANDIDATES)]}m).")

            # Construct Telemetry for LLM
            last_stuck_str = f"(X={last_stuck_x:.2f}, Z={last_stuck_z:.2f})" if last_stuck_x != -999 else "None"
            telemetry = {
                "stuck_flag": (stuck_counter > 0),
                "stuck_count": stuck_counter,
                "last_stuck_str": last_stuck_str,
                "last_escape_dir": last_escape_dir,
                "subgoal": subgoal_str
            }
            
            # --- ACTION SELECTION (Planner vs Probe vs Retreat) ---
            forced_action = False
            move_x, move_z, strength = 0.0, 0.0, "medium"
            
            # 1. FORCED RETREAT (Safety Mechanism) -- DISABLED (Caused Regression in N=30)
            if False and mode == "REROUTE" and reroute_stage == "OFFSET":
                # FIX: Relaxed from 1.2 to 0.85 to prevent "Fighting the Policy" loops.
                # Agent radius is 0.4, so 0.85 allows it to get within ~45cm of the wall before retreating.
                safe_offset_x = obstacle_plane_x - 0.85
                # DEBUG PRINT
                # print(f"DEBUG: X={current_x:.2f} Safe={safe_offset_x:.2f} (Obst={obstacle_plane_x:.2f})")
                
                if current_x > safe_offset_x + 0.1:
                    move_x = -0.5 # Strong Back
                    move_z = 0.0
                    strength = "medium"
                    forced_action = True
                    # print(f"  [Executive] Forced Retreat. X={current_x:.2f} > Safe={safe_offset_x:.2f}.")

            # 2. PROBE LOGIC
            if (not forced_action) and mode == "REROUTE" and reroute_stage == "PROBE":
                # Deterministic Probe
                move_x = 0.5
                move_z = 0.0 
                strength = "medium"
                forced_action = True
                
                # Check probe result (from PREVIOUS step)
                # But wait, we need to execute this step first.
                # The state transition logic happens at TOP of loop.
                # So here we just Command.
                # Next loop iteration will check Delta.
            
            if not forced_action:
                plan = self.run_planner(obs, [], gain, telemetry)
                move_x = float(plan.get("move_x_meters", 0.0))
                move_z = float(plan.get("move_z_meters", 0.0))
                strength = plan.get("throw_strength", "medium")

            # --- DYNAMIC SAFEGUARDS ---
            # Distance to subgoal
            dist_x = abs(current_subgoal_x - current_x)
            dist_z = abs(current_subgoal_z - current_z)
            dist_sub = np.sqrt(dist_x**2 + dist_z**2)
            
            # Dynamic Clamp
            MAX_MOVE = 1.5
            if dist_sub > 2.0: 
                MAX_MOVE = 4.0 
            
            # Clamp
            move_x = max(min(move_x, MAX_MOVE), -MAX_MOVE)
            move_z = max(min(move_z, MAX_MOVE), -MAX_MOVE)
            
            steps_x = int(round(move_x / max(gain, 0.01)))
            steps_z = int(round(move_z / max(gain, 0.01)))
            
            actions = []
            if steps_x > 0: actions += ["forward"] * steps_x
            elif steps_x < 0: actions += ["back"] * abs(steps_x)
            if steps_z > 0: actions += ["right"] * steps_z
            elif steps_z < 0: actions += ["left"] * abs(steps_z)
            
            actions += ["pick", strength, "idle", "idle"]
            
            # --- EXECUTION ---
            start_x_step = obs["agentPosition"][0]
            start_z_step = obs["agentPosition"][2]
            
            executed_steps_count = 0
            block_land_x = 0.0
            
            for act in actions:
                obs = self.env.step(act, self.config.granularity)
                if act in ["forward", "back", "left", "right"]: executed_steps_count += 1
                if not obs["holdingBlock"] and obs["blockPosition"][1] < 0.2:
                    block_land_x = obs["blockPosition"][0]
                if obs.get("isTaskComplete"): success = True
            
            if block_land_x == 0: block_land_x = obs["blockPosition"][0]
            
            # --- ADAPTIVE CONTROL & STUCK ---
            end_x_step = obs["agentPosition"][0]
            end_z_step = obs["agentPosition"][2]
            
            delta_x = end_x_step - start_x_step
            delta_z = end_z_step - start_z_step
            total_delta = np.sqrt(delta_x**2 + delta_z**2)
            
            # Check for Teleport/Anomaly
            if total_delta > 0.8: # Physical limit check
                print(f"  [ANOMALY] Jump > 0.8m ({total_delta:.2f}). Pos ({start_x_step:.2f},{start_z_step:.2f}) -> ({end_x_step:.2f},{end_z_step:.2f})")
                anomaly = True
            else:
                anomaly = False
            
            EPS_MOTION = 0.02
            is_stuck = False
            
            # Track Hesitation in PASS
            if mode == "REROUTE" and reroute_stage == "PASS":
                if delta_x < 0.05: 
                    pass_hesitation_counter += 1
                else:
                    pass_hesitation_counter = 0

            # PROBE RESULTS (Logic)
            if mode == "REROUTE" and reroute_stage == "PROBE":
                probe_step_count += 1
                probe_accum_dx += delta_x
                print(f"  [Probe] Step {probe_step_count}: dx={delta_x:.4f} (Accum={probe_accum_dx:.4f})")
                
                if probe_step_count >= 2:
                    # Decision Time
                    if probe_accum_dx > 0.10: # Threshold
                        print(f"  [Executive] Lane {lane_index} VALIDATED (dx={probe_accum_dx:.2f}). Proceeding to PASS.")
                        reroute_stage = "PASS"
                    else:
                        print(f"  [Executive] Lane {lane_index} BLOCKED (dx={probe_accum_dx:.2f} < 0.1). Switching.")
                        lane_index += 1
                        reroute_stage = "OFFSET"
                        lane_attempts = 0
                    probe_step_count = 0
            
            if executed_steps_count > 0:
                # ROBUST STUCK LOGIC
                x_progress_stalled = abs(delta_x) < EPS_MOTION
                behind_wall = current_x < obstacle_plane_x
                
                stuck_condition = False
                if mode == "APPROACH":
                     if x_progress_stalled and behind_wall:
                         stuck_condition = True
                else:
                     # In Reroute, stuck condition is less strictly gating gain, but tracking progress
                     # Except PROBE already handles itself.
                     # OFFSET handles itself (timeout).
                     if total_delta < EPS_MOTION:
                         stuck_condition = True
                
                if stuck_condition:
                    stuck_counter += 1
                    is_stuck = True
                    prev_last_stuck_x = last_stuck_x
                    last_stuck_x = start_x_step
                    last_stuck_z = start_z_step
                    print(f"  [Stuck] Mode={mode} dx={delta_x:.4f} dz={delta_z:.4f}. Counter={stuck_counter}")
                    # No escape needed in REROUTE, lane switch handles it.
                else:
                    # Valid Motion -> Update Gain if no Anomaly
                    if self.config.mode in ["adaptive", "ema_only"] and not anomaly:
                        inst_gain = total_delta / executed_steps_count
                        gain = 0.7*gain + 0.3*inst_gain
                        # CLAMP GAIN
                        gain = max(0.05, min(0.25, gain))
                        
                        stuck_counter = 0 
                        last_escape_dir = "none"

            # Trace Logging
            step_trace = {
                "attempt": attempts,
                "mode": mode,
                "reroute_stage": reroute_stage,
                "lane_index": lane_index,
                "agent_pos": [round(start_x_step, 3), round(start_z_step, 3)],
                "delta": [round(delta_x, 3), round(delta_z, 3)],
                "gain": round(gain, 4),
                "stuck_counter": stuck_counter,
                "plan": {"x": round(move_x, 2), "z": round(move_z, 2)},
                "status": "ANOMALY" if anomaly else ("STUCK" if is_stuck else "OK")
            }
            if mode == "DIAGNOSTIC": step_trace["status"] = "DIAGNOSTIC"
            history.append(step_trace)
            
            print(f"  Attempt {attempts}: Mode={mode} Stg={reroute_stage}. Plan X={move_x:.2f} Z={move_z:.2f}. Delta={total_delta:.4f}. Gain={gain:.4f}")
            
            if success: break
            
        return {
            "seed": seed, 
            "success": success, 
            "attempts": attempts,
            "dist": initial_dist, 
            "mode": self.config.mode,
            "init_gain": initial_gain, 
            "final_gain": gain,
            "reroute_entered": "REROUTE_OFFSET" in phases_visited,
            "offset_reached": "REROUTE_PASS" in phases_visited,
            "pass_completed": "REROUTE_RETURN" in phases_visited,
            "stuck_events": stuck_counter,
            "final_dist_from_basket": abs(obs["agentPosition"][0] - obs["basketPosition"][0]),
            "trace": history
        }

if __name__ == "__main__":
    import sys
    
    # --- SMOKE TEST MODE ---
    if "--smoke" in sys.argv:
        print("--- SMOKE TEST: Verifying Environment & Config ---")
        try:
            # 1. Check Server
            print("[1/3] Checking Environment Server...", end=" ")
            env_client = WebEnvClient()
            info = env_client.get_info()
            print(f"OK (Task: {info.get('taskName','Unknown')})")
            
            # 2. Check Model
            print(f"[2/3] Checking Model Configuration ({ExperimentConfig.model})...", end=" ")
            if not GEMINI_API_KEY: raise ValueError("GEMINI_API_KEY not found in config.")
            print("OK")
            
            # 3. Running Golden Seed 2005 (Sanity Check)...
            print("[3/3] Running Golden Seed 2005 (Sanity Check)...")
            runner = ExperimentRunner(ExperimentConfig(mode="adaptive", num_seeds=1))
            res = runner.run_seed(2005)
            
            # For a Smoke Test, we only care that the Environment/LLM pipeline ran without crashing.
            # Task failure (WEDGED/POLICY) is an agent issue, not an infra issue.
            print(f"\nEpisode Complete. Status: {res.get('status','UNKNOWN')}")
            print("SMOKE TEST PASSED (Infra Verified).")
            sys.exit(0)
                
        except Exception as e:
            print(f"\nSMOKE TEST FAILED: {str(e)}")
            sys.exit(1)

    print("--- Running Final Validation Sweep (N=100) [PARALLEL] ---")
    
    # Standard Benchmark Range: 2000-2099
    seeds = list(range(2000, 2100))
    
    import concurrent.futures
    import threading

    print_lock = threading.Lock()
    results = []
    counts = {"SUCCESS": 0, "UNSAT_WEDGED": 0, "FAIL_POLICY": 0, "FAIL_INSTABILITY": 0}
    attempts_success = []
    
    def run_single_seed(seed):
        # Create a unique session ID for this seed execution
        session_id = f"seed_{seed}"
        # Runner needs its own instance to have its own WebEnvClient(session_id)
        runner = ExperimentRunner(ExperimentConfig(mode="adaptive", num_seeds=1), session_id=session_id)
        
        with print_lock:
            print(f"[{session_id}] Starting...")
            
        res = runner.run_seed(seed)
        
        # Normalize Status
        status = res.get('status', 'SUCCESS' if res['success'] else 'FAIL_POLICY')
        if status == "FAIL (Policy)": status = "FAIL_POLICY"
        
        # Log to shared structures (thread-safe lock needed for counts/lists?)
        # Lists/Dicts append/update is atomic in Python for single items, but let's lock for output safety
        res['status'] = status
        
        with print_lock:
            print(f"SEED {seed}: {status} ({res.get('attempts',0)} steps)")
            if status not in counts: counts[status] = 0
            counts[status] += 1
            results.append(res)
            if status == "SUCCESS":
                attempts_success.append(res.get('attempts',0))
                
        return res

    # Run in Parallel (Max Workers = 10 for LLM Latency Optimization)
    MAX_WORKERS = 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single_seed, seed): seed for seed in seeds}
        for future in concurrent.futures.as_completed(futures):
            # Just consuming to ensure exceptions are raised if any
            try:
                future.result()
            except Exception as e:
                seed = futures[future]
                print(f"SEED {seed} CRASHED: {e}")

    # Sorting results by seed for clean JSON
    results.sort(key=lambda x: x['seed'])

    print("\n--- SWEEP RESULTS ---")
    for k, v in counts.items():
        print(f"{k}: {v}/{len(seeds)} ({v/len(seeds)*100:.1f}%)")
    
    if attempts_success:
        import statistics
        p50 = statistics.median(attempts_success)
        attempts_success.sort()
        p90_idx = int(len(attempts_success) * 0.9)
        p90 = attempts_success[min(p90_idx, len(attempts_success)-1)]
        print(f"Attempts (Correct): P50={p50}, P90={p90}")
    
    import json
    with open("final_classified_results_N100.json", "w") as f:
        json.dump(results, f, indent=2)

