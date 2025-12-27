import requests
import time

ENV_SERVER_URL = "http://localhost:3002"

def debug_seed_2000():
    print("--- Debugging Seed 2000 Calibration ---")
    
    # 1. Reset
    r = requests.post(f"{ENV_SERVER_URL}/reset", json={"task": "throw", "taskVersion": "v2", "gravity": 9.81, "seed": 2000}, timeout=10)
    obs = r.json().get("observation", {})
    start_x = obs["agentPosition"][0]
    block_x = obs["blockPosition"][0]
    print(f"Start Agent X: {start_x:.4f}")
    print(f"Start Block X: {block_x:.4f}")
    
    current_x = start_x
    
    # 2. Forward Loop
    print("\n--- Forward Sequence (10 steps) ---")
    for i in range(10):
        r = requests.post(f"{ENV_SERVER_URL}/step", json={"action": "forward", "durationScale": 0.25})
        obs = r.json().get("observation", {})
        new_x = obs["agentPosition"][0]
        block_pos = obs["blockPosition"][0]
        delta = new_x - current_x
        print(f"Step {i+1}: X={new_x:.4f} (Delta={delta:.4f}). Block X={block_pos:.4f}")
        current_x = new_x
        
    # 3. Settle
    r = requests.post(f"{ENV_SERVER_URL}/step", json={"action": "idle", "durationScale": 1.0})
    obs = r.json().get("observation", {})
    settle_x = obs["agentPosition"][0]
    print(f"\nAfter Settle: X={settle_x:.4f}")
    
    # 4. Backward Loop
    print("\n--- Backward Sequence (10 steps) ---")
    current_x = settle_x
    for i in range(10):
        r = requests.post(f"{ENV_SERVER_URL}/step", json={"action": "back", "durationScale": 0.25})
        obs = r.json().get("observation", {})
        new_x = obs["agentPosition"][0]
        delta = new_x - current_x
        # print(f"Step {i+1}: X={new_x:.4f} (Delta={delta:.4f})")
        current_x = new_x

    print("\n--- Phase 1: Long Move (52 steps forward) ---")
    start_run_x = current_x
    for i in range(52):
        r = requests.post(f"{ENV_SERVER_URL}/step", json={"action": "forward", "durationScale": 0.25})
        obs = r.json().get("observation", {})
        new_x = obs["agentPosition"][0]
        delta = new_x - current_x
        current_x = new_x
        
        # Log every 5 steps or if delta is tiny
        if i % 5 == 0 or abs(delta) < 0.01:
            print(f"Exec Step {i+1}: X={new_x:.4f} (Delta={delta:.4f})")
            
    print(f"Total Run Dist: {total_run_dist:.4f}m / 52 steps")
    print(f"Average Gain: {total_run_dist/52:.4f}")
    
if __name__ == "__main__":
    debug_seed_2000()
