import requests
import time

ENV_SERVER_URL = "http://localhost:3002"

def verify_detour():
    print("--- Verifying 2D Detour Capability (Seed 2000) ---")
    
    # 1. Reset
    r = requests.post(f"{ENV_SERVER_URL}/reset", json={"task": "throw", "taskVersion": "v2", "gravity": 9.81, "seed": 2000}, timeout=10)
    obs = r.json().get("observation", {})
    start_x = obs["agentPosition"][0]
    basket_x = obs["basketPosition"][0]
    print(f"Start: Agent X={start_x:.4f}, Z={obs['agentPosition'][2]:.4f}")
    print(f"Target: Basket X={basket_x:.4f}, Z={obs['basketPosition'][2]:.4f}")
    
    # 2. Move Forward to X=2.0 (Before collision at 2.97)
    print("\n--- Phase 1: Approach (Forward 30 steps) ---")
    current_x = start_x
    for i in range(30):
        r = requests.post(f"{ENV_SERVER_URL}/step", json={"action": "forward", "durationScale": 0.25})
        obs = r.json().get("observation", {})
        delta = obs["agentPosition"][0] - current_x
        current_x = obs["agentPosition"][0]
        # if i % 10 == 0: print(f"Step {i}: X={current_x:.4f}")
    
    print(f"At Staging Point: X={current_x:.4f}, Z={obs['agentPosition'][2]:.4f}")
    
    # 3. Lateral Move (Left/Right)
    # Move 'left' (negative Z?)
    print("\n--- Phase 2: Detour LEFT (20 steps) ---")
    start_z = obs['agentPosition'][2]
    for i in range(20):
        r = requests.post(f"{ENV_SERVER_URL}/step", json={"action": "left", "durationScale": 0.25})
        obs = r.json().get("observation", {})
    
    new_z = obs['agentPosition'][2]
    print(f"After Lateral: X={obs['agentPosition'][0]:.4f}, Z={new_z:.4f}")
    print(f"Lateral Shift: {new_z - start_z:.4f}m")
    
    # 4. Pass the Obstacle
    print("\n--- Phase 3: Pass (Forward 30 steps) ---")
    start_pass_x = obs['agentPosition'][0]
    for i in range(30):
        r = requests.post(f"{ENV_SERVER_URL}/step", json={"action": "forward", "durationScale": 0.25})
        obs = r.json().get("observation", {})
        delta = obs["agentPosition"][0] - start_pass_x
    
    end_x = obs["agentPosition"][0]
    print(f"Final Pos: X={end_x:.4f}, Z={obs['agentPosition'][2]:.4f}")
    
    # Check if we passed the wall (X > 3.5)
    wall_x_est = basket_x - 0.75 # Approx
    if end_x > wall_x_est:
        print(f"SUCCESS: Passed the wall (X > {wall_x_est:.2f})!")
    else:
        print(f"FAILURE: Getting stuck at X={end_x:.4f} (Wall ~{wall_x_est:.2f})")

if __name__ == "__main__":
    verify_detour()
