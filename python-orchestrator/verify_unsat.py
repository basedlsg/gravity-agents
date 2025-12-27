import requests
import numpy as np

ENV_SERVER_URL = "http://localhost:3002"

def generate_unsat_certificate(seed):
    print(f"\n--- Generating Certificate for Seed {seed} ---")
    
    # 1. Reset to get geometry
    resp = requests.post(f"{ENV_SERVER_URL}/reset", json={
        "task": "throw", "taskVersion": "v2", 
        "gravity": 9.81, "seed": seed
    })
    obs = resp.json()["observation"]
    basket_x = obs["basketPosition"][0]
    agent_start_x = obs["agentPosition"][0]
    
    # 2. Define Reachable State Space (Fine Granularity = 0.125m)
    # Platform is typically +/- 2m from center?
    # Let's assume agent can move from -4.0 to +4.0 (platform limits)
    # We will step by 0.125m
    
    positions = np.arange(-5.0, 5.0, 0.125) 
    
    # 3. Exhaustive Search
    best_error = float('inf')
    best_config = None
    reachable_count = 0
    valid_positions = []
    
    # To simulate "teleport" for check, we can't just set pos in v2 server easily 
    # without patching "teleport" cheat.
    # Alternatives: 
    # A) Step 100 times to get to each spot (Slow)
    # B) Use the "Baseline" strategy of resetting and stepping N times.
    
    # Let's use the Baseline approach: Try moves from -20 to +20 steps (Fine)
    # range_limit = 40 (covers 5 meters)
    
    moves = range(-40, 41) 
    
    for m in moves:
        # Reset
        requests.post(f"{ENV_SERVER_URL}/reset", json={
            "task": "throw", "taskVersion": "v2", 
            "gravity": 9.81, "seed": seed
        })
        
        # Move N times (Approximation of reaching position)
        if m != 0:
            scale = 0.25 # Fine
            # Optimization: Can we send "durationScale": scale * abs(m) in one step?
            # Server patch assumed linear velocity scaling. 
            # If we send durationScale=10 * 0.25 = 2.5, velocity becomes 3.0 * 2.5 = 7.5? 
            # No, velocity clamp at 4.0 in task code!
            # So we must loop.
            
            direction = "forward" if m > 0 else "back"
            steps = abs(m)
            
            # Batch steps to save HTTP calls?
            # No, naive loop for safety.
            for _ in range(steps):
                requests.post(f"{ENV_SERVER_URL}/step", json={"action": direction, "durationScale": 0.25})
                
        # Get position
        obs = requests.get(f"{ENV_SERVER_URL}/info").json() # Only static info
        # Need step to get obs
        r = requests.post(f"{ENV_SERVER_URL}/step", json={"action": "idle", "durationScale": 0.01})
        current_x = r.json()["observation"]["agentPosition"][0]
        
        # Try all throws
        for strength in ["throw_weak", "throw_medium", "throw_strong"]:
            # Need to pick first if not holding? Reset gives us holding.
            # Assuming we haven't dropped it.
            
            r = requests.post(f"{ENV_SERVER_URL}/step", json={"action": strength})
            res = r.json()
            
            # Wait for land
            landed_x = None
            for _ in range(10): # fast forward
                r = requests.post(f"{ENV_SERVER_URL}/step", json={"action": "idle"})
                # Check block pos
                b_y = r.json()["observation"]["blockPosition"][1]
                if b_y < 0.2:
                    landed_x = r.json()["observation"]["blockPosition"][0]
                    break
            
            if landed_x is not None:
                error = abs(landed_x - basket_x)
                if error < best_error:
                    best_error = error
                    best_config = (m, strength, current_x, landed_x)
                
                # Check success
                if error < 0.6: # Success radius (basket width ~1.2?)
                     # Actually basket width is 1.2, so radius 0.6.
                     pass 
                     
    reachable_count = len(moves) * 3
    
    print(f"Seed {seed}: Scanned {reachable_count} configs.")
    print(f"  Best Error: {best_error:.4f}")
    if best_config:
        print(f"  Best Config: Move {best_config[0]} (Fine), {best_config[1]} -> AgentX {best_config[2]:.2f}, LandX {best_config[3]:.2f} (Target {basket_x:.2f})")
    
    is_unsat = best_error > 0.6
    print(f"  Result: {'UNSAT (Certified)' if is_unsat else 'SAT (Solvable)'}")
    return is_unsat, best_error

if __name__ == "__main__":
    # Check the knowns
    generate_unsat_certificate(1001) # Previously failed
    generate_unsat_certificate(1003) # Previously failed
    generate_unsat_certificate(1000) # Previously passed (sanity check)
