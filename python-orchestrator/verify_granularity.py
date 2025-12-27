import requests
import numpy as np
import time

ENV_SERVER_URL = "http://localhost:3002"

def measure_displacement(granularity, trials=20):
    displacements = []
    
    # Scale logic matching experiment_v4.py
    scale = 1.0
    if granularity == "medium": scale = 0.5
    if granularity == "fine": scale = 0.25
    
    print(f"\n--- Measuring {granularity} (scale={scale}) ---")
    
    for i in range(trials):
        # Reset
        requests.post(f"{ENV_SERVER_URL}/reset", json={
            "task": "throw", "taskVersion": "v2", 
            "gravity": 9.81, "seed": 1000 + i
        })
        
        # Get initial pos
        obs1 = requests.get(f"{ENV_SERVER_URL}/info").json() # Wait, info doesn't give pos. Need to step or store from reset.
        # Actually reset returns obs.
        resp = requests.post(f"{ENV_SERVER_URL}/reset", json={
             "task": "throw", "taskVersion": "v2", 
            "gravity": 9.81, "seed": 1000 + i
        })
        start_x = resp.json()["observation"]["agentPosition"][0]
        
        # Step Forward
        req_body = {"action": "forward", "durationScale": scale}
        resp = requests.post(f"{ENV_SERVER_URL}/step", json=req_body)
        end_x = resp.json()["observation"]["agentPosition"][0]
        
        delta = end_x - start_x
        displacements.append(delta)
        # print(f"Trial {i}: {start_x:.4f} -> {end_x:.4f} (Delta={delta:.4f})")

    arr = np.array(displacements)
    print(f"Stats for {granularity}:")
    print(f"  Mean: {np.mean(arr):.4f}")
    print(f"  Std : {np.std(arr):.6f}")
    print(f"  Min : {np.min(arr):.4f}")
    print(f"  Max : {np.max(arr):.4f}")
    return np.mean(arr)

if __name__ == "__main__":
    measure_displacement("coarse")
    measure_displacement("medium")
    measure_displacement("fine")
