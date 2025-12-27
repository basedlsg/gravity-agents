import requests
import time

ENV_SERVER_URL = "http://localhost:3002"

def test_steps():
    print("Testing Reset...")
    r = requests.post(f"{ENV_SERVER_URL}/reset", json={"task": "throw", "taskVersion": "v2", "gravity": 9.81, "seed": 2000})
    start_x = r.json()["observation"]["agentPosition"][0]
    print(f"Start X: {start_x}")
    
    print("Testing Step Forward (Fine)...")
    r = requests.post(f"{ENV_SERVER_URL}/step", json={"action": "forward", "durationScale": 0.25})
    # Settle
    r = requests.post(f"{ENV_SERVER_URL}/step", json={"action": "idle", "durationScale": 1.0})
    end_x = r.json()["observation"]["agentPosition"][0]
    
    print(f"End X: {end_x}")
    print(f"Delta: {end_x - start_x}")

if __name__ == "__main__":
    test_steps()
