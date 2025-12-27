import requests
from dataclasses import dataclass

ENV_SERVER_URL = "http://localhost:3002"

class WebEnvClient:
    def reset(self, seed: int) -> dict:
        response = requests.post(f"{ENV_SERVER_URL}/reset", json={"task": "throw", "taskVersion": "v2", "gravity": 9.81, "seed": seed}, timeout=10)
        response.raise_for_status()
        return response.json().get("observation", {})

    def step(self, action: str, granularity: str = "coarse") -> dict:
        scale = {"coarse": 1.0, "medium": 0.5, "fine": 0.25}[granularity]
        req = {"action": action}
        if action in ["forward", "back", "left", "right"]: req["durationScale"] = scale
        
        response = requests.post(f"{ENV_SERVER_URL}/step", json=req)
        response.raise_for_status()
        return response.json().get("observation", {})

    def get_info(self) -> dict:
        response = requests.get(f"{ENV_SERVER_URL}/info")
        response.raise_for_status()
        return response.json()

def test_class():
    env = WebEnvClient()
    obs = env.reset(2000)
    start_x = obs["agentPosition"][0]
    print(f"Start X: {start_x}")
    
    for i in range(10):
        env.step("forward", "fine")
    
    # Settle
    env.step("idle")
    
    info = env.get_info()
    print(f"Info dump: {info}")
    mid_x = info.get("observation", {}).get("agentPosition", [0])[0]
    print(f"Mid X: {mid_x}")
    print(f"Delta: {mid_x - start_x}")

if __name__ == "__main__":
    test_class()
