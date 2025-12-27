"""Trace what the heuristic does."""
import requests
from env_config_v2 import get_heuristic_action, reset_heuristic_state

ENV_URL = "http://localhost:3002"

# Reset
r = requests.post(f"{ENV_URL}/reset", json={"task": "gap", "taskVersion": "v2", "gravity": 9.81})
obs = r.json()["observation"]
reset_heuristic_state()

print("=== Heuristic Trace ===")
print(f"Gap: x={obs['gapStart']:.2f} to x={obs['gapEnd']:.2f}")
print()

for step in range(30):
    x, y, z = obs['agentPosition']
    vx, vy, vz = obs['agentVelocity']
    grounded = obs['isGrounded']
    gap_start = obs['gapStart']

    action = get_heuristic_action(obs)
    print(f"Step {step+1}: x={x:.2f}, grounded={grounded}, action={action}")

    r = requests.post(f"{ENV_URL}/step", json={"action": action})
    obs = r.json()["observation"]
    done = r.json().get("done", False)

    if done:
        info = r.json().get("info", {})
        print(f"\nDONE: {info.get('reason')} (success={info.get('success')})")
        break
