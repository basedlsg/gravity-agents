"""Debug physics - check velocity and jump behavior"""
import requests

ENV_URL = "http://localhost:3002"

# Reset
r = requests.post(f"{ENV_URL}/reset", json={"task": "gap", "taskVersion": "v2", "gravity": 9.81})
obs = r.json()["observation"]

print("=== Gap Crossing Physics Test ===")
print(f"Gap: x={obs['gapStart']:.1f} to x={obs['gapEnd']:.1f}")
print(f"Goal zone: x={obs['goalZone']['minX']:.1f} to {obs['goalZone']['maxX']:.1f}")
print(f"           z={obs['goalZone']['minZ']:.1f} to {obs['goalZone']['maxZ']:.1f}")
print()

print("Simulating optimal run: 5 forward -> jump -> forward until done")
for i in range(5):
    r = requests.post(f"{ENV_URL}/step", json={"action": "forward"})
    obs = r.json()["observation"]

r = requests.post(f"{ENV_URL}/step", json={"action": "jump"})
obs = r.json()["observation"]
print(f"After jump: x={obs['agentPosition'][0]:.2f}, y={obs['agentPosition'][1]:.2f}, z={obs['agentPosition'][2]:.2f}")

for i in range(30):
    r = requests.post(f"{ENV_URL}/step", json={"action": "forward"})
    obs = r.json()["observation"]
    result = r.json()
    done = result.get("done", False)
    success = result.get("info", {}).get("success", False)
    reason = result.get("info", {}).get("reason", "ongoing")

    x, y, z = obs['agentPosition']
    print(f"Step {i+1}: x={x:.2f}, y={y:.2f}, z={z:.2f}, grounded={obs['isGrounded']}")

    if done:
        print(f"\n=== Episode ended: {reason} (success={success}) ===")
        break
