"""Trace exactly what happens during a jump."""
import requests

ENV_URL = "http://localhost:3002"

# Reset
r = requests.post(f"{ENV_URL}/reset", json={"task": "gap", "taskVersion": "v2", "gravity": 9.81})
obs = r.json()["observation"]

print("=== Jump Trace ===")
print(f"Gap: x={obs['gapStart']:.2f} to x={obs['gapEnd']:.2f} (width={obs['gapWidth']:.2f}m)")
print(f"Goal: x={obs['goalZone']['minX']:.2f} to x={obs['goalZone']['maxX']:.2f}")
print()

# Build momentum - 6 forwards to get closer to edge
print("Building momentum (6 forward):")
for i in range(6):
    r = requests.post(f"{ENV_URL}/step", json={"action": "forward"})
    obs = r.json()["observation"]
    x, y, z = obs['agentPosition']
    vx, vy, vz = obs['agentVelocity']
    print(f"  Step {i+1}: x={x:.2f}, vx={vx:.2f}")

# Jump
print("\nJumping:")
r = requests.post(f"{ENV_URL}/step", json={"action": "jump"})
obs = r.json()["observation"]
x_jump, y_jump, _ = obs['agentPosition']
vx_jump, vy_jump, _ = obs['agentVelocity']
print(f"  At jump: x={x_jump:.2f}, y={y_jump:.2f}, vx={vx_jump:.2f}, vy={vy_jump:.2f}")

# Track flight
print("\nIn flight (forward):")
max_y = y_jump
landed = False
for i in range(30):
    r = requests.post(f"{ENV_URL}/step", json={"action": "forward"})
    obs = r.json()["observation"]
    x, y, z = obs['agentPosition']
    vx, vy, vz = obs['agentVelocity']
    grounded = obs['isGrounded']
    done = r.json().get("done", False)

    if y > max_y:
        max_y = y

    status = "LANDED" if grounded else "flying"
    print(f"  Step {i+1}: x={x:.2f}, y={y:.2f}, vx={vx:.2f}, vy={vy:.2f} [{status}]")

    if done:
        info = r.json().get("info", {})
        print(f"\n  DONE: {info.get('reason')} (success={info.get('success')})")
        break

    if grounded and not landed:
        landed = True
        x_land = x
        print(f"\n  *** Landed at x={x_land:.2f} ***")
        print(f"  Jump distance: {x_land - x_jump:.2f}m")
        print(f"  Max height: {max_y:.2f}m")
