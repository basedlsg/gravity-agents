"""Analyze the physics to understand the difficulty margin."""
import math

# Current parameters
g = 9.81  # gravity
jump_height = 1.2  # meters
move_speed = 3.0  # m/s
run_up_bonus = 1.3  # 30% bonus

# Calculate jump velocity
# v = sqrt(2 * g * h)
base_jump_vel = math.sqrt(2 * g * jump_height)
bonus_jump_vel = base_jump_vel * run_up_bonus

print("=== Physics Analysis ===\n")
print(f"Gravity: {g} m/s²")
print(f"Jump height: {jump_height} m")
print(f"Move speed: {move_speed} m/s")
print(f"Run-up bonus: {run_up_bonus}x")

print(f"\nBase jump velocity: {base_jump_vel:.2f} m/s")
print(f"Bonus jump velocity: {bonus_jump_vel:.2f} m/s")

# Time in air
# Total flight time = 2 * v_y / g
base_flight_time = 2 * base_jump_vel / g
bonus_flight_time = 2 * bonus_jump_vel / g

print(f"\nBase flight time: {base_flight_time:.2f} s")
print(f"Bonus flight time: {bonus_flight_time:.2f} s")

# Horizontal distance during jump
# Assuming constant vx = move_speed during flight
base_distance = move_speed * base_flight_time
bonus_distance = move_speed * bonus_flight_time

print(f"\nBase jump distance (vx={move_speed}): {base_distance:.2f} m")
print(f"Bonus jump distance (vx={move_speed}): {bonus_distance:.2f} m")

# Gap analysis
gap_width = 4.2  # average
print(f"\n=== Gap Analysis ===")
print(f"Gap width: {gap_width} m")
print(f"Base margin: {base_distance - gap_width:.2f} m")
print(f"Bonus margin: {bonus_distance - gap_width:.2f} m")

if bonus_distance > gap_width:
    print(f"\n✓ Gap is crossable with {((bonus_distance / gap_width) - 1) * 100:.1f}% margin")
else:
    print(f"\n✗ Gap is NOT crossable (need {gap_width - bonus_distance:.2f}m more)")

# What gap would be challenging?
print(f"\n=== Recommended Gap Sizes ===")
print(f"For ~50% success rate, gap should be close to max jump distance")
print(f"  Base max: {base_distance:.2f} m")
print(f"  Bonus max: {bonus_distance:.2f} m")
print(f"\nSuggested gaps:")
print(f"  Easy (100%): < {base_distance * 0.9:.1f} m")
print(f"  Medium (70%): ~ {base_distance:.1f} m")
print(f"  Hard (40%): ~ {base_distance * 1.1:.1f} m to {bonus_distance * 0.95:.1f} m")
print(f"  Very hard (10%): > {bonus_distance:.1f} m")
