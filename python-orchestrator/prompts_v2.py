"""
Prompts V2 - Balanced Factorial Design
======================================
Clean 2x2 design: Representation (Formula/Story) × Guidance (Neutral/Guided)

Key principle: Structural parallelism
- Formula and Story prompts have identical structure
- Only the content of the physics block differs
- Guidance hints are structurally identical across representations
"""

# =============================================================================
# FACTOR A: REPRESENTATION
# =============================================================================

# Formula representation - explicit equations
PHYSICS_FORMULA = {
    "training": """
PHYSICS EQUATIONS:
- Gravity: g = 9.81 m/s²
- Jump initial velocity: vᵧ = 4.85 m/s
- Forward velocity: vₓ = 3.0 m/s (max 4.0 m/s with air control)
- Flight time: T = 2 × vᵧ / g = 2 × 4.85 / 9.81 = 0.99 seconds
- Horizontal range: R = vₓ × T = 3.0 × 0.99 = 2.97 meters
- With air control (vₓ → 4.0): R_max ≈ 4.0 meters
""",

    "test": """
PHYSICS EQUATIONS:
- Gravity: g = 4.905 m/s²
- Jump initial velocity: vᵧ = 4.85 m/s
- Forward velocity: vₓ = 3.0 m/s (max 4.0 m/s with air control)
- Flight time: T = 2 × vᵧ / g = 2 × 4.85 / 4.905 = 1.98 seconds
- Horizontal range: R = vₓ × T = 3.0 × 1.98 = 5.94 meters
- With air control (vₓ → 4.0): R_max ≈ 7.9 meters
"""
}

# Throw physics - explicit equations for v3
PHYSICS_THROW_FORMULA = {
    "training": """
PHYSICS EQUATIONS (THROWING):
- Gravity: g = 9.81 m/s²
- Throw Angle: θ = 45° (optimal for range)
- Range Formula: R = (v² × sin(2θ)) / g = v² / 9.81 (since sin(90°) = 1)
- Strengths & Velocities:
  * weak: v = 4.0 m/s  →  Range R ≈ 1.63 meters
  * medium: v = 6.0 m/s  →  Range R ≈ 3.67 meters
  * strong: v = 8.5 m/s  →  Range R ≈ 7.36 meters
""",

    "test": """
PHYSICS EQUATIONS (THROWING):
- Gravity: g = 4.905 m/s² (Moon-like)
- Throw Angle: θ = 45° (optimal for range)
- Range Formula: R = (v² × sin(2θ)) / g = v² / 4.905 (since sin(90°) = 1)
- Strengths & Velocities:
  * weak: v = 4.0 m/s  →  Range R ≈ 3.26 meters
  * medium: v = 6.0 m/s  →  Range R ≈ 7.34 meters
  * strong: v = 8.5 m/s  →  Range R ≈ 14.73 meters
"""
}

# Story representation - intuitive descriptions
PHYSICS_STORY = {
    "training": """
PHYSICS ENVIRONMENT:
- Gravity: Normal Earth-like gravity
- Your character moves at walking speed (~3 m/s)
- Jump height: About 1.2 meters
- Jump distance: With a running start, roughly 3-4 meters
- Air control: You can push forward while airborne
""",

    "test_silent": """
PHYSICS ENVIRONMENT:
- Gravity: Normal Earth-like gravity
- Your character moves at walking speed (~3 m/s)
- Jump height: About 1.2 meters
- Jump distance: With a running start, roughly 3-4 meters
- Air control: You can push forward while airborne
""",

    "test_explained": """
PHYSICS ENVIRONMENT:
- Gravity: Weak, like on a low-gravity moon (roughly half Earth gravity)
- Your character moves at walking speed (~3 m/s)
- Jump height: About 1.2 meters (same as before)
- Jump distance: With a running start, roughly 6-8 meters (longer due to weak gravity)
- Air control: You can push forward while airborne
"""
}

# Throw story - intuitive descriptions for v3
PHYSICS_THROW_STORY = {
    "training": """
PHYSICS ENVIRONMENT (THROWING):
- Gravity: Normal Earth-like gravity
- You have three throw strengths: weak, medium, and strong.
- Weak throw: Reaches roughly 1.5 - 2 meters.
- Medium throw: Reaches roughly 3.5 - 4.5 meters.
- Strong throw: Reaches roughly 7 - 8 meters.
""",

    "test_silent": """
PHYSICS ENVIRONMENT (THROWING):
- Gravity: Normal Earth-like gravity
- You have three throw strengths: weak, medium, and strong.
- Weak throw: Reaches roughly 1.5 - 2 meters.
- Medium throw: Reaches roughly 3.5 - 4.5 meters.
- Strong throw: Reaches roughly 7 - 8 meters.
""",

    "test_explained": """
PHYSICS ENVIRONMENT (THROWING):
- Gravity: Weak, Moon-like gravity.
- Things stay in the air much longer and travel MUCH further.
- Weak throw: Now reaches roughly 3 - 4 meters (used to be 2m).
- Medium throw: Now reaches roughly 7 - 8 meters (used to be 4m).
- Strong throw: Now reaches roughly 14 - 15 meters (used to be 7.5m).
"""
}

# =============================================================================
# FACTOR B: GUIDANCE
# =============================================================================

# Neutral guidance - no behavioral hints
GUIDANCE_NEUTRAL = {
    "training": "",
    "test": ""
}

# Strategy-hinted guidance - parallel structure for both representations
GUIDANCE_HINTED = {
    "formula": {
        "training": "",
        "test": """
ADAPTATION NOTE:
Because flight time is longer than before (1.98s vs 0.99s), the timing window
for a successful jump is more forgiving. Jumping slightly later than in the
higher-gravity case can still succeed.
"""
    },
    "story": {
        "training": "",
        "test": """
ADAPTATION NOTE:
Because gravity is weaker than before, you stay in the air longer. The timing
window for a successful jump is more forgiving. Jumping slightly later than
in the normal-gravity case can still succeed.
"""
    }
}

# =============================================================================
# TASK DESCRIPTIONS
# =============================================================================

# Shared task structure - only geometry differs
TASK_TEMPLATE = """
TASK: Jump across a gap between two platforms.

GEOMETRY:
- Start position: x = {start_x:.2f}m
- Gap: x = {gap_start:.2f}m to x = {gap_end:.2f}m (width = {gap_width:.2f}m)
- Landing zone: x = {land_start:.2f}m to x = {land_end:.2f}m (width = {land_width:.2f}m)

STEP-TO-DISTANCE MAPPING:
- Each "forward" action moves you approximately 0.5 meters
- Distance to gap edge: {dist_to_gap:.2f}m
- Steps to reach gap edge: approximately {steps_to_gap} forward actions

OBJECTIVE: Land anywhere on the landing zone.

AVAILABLE ACTIONS: forward, jump, idle, back
"""

# Throw task structure
TASK_THROW_TEMPLATE = """
TASK: Pick up a block and throw it into a basket.

GEOMETRY:
- Agent start position: x = {agent_x:.2f}m
- Block start position: x = {block_x:.2f}m
- Basket position: x = {basket_x:.2f}m, y = {basket_y:.2f}m (height)
- Current distance agent to block: {dist_to_block:.2f}m
- Current distance agent to basket: {dist_to_basket:.2f}m

THROW OPTIONS: weak, medium, strong

OBJECTIVE: Get the block into the basket.

AVAILABLE ACTIONS: forward, back, left, right, pick, throw_weak, throw_medium, throw_strong, idle
"""

# =============================================================================
# TWO-STAGE PLANNING INTERFACE
# =============================================================================

# Stage 1: Planning (physics reasoning → jump step prediction)
STAGE1_PLANNING_PROMPT = """
You are planning a gap-crossing maneuver for a simulated character.

{physics_block}

{guidance_block}

{task_block}

PLANNING TASK:
Based on the physics and geometry, determine the optimal step at which to jump.
Consider:
1. How many forward steps to build momentum?
2. When should you jump relative to the gap edge?
3. Will you clear the gap and land on the target?

Respond with a JSON object:
{{
  "physics_reasoning": "<your calculations and reasoning>",
  "jump_step": <integer: the step number at which to execute jump>,
  "confidence": "<high/medium/low>",
  "notes": "<any additional considerations>"
}}
"""

# Stage 1: Planning for v3 (physics reasoning → throw strength prediction)
STAGE1_THROW_PLANNING_PROMPT = """
You are planning a block-throwing task for a simulated character.

{physics_block}

{guidance_block}

{task_block}

PLANNING TASK:
Based on the physics and geometry, determine:
1. How many steps should you move forward (if any) to get into the optimal range?
2. Which throw strength should you use once you have picked up the block?

Respond with a JSON object:
{{
  "physics_reasoning": "<your calculations and reasoning>",
  "move_steps": <integer: forward steps to take before picking/throwing>,
  "throw_strength": "<weak/medium/strong>",
  "confidence": "<high/medium/low>",
  "notes": "<any additional considerations>"
}}
"""

# Stage 2: Feedback (Visual Encoder Output → Correction Request)
STAGE2_FEEDBACK_PROMPT = """
You just attempted the task, but it FAILED.

PREVIOUS PLAN:
- Move Steps: {moved_steps}
- Throw Strength: {throw_strength}

RESULT:
- Target Distance (Basket): {target_dist:.2f}m
- Actual Throw Distance: {actual_dist:.2f}m
- Error: {error_desc} ({error_val:.2f}m)

PHYSICS ANALYSIS:
{physics_hint}

ADJUSTMENT TASK:
Based on the error, calculate the necessary adjustment.
1. If you overshot (+), you threw too far. You must either throw weaker OR move backward (increase distance).
2. If you undershot (-), you threw too short. You must either throw stronger OR move forward (decrease distance).
3. HINT: Changing throw strength is coarse (large jumps). Moving is fine (small steps).

Respond with a JSON object for your NEXT attempt:
{{
  "error_analysis": "<explain why the previous attempt failed>",
  "adjustment_strategy": "<explain your math for the fix>",
  "move_steps": <integer: NEW total forward steps from start (NOT additive)>
  "throw_strength": "<weak/medium/strong>",
  "confidence": "<high/medium/low>"
}}
"""

# Helper to generate the semantic hint
def get_feedback_hint(representation: str, error_val: float) -> str:
    if representation == "formula":
        return f"""The Range Formula is R = v²/g.
You missed by {error_val:.2f}m.
To fix this, calculate the position shift Δx needed such that the new distance matches your fixed throw range."""
    else:
        return f"""You missed by {error_val:.2f}m.
If you overshot, you are too close. Back up!
If you undershot, you are too far. Get closer!"""

# Stage 2: Execution (jump_step → action sequence)
# Note: This can be deterministic (no LLM) or LLM-based
def compile_sequence(jump_step: int, total_steps: int = 25) -> list[str]:
    """
    Deterministic sequence compiler.
    Given a jump_step, produce the action sequence.
    """
    sequence = []
    for i in range(total_steps):
        if i < jump_step:
            sequence.append("forward")
        elif i == jump_step:
            sequence.append("jump")
        else:
            sequence.append("forward")  # Air control / continue
    return sequence


# =============================================================================
# TASK GEOMETRIES
# =============================================================================

# Task A: Invariant - optimal jump timing same at 1g and 0.5g
# UPDATED based on physics sweep:
# - At 1g: jump@6 lands at ~9.3m
# - At 0.5g: jump@6 lands at ~17.2m (massive overshoot due to runUpBonus)
# To make this truly invariant, we need a VERY wide landing zone: 7-20m
TASK_A_GEOMETRY = {
    "name": "invariant",
    "description": "Very wide landing zone - adaptation not required",
    "start_x": -1.33,
    "gap_start": 2.0,
    "gap_end": 6.5,
    "gap_width": 4.5,
    "land_start": 7.0,
    "land_end": 20.0,  # Extended to accommodate 0.5g overshoot
    "land_width": 13.0,
    # At 1g: jump@6 → land ~9.3m ✓
    # At 0.5g: jump@6 → land ~17.2m ✓ (still in zone!)
    "optimal_jump_1g": 6,
    "optimal_jump_0p5g": 6,  # Now actually same!
}

# Task B: Adaptive - must adjust timing when g changes
# UPDATED based on physics sweep:
# - At 1g: jump@6 lands at ~9.3m
# - At 0.5g: jump@3 lands at ~9.7m (the ONLY success at 0.5g!)
# - At 0.5g: jump@6 lands at ~17.2m (massive overshoot)
# Landing zone 8-11m allows jump@6 at 1g but NOT at 0.5g
TASK_B_GEOMETRY = {
    "name": "adaptive",
    "description": "Medium landing zone - requires adaptation at 0.5g",
    "start_x": -1.33,
    "gap_start": 2.0,
    "gap_end": 6.5,
    "gap_width": 4.5,
    "land_start": 8.0,   # Shifted to exclude 0.5g jump@3 (lands at 9.7)
    "land_end": 11.0,    # Excludes 0.5g jump@6 (lands at 17.2)
    "land_width": 3.0,
    # At 1g: jump@6 → land ~9.3m ✓ (in 8-11 zone)
    # At 0.5g: jump@6 → land ~17.2m ✗ (overshoots!)
    # At 0.5g: jump@3 → land ~9.7m ✓ (in 8-11 zone)
    "optimal_jump_1g": 6,
    "optimal_jump_0p5g": 3,  # Must jump MUCH earlier!
}


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_planning_prompt(
    representation: str,  # "formula" or "story"
    guidance: str,        # "neutral" or "guided"
    gravity: str,         # "training" or "test"
    task_geometry: dict,
    gravity_explained: bool = True  # For story, whether to explain the change
) -> str:
    """
    Build a complete Stage 1 planning prompt with balanced factors.
    """
    # Select physics block
    if representation == "formula":
        if gravity == "training":
            physics_block = PHYSICS_FORMULA["training"]
        else:
            physics_block = PHYSICS_FORMULA["test"]
    else:  # story
        if gravity == "training":
            physics_block = PHYSICS_STORY["training"]
        elif gravity_explained:
            physics_block = PHYSICS_STORY["test_explained"]
        else:
            physics_block = PHYSICS_STORY["test_silent"]

    # Select guidance block
    if guidance == "neutral":
        guidance_block = ""
    else:  # guided
        if representation == "formula":
            guidance_block = GUIDANCE_HINTED["formula"].get(gravity, "")
        else:
            guidance_block = GUIDANCE_HINTED["story"].get(gravity, "")

    # Build task block
    g = task_geometry
    dist_to_gap = g["gap_start"] - g["start_x"]
    steps_to_gap = int(dist_to_gap / 0.5)

    task_block = TASK_TEMPLATE.format(
        start_x=g["start_x"],
        gap_start=g["gap_start"],
        gap_end=g["gap_end"],
        gap_width=g["gap_width"],
        land_start=g["land_start"],
        land_end=g["land_end"],
        land_width=g["land_width"],
        dist_to_gap=dist_to_gap,
        steps_to_gap=steps_to_gap
    )

    # Assemble full prompt
    return STAGE1_PLANNING_PROMPT.format(
        physics_block=physics_block.strip(),
        guidance_block=guidance_block.strip(),
        task_block=task_block.strip()
    )


def build_throw_planning_prompt(
    representation: str,
    guidance: str,
    gravity: str,
    obs: dict,
    gravity_explained: bool = True
) -> str:
    """
    Build a v3 planning prompt for the throw task.
    """
    # Select physics block
    if representation == "formula":
        if gravity == "training":
            physics_block = PHYSICS_THROW_FORMULA["training"]
        else:
            physics_block = PHYSICS_THROW_FORMULA["test"]
    else:  # story
        if gravity == "training":
            physics_block = PHYSICS_THROW_STORY["training"]
        elif gravity_explained:
            physics_block = PHYSICS_THROW_STORY["test_explained"]
        else:
            physics_block = PHYSICS_THROW_STORY["test_silent"]

    # Select guidance block (stubs for now)
    guidance_block = ""

    # Build task block
    ax = obs["agentPosition"][0]
    bx = obs["blockPosition"][0]
    kx = obs["basketPosition"][0]
    ky = obs["basketPosition"][1]

    task_block = TASK_THROW_TEMPLATE.format(
        agent_x=ax,
        block_x=bx,
        basket_x=kx,
        basket_y=ky,
        dist_to_block=abs(bx - ax),
        dist_to_basket=abs(kx - ax)
    )

    return STAGE1_THROW_PLANNING_PROMPT.format(
        physics_block=physics_block.strip(),
        guidance_block=guidance_block,
        task_block=task_block.strip()
    )


# =============================================================================
# EXAMPLE PROMPTS FOR VERIFICATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EXAMPLE: Formula + Neutral + Training + Task A")
    print("=" * 70)
    print(build_planning_prompt("formula", "neutral", "training", TASK_A_GEOMETRY))

    print("\n" + "=" * 70)
    print("EXAMPLE: Story + Guided + Test + Task B")
    print("=" * 70)
    print(build_planning_prompt("story", "guided", "test", TASK_B_GEOMETRY))
