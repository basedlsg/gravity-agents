# Gravity Agents Experiment v2 Specification

## Executive Summary

The v1 experiment had fundamental methodological flaws that invalidated the results:
1. Gap task was trivially easy (any jump clears it, forward-only works at 0.5g)
2. No actual RL training occurred (all agents were pure LLM policies)
3. Prompts used prose instead of structured format
4. Baseline calibration was never performed

This document provides specifications to fix all issues.

---

## Part 1: Calibrated Environment Specification

### Gap Crossing Task v2

**Current Problem:** Gap is 3m but physics allows 5.1m horizontal travel. Task is trivially easy.

**Calibrated Parameters:**

```javascript
// GapCrossingTask v2 parameters
const GAP_PARAMS = {
  // Gap configuration - HARDER
  gapWidth: 4.5,          // Was 3.0 - now requires precise timing
  gapVariance: 0.3,       // Was 0.5 - less variance for consistency

  // Platform configuration
  platformWidth: 4.0,     // Was 5.0 - less runway
  platformDepth: 3.0,     // Was 5.0 - narrower to punish lateral drift

  // Agent physics - NERFED
  agentMass: 80,          // Was 70
  moveSpeed: 3.0,         // Was 4.0 - slower horizontal
  maxSpeed: 4.0,          // Was 5.0

  // Jump physics - CRITICAL CHANGES
  jumpHeight: 1.2,        // Was 2.0 - lower jump
  jumpCooldown: 500,      // ms - prevent bunny hopping

  // Anti-cheese measures
  requireRunUp: true,     // Must be moving forward to jump effectively
  runUpBonus: 1.3,        // Jump is 30% further with forward momentum
  airControl: 0.3,        // Was implicit 1.0 - reduced air steering
};

// Physics calculations at these values:
// At 1g (9.81): jumpVel = sqrt(2*9.81*1.2) = 4.85 m/s
//   Time in air = 2 * 4.85 / 9.81 = 0.99s
//   Horizontal = 3.0 * 0.99 = 2.97m (WITHOUT run-up)
//   With run-up = 2.97 * 1.3 = 3.86m (barely clears 4.5m gap with precision)
//
// At 0.5g (4.9): jumpVel = sqrt(2*4.9*1.2) = 3.43 m/s
//   Time in air = 2 * 3.43 / 4.9 = 1.40s
//   Horizontal = 3.0 * 1.40 = 4.2m (WITHOUT run-up)
//   With run-up = 4.2 * 1.3 = 5.46m (easy clear)
//
// This creates DIFFERENTIAL DIFFICULTY that tests adaptation!
```

**Calibration Targets:**
- NRL-F at 1g baseline: 20-40% success rate
- RL agents at 1g after training: 70-85% success rate
- Adaptation gap (0.5g silent vs explained): should differ by 15-25%

### Throw Block Task v2

**Current Problem:** Task is too hard (max 10% success), provides no signal.

**Calibrated Parameters:**

```javascript
const THROW_PARAMS = {
  // Basket configuration - EASIER
  basketDistance: 4.0,        // Was 6.0
  basketDistanceVariance: 0.5,// Was 1.0
  basketWidth: 1.2,           // Was 0.6 (implicit)
  basketHeight: 1.0,          // Lower target

  // Throw physics - SIMPLIFIED
  throwStrengths: {
    weak: 4.0,    // Was 5
    medium: 6.5,  // Was 8
    strong: 9.0   // Was 12
  },
  throwAngle: 45,  // Fixed optimal angle, not variable

  // Block physics
  blockMass: 0.5,  // Lighter for easier throws
  blockSize: 0.3,  // Smaller to fit in basket

  // Quality of life
  autoAlign: true,     // Agent faces basket when throwing
  pickupRange: 1.5,    // Was 1.0
};
```

**Calibration Targets:**
- NRL-F at 1g baseline: 30-50% success rate
- RL agents at 1g after training: 75-90% success rate

---

## Part 2: Revised Prompt Template

### Problem with v1 Prompts

v1 used prose-style prompts that:
1. Buried physics info in paragraphs
2. Didn't require structured reasoning
3. Made "forward" the path of least resistance
4. Didn't test if agent actually uses physics formulas

### v2 Structured Prompt Format

```python
SYSTEM_PROMPT_V2 = """You are an AI agent in a physics simulation.

TASK: {task_description}

PHYSICS LAWS IN THIS WORLD:
{physics_laws_block}

You MUST respond in this exact JSON format:
{
  "physics_calculation": "<show your work using the physics laws>",
  "predicted_outcome": "<what will happen if you take this action>",
  "action": "<exactly one action from the list>"
}

AVAILABLE ACTIONS: {action_list}
"""

# Physics law blocks by condition
PHYSICS_BLOCKS = {
    "formula_training": """
Gravity: F = m * g where g = 9.81 m/s²
Projectile motion:
  - Horizontal: x = v_x * t
  - Vertical: y = v_y * t - 0.5 * g * t²
  - Time to peak: t_peak = v_y / g
  - Max height: h = v_y² / (2 * g)
Jump velocity in this world: {jump_velocity:.2f} m/s
Your move speed: {move_speed:.1f} m/s
""",

    "formula_test_silent": """
Gravity: F = m * g where g = 9.81 m/s²  # LIE - actually 4.9
Projectile motion:
  - Horizontal: x = v_x * t
  - Vertical: y = v_y * t - 0.5 * g * t²
  - Time to peak: t_peak = v_y / g
  - Max height: h = v_y² / (2 * g)
Jump velocity in this world: {jump_velocity:.2f} m/s
Your move speed: {move_speed:.1f} m/s
""",

    "formula_test_explained": """
Gravity: F = m * g where g = 4.9 m/s² (CHANGED from 9.81!)
Projectile motion:
  - Horizontal: x = v_x * t
  - Vertical: y = v_y * t - 0.5 * g * t²
  - Time to peak: t_peak = v_y / g
  - Max height: h = v_y² / (2 * g)
Jump velocity in this world: {jump_velocity:.2f} m/s
Your move speed: {move_speed:.1f} m/s

NOTE: With lower gravity, objects stay in the air longer!
  - Old jump distance: ~{old_jump_dist:.1f}m
  - New jump distance: ~{new_jump_dist:.1f}m
""",

    "normal_training": """
Physics: Normal Earth gravity. Objects fall at a standard rate.
Jump height: About {jump_height:.1f} meters
Move speed: About {move_speed:.1f} m/s
""",

    "normal_test_silent": """
Physics: Normal Earth gravity. Objects fall at a standard rate.
Jump height: About {jump_height:.1f} meters
Move speed: About {move_speed:.1f} m/s
""",

    "normal_test_explained": """
Physics: Gravity is now WEAKER than before (like on the Moon).
Objects fall more slowly and stay in the air longer.
Jump height: About {new_jump_height:.1f} meters (was {old_jump_height:.1f}m)
Move speed: About {move_speed:.1f} m/s
"""
}
```

### Observation Format v2

```python
OBSERVATION_TEMPLATE = """
CURRENT STATE (step {step}/{max_steps}):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Position: x={pos_x:.2f}, y={pos_y:.2f}, z={pos_z:.2f}
Velocity: vx={vel_x:.2f}, vy={vel_y:.2f}, vz={vel_z:.2f}
Grounded: {grounded}

TASK TARGET:
Gap: x={gap_start:.2f} to x={gap_end:.2f} (width={gap_width:.2f}m)
Goal: x={goal_x:.2f}

DISTANCE TO GAP EDGE: {dist_to_gap:.2f}m
DISTANCE TO GOAL: {dist_to_goal:.2f}m

Your {action_count} attempts remaining. Choose wisely.

Available actions: {actions}

Respond with JSON only.
"""
```

---

## Part 3: Methodological v2 Plan

### Phase 1: Environment Calibration (Day 1)

1. **Implement v2 physics parameters** in GapCrossingTask.js and ThrowBlockTask.js

2. **Run human baseline tests:**
   - Play each task 20 times yourself
   - Verify gap task requires timing/skill at 1g
   - Verify throw task is achievable but not trivial

3. **Run NRL-F calibration:**
   - NRL-F at 1g, 50 episodes
   - Target: 25-40% success (currently 0%)
   - If too hard: increase jump height slightly
   - If too easy: increase gap width

4. **Document calibration:**
   - Record exact parameters used
   - Record success rates at each step
   - Save as `calibration_log.json`

### Phase 2: Implement Actual RL Training (Days 2-3)

**Current issue:** "RL-F" and "RL-N" don't actually do RL. They're just LLM policies.

**Options for real RL:**

**Option A: Policy Gradient with LLM (Recommended)**
```python
class RLPolicy:
    def __init__(self, base_llm_policy):
        self.base = base_llm_policy
        self.action_log_probs = []
        self.rewards = []

    def select_action(self, obs, gravity_condition):
        # Get LLM's action distribution
        logits = self.base.get_action_logits(obs, gravity_condition)
        probs = softmax(logits)

        # Sample action
        action = np.random.choice(actions, p=probs)

        # Store for training
        self.action_log_probs.append(log(probs[action]))

        return action

    def update_after_episode(self, total_reward):
        # REINFORCE update
        self.rewards.append(total_reward)

        if len(self.rewards) >= self.batch_size:
            # Compute advantage
            # Update LLM prompt weights or fine-tune
            pass
```

**Option B: Prompt Optimization (Simpler)**
```python
class PromptEvolver:
    """Evolve prompt based on episode success"""

    def __init__(self):
        self.prompt_templates = [initial_prompt]
        self.scores = []

    def mutate_prompt(self, prompt, success_rate):
        # Add successful examples
        # Remove unhelpful phrases
        # Adjust emphasis on physics
        pass

    def select_best_prompts(self):
        # Keep top 3 prompts
        # Combine successful elements
        pass
```

**Option C: State-Action Value Cache (Hybrid)**
```python
class ValueCachedPolicy:
    """Cache successful action patterns"""

    def __init__(self):
        self.state_action_values = {}  # state_hash -> action -> avg_reward

    def select_action(self, obs, gravity_condition):
        state_hash = self.hash_state(obs)

        if state_hash in self.state_action_values:
            # Epsilon-greedy using cached values
            if random() < self.epsilon:
                return self.llm_action(obs)
            else:
                return max(self.state_action_values[state_hash], key=lambda a: values[a])
        else:
            return self.llm_action(obs)

    def update(self, trajectory):
        for state, action, reward in trajectory:
            # Update running average
            pass
```

### Phase 3: Full Experiment (Days 4-6)

1. **Training Phase:**
   - Each RL agent: 200 training episodes at 1g
   - Log all trajectories
   - Save checkpoints every 50 episodes

2. **Evaluation Phase:**
   - All 3 agents × 2 tasks × 3 conditions
   - 100 eval episodes each
   - Fixed seeds for reproducibility

3. **Data Collection:**
   - Full trajectory logging (every step)
   - LLM response logging (full JSON)
   - Physics calculations from agent
   - Action distributions per step

### Phase 4: Analysis (Day 7)

**Key Metrics:**

```python
METRICS = {
    "primary": {
        "success_rate": "Episodes where task completed / total",
        "adaptation_gap": "0.5g_explained - 0.5g_silent success rate",
        "learning_curve": "Success rate over training episodes",
    },
    "secondary": {
        "action_entropy": "Diversity of actions taken",
        "physics_usage": "% of responses with correct physics calculations",
        "jump_timing": "Distribution of jump positions relative to gap",
        "adaptation_speed": "Episodes to reach stable performance at 0.5g",
    },
    "diagnostic": {
        "calculation_accuracy": "Agent's predicted vs actual trajectory",
        "formula_application": "Did agent use g=9.81 vs g=4.9 correctly",
        "reasoning_chain": "Quality of physics_calculation field",
    }
}
```

---

## Part 4: Interpretation Guide for Current (v1) Data

### What the v1 Data Actually Shows

The v1 data is NOT usable for the intended hypotheses, but it does tell us:

1. **LLM Action Bias:**
   - Gemini 2.0 Flash outputs "forward" 97-99% of the time
   - This is independent of physics description
   - Prompt engineering is critical for action diversity

2. **Task Difficulty Estimation:**
   - Gap task at current params: 0% at 1g, 100% at 0.5g
   - This confirms the task is broken, not that agents adapted

3. **Physics Doesn't Matter Yet:**
   - Agents don't need to understand physics when:
     - Tasks are trivially easy, or
     - Tasks are impossibly hard

4. **What Would Need to be True for Valid Results:**
   - Tasks calibrated so NRL-F gets 30-50% at 1g baseline
   - RL agents show learning curves during training
   - Action distributions show meaningful variation
   - Physics calculations in responses are verifiable

### Reinterpreting Each Finding

| v1 Finding | What It Actually Means |
|------------|----------------------|
| NRL-F: 0% baseline, 100% 0.5g | Task broken, not adaptation |
| RL-F: 0% baseline, 0% 0.5g | No RL happened, LLM just outputs forward |
| RL-N: 0% baseline, 0% 0.5g | Same as RL-F, description doesn't matter |
| All agents forward-biased | Prompt issue, not physics understanding |

### Salvageable Insights

1. **Prompt sensitivity is high:** Small prompt changes cause large behavior changes
2. **LLM action diversity is low:** Need explicit prompting for varied actions
3. **Physics context is ignored:** LLM doesn't naturally reason about physics
4. **Structured output is necessary:** Free-form responses bias toward simple actions

---

## Implementation Checklist

### Immediate (Day 1)
- [ ] Update GapCrossingTask.js with v2 parameters
- [ ] Update ThrowBlockTask.js with v2 parameters
- [ ] Test physics math matches expected trajectories
- [ ] Run 20 manual trials to verify difficulty

### Short-term (Days 2-3)
- [ ] Implement structured JSON prompts
- [ ] Add physics_calculation parsing
- [ ] Implement one of the RL options (recommend Option C)
- [ ] Verify RL agents actually improve over episodes

### Medium-term (Days 4-6)
- [ ] Run full experiment matrix
- [ ] Monitor for anomalies during runs
- [ ] Collect all diagnostic data

### Analysis (Day 7)
- [ ] Generate success rate plots
- [ ] Analyze physics calculation accuracy
- [ ] Test hypotheses H1, H2, H3
- [ ] Write up findings

---

## Quick Start for v2

```bash
# 1. Update the task files
# (Apply changes from Part 1)

# 2. Test calibration
cd gravity-agents/web-env
PORT=3002 node server.js &

cd ../python-orchestrator
python3 -c "
from run_experiments import run_single_episode
result = run_single_episode('NRL-F', 'gap', 'training')
print(f'Success: {result.success}')
"

# 3. Run calibration sweep
python3 calibrate_tasks.py  # New script needed

# 4. Run full v2 experiments
python3 run_all_experiments_v2.py
```
