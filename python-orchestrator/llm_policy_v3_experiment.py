"""
LLM Policy V3 for Gravity Experiment
====================================
Tests the 3 hypotheses:
- H1: Formula vs Story (does explicit physics help?)
- H2: RL vs No-RL (does learning from experience help?)
- H3: Explained vs Silent gravity change

Key difference from llm_policy_v3.py: NO WINNING STRATEGY HINT
The LLM must figure out the strategy from physics understanding.
"""

import json
import re
from typing import Literal
from dataclasses import dataclass
import google.generativeai as genai
from groq import Groq

from config import GEMINI_API_KEY, GROQ_API_KEY, ACTION_SPACES


@dataclass
class ExperimentConfig:
    """Configuration for gravity experiment"""
    agent_type: Literal["NRL-F", "NRL-N"]  # Formula vs Normal (story)
    gravity_condition: Literal["training", "test_silent", "test_explained"]
    model: str = "gemini-2.0-flash"
    use_groq: bool = False
    temperature: float = 0.2


# H1: Formula-based physics (explicit equations)
PHYSICS_FORMULA = """
PHYSICS LAWS (use for calculations):
- Gravity: g = {gravity} m/s²
- Projectile motion:
  * Horizontal: x(t) = x₀ + vₓ * t
  * Vertical: y(t) = y₀ + vᵧ * t - 0.5 * g * t²
  * Flight time: T = 2 * vᵧ / g
  * Jump range: R = vₓ * T
- Jump gives initial vᵧ ≈ 4.85 m/s (from jump_height ~1.2m)
- Forward gives vₓ ≈ 3.0 m/s (max 4.0 m/s)
- Air control factor: 0.3 (can adjust mid-flight)

CALCULATE: Given g={gravity}, flight time T = 2 * 4.85 / {gravity} = {flight_time:.2f}s
With vₓ=3 and T={flight_time:.2f}s, jump range ≈ {jump_range:.1f}m
"""

# H1: Story-based physics (intuitive description)
PHYSICS_STORY = """
PHYSICS ENVIRONMENT:
{gravity_description}

Your character:
- Moves forward at walking speed (~3 m/s)
- Can jump about 1.2 meters high
- Has some air control (can push forward while jumping)
- Needs momentum before jumping for distance

{adaptation_hint}
"""

# Gravity descriptions for story mode
GRAVITY_DESCRIPTIONS = {
    "training": "Normal Earth-like gravity. Objects fall at the standard rate you're used to.",
    "test_silent": "Normal Earth-like gravity. Objects fall at the standard rate you're used to.",
    "test_explained": "WEAK GRAVITY - like on a low-gravity moon! Objects fall more slowly and stay in the air much longer. Your jumps will go further than normal - roughly 1.4x the distance!"
}

ADAPTATION_HINTS = {
    "training": "",
    "test_silent": "",
    "test_explained": "IMPORTANT: Because gravity is weaker, you can jump later and still make it. The timing window is more forgiving."
}

# Task description - with step-to-distance mapping (no optimal hint)
TASK_PROMPT = """TASK: Jump across a gap between two platforms.

SETUP:
- You start on Platform A
- There's a gap to cross
- Platform B has the goal zone

OBJECTIVE: Land on the goal zone on Platform B.

CRITICAL STEP MAPPING:
- Each "forward" action moves you approximately 0.5 meters
- You start at x=-1.33m
- The gap edge is at x=2.0m
- Therefore: reaching the gap edge takes about (2.0 - (-1.33)) / 0.5 = 6-7 forward steps

STRATEGY REQUIREMENTS:
1. Build forward momentum before jumping (stationary jumps don't go far)
2. Jump JUST BEFORE the gap edge (around step 6-7 when starting from x=-1.33m)
3. Use forward input during flight for air control

OUTPUT: A complete sequence of actions to cross the gap.
Valid actions: forward, jump, idle, back
"""


class GravityExperimentPolicy:
    """Policy for testing gravity adaptation hypotheses"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.action_sequence = []
        self.current_step = 0
        self.total_llm_calls = 0
        self.plan_response = None

        # Set up LLM client
        if config.use_groq:
            self.client = Groq(api_key=GROQ_API_KEY)
        else:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(config.model)

    def _get_gravity_value(self) -> float:
        """Get gravity based on condition"""
        if self.config.gravity_condition == "training":
            return 9.81
        else:
            return 4.905  # Half gravity for test conditions

    def _build_physics_block(self) -> str:
        """Build physics description based on agent type and condition"""
        gravity = self._get_gravity_value()

        if self.config.agent_type == "NRL-F":
            # Formula-based: explicit equations
            flight_time = 2 * 4.85 / gravity
            jump_range = 3.0 * flight_time
            return PHYSICS_FORMULA.format(
                gravity=gravity,
                flight_time=flight_time,
                jump_range=jump_range
            )
        else:
            # Story-based: intuitive descriptions
            gravity_desc = GRAVITY_DESCRIPTIONS[self.config.gravity_condition]
            adaptation_hint = ADAPTATION_HINTS[self.config.gravity_condition]
            return PHYSICS_STORY.format(
                gravity_description=gravity_desc,
                adaptation_hint=adaptation_hint
            )

    def _build_prompt(self, obs: dict) -> str:
        """Build the planning prompt"""
        gap_start = obs['gapStart']
        gap_end = obs['gapEnd']
        gap_width = obs['gapWidth']
        goal_min = obs['goalZone']['minX']
        goal_max = obs['goalZone']['maxX']
        x = obs['agentPosition'][0]

        physics_block = self._build_physics_block()

        return f"""You are an AI controlling a character in a physics simulation.

{physics_block}

{TASK_PROMPT}

CURRENT EPISODE:
- Start position: x={x:.2f}m
- Gap: x={gap_start:.2f}m to x={gap_end:.2f}m (width={gap_width:.2f}m)
- Goal zone: x={goal_min:.2f}m to x={goal_max:.2f}m
- Distance from start to gap: {gap_start - x:.2f}m

Think carefully about the physics. Calculate how far you can jump and plan accordingly.

RESPOND with a JSON object:
{{
  "physics_reasoning": "<show your calculations for jump distance>",
  "strategy": "<explain when to jump and why>",
  "sequence": ["forward", "forward", ..., "jump", "forward", ...]
}}

Include ~20-25 actions total. The sequence should get you from start to goal.
"""

    def plan_episode(self, initial_obs: dict) -> dict:
        """Generate the full action sequence for this episode"""
        self.total_llm_calls += 1

        prompt = self._build_prompt(initial_obs)

        try:
            if self.config.use_groq:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=800
                )
                response_text = response.choices[0].message.content.strip()
            else:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.config.temperature,
                        max_output_tokens=800
                    )
                )
                response_text = response.text.strip()

            # Parse response
            response_data = self._parse_sequence_response(response_text)
            self.action_sequence = response_data.get("sequence", [])
            self.plan_response = response_data

            return response_data

        except Exception as e:
            print(f"LLM error: {e}")
            self.action_sequence = ["forward"] * 10  # No hint fallback
            return {"error": str(e), "sequence": self.action_sequence}

    def select_action(self, observation: dict, gravity_condition: str = "training") -> tuple[str, dict]:
        """Get next action from pre-planned sequence."""
        if not self.action_sequence:
            self.plan_episode(observation)

        if self.current_step < len(self.action_sequence):
            action = self.action_sequence[self.current_step]
            action = self._validate_action(action)
        else:
            action = "forward"

        response_data = {
            "source": "planned_sequence",
            "step": self.current_step,
            "action": action,
            "remaining": len(self.action_sequence) - self.current_step - 1
        }

        self.current_step += 1
        return action, response_data

    def _parse_sequence_response(self, text: str) -> dict:
        """Parse action sequence from LLM response"""
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            data = json.loads(text.strip())
            if "sequence" in data and isinstance(data["sequence"], list):
                return data

        except json.JSONDecodeError:
            pass

        # Fallback parsing
        match = re.search(r'\[([^\]]+)\]', text)
        if match:
            items = match.group(1)
            actions = re.findall(r'"([^"]+)"', items)
            if actions:
                return {"sequence": actions, "parse_fallback": True, "raw": text[:500]}

        # Last resort - extract action words
        actions = []
        for word in text.lower().split():
            clean = word.strip('",[]')
            if clean in ["forward", "jump", "idle", "back"]:
                actions.append(clean)

        if len(actions) >= 3:
            return {"sequence": actions, "parse_fallback": True, "raw": text[:500]}

        return {"sequence": ["forward"] * 10, "parse_error": True, "raw": text[:500]}

    def _validate_action(self, action: str) -> str:
        """Validate action is in action space"""
        action = str(action).strip().lower()
        valid = ACTION_SPACES["gap"]
        if action in valid:
            return action
        for v in valid:
            if v in action or action in v:
                return v
        return "forward"

    def reset(self):
        """Reset for new episode"""
        self.action_sequence = []
        self.current_step = 0
        self.plan_response = None

    def get_stats(self) -> dict:
        return {
            "total_llm_calls": self.total_llm_calls,
            "sequence_length": len(self.action_sequence),
            "current_step": self.current_step
        }
