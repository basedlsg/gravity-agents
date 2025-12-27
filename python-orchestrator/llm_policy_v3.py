"""
LLM Policy V3 - Full sequence planning to reduce episode cost
=============================================================
Key change: LLM outputs COMPLETE action sequence for entire episode in ONE call.
This reduces LLM calls per episode from ~20 to exactly 1.
"""

import json
import re
from typing import Literal
from dataclasses import dataclass
import google.generativeai as genai
from groq import Groq

from config import GEMINI_API_KEY, GROQ_API_KEY, ACTION_SPACES


@dataclass
class PolicyConfigV3:
    """Configuration for V3 LLM policy with full sequence planning"""
    agent_type: Literal["RL-F", "RL-N", "NRL-F"]
    task: Literal["gap", "throw"]
    model: str = "gemini-2.0-flash"
    use_groq: bool = False
    temperature: float = 0.2
    max_steps: int = 80  # Max episode length


# Physics block with concrete numbers
PHYSICS_BLOCK = """
PHYSICS PARAMETERS:
- Gravity: g = 9.81 m/s²
- Forward speed: 3.0 m/s (accumulated over steps)
- Max speed: 4.0 m/s (capped)
- Jump initial velocity: ~4.85 m/s upward
- Flight time: ~1.0 second
- With vx=3 and flight time=1.0s: jump distance ~5m

MECHANICS:
- Each "forward" step moves you ~0.5m and builds velocity
- "jump" launches you upward while preserving horizontal velocity
- Air control allows forward input during flight
- You start at x=-1.33m, gap starts at x=2.0m
"""

# V3: Planning prompt that asks for FULL sequence
TASK_PROMPT_V3 = """TASK: Cross a 4.5m gap between two platforms.

START: x=-1.33m (on platform A)
GAP: starts at x=2.0m, ends at x=6.5m
GOAL: x=7.3m to x=9.7m (on platform B)

WINNING STRATEGY (verified to work):
1. Forward x6 steps: builds momentum, reaches x≈1.7m with vx≈3m/s
2. Jump x1 step: launches from near edge (jump activates near x≈2.0m)
3. Forward x10+ steps: maintain air control until landing on goal

OUTPUT: Give me the COMPLETE action sequence as a list.
Each action is one of: forward, jump, idle, back
"""


class LLMPolicyV3:
    """V3 LLM policy with full sequence planning - ONE call per episode"""

    def __init__(self, config: PolicyConfigV3):
        self.config = config
        self.action_sequence = []  # Full sequence for episode
        self.current_step = 0
        self.total_llm_calls = 0
        self.plan_response = None

        # Set up LLM client
        if config.use_groq:
            self.client = Groq(api_key=GROQ_API_KEY)
        else:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(config.model)

    def _build_prompt(self, obs: dict) -> str:
        """Build the full planning prompt"""
        gap_start = obs['gapStart']
        gap_end = obs['gapEnd']
        gap_width = obs['gapWidth']
        goal_min = obs['goalZone']['minX']
        goal_max = obs['goalZone']['maxX']
        x = obs['agentPosition'][0]

        return f"""You are controlling a character in a physics simulation.

{PHYSICS_BLOCK}

{TASK_PROMPT_V3}

CURRENT EPISODE:
- Start position: x={x:.2f}m
- Gap: x={gap_start:.2f}m to x={gap_end:.2f}m (width={gap_width:.2f}m)
- Goal zone: x={goal_min:.2f}m to x={goal_max:.2f}m

RESPOND with a JSON object containing your action sequence:
{{
  "reasoning": "<explain your strategy>",
  "sequence": ["forward", "forward", ..., "jump", "forward", ...]
}}

The sequence should have ~20 actions. Include enough forward actions after jump for air control.
Valid actions: forward, jump, idle, back
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
                    max_tokens=600
                )
                response_text = response.choices[0].message.content.strip()
            else:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.config.temperature,
                        max_output_tokens=600
                    )
                )
                response_text = response.text.strip()

            # Parse response
            response_data = self._parse_sequence_response(response_text)
            self.action_sequence = response_data.get("sequence", ["forward"] * 20)
            self.plan_response = response_data

            return response_data

        except Exception as e:
            print(f"LLM error: {e}")
            # Fallback to optimal sequence
            self.action_sequence = ["forward"] * 6 + ["jump"] + ["forward"] * 13
            return {"error": str(e), "sequence": self.action_sequence}

    def select_action(
        self,
        observation: dict,
        gravity_condition: str = "training"
    ) -> tuple[str, dict]:
        """
        Get next action from pre-planned sequence.
        If no sequence planned yet, plan now.
        """
        # Plan on first call
        if not self.action_sequence:
            self.plan_episode(observation)

        # Get next action from sequence
        if self.current_step < len(self.action_sequence):
            action = self.action_sequence[self.current_step]
            action = self._validate_action(action)
        else:
            action = "forward"  # Default if sequence exhausted

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
            # Remove markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            data = json.loads(text.strip())

            # Validate sequence
            if "sequence" in data and isinstance(data["sequence"], list):
                return data

        except json.JSONDecodeError:
            pass

        # Try to extract sequence from text
        # Look for array pattern
        match = re.search(r'\[([^\]]+)\]', text)
        if match:
            items = match.group(1)
            # Extract quoted strings
            actions = re.findall(r'"([^"]+)"', items)
            if actions:
                return {"sequence": actions, "parse_fallback": True}

        # Ultimate fallback - look for action words
        actions = []
        for word in text.lower().split():
            if word in ["forward", "jump", "idle", "back"]:
                actions.append(word)

        if len(actions) >= 5:
            return {"sequence": actions, "parse_fallback": True}

        # Fallback to optimal
        return {"sequence": ["forward"] * 6 + ["jump"] + ["forward"] * 13, "parse_error": True}

    def _validate_action(self, action: str) -> str:
        """Validate action is in action space"""
        action = str(action).strip().lower()
        valid = ACTION_SPACES[self.config.task]
        if action in valid:
            return action
        # Fuzzy match
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
        """Get policy statistics"""
        return {
            "total_llm_calls": self.total_llm_calls,
            "sequence_length": len(self.action_sequence),
            "current_step": self.current_step
        }
