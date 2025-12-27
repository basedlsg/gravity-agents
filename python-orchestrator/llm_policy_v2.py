"""
LLM Policy V2 - Structured JSON prompts that test physics understanding

Changes from V1:
- Requires JSON response with physics_calculation field
- Explicit physics formulas in prompts
- Verifiable reasoning chains
- Better action diversity through structured format
"""

import json
import re
from typing import Literal
from dataclasses import dataclass
import google.generativeai as genai
from groq import Groq

from config import GEMINI_API_KEY, GROQ_API_KEY, ACTION_SPACES


@dataclass
class PolicyConfigV2:
    """Configuration for V2 LLM policy"""
    agent_type: Literal["RL-F", "RL-N", "NRL-F"]
    task: Literal["gap", "throw"]
    model: str = "gemini-2.0-flash"
    use_groq: bool = False
    history_length: int = 3
    temperature: float = 0.2


# V2: Structured physics blocks
PHYSICS_BLOCKS = {
    "formula": {
        "training": """
PHYSICS LAWS (use these for calculations):
- Gravity: g = 9.81 m/s²
- Projectile motion:
  * Horizontal: x(t) = x₀ + vₓ * t
  * Vertical: y(t) = y₀ + vᵧ * t - 0.5 * g * t²
  * Time to peak: t_peak = vᵧ / g
  * Max height: h = vᵧ² / (2 * g)
- Jump velocity: v = sqrt(2 * g * jump_height)
- Range: R = vₓ * (2 * vᵧ / g)
""",
        "test_silent": """
PHYSICS LAWS (use these for calculations):
- Gravity: g = 9.81 m/s²
- Projectile motion:
  * Horizontal: x(t) = x₀ + vₓ * t
  * Vertical: y(t) = y₀ + vᵧ * t - 0.5 * g * t²
  * Time to peak: t_peak = vᵧ / g
  * Max height: h = vᵧ² / (2 * g)
- Jump velocity: v = sqrt(2 * g * jump_height)
- Range: R = vₓ * (2 * vᵧ / g)
""",
        "test_explained": """
PHYSICS LAWS (UPDATED - gravity has changed!):
- Gravity: g = 4.9 m/s² (was 9.81, now HALF as strong!)
- Projectile motion:
  * Horizontal: x(t) = x₀ + vₓ * t
  * Vertical: y(t) = y₀ + vᵧ * t - 0.5 * g * t²
  * Time to peak: t_peak = vᵧ / g (LONGER with weaker gravity)
  * Max height: h = vᵧ² / (2 * g) (HIGHER with weaker gravity)
- Jump velocity: v = sqrt(2 * g * jump_height)
- Range: R = vₓ * (2 * vᵧ / g) (LONGER with weaker gravity)

IMPORTANT: With g=4.9 instead of 9.81:
- Objects stay in the air ~1.4x longer
- Jumps go ~1.4x further
- Throws travel ~2x the distance
"""
    },
    "normal": {
        "training": """
PHYSICS:
Normal Earth-like gravity. Objects fall at the standard rate.
Your character can jump about 1.2 meters high.
Movement speed is about 3 m/s.
""",
        "test_silent": """
PHYSICS:
Normal Earth-like gravity. Objects fall at the standard rate.
Your character can jump about 1.2 meters high.
Movement speed is about 3 m/s.
""",
        "test_explained": """
PHYSICS (CHANGED!):
Gravity is now WEAKER - like on a low-gravity moon!
Objects fall more slowly and stay in the air longer.
Your jumps now go further - about 1.4x the normal distance.
Throws travel about 2x as far.
Adjust your timing accordingly!
"""
    }
}

# V2: Task descriptions with structured format requirement
TASK_PROMPTS = {
    "gap": """TASK: Cross the gap between two platforms.

You stand on platform A. There is a gap, then platform B with a goal zone.
You must jump across the gap and land on the goal zone.

CRITICAL: You must time your jump correctly. Too early and you won't clear the gap. Too late and you'll run off the edge.

STRATEGY HINTS:
- Build up forward momentum before jumping
- Jump when you're close to (but not past) the gap edge
- A running jump goes further than a standing jump
""",
    "throw": """TASK: Throw the block into the basket.

1. Pick up the block near you
2. Move to a good throwing position
3. Throw with the right strength to land in the basket

CRITICAL: The right throw strength depends on your distance from the basket.
- throw_weak: ~2m range
- throw_medium: ~4m range
- throw_strong: ~6m range

STRATEGY HINTS:
- Get close enough that medium throw can reach
- The basket is elevated, so you need arc in your throw
"""
}


class LLMPolicyV2:
    """V2 LLM policy with structured JSON output"""

    def __init__(self, config: PolicyConfigV2):
        self.config = config
        self.history = []

        # Set up LLM client
        if config.use_groq:
            self.client = Groq(api_key=GROQ_API_KEY)
        else:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(config.model)

        # Get gravity style
        self.gravity_style = "formula" if config.agent_type in ["RL-F", "NRL-F"] else "normal"

    def _build_system_prompt(self) -> str:
        """Build structured system prompt"""
        actions = ACTION_SPACES[self.config.task]
        task_prompt = TASK_PROMPTS[self.config.task]

        return f"""You are an AI agent in a 3D physics simulation.

{task_prompt}

AVAILABLE ACTIONS: {', '.join(actions)}

You MUST respond with ONLY a JSON object in this exact format:
{{
  "observation_summary": "<1-2 sentence summary of current state>",
  "physics_calculation": "<show your math using the physics laws>",
  "reasoning": "<why this action is best right now>",
  "action": "<exactly one action from the list>"
}}

RULES:
1. Always include physics_calculation with actual numbers
2. Action must be exactly one of the available actions
3. No text outside the JSON object
4. Think step by step about the physics"""

    def _format_observation(self, obs: dict) -> str:
        """Format observation for structured prompt"""
        if self.config.task == "gap":
            return f"""CURRENT STATE:
Position: x={obs['agentPosition'][0]:.2f}m, y={obs['agentPosition'][1]:.2f}m
Velocity: vx={obs['agentVelocity'][0]:.2f} m/s, vy={obs['agentVelocity'][1]:.2f} m/s
Grounded: {obs['isGrounded']}

GAP: from x={obs['gapStart']:.2f}m to x={obs['gapEnd']:.2f}m (width={obs['gapWidth']:.2f}m)
GOAL: x from {obs['goalZone']['minX']:.2f}m to {obs['goalZone']['maxX']:.2f}m

DISTANCES:
- To gap edge: {obs['gapStart'] - obs['agentPosition'][0]:.2f}m
- To goal center: {(obs['goalZone']['minX'] + obs['goalZone']['maxX'])/2 - obs['agentPosition'][0]:.2f}m"""
        else:
            return f"""CURRENT STATE:
Agent Position: x={obs['agentPosition'][0]:.2f}m, y={obs['agentPosition'][1]:.2f}m
Agent Velocity: vx={obs['agentVelocity'][0]:.2f} m/s
Holding Block: {obs['holdingBlock']}

Block Position: x={obs['blockPosition'][0]:.2f}m, y={obs['blockPosition'][1]:.2f}m
Basket Position: x={obs['basketPosition'][0]:.2f}m, y={obs['basketPosition'][1]:.2f}m

DISTANCES:
- Agent to block: {abs(obs['blockPosition'][0] - obs['agentPosition'][0]):.2f}m
- Agent to basket: {obs['basketPosition'][0] - obs['agentPosition'][0]:.2f}m"""

    def _format_history(self) -> str:
        """Format action history"""
        if not self.history:
            return "HISTORY: (first step)"

        recent = self.history[-self.config.history_length:]
        lines = ["RECENT ACTIONS:"]
        for i, (action, result) in enumerate(recent):
            lines.append(f"  {i+1}. {action} -> {result}")
        return "\n".join(lines)

    def select_action(
        self,
        observation: dict,
        gravity_condition: Literal["training", "test_silent", "test_explained"] = "training"
    ) -> tuple[str, dict]:
        """
        Select action with full reasoning.
        Returns (action, response_data) where response_data contains the LLM's reasoning.
        """
        actions = ACTION_SPACES[self.config.task]

        # Build prompts
        system_prompt = self._build_system_prompt()
        physics_block = PHYSICS_BLOCKS[self.gravity_style][gravity_condition]
        obs_text = self._format_observation(observation)
        history_text = self._format_history()

        user_prompt = f"""{physics_block}

{obs_text}

{history_text}

Respond with JSON only. Choose from: {', '.join(actions)}"""

        # Call LLM
        try:
            if self.config.use_groq:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=300
                )
                response_text = response.choices[0].message.content.strip()
            else:
                response = self.model.generate_content(
                    f"{system_prompt}\n\n{user_prompt}",
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.config.temperature,
                        max_output_tokens=300
                    )
                )
                response_text = response.text.strip()

            # Parse JSON response
            response_data = self._parse_json_response(response_text)
            action = self._validate_action(response_data.get("action", "idle"), actions)

            # Update history
            result_summary = f"pos=({observation.get('agentPosition', [0,0,0])[0]:.1f}, {observation.get('agentPosition', [0,0,0])[1]:.1f})"
            self.history.append((action, result_summary))

            return action, response_data

        except Exception as e:
            print(f"LLM error: {e}")
            return "idle", {"error": str(e), "action": "idle"}

    def _parse_json_response(self, text: str) -> dict:
        """Parse JSON from LLM response"""
        # Try to extract JSON from response
        try:
            # Remove markdown code blocks if present
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            return json.loads(text.strip())
        except json.JSONDecodeError:
            # Try to find JSON object in text
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

            # Return parsed fields if possible
            action_match = re.search(r'"action"\s*:\s*"([^"]+)"', text)
            return {
                "action": action_match.group(1) if action_match else "idle",
                "raw_response": text,
                "parse_error": True
            }

    def _validate_action(self, action: str, valid_actions: list[str]) -> str:
        """Validate and normalize action"""
        action = action.strip().lower()

        if action in valid_actions:
            return action

        # Handle throw variants
        if "throw" in action:
            if "weak" in action:
                return "throw_weak"
            elif "strong" in action:
                return "throw_strong"
            elif "medium" in action:
                return "throw_medium"

        # Fuzzy match
        for valid in valid_actions:
            if valid in action or action in valid:
                return valid

        return "idle"

    def reset(self):
        """Reset for new episode"""
        self.history = []


class ValueCachedPolicy(LLMPolicyV2):
    """
    V2 RL policy: Caches state-action values from experience.
    This is Option C from the v2 spec - a hybrid approach.
    """

    def __init__(self, config: PolicyConfigV2, epsilon: float = 0.3):
        super().__init__(config)
        self.state_action_values = {}  # state_hash -> action -> running average
        self.state_action_counts = {}
        self.epsilon = epsilon
        self.episode_trajectory = []

    def _hash_state(self, obs: dict) -> str:
        """Create a discretized state hash"""
        if self.config.task == "gap":
            # Discretize position to 0.5m bins
            x_bin = round(obs['agentPosition'][0] * 2) / 2
            y_bin = round(obs['agentPosition'][1] * 2) / 2
            vx_sign = 1 if obs['agentVelocity'][0] > 0.5 else (-1 if obs['agentVelocity'][0] < -0.5 else 0)
            grounded = 1 if obs['isGrounded'] else 0
            return f"gap_{x_bin}_{y_bin}_{vx_sign}_{grounded}"
        else:
            x_bin = round(obs['agentPosition'][0] * 2) / 2
            holding = 1 if obs['holdingBlock'] else 0
            dist_bin = round((obs['basketPosition'][0] - obs['agentPosition'][0]) * 2) / 2
            return f"throw_{x_bin}_{holding}_{dist_bin}"

    def select_action(
        self,
        observation: dict,
        gravity_condition: Literal["training", "test_silent", "test_explained"] = "training"
    ) -> tuple[str, dict]:
        """Select action using epsilon-greedy with cached values"""
        state_hash = self._hash_state(observation)
        actions = ACTION_SPACES[self.config.task]

        # Epsilon-greedy
        import random
        if state_hash in self.state_action_values and random.random() > self.epsilon:
            # Exploit: choose best cached action
            values = self.state_action_values[state_hash]
            best_action = max(values, key=values.get)
            response_data = {
                "action": best_action,
                "source": "cached_value",
                "value": values[best_action],
                "state_hash": state_hash
            }
        else:
            # Explore: use LLM
            best_action, response_data = super().select_action(observation, gravity_condition)
            response_data["source"] = "llm_exploration"
            response_data["state_hash"] = state_hash

        # Record for trajectory
        self.episode_trajectory.append((state_hash, best_action))

        return best_action, response_data

    def update_episode(self, total_reward: float):
        """Update values based on episode outcome"""
        # Simple: assign episode reward to all state-action pairs
        for state_hash, action in self.episode_trajectory:
            if state_hash not in self.state_action_values:
                self.state_action_values[state_hash] = {}
                self.state_action_counts[state_hash] = {}

            if action not in self.state_action_values[state_hash]:
                self.state_action_values[state_hash][action] = 0.0
                self.state_action_counts[state_hash][action] = 0

            # Running average update
            count = self.state_action_counts[state_hash][action]
            old_value = self.state_action_values[state_hash][action]
            self.state_action_values[state_hash][action] = (old_value * count + total_reward) / (count + 1)
            self.state_action_counts[state_hash][action] = count + 1

        self.episode_trajectory = []

    def reset(self):
        """Reset for new episode"""
        super().reset()
        self.episode_trajectory = []

    def get_learning_stats(self) -> dict:
        """Get statistics about learned values"""
        total_states = len(self.state_action_values)
        total_pairs = sum(len(v) for v in self.state_action_values.values())
        return {
            "states_visited": total_states,
            "state_action_pairs": total_pairs,
            "epsilon": self.epsilon
        }
