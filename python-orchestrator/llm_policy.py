"""
LLM Policy Server - Interfaces with Gemini/Groq for action selection
"""

import json
import re
from typing import Literal
from dataclasses import dataclass
import google.generativeai as genai
from groq import Groq

from config import (
    GEMINI_API_KEY, GROQ_API_KEY,
    GRAVITY_DESCRIPTIONS, TASK_DESCRIPTIONS, ACTION_SPACES,
    get_agent_gravity_style
)


@dataclass
class PolicyConfig:
    """Configuration for LLM policy"""
    agent_type: Literal["RL-F", "RL-N", "NRL-F"]
    task: Literal["gap", "throw"]
    model: str = "gemini-2.0-flash"
    use_groq: bool = False
    history_length: int = 5
    temperature: float = 0.3


class LLMPolicy:
    """LLM-based policy for action selection"""

    def __init__(self, config: PolicyConfig):
        self.config = config
        self.history = []

        # Set up LLM client
        if config.use_groq:
            self.client = Groq(api_key=GROQ_API_KEY)
        else:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(config.model)

        # Get gravity style for this agent type
        self.gravity_style = get_agent_gravity_style(config.agent_type)

        # Build system prompt
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent"""
        task_desc = TASK_DESCRIPTIONS[self.config.task]
        actions = ACTION_SPACES[self.config.task]

        return f"""You are an AI agent controlling a character in a 3D physics simulation.

TASK: {task_desc}

AVAILABLE ACTIONS: {', '.join(actions)}

ACTION EFFECTS:
- forward/back/left/right: Move in that direction
- jump: Jump upward (only works when grounded) [gap task only]
- pick: Pick up the block if close enough [throw task only]
- drop: Release held block [throw task only]
- throw_weak/throw_medium/throw_strong: Throw held block with varying force [throw task only]
- idle: Do nothing, slow down

REWARD:
- You receive +1 for successfully completing the task, 0 otherwise.
- Your goal is to maximize your reward by completing the task efficiently.

RULES:
1. Analyze the current observation carefully
2. Consider the physics (gravity, momentum, trajectories)
3. Choose the single best action for this step
4. Reply with ONLY the action name, nothing else"""

    def _format_observation(self, obs: dict) -> str:
        """Format observation for the LLM"""
        if self.config.task == "gap":
            return f"""Current State:
- Position: x={obs['agentPosition'][0]:.2f}, y={obs['agentPosition'][1]:.2f}, z={obs['agentPosition'][2]:.2f}
- Velocity: vx={obs['agentVelocity'][0]:.2f}, vy={obs['agentVelocity'][1]:.2f}, vz={obs['agentVelocity'][2]:.2f}
- Gap: starts at x={obs['gapStart']:.2f}, ends at x={obs['gapEnd']:.2f} (width={obs['gapWidth']:.2f})
- Goal zone: x from {obs['goalZone']['minX']:.2f} to {obs['goalZone']['maxX']:.2f}
- Grounded: {obs['isGrounded']}"""
        else:
            return f"""Current State:
- Agent Position: x={obs['agentPosition'][0]:.2f}, y={obs['agentPosition'][1]:.2f}, z={obs['agentPosition'][2]:.2f}
- Agent Velocity: vx={obs['agentVelocity'][0]:.2f}, vy={obs['agentVelocity'][1]:.2f}
- Block Position: x={obs['blockPosition'][0]:.2f}, y={obs['blockPosition'][1]:.2f}, z={obs['blockPosition'][2]:.2f}
- Holding Block: {obs['holdingBlock']}
- Basket Position: x={obs['basketPosition'][0]:.2f}, y={obs['basketPosition'][1]:.2f}
- Basket Bounds: x from {obs['basketBounds']['minX']:.2f} to {obs['basketBounds']['maxX']:.2f}"""

    def _format_history(self) -> str:
        """Format recent action history"""
        if not self.history:
            return ""

        recent = self.history[-self.config.history_length:]
        lines = ["Recent actions:"]
        for i, (obs_summary, action) in enumerate(recent):
            lines.append(f"  Step -{len(recent)-i}: {action} -> {obs_summary}")
        return "\n".join(lines)

    def get_gravity_description(self, condition: Literal["training", "test_silent", "test_explained"]) -> str:
        """Get appropriate gravity description for current condition"""
        return GRAVITY_DESCRIPTIONS[self.gravity_style][condition]

    def select_action(
        self,
        observation: dict,
        gravity_condition: Literal["training", "test_silent", "test_explained"] = "training"
    ) -> str:
        """Select an action given the current observation"""
        actions = ACTION_SPACES[self.config.task]

        # Build the prompt
        gravity_desc = self.get_gravity_description(gravity_condition)
        obs_text = self._format_observation(observation)
        history_text = self._format_history()

        user_prompt = f"""{gravity_desc}

{obs_text}

{history_text}

Choose the best action from: {', '.join(actions)}
Reply with ONLY the action name."""

        # Call LLM
        try:
            if self.config.use_groq:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=20
                )
                action_text = response.choices[0].message.content.strip().lower()
            else:
                response = self.model.generate_content(
                    f"{self.system_prompt}\n\n{user_prompt}",
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.config.temperature,
                        max_output_tokens=20
                    )
                )
                action_text = response.text.strip().lower()

            # Parse action
            action = self._parse_action(action_text, actions)

            # Update history
            obs_summary = f"pos=({observation.get('agentPosition', [0,0,0])[0]:.1f}, {observation.get('agentPosition', [0,0,0])[1]:.1f})"
            self.history.append((obs_summary, action))

            return action

        except Exception as e:
            print(f"LLM error: {e}, defaulting to idle")
            return "idle"

    def _parse_action(self, text: str, valid_actions: list[str]) -> str:
        """Parse LLM output to valid action"""
        text = text.strip().lower()

        # Direct match
        if text in valid_actions:
            return text

        # Handle throw variants
        if "throw" in text:
            if "weak" in text:
                return "throw_weak"
            elif "strong" in text:
                return "throw_strong"
            elif "medium" in text:
                return "throw_medium"

        # Fuzzy match
        for action in valid_actions:
            if action in text or text in action:
                return action

        # Default
        return "idle"

    def reset(self):
        """Reset history for new episode"""
        self.history = []


class FewShotPolicy(LLMPolicy):
    """Non-RL policy with few-shot examples"""

    def __init__(self, config: PolicyConfig, examples: list[dict] = None):
        # Set examples BEFORE calling super().__init__ since it calls _build_system_prompt
        self.examples = examples
        super().__init__(config)
        # Now set default examples if not provided
        if self.examples is None:
            self.examples = self._get_default_examples()
            self.system_prompt = self._build_system_prompt()

    def _get_default_examples(self) -> list[dict]:
        """Get default few-shot examples for each task"""
        if self.config.task == "gap":
            return [
                {
                    "observation": "Position: x=-1.0, y=1.0 | Gap: 2.5 to 5.5 | Grounded: True",
                    "reasoning": "I'm before the gap and grounded. Need to build speed first.",
                    "action": "forward"
                },
                {
                    "observation": "Position: x=1.5, y=1.0 | Gap: 2.5 to 5.5 | Grounded: True",
                    "reasoning": "I'm close to the gap edge with good speed. Time to jump.",
                    "action": "jump"
                },
                {
                    "observation": "Position: x=4.0, y=2.5 | Gap: 2.5 to 5.5 | Grounded: False",
                    "reasoning": "I'm in the air over the gap. Keep forward momentum.",
                    "action": "forward"
                }
            ]
        else:
            return [
                {
                    "observation": "Agent: x=-2.0 | Block: x=-1.5 | Holding: False",
                    "reasoning": "Block is nearby but I'm not holding it. Move closer and pick it up.",
                    "action": "forward"
                },
                {
                    "observation": "Agent: x=-1.2 | Block: x=-1.5 | Holding: False",
                    "reasoning": "I'm close enough to the block now. Pick it up.",
                    "action": "pick"
                },
                {
                    "observation": "Agent: x=2.0 | Block: held | Basket: x=6.0, y=1.5",
                    "reasoning": "I'm holding the block. Need to get closer before throwing.",
                    "action": "forward"
                },
                {
                    "observation": "Agent: x=4.0 | Block: held | Basket: x=6.0, y=1.5",
                    "reasoning": "Good distance from basket. Medium throw should work.",
                    "action": "throw_medium"
                }
            ]

    def _build_system_prompt(self) -> str:
        """Build system prompt with few-shot examples"""
        base_prompt = super()._build_system_prompt()

        if not self.examples:
            return base_prompt

        examples_text = "\n\nEXAMPLES:\n"
        for ex in self.examples:
            examples_text += f"\nObservation: {ex['observation']}\n"
            examples_text += f"Reasoning: {ex['reasoning']}\n"
            examples_text += f"Action: {ex['action']}\n"

        return base_prompt + examples_text
