"""
Environment Configuration V2 - LOCKED
======================================
DO NOT MODIFY during experiments.

VALIDATED BASELINE RESULTS (Dec 2024):
- Optimal agent: 100% success
- Heuristic agent: 44% success (random threshold 0-1.2m)
- Random agent: 0% success

Physics Settings:
- friction = 0.01 (very low, game-like movement)
- movement = velocity applied every physics tick while action held
- linearDamping = 0 on agent body

Gap Task Geometry:
- Platform A: x = -2.0 to 2.0 (width 4.0m)
- Gap: x = 2.0 to ~6.5 (width 4.6m +/- 0.1m variance)
- Platform B: starts at gap end
- Goal zone: x = 7.3 to 9.7, z = -0.7 to 0.7
- Agent start: x = -1.33

Agent Parameters:
- moveSpeed = 3.0 m/s
- maxSpeed = 4.0 m/s (can reach via air control during jump)
- jumpHeight = 1.2m
- airControl = 0.3 (allows velocity gain during flight)
- runUpBonus = 1.3 (30% bonus with momentum)

Max Jump Distance: ~5.7m (with full air control)

Optimal Policy:
- 6x forward (build momentum, reach x ≈ 1.7)
- 1x jump (just before gap edge)
- forward until goal (land on platform B, walk to goal zone)
- Expected steps: ~19

Episode Limits:
- max_steps = 80 (sufficient for ~4 attempts)
- timeout triggers failure
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EnvConfigV2:
    """Immutable environment configuration"""

    # Physics
    friction: float = 0.01
    time_step: float = 1/60
    physics_ticks_per_step: int = 10

    # Gap task geometry
    platform_width: float = 4.0
    gap_width: float = 4.5
    gap_variance: float = 0.3
    gap_start: float = 2.0  # platform_width / 2

    # Agent
    agent_start_x: float = -1.33  # -platform_width / 3
    move_speed: float = 3.0
    max_speed: float = 4.0
    jump_height: float = 1.2
    air_control: float = 0.3
    run_up_bonus: float = 1.3

    # Episode
    max_steps: int = 80

    # Gravity
    training_gravity: float = 9.81
    test_gravity: float = 4.905  # 0.5g


@dataclass
class BaselineResults:
    """Results from baseline agent runs"""
    agent_name: str
    num_episodes: int
    successes: int
    failures: int
    success_rate: float
    avg_steps_to_success: float
    avg_steps_to_failure: float
    failure_reasons: dict


# Singleton config instance
ENV_CONFIG = EnvConfigV2()


# Optimal action sequence (validated for 5.0m gap)
# Need 6 forwards to get closer to edge before jumping
OPTIMAL_SEQUENCE = ['forward'] * 6 + ['jump'] + ['forward'] * 73  # 80 total


def get_heuristic_action(obs: dict, state: dict = None) -> str:
    """
    Dumb heuristic with randomness: move forward, jump once somewhere near gap.

    The jump threshold is randomly chosen per episode between 0.0 and 1.2m
    before the gap edge. This introduces variance in success rates.

    Args:
        obs: Environment observation
        state: Mutable dict to track jumped and threshold

    Returns:
        Action string
    """
    import random

    # Initialize state on first call
    if state is None:
        state = {}
    if 'jumped' not in state:
        state['jumped'] = False
        # Random threshold: jump somewhere between gap_start-1.2 and gap_start
        # Optimal is around gap_start - 0.3 to gap_start - 0.1
        # So this will sometimes be too early (fail) or just right (succeed)
        state['threshold'] = random.uniform(0.0, 1.2)

    # Store state for future calls (via default argument hack)
    get_heuristic_action.__defaults__ = (state,)

    x = obs['agentPosition'][0]
    gap_start = obs['gapStart']
    is_grounded = obs['isGrounded']

    # Jump once when at threshold
    if is_grounded and not state['jumped'] and x >= gap_start - state['threshold']:
        state['jumped'] = True
        return 'jump'

    return 'forward'


def reset_heuristic_state():
    """Reset the heuristic's state between episodes."""
    get_heuristic_action.__defaults__ = (None,)


def get_random_action() -> str:
    """
    Random agent: uniform sample from action subset.

    We exclude 'left' and 'right' since they don't help with gap crossing
    and would just add noise to the baseline.
    """
    import random
    return random.choice(['forward', 'idle', 'jump', 'back'])


def get_optimal_action(step: int) -> str:
    """
    Optimal scripted agent.

    Args:
        step: Current step number (0-indexed)

    Returns:
        Action string
    """
    if step < len(OPTIMAL_SEQUENCE):
        return OPTIMAL_SEQUENCE[step]
    return 'forward'  # Keep going if somehow we need more steps
