"""
LLM Baseline V3 Runner - Full Sequence Planning
================================================
LLM outputs COMPLETE action sequence for episode in ONE call.
Expected: exactly 1 LLM call per episode.
"""

import json
import time
import requests
from dataclasses import dataclass, asdict

from env_config_v2 import ENV_CONFIG
from config import ENV_SERVER_URL
from llm_policy_v3 import PolicyConfigV3, LLMPolicyV3

TASK_VERSION = "v2"


@dataclass
class V3EpisodeResult:
    episode_id: int
    success: bool
    steps: int
    reason: str
    final_x: float
    llm_calls: int
    planned_sequence: list[str]
    actions_taken: list[str]
    llm_latency: float
    trajectory: list[dict]
    reasoning: str


def run_v3_episode(
    policy: LLMPolicyV3,
    episode_id: int,
    gravity: float = ENV_CONFIG.training_gravity,
    max_steps: int = ENV_CONFIG.max_steps,
    verbose: bool = False
) -> V3EpisodeResult:
    """Run a single episode with the V3 full-sequence planning policy."""

    # Reset
    response = requests.post(
        f"{ENV_SERVER_URL}/reset",
        json={
            "task": "gap",
            "taskVersion": TASK_VERSION,
            "gravity": gravity
        },
        timeout=10
    )
    response.raise_for_status()
    obs = response.json().get("observation", response.json())

    policy.reset()

    # Get the full plan (single LLM call)
    start = time.time()
    plan_response = policy.plan_episode(obs)
    llm_latency = time.time() - start

    planned_sequence = plan_response.get("sequence", [])
    reasoning = plan_response.get("reasoning", "")

    if verbose:
        print(f"  Plan: {planned_sequence[:10]}... ({len(planned_sequence)} actions)")
        if reasoning:
            print(f"  Reasoning: {reasoning[:100]}...")

    actions_taken = []
    trajectory = []

    step = 0
    while step < max_steps:
        # Record state
        trajectory.append({
            "step": step,
            "x": obs["agentPosition"][0],
            "y": obs["agentPosition"][1],
            "vx": obs["agentVelocity"][0],
            "grounded": obs["isGrounded"]
        })

        # Get action from pre-planned sequence
        action, _ = policy.select_action(obs, "training")
        actions_taken.append(action)

        if verbose and (step < 3 or step % 5 == 0):
            x = obs["agentPosition"][0]
            vx = obs["agentVelocity"][0]
            print(f"  Step {step+1}: x={x:.2f}, vx={vx:.2f}, action={action}")

        # Step environment
        response = requests.post(
            f"{ENV_SERVER_URL}/step",
            json={"action": action},
            timeout=10
        )
        response.raise_for_status()
        result = response.json()

        obs = result.get("observation", {})
        done = result.get("done", False)
        info = result.get("info", {})

        step += 1

        if done:
            return V3EpisodeResult(
                episode_id=episode_id,
                success=info.get("success", False),
                steps=step,
                reason=info.get("reason", "unknown"),
                final_x=obs.get("agentPosition", [0, 0, 0])[0],
                llm_calls=1,
                planned_sequence=planned_sequence,
                actions_taken=actions_taken,
                llm_latency=llm_latency,
                trajectory=trajectory,
                reasoning=reasoning
            )

    # Timeout
    return V3EpisodeResult(
        episode_id=episode_id,
        success=False,
        steps=step,
        reason="timeout",
        final_x=obs.get("agentPosition", [0, 0, 0])[0],
        llm_calls=1,
        planned_sequence=planned_sequence,
        actions_taken=actions_taken,
        llm_latency=llm_latency,
        trajectory=trajectory,
        reasoning=reasoning
    )


def run_v3_baseline(
    num_episodes: int = 20,
    use_groq: bool = False,
    verbose: bool = True
) -> dict:
    """Run the V3 LLM baseline with full sequence planning."""

    print("=" * 70)
    print("LLM BASELINE V3 (Full Sequence Planning)")
    print("=" * 70)
    print(f"Agent: NRL-F with full sequence output")
    print(f"Model: {'Groq' if use_groq else 'Gemini 2.0 Flash'}")
    print(f"Episodes: {num_episodes}")
    print(f"Max steps: {ENV_CONFIG.max_steps}")
    print()

    # Check server
    try:
        response = requests.get(f"{ENV_SERVER_URL}/health", timeout=5)
        print(f"Server: {response.json()}")
    except Exception as e:
        print(f"ERROR: Cannot connect to server: {e}")
        return {}

    results = []
    successes = 0
    total_steps = 0
    total_llm_calls = 0
    total_latency = 0

    print(f"\nRunning {num_episodes} episodes...")
    print("-" * 70)

    for ep in range(num_episodes):
        # New policy per episode
        config = PolicyConfigV3(
            agent_type="NRL-F",
            task="gap",
            use_groq=use_groq,
            model="llama3-70b-8192" if use_groq else "gemini-2.0-flash"
        )
        policy = LLMPolicyV3(config)

        if verbose:
            print(f"\nEpisode {ep+1}/{num_episodes}")

        result = run_v3_episode(policy, ep, verbose=verbose)
        results.append(result)

        if result.success:
            successes += 1

        total_steps += result.steps
        total_llm_calls += result.llm_calls
        total_latency += result.llm_latency

        if verbose:
            status = "SUCCESS" if result.success else f"FAIL ({result.reason})"
            print(f"  -> {status} in {result.steps} steps")

        # Progress update every 5 episodes
        if (ep + 1) % 5 == 0:
            rate = successes / (ep + 1) * 100
            print(f"\n  Progress: {ep+1}/{num_episodes}, Success: {rate:.1f}%")

    # Summary
    success_rate = successes / num_episodes
    avg_steps = total_steps / num_episodes
    avg_latency = total_latency / total_llm_calls if total_llm_calls > 0 else 0

    success_results = [r for r in results if r.success]
    failure_results = [r for r in results if not r.success]

    failure_reasons = {}
    for r in failure_results:
        failure_reasons[r.reason] = failure_reasons.get(r.reason, 0) + 1

    # Analyze planned sequences
    jump_positions = []
    for r in results:
        if "jump" in r.planned_sequence:
            jump_idx = r.planned_sequence.index("jump")
            jump_positions.append(jump_idx)

    summary = {
        "agent": "NRL-F-V3",
        "model": "gemini-2.0-flash",
        "num_episodes": num_episodes,
        "successes": successes,
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "avg_latency_per_call": avg_latency,
        "total_llm_calls": total_llm_calls,
        "avg_steps_success": sum(r.steps for r in success_results) / len(success_results) if success_results else 0,
        "avg_steps_failure": sum(r.steps for r in failure_results) / len(failure_results) if failure_results else 0,
        "failure_reasons": failure_reasons,
        "avg_jump_position": sum(jump_positions) / len(jump_positions) if jump_positions else 0
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Success rate: {success_rate*100:.1f}%")
    print(f"Avg steps per episode: {avg_steps:.1f}")
    print(f"Total LLM calls: {total_llm_calls} (1 per episode)")
    print(f"Avg latency per call: {avg_latency:.2f}s")
    print(f"Failure reasons: {failure_reasons}")
    print(f"Avg jump position in sequence: {summary['avg_jump_position']:.1f} (optimal is ~6)")

    # Action distribution from planned sequences
    all_planned = []
    for r in results:
        all_planned.extend(r.planned_sequence)
    planned_counts = {}
    for a in all_planned:
        planned_counts[a] = planned_counts.get(a, 0) + 1
    print(f"\nPlanned action distribution: {planned_counts}")

    # Compare to baselines
    print("\n" + "-" * 70)
    print("COMPARISON TO BASELINES")
    print("-" * 70)
    print(f"Optimal:      100.0%")
    print(f"LLM V3:       {success_rate*100:>5.1f}%  (full sequence)")
    print(f"LLM V2:         0.0%  (single-tick)")
    print(f"Heuristic:     44.0%")
    print(f"Random:         0.0%")
    print()
    print(f"LLM calls: V2 ~340  ->  V3 {total_llm_calls}")

    # Save results
    output = {
        "summary": summary,
        "episodes": [asdict(r) for r in results]
    }
    with open("llm_v3_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to llm_v3_results.json")

    return summary


if __name__ == "__main__":
    run_v3_baseline(num_episodes=20, use_groq=False, verbose=True)
