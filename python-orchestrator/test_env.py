#!/usr/bin/env python3
"""
Quick test to verify environment setup
"""

import sys
from env_client import GravityEnvClient, MockGravityEnvClient


def test_mock_client():
    """Test with mock client (no server needed)"""
    print("Testing mock client...")

    client = MockGravityEnvClient()

    # Test reset
    obs = client.reset(task="gap", gravity=9.81, seed=42)
    assert "agentPosition" in obs
    assert "gravity" in obs
    print(f"  Reset OK: agent at {obs['agentPosition']}")

    # Test step
    result = client.step("forward")
    assert result.observation is not None
    print(f"  Step OK: done={result.done}, reward={result.reward}")

    print("Mock client tests passed!")
    return True


def test_real_client(server_url: str = "http://localhost:3000"):
    """Test with real server"""
    print(f"\nTesting real client at {server_url}...")

    client = GravityEnvClient(server_url)

    # Health check
    if not client.health_check():
        print("  Server not running - skipping real client tests")
        return False

    print("  Server is healthy")

    # Test gap task
    print("\n  Testing Gap task...")
    obs = client.reset(task="gap", gravity=9.81, seed=42)
    print(f"    Initial position: {obs['agentPosition']}")
    print(f"    Gap: {obs['gapStart']:.1f} to {obs['gapEnd']:.1f}")

    # Take a few steps
    for action in ["forward", "forward", "forward", "jump", "forward"]:
        result = client.step(action)
        print(f"    {action}: pos={result.observation['agentPosition'][0]:.2f}, done={result.done}")
        if result.done:
            break

    # Test throw task
    print("\n  Testing Throw task...")
    obs = client.reset(task="throw", gravity=9.81, seed=42)
    print(f"    Agent: {obs['agentPosition']}")
    print(f"    Block: {obs['blockPosition']}")
    print(f"    Basket: {obs['basketPosition']}")

    # Take a few steps
    for action in ["forward", "pick", "forward", "forward", "throw_medium"]:
        result = client.step(action)
        print(f"    {action}: holding={result.observation.get('holdingBlock', '?')}, done={result.done}")
        if result.done:
            break

    # Test gravity change
    print("\n  Testing 0.5g gravity...")
    obs = client.reset(task="gap", gravity=4.9, seed=42)
    print(f"    Gravity: {obs['gravity']}")

    print("\nReal client tests passed!")
    return True


def test_llm_policy():
    """Test LLM policy (requires API key)"""
    print("\nTesting LLM policy...")

    try:
        from llm_policy import LLMPolicy, PolicyConfig

        config = PolicyConfig(
            agent_type="RL-F",
            task="gap",
            model="gemini-1.5-flash"
        )

        policy = LLMPolicy(config)

        # Test with mock observation
        obs = {
            "agentPosition": [0, 1, 0],
            "agentVelocity": [0, 0, 0],
            "gapStart": 2.5,
            "gapEnd": 5.5,
            "gapWidth": 3.0,
            "goalZone": {"minX": 6.0, "maxX": 9.0},
            "gravity": 9.81,
            "isGrounded": True
        }

        action = policy.select_action(obs, "training")
        print(f"  Selected action: {action}")

        print("LLM policy test passed!")
        return True

    except Exception as e:
        print(f"  LLM policy test failed: {e}")
        return False


def main():
    print("="*50)
    print("Gravity Agents - Environment Tests")
    print("="*50)

    results = []

    # Always test mock
    results.append(("Mock Client", test_mock_client()))

    # Test real server if available
    results.append(("Real Client", test_real_client()))

    # Test LLM if requested
    if "--with-llm" in sys.argv:
        results.append(("LLM Policy", test_llm_policy()))

    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)

    for name, passed in results:
        status = "PASSED" if passed else "SKIPPED/FAILED"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results if r[1] is not None)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
