
import numpy as np
import json
import time
from experiment_final_sweep import ExperimentRunner, ExperimentConfig
from config import GravityConfig

def run_determinism_check():
    print("=== STARTING DETERMINISM CHECK (SEED 2000 x 3) ===")
    
    config = ExperimentConfig(mode="adaptive", num_seeds=1)
    agent = ExperimentRunner(config)
    results = []
    
    for i in range(3):
        print(f"\n--- RUN {i+1} ---")
        obs = agent.env.reset(2000)
        
        start_agent_pos = obs["agentPosition"]
        start_basket_pos = obs["basketPosition"]
        
        print(f"Spawn Agent: {start_agent_pos}")
        print(f"Spawn Basket: {start_basket_pos}")
        
        # We will run a fixed sequence of actions to test physics determinism
        # consistently, rather than the LLM which might have temperature=0 but still...
        # Actually, let's run the actual agent logic (LLM should be temp=0 deterministic).
        # We want to see if the *Outcome* is deterministic.
        
        # We'll just run the agent.run_seed(2000) method
        # logic is inside run_seed
        
        # Wait, run_seed creates its own reset. 
        # We will instantiate a new agent or just call run_seed?
        # run_seed calls env.reset(seed).
        
        # Capture stdout? Or just let it print?
        # We rely on the logs printed by run_seed.
        
        # To strictly verify, we should maybe hash the trajectory?
        # But for now, let's just see if they all fail in the same way or succeed.
        
        res = agent.run_seed(2000)
        results.append(res)
        
    print("\n=== SUMMARY ===")
    for i, r in enumerate(results):
        print(f"Run {i+1}: {r}")

if __name__ == "__main__":
    run_determinism_check()
