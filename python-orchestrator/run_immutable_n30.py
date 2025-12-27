
import sys
import json
import concurrent.futures
from experiment_final_sweep import ExperimentConfig, ExperimentRunner
from config import GEMINI_API_KEY # Ensure config is available

# Force Serial Execution (No ThreadPool, just loop)
if __name__ == "__main__":
    print("--- Running IMMUTABLE Final Validation Sweep (N=30) [SERIAL] ---")
    
    # Standard Benchmark Range: 2000-2029 (30 Seeds)
    seeds = list(range(2000, 2030))
    
    results = []
    counts = {"SUCCESS": 0, "UNSAT_WEDGED": 0, "FAIL_POLICY": 0, "FAIL_INSTABILITY": 0}
    attempts_success = []
    
    for seed in seeds:
        print(f"\n[SERIAL] Starting Seed {seed}...")
        try:
            # Create runner (Session ID irrelevant for Serial but good hygiene)
            runner = ExperimentRunner(ExperimentConfig(mode="adaptive", num_seeds=1), session_id=f"serial_{seed}")
            
            res = runner.run_seed(seed)
            if not res: 
                print(f"SEED {seed} FAILED TO START (No Observation)")
                continue

            # Normalize Status
            status = res.get('status', 'SUCCESS' if res['success'] else 'FAIL_POLICY')
            if status == "FAIL (Policy)": status = "FAIL_POLICY"
            
            res['status'] = status
            
            print(f"SEED {seed}: {status} ({res.get('attempts',0)} steps)")
            if status not in counts: counts[status] = 0
            counts[status] += 1
            results.append(res)
            if status == "SUCCESS":
                attempts_success.append(res.get('attempts',0))
                
        except Exception as e:
            print(f"SEED {seed} CRASHED: {e}")

    # Sorting results by seed for clean JSON
    results.sort(key=lambda x: x['seed'])

    print("\n--- SWEEP RESULTS (IMMUTABLE N=30) ---")
    for k, v in counts.items():
        print(f"{k}: {v}/{len(seeds)} ({v/len(seeds)*100:.1f}%)")
    
    if attempts_success:
        import statistics
        p50 = statistics.median(attempts_success)
        attempts_success.sort()
        p90_idx = int(len(attempts_success) * 0.9)
        p90 = attempts_success[min(p90_idx, len(attempts_success)-1)]
        print(f"Attempts (Correct): P50={p50}, P90={p90}")
    
    # Save to the specific requested filename structure
    # We will use a placeholder 'commit_current' since we don't have git sha in python easily available without subprocess
    # But I can fetch it via subprocess if needed.
    import subprocess
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except:
        commit_hash = "unknown"
        
    filename = f"experiment_data_archive/final_classified_results_N30_commit{commit_hash}.json"
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved Immutable Results by Commit to: {filename}")
