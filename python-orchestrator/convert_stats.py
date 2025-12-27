
import json
import os
import glob
import numpy as np

def json_to_md_table(data, filename, title, description):
    """Converts a list of result objects into a Markdown table."""
    md = f"# {title}\n\n{description}\n\n"
    
    # Calculate Summary Stats
    total = len(data)
    if total == 0:
        md += "No data found.\n"
        with open(filename, 'w') as f: f.write(md)
        return

    successes = [d for d in data if d.get('success', False)]
    success_rate = (len(successes) / total) * 100
    
    success_attempts = [d['attempts'] for d in successes]
    p50 = np.median(success_attempts) if success_attempts else 0
    p90 = np.percentile(success_attempts, 90) if success_attempts else 0
    
    # Failure Breakdown
    statuses = [d.get('status', 'UNKNOWN') for d in data]
    status_counts = {s: statuses.count(s) for s in set(statuses)}
    
    md += "## Executive Summary\n"
    md += f"- **Total Episodes**: {total}\n"
    md += f"- **Success Rate**: {success_rate:.1f}%\n"
    if success_attempts:
        md += f"- **Attempts (Success P50)**: {p50:.1f}\n"
        md += f"- **Attempts (Success P90)**: {p90:.1f}\n"
    
    md += "\n### Status Breakdown\n"
    for s, c in status_counts.items():
        md += f"- **{s}**: {c} ({c/total*100:.1f}%)\n"
        
    md += "\n## Episode Log\n"
    md += "| Seed | Status | Attempts | Final Dist | Stuck Events | Notes |\n"
    md += "|---|---|---|---|---|---|\n"
    
    for d in data:
        seed = d.get('seed', 'N/A')
        status = d.get('status', 'SUCCESS' if d.get('success') else 'FAIL')
        attempts = d.get('attempts', 0)
        final_dist = d.get('final_dist_from_basket', 0.0)
        stuck = d.get('stuck_events', 0)
        
        note = ""
        if status == "UNSAT_WEDGED":
            note = "4-way probe failed"
        elif status == "FAIL_INSTABILITY":
            note = "Teleport > 0.8m"
            
        md += f"| {seed} | {status} | {attempts} | {final_dist:.2f}m | {stuck} | {note} |\n"
        
    with open(filename, 'w') as f:
        f.write(md)
    print(f"Generated {filename}")

def convert_calibration(json_path, out_path):
    """Converts physics sweep data (Jump Steps) to MD."""
    if not os.path.exists(json_path): return
    
    with open(json_path) as f:
        data = json.load(f)
    
    # Structure: {"Task_A_1g": [ {jump_step, avg_landed_x, ...} ]}
    md = "# System ID: Actuation Calibration Data (Jump Impulse)\n\n"
    md += "This dataset correlates 'Jump Step' (Impulse Duration) with 'Landed Distance' (Output).\n"
    md += "It demonstrates the linearity of the physics engine for high-force actions.\n\n"
    
    md += "## Calibration Table\n"
    md += "| Jump Step (Impulse) | Average Distance (m) | Success Rate |\n"
    md += "|---|---|---|\n"
    
    # Handle the specific structure
    if "Task_A_1g" in data:
        items = data["Task_A_1g"]
        # Sort by step
        items.sort(key=lambda x: x.get('jump_step', 0))
        
        for item in items:
            step = item.get('jump_step')
            dist = item.get('avg_landed_x', 0)
            succ = item.get('successes', 0)
            trials = len(item.get('trials', []))
            rate = succ/trials if trials > 0 else 0
            
            md += f"| {step} | {dist:.4f}m | {rate*100:.0f}% |\n"
            
    with open(out_path, 'w') as f: f.write(md)
    print(f"Generated {out_path}")

# --- execution ---
if __name__ == "__main__":
    OUT_DIR = "notebook_lm_import"
    DATA_DIR = "experiment_data_archive"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. N=100 Failure
    try:
        with open(f"{DATA_DIR}/final_classified_results_N100.json") as f:
            n100 = json.load(f)
        json_to_md_table(n100, f"{OUT_DIR}/Report_StressTest_N100_Parallel.md", 
                        "Experiment Report: N=100 Stress Test (Parallel)", 
                        "Results of running 100 seeds in parallel. Result: 0% SUCCESS due to event loop starvation.")
    except Exception as e: print(f"Skipping N100: {e}")

    # 2. Adaptive Gain (Successes)
    try:
        with open(f"{DATA_DIR}/experiment_v2_results.json") as f:
            v2 = json.load(f)
            
            # v2 is { condition_name: { summary: {}, episodes: [] } }
            all_episodes = []
            for cond, content in v2.items():
                if isinstance(content, dict) and 'episodes' in content:
                    # Enrich with condition name
                    eps = content['episodes']
                    for e in eps: e['condition'] = cond
                    all_episodes.extend(eps)
            
            # Filter for a sample (First 50)
            v2_sample = all_episodes[:50] 
            
            # Adapt schema if needed (v2 has 'episode_id' not 'seed' sometimes)
            # Normalizing keys for the table generator
            for e in v2_sample:
                if 'episode_id' in e and 'seed' not in e: e['seed'] = e['episode_id']
                # status might be missing, assume SUCCESS if success=True
                if 'status' not in e: e['status'] = "SUCCESS" if e.get('success') else "FAIL"
                
            json_to_md_table(v2_sample, f"{OUT_DIR}/Report_Algorithm_Performance.md",
                             "Experiment Report: Adaptive Gain Algorithm",
                             "Results from the development phase showing the effectiveness of the Adaptive Gain Controller.")
    except Exception as e:
        print(f"Skipping V2: {e}")

    # 3. Calibration
    convert_calibration(f"{DATA_DIR}/physics_sweep_results.json", f"{OUT_DIR}/Report_Physics_Calibration.md")

