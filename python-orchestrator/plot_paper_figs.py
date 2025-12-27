
import matplotlib.pyplot as plt
import json
import os
import numpy as np

def plot_success_comparsion():
    # Data manually aggregated from our known results
    conditions = ['Baseline (N=30, Serial)', 'Stress Test (N=100, Parallel)']
    success_rates = [85.0, 0.0]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(conditions, success_rates, color=['green', 'red'])
    plt.title('Success Rate: Serial vs Parallel Execution')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 100)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom')
                 
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('notebook_lm_import/Fig1_Success_Comparison.png')
    plt.close()

def plot_calibration_curve():
    try:
        with open('experiment_data_archive/physics_sweep_results.json') as f:
            data = json.load(f)
    except: return

    # Handle {"Task_A_1g": [...]}
    if "Task_A_1g" in data:
        items = data["Task_A_1g"]
        items.sort(key=lambda x: x.get('jump_step', 0))
        
        steps = [x['jump_step'] for x in items]
        observed = [x['avg_landed_x'] for x in items]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, observed, 'b-o', linewidth=2, label='Observed Jump Distance')
        
        plt.xlabel('Jump Step Parameter')
        plt.ylabel('Average Landed Distance (m)')
        plt.title('System Identification: Jump Strength Response')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('notebook_lm_import/Fig2_SystemID_Response.png')
        plt.close()

if __name__ == "__main__":
    os.makedirs('notebook_lm_import', exist_ok=True)
    plot_success_comparsion()
    plot_calibration_curve()
    print("Generated Figures.")
