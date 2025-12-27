
import json
import random

def extract_traces():
    # Find the latest commit-based file or fall back
    import glob
    files = glob.glob("experiment_data_archive/final_classified_results_N30_*.json")
    if not files:
        print("No N30 results found yet.")
        return

    # Pick the newest
    latest_file = max(files, key=lambda f: f)
    print(f"Extracting from: {latest_file}")
    
    with open(latest_file) as f:
        data = json.load(f)
        
    categories = {
        "SUCCESS": [],
        "UNSAT_WEDGED": [],
        "FAIL_INSTABILITY": [],
        "FAIL_POLICY": []
    }
    
    for entry in data:
        status = entry.get('status', 'UNKNOWN')
        if status in categories:
            categories[status].append(entry)
            
    # Select one of each
    selected_traces = {}
    for cat, items in categories.items():
        if items:
            # Prefer short/interesting ones?
            # For Success, maybe one with decent gain?
            # Just random is fine, or first.
            selected = items[0]
            # Prune to just trace + metadata
            clean = {
                "seed": selected['seed'],
                "status": selected['status'],
                "attempts": selected['attempts'],
                "trace": selected.get('trace', [])
            }
            selected_traces[cat] = clean
        else:
            print(f"Warning: No examples found for {cat}")

    # Save to file
    with open("experiment_data_archive/representative_traces_N30.json", "w") as f:
        json.dump(selected_traces, f, indent=2)
    print("Saved experiment_data_archive/representative_traces_N30.json")

if __name__ == "__main__":
    extract_traces()
