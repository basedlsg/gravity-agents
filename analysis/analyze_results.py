#!/usr/bin/env python3
"""
Analysis script for Gravity Agents experiments

Generates plots and statistics for hypothesis testing:
- H1: Law vs Story
- H2: RL vs No-RL
- H3: Information vs Silence
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_all_experiments(log_dir: str = "./logs") -> pd.DataFrame:
    """Load all experiment data into a single DataFrame"""
    log_path = Path(log_dir)
    all_episodes = []

    for exp_dir in log_path.iterdir():
        if not exp_dir.is_dir():
            continue

        parquet_file = exp_dir / "episodes.parquet"
        if parquet_file.exists():
            df = pd.read_parquet(parquet_file)
            df["experiment"] = exp_dir.name
            all_episodes.append(df)

    if not all_episodes:
        return pd.DataFrame()

    return pd.concat(all_episodes, ignore_index=True)


def compute_success_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute success rates grouped by agent, task, condition"""
    grouped = df.groupby(["agent_type", "task", "condition"]).agg({
        "success": ["mean", "std", "count"],
        "num_steps": ["mean", "std"]
    }).reset_index()

    grouped.columns = [
        "agent_type", "task", "condition",
        "success_rate", "success_std", "n_episodes",
        "avg_steps", "steps_std"
    ]

    # Compute 95% confidence intervals
    grouped["success_ci"] = 1.96 * grouped["success_std"] / np.sqrt(grouped["n_episodes"])

    return grouped


def plot_success_rates_by_condition(df: pd.DataFrame, output_dir: str = "./plots"):
    """Plot success rates comparing conditions"""
    os.makedirs(output_dir, exist_ok=True)

    summary = compute_success_rates(df)

    for task in ["gap", "throw"]:
        task_data = summary[summary["task"] == task]

        fig, ax = plt.subplots(figsize=(10, 6))

        agents = ["RL-F", "RL-N", "NRL-F"]
        conditions = ["baseline", "silent", "explained"]
        x = np.arange(len(agents))
        width = 0.25

        colors = {"baseline": "#2ecc71", "silent": "#e74c3c", "explained": "#3498db"}

        for i, condition in enumerate(conditions):
            cond_data = task_data[task_data["condition"] == condition]
            # Align data with agents order
            rates = []
            errors = []
            for agent in agents:
                agent_data = cond_data[cond_data["agent_type"] == agent]
                if len(agent_data) > 0:
                    rates.append(agent_data["success_rate"].values[0])
                    errors.append(agent_data["success_ci"].values[0])
                else:
                    rates.append(0)
                    errors.append(0)

            bars = ax.bar(
                x + i * width,
                rates,
                width,
                label=condition.capitalize(),
                color=colors[condition],
                yerr=errors,
                capsize=3
            )

        ax.set_xlabel("Agent Type")
        ax.set_ylabel("Success Rate")
        ax.set_title(f"Success Rates - {task.capitalize()} Task")
        ax.set_xticks(x + width)
        ax.set_xticklabels(agents)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/success_rates_{task}.png", dpi=150)
        plt.close()

    print(f"Success rate plots saved to {output_dir}")


def plot_behavioral_metrics(df: pd.DataFrame, output_dir: str = "./plots"):
    """Plot behavioral metrics (jumps, throws, etc.)"""
    os.makedirs(output_dir, exist_ok=True)

    # Gap task: Jump analysis
    gap_df = df[df["task"] == "gap"]
    if len(gap_df) > 0 and "metric_num_jumps" in gap_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Jumps per episode
        sns.boxplot(
            data=gap_df,
            x="agent_type",
            y="metric_num_jumps",
            hue="condition",
            ax=axes[0]
        )
        axes[0].set_title("Jumps per Episode - Gap Task")
        axes[0].set_ylabel("Number of Jumps")

        # First jump position
        if "metric_first_jump_x" in gap_df.columns:
            gap_with_jumps = gap_df[gap_df["metric_num_jumps"] > 0]
            sns.boxplot(
                data=gap_with_jumps,
                x="agent_type",
                y="metric_first_jump_x",
                hue="condition",
                ax=axes[1]
            )
            axes[1].set_title("First Jump Position (X) - Gap Task")
            axes[1].axhline(y=2.5, color="red", linestyle="--", label="Gap Start")
            axes[1].legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/behavioral_gap.png", dpi=150)
        plt.close()

    # Throw task: Throw analysis
    throw_df = df[df["task"] == "throw"]
    if len(throw_df) > 0 and "metric_num_throws" in throw_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.boxplot(
            data=throw_df,
            x="agent_type",
            y="metric_num_throws",
            hue="condition",
            ax=ax
        )
        ax.set_title("Throws per Episode - Throw Task")
        ax.set_ylabel("Number of Throws")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/behavioral_throw.png", dpi=150)
        plt.close()

    print(f"Behavioral plots saved to {output_dir}")


def test_hypothesis_h1(df: pd.DataFrame) -> Dict:
    """
    H1 - Law vs Story:
    Agents with explicit physical law adapt better than story-based agents
    Compare RL-F vs RL-N at 0.5g explained condition
    """
    results = {}

    for task in ["gap", "throw"]:
        task_df = df[(df["task"] == task) & (df["condition"] == "explained")]

        rl_f = task_df[task_df["agent_type"] == "RL-F"]["success"].values
        rl_n = task_df[task_df["agent_type"] == "RL-N"]["success"].values

        if len(rl_f) > 0 and len(rl_n) > 0:
            # Fisher's exact test or chi-squared
            contingency = [
                [sum(rl_f), len(rl_f) - sum(rl_f)],
                [sum(rl_n), len(rl_n) - sum(rl_n)]
            ]
            _, p_value = stats.fisher_exact(contingency)

            results[task] = {
                "rl_f_success_rate": np.mean(rl_f),
                "rl_n_success_rate": np.mean(rl_n),
                "difference": np.mean(rl_f) - np.mean(rl_n),
                "p_value": p_value,
                "significant": p_value < 0.05,
                "n_rl_f": len(rl_f),
                "n_rl_n": len(rl_n)
            }

    return {"H1_law_vs_story": results}


def test_hypothesis_h2(df: pd.DataFrame) -> Dict:
    """
    H2 - RL vs No-RL:
    RL-trained agents adapt more robustly than non-learning LLM controller
    Compare RL-F vs NRL-F at 0.5g explained condition
    """
    results = {}

    for task in ["gap", "throw"]:
        task_df = df[(df["task"] == task) & (df["condition"] == "explained")]

        rl_f = task_df[task_df["agent_type"] == "RL-F"]["success"].values
        nrl_f = task_df[task_df["agent_type"] == "NRL-F"]["success"].values

        if len(rl_f) > 0 and len(nrl_f) > 0:
            contingency = [
                [sum(rl_f), len(rl_f) - sum(rl_f)],
                [sum(nrl_f), len(nrl_f) - sum(nrl_f)]
            ]
            _, p_value = stats.fisher_exact(contingency)

            results[task] = {
                "rl_f_success_rate": np.mean(rl_f),
                "nrl_f_success_rate": np.mean(nrl_f),
                "difference": np.mean(rl_f) - np.mean(nrl_f),
                "p_value": p_value,
                "significant": p_value < 0.05,
                "n_rl_f": len(rl_f),
                "n_nrl_f": len(nrl_f)
            }

    return {"H2_rl_vs_no_rl": results}


def test_hypothesis_h3(df: pd.DataFrame) -> Dict:
    """
    H3 - Information vs Silence:
    Agents given updated gravity description adapt better than silent change
    Compare explained vs silent condition within each agent type
    """
    results = {}

    for agent_type in ["RL-F", "RL-N", "NRL-F"]:
        agent_results = {}

        for task in ["gap", "throw"]:
            agent_df = df[(df["agent_type"] == agent_type) & (df["task"] == task)]

            silent = agent_df[agent_df["condition"] == "silent"]["success"].values
            explained = agent_df[agent_df["condition"] == "explained"]["success"].values

            if len(silent) > 0 and len(explained) > 0:
                contingency = [
                    [sum(explained), len(explained) - sum(explained)],
                    [sum(silent), len(silent) - sum(silent)]
                ]
                _, p_value = stats.fisher_exact(contingency)

                agent_results[task] = {
                    "silent_success_rate": np.mean(silent),
                    "explained_success_rate": np.mean(explained),
                    "difference": np.mean(explained) - np.mean(silent),
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }

        results[agent_type] = agent_results

    return {"H3_information_vs_silence": results}


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary table for all conditions"""
    summary = compute_success_rates(df)

    # Pivot for nice display
    pivot = summary.pivot_table(
        index=["agent_type", "task"],
        columns="condition",
        values="success_rate",
        aggfunc="first"
    ).round(3)

    return pivot


def run_analysis(log_dir: str = "./logs", output_dir: str = "./plots"):
    """Run full analysis pipeline"""
    print("Loading experiment data...")
    df = load_all_experiments(log_dir)

    if len(df) == 0:
        print("No experiment data found!")
        return

    print(f"Loaded {len(df)} episodes")
    print(f"Agents: {df['agent_type'].unique()}")
    print(f"Tasks: {df['task'].unique()}")
    print(f"Conditions: {df['condition'].unique()}")

    # Generate plots
    print("\nGenerating plots...")
    plot_success_rates_by_condition(df, output_dir)
    plot_behavioral_metrics(df, output_dir)

    # Run hypothesis tests
    print("\nRunning hypothesis tests...")

    h1_results = test_hypothesis_h1(df)
    h2_results = test_hypothesis_h2(df)
    h3_results = test_hypothesis_h3(df)

    all_results = {**h1_results, **h2_results, **h3_results}

    # Print results
    print("\n" + "="*60)
    print("HYPOTHESIS TEST RESULTS")
    print("="*60)

    for hypothesis, data in all_results.items():
        print(f"\n{hypothesis}:")
        print(json.dumps(data, indent=2, default=str))

    # Save results
    with open(f"{output_dir}/hypothesis_tests.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary table
    print("\n" + "="*60)
    print("SUCCESS RATE SUMMARY")
    print("="*60)

    summary = generate_summary_table(df)
    print(summary.to_string())

    summary.to_csv(f"{output_dir}/summary_table.csv")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Gravity Agents experiment results")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--output-dir", type=str, default="./plots", help="Output directory for plots")

    args = parser.parse_args()

    run_analysis(args.log_dir, args.output_dir)
