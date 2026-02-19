import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    "font.size": 32,
})


HUMAN_READABLE = {
    "actor_loss": "Actor Loss",
    "value_loss": "Value Loss",
    "total_loss": "Total Loss",

    "entropy": "Policy Entropy",

    "return_mean": "Mean Return",
    "return_std": "Return Standard Deviation",
    "return_max": "Maximum Return",
    "return_min": "Minimum Return",

    "disc_return_mean": "Mean Discounted Return",

    "ep_len_mean": "Mean Episode Length",
    "ep_len_std": "Episode Length Standard Deviation",
    "ep_len_max": "Maximum Episode Length",
    "ep_len_min": "Minimum Episode Length",

    "prob_success": "Probability of Success",
    "prob_fail": "Probability of Failure",

    "fps": "Frames Per Second",
}

# Parse arguments: accept multiple log directories
parser = argparse.ArgumentParser(description="Plot mean and std from multiple log CSVs across different log directories.")
parser.add_argument("--n", type=int, required=True, help="Number of sampled in the dataset (e.g., 100)")
parser.add_argument("--log-dirs", nargs="+", help="One or more directories containing log files", required=True)
parser.add_argument("--save-dir", help="Directory to save plots", default="plots/")
parser.add_argument("--names", nargs="+", help="Custom names for each log directory (must match number of log-dirs)")
parser.add_argument("--colors", nargs="+", help="Hex colors for each log directory (must match number of log-dirs). E.g., #FF0000 #00FF00 #0000FF")

args = parser.parse_args()
log_dirs = args.log_dirs
save_dir = args.save_dir
custom_names = args.names or []
custom_colors = args.colors or []

# Validate that if names/colors are provided, they match the number of log dirs
if custom_names and len(custom_names) != len(log_dirs):
    raise ValueError(f"Number of names ({len(custom_names)}) must match number of log-dirs ({len(log_dirs)})")
if custom_colors and len(custom_colors) != len(log_dirs):
    raise ValueError(f"Number of colors ({len(custom_colors)}) must match number of log-dirs ({len(log_dirs)})")

# Dictionary to hold results for each log directory + task combination
all_results = {}

for log_dir in log_dirs:
    if not os.path.isdir(log_dir):
        print(f"Warning: Provided path is not a directory: {log_dir}")
        continue

    # Determine tasks by inspecting the provided directory. If it contains subdirectories
    # with names containing 'reach' or 'reach_avoid' we use those; otherwise if the
    # directory directly contains CSVs we treat it as a single task.
    tasks = {}
    subdirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    for d in subdirs:
        low = d.lower()
        if "reach_avoid" in low or "reachavoid" in low:
            tasks["ReachAvoid"] = os.path.join(log_dir, d, "log_*.csv")
        elif "reach" in low:
            tasks["Reach"] = os.path.join(log_dir, d, "log_*.csv")

    # If no matching subdirectories, look for CSVs directly inside log_dir and treat
    # the directory basename as the task name.
    if not tasks:
        pattern = os.path.join(log_dir, "*.csv")
        if glob.glob(pattern):
            tasks[os.path.basename(os.path.abspath(log_dir))] = pattern
        else:
            print(f"Warning: No log files or task subdirectories found in {log_dir}")
            continue

    # Dictionary to hold mean/std for each task in this log_dir
    results = {}

    for task, pattern in tasks.items():
        log_files = glob.glob(pattern)
        if not log_files:
            print(f"Warning: No files found for task={task} with pattern {pattern}")
            continue

        dfs = [pd.read_csv(f) for f in log_files]

        # ensure unique timesteps per df (take mean if duplicates exist)
        for i in range(len(dfs)):
            if dfs[i]["timestep"].duplicated().any():
                dfs[i] = dfs[i].groupby("timestep", as_index=False).mean()

        # union of all timesteps
        all_timesteps = sorted(set().union(*[df["timestep"].values for df in dfs]))

        # reindex each df to have the full timestep range
        reindexed_dfs = []
        for df in dfs:
            df = df.set_index("timestep").reindex(all_timesteps)
            reindexed_dfs.append(df.reset_index().rename(columns={"index": "timestep"}))

        timesteps = np.array(all_timesteps)
        base_columns = [c for c in dfs[0].columns if c != "timestep"]

        data_mean, data_std = {}, {}
        for col in base_columns:
            values = np.stack([df[col].values for df in reindexed_dfs], axis=1)  # (T, num_seeds)
            data_mean[col] = np.nanmean(values, axis=1)
            data_std[col] = np.nanstd(values, axis=1)

        results[task] = {
            "timesteps": timesteps,
            "mean": data_mean,
            "std": data_std,
            "columns": base_columns,
        }

    # Store results keyed by (log_dir, task)
    for task, result in results.items():
        all_results[(log_dir, task)] = result

os.makedirs(save_dir, exist_ok=True)

# Create a mapping of log_dir to custom name and color
dir_to_custom_name = {}
dir_to_custom_color = {}
for i, log_dir in enumerate(log_dirs):
    if custom_names:
        dir_to_custom_name[log_dir] = custom_names[i]
    if custom_colors:
        dir_to_custom_color[log_dir] = custom_colors[i]

# Consistent colors and markers for each (log_dir, task) pair
default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
colors = {}
for i, key in enumerate(sorted(all_results.keys())):
    log_dir, task = key
    if log_dir in dir_to_custom_color:
        colors[key] = dir_to_custom_color[log_dir]
    else:
        # Find the index of this log_dir to cycle through default colors
        dir_index = log_dirs.index(log_dir)
        colors[key] = default_colors[dir_index % len(default_colors)]

# Get all unique columns across all results
all_columns = set()
for result in all_results.values():
    all_columns.update(result["columns"])
all_columns = sorted(all_columns)

# Plot each column with all (log_dir, task) series
for col in all_columns:
    plt.figure(figsize=(12, 8))
    # Iterate through log_dirs in provided order to maintain legend order
    for log_dir in log_dirs:
        for task in sorted([t for (ld, t) in all_results if ld == log_dir]):
            key = (log_dir, task)
            if key not in all_results:
                continue
            result = all_results[key]
            if col not in result["columns"]:
                continue
            timesteps = result["timesteps"]
            mean = result["mean"][col]
            std = result["std"][col]
            
            # Create label from custom name or log_dir basename
            if log_dir in dir_to_custom_name:
                label = dir_to_custom_name[log_dir]
            else:
                label = os.path.basename(os.path.abspath(log_dir))
            
            plt.plot(
                timesteps,
                mean,
                label=label,
                color=colors[key],
                linewidth=3,
            )

            plt.fill_between(
                timesteps,
                mean - std,
                mean + std,
                color=colors[key],
                alpha=0.20,
            )


    plt.xlabel("Timestep")

    pretty_col = HUMAN_READABLE.get(col, col.replace("_", " ").title())
    plt.ylabel(pretty_col)
    plt.title(f"{args.n} " + r"$\mathtt{ReachAvoid}$ Tasks in $\mathtt{TokenEnv}$")

    if "prob_" in col:
        plt.ylim(0.45, 1.05)
    # plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=len(log_dirs), frameon=True, fancybox=True, shadow=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    pdf_path = os.path.join(save_dir, f"{col}.pdf")
    plt.savefig(pdf_path)
    plt.close()

print(f"✅ Plots saved in {save_dir}")
