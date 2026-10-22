"""Gradient-norm statistics and Fig. 3 from the train.log files of several runs.

The logged norm is that of the PROPOSAL network's gradient, measured before
clipping (train.py calls GradientTracker.update before clip_grad_norm_), once
per batch; per epoch the log holds its mean and within-epoch std.

    python analysis/read_gradient_norms.py                 # Table 3, Fig. 3 (mean over runs)
    python analysis/read_gradient_norms.py --run 5         # Fig. 3 for run 5 of every method
    python analysis/read_gradient_norms.py --experiments \
        0.01=runs/ablations/tau/dsr_tau0.01_10db ...        # Table 8 rows

Table 3 and the gradient columns of Table 8 should both come from this script:
the summary prints the mean and median, over all runs and epochs, of the
per-epoch mean norm and of the within-epoch std.
"""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt


def read_gradient_norms(folder_paths, required_epochs=500):
    """
    Read mean and within-epoch std gradient norm per epoch from train.log files.
    Folders whose run did not reach required_epochs are skipped.

    Returns:
        means_per_run: list of lists of per-epoch mean gradient norms
        stds_per_run:  list of lists of per-epoch within-epoch std gradient norms
    """
    means_per_run, stds_per_run = [], []
    for folder in folder_paths:
        log_path = f'{folder}/train.log'
        try:
            with open(log_path, 'r') as f:
                content = f.read()
            means = [float(v) for v in re.findall(r'Mean gradient norm=([\d.]+)', content)]
            stds  = [float(v) for v in re.findall(r'Mean gradient norm=[\d.]+ \(std=([\d.]+)\)', content)]
            if len(means) < required_epochs:
                print(f"Skipping {folder}: only {len(means)} epochs (need {required_epochs})")
                continue
            means_per_run.append(means)
            stds_per_run.append(stds)
        except FileNotFoundError:
            print(f"Log file not found: {log_path}")
    return means_per_run, stds_per_run


def read_gradient_norms_single(folder_path):
    """
    Read mean and std gradient norm per epoch from a single train.log.

    Returns:
        means: list of per-epoch mean gradient norms
        stds:  list of per-epoch std gradient norms (from the log file)
    """
    log_path = f'{folder_path}/train.log'
    with open(log_path, 'r') as f:
        content = f.read()
    means = [float(v) for v in re.findall(r'Mean gradient norm=([\d.]+)', content)]
    stds  = [float(v) for v in re.findall(r'Mean gradient norm=[\d.]+ \(std=([\d.]+)\)', content)]
    return means, stds


def compute_stats(means_per_run, start_epoch=1):
    """Return (epochs, mean, std) arrays aligned to the shortest run."""
    n_epochs = min(len(r) for r in means_per_run)
    array = np.array([r[start_epoch:n_epochs] for r in means_per_run])  # (n_runs, n_epochs)
    return np.arange(start_epoch + 1, n_epochs + 1), array.mean(axis=0), array.std(axis=0)


def summarize(label, means_per_run, stds_per_run):
    """Prints mean and median over all runs and epochs."""
    all_means = np.concatenate([np.asarray(r) for r in means_per_run])
    all_stds = np.concatenate([np.asarray(r) for r in stds_per_run])
    print(
        f"{label:<12}{len(means_per_run):>5}"
        f"{np.mean(all_means):>12.0f}{np.median(all_means):>12.0f}"
        f"{np.mean(all_stds):>12.0f}{np.median(all_stds):>12.0f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiments',
        nargs='+',
        default=[
            'Hard=runs/lorenz63/hard_10db',
            'Soft=runs/lorenz63/soft_10db',
            'EROT=runs/lorenz63/erot_10db',
            'DSR=runs/lorenz63/dsr_10db',
        ],
        help='label=run_prefix pairs; runs are <prefix>_0 ... <prefix>_<num_runs - 1>.',
    )
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument(
        '--run',
        type=int,
        default=None,
        help='Plot this run of every method; default: mean over runs.',
    )
    parser.add_argument('--output', type=str, default='gradient_norms')
    args = parser.parse_args()
    experiments = dict(e.split('=', 1) for e in args.experiments)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'text.usetex': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'lines.linewidth': 1.2,
        'lines.markersize': 4.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    colors = ["#0072B2", "#CC79A7", "#E69F00", "#009E73"]
    fig, ax = plt.subplots(figsize=(5, 3.5))

    print("\nProposal gradient norm before clipping; runs with all 500 epochs,")
    print("statistics over all runs and epochs.")
    print(f"{'Method':<12}{'Runs':>5}{'Mean':>12}{'Median':>12}"
          f"{'Std mean':>12}{'Std median':>12}")
    print("-" * 65)
    for k, (label, base) in enumerate(experiments.items()):
        color = colors[k % len(colors)]
        folders = [f"{base}_{i}" for i in range(args.num_runs)]
        means_per_run, stds_per_run = read_gradient_norms(folders)
        if not means_per_run:
            continue
        summarize(label, means_per_run, stds_per_run)
        if args.run is not None:
            # The same run index for every method.
            means, stds = read_gradient_norms_single(f"{base}_{args.run}")
            means, stds = np.array(means), np.array(stds)
            epochs = np.arange(1, len(means) + 1)
            ax.semilogy(epochs, means, label=label, color=color)
            ax.fill_between(epochs, np.maximum(means - stds, 1e-9), means + stds,
                            alpha=0.15, color=color, linewidth=0)
        else:
            epochs, mean, std = compute_stats(means_per_run, start_epoch=0)
            ax.semilogy(epochs, mean, label=label, color=color)
            ax.fill_between(epochs, np.maximum(mean - std, 1e-9), mean + std,
                            alpha=0.15, color=color, linewidth=0)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Proposal gradient norm")
    ax.set_ylim(bottom=1.0)
    ax.legend()
    plt.tight_layout()
    for extension in ("png", "pdf"):
        plt.savefig(f"{args.output}.{extension}")
        print(f"\nPlot saved to {args.output}.{extension}")


if __name__ == "__main__":
    main()
