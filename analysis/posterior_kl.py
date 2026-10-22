"""Training-time posterior quality on the linear Gaussian SSM ("KLD KF", Table 1).

The KLD KF column of Table 1 was logged during training by comparing the
particles after resampling at step t, which approximate p(x_(t-1) | y_(1:t-1)),
with the Kalman posterior at index t, one step ahead, and by ignoring the
weights of the sets that were not resampled. train.py now compares them with
the posterior of the same index, using the weights. This script recomputes the
corrected value from the saved checkpoints, so no retraining is needed: every
run is filtered exactly as in training (its own resampling method, N, tau,
ESS threshold and log-density floor) on fresh test sequences.

From the repository root:

    python analysis/posterior_kl.py runs/linear/hard_10db runs/linear/soft_10db \
        runs/linear/erot_10db runs/linear/dsr_10db --num_runs 3

    # Reference: the true model, filtered with each method's resampling.
    python analysis/posterior_kl.py --oracle configs/linear/hard_10db.gin \
        configs/linear/soft_10db.gin configs/linear/erot_10db.gin \
        configs/linear/dsr_10db.gin

The run directories need gin_config.gin, transition_model_epoch_last.pt and
proposal_model_epoch_last.pt, as written by train.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import gin
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataset as dataset_lib  # noqa: E402
import train as train_lib  # noqa: E402


def build_loop(gin_path: str, batch_size: int) -> train_lib.TrainingLoop:
    """A TrainingLoop of the config, with a small training set.

    Args:
        gin_path: Gin config of the run.
        batch_size: Number of training sequences to generate; only the models
            and the filter settings of the loop are used.

    Returns:
        The training loop.
    """
    gin.clear_config()
    gin.parse_config_file(gin_path)
    gin.bind_parameter('LinearSSMRotationalDataset.num_samples', batch_size)
    return train_lib.TrainingLoop(save_path=tempfile.mkdtemp())


def set_true_model(loop: train_lib.TrainingLoop) -> None:
    """Sets the transition and proposal models to the data-generating model."""
    true_model = loop._train_dataset._linear_model
    A = torch.tensor(true_model.A)
    # softplus(log_diag) = diag(Ce).
    log_diag = torch.log(torch.expm1(torch.tensor(np.diag(true_model.Ce).copy())))
    for model in (loop._transition_model, loop._proposal_model):
        with torch.no_grad():
            model.A.weight.copy_(A)
            model.log_diag.copy_(log_diag)


def load_checkpoints(loop: train_lib.TrainingLoop, run_dir: str) -> None:
    """Loads the final transition and proposal models of a run."""
    for name, model in (
        ('transition', loop._transition_model),
        ('proposal', loop._proposal_model),
    ):
        path = os.path.join(run_dir, f'{name}_model_epoch_last.pt')
        model.load_state_dict(torch.load(path, map_location=loop._device))


@torch.no_grad()
def posterior_kl(
    loop: train_lib.TrainingLoop, num_sequences: int, seed: int
) -> float:
    """Mean KLD KF of the loop's models on fresh test sequences.

    Args:
        loop: Training loop with the models to evaluate.
        num_sequences: Number of test sequences.
        seed: Seed of the test sequences and of the filter.

    Returns:
        The KL divergence, averaged over steps t >= 2 and sequences.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_dataset = loop._train_dataset
    test_dataset = dataset_lib.LinearSSMRotationalDataset(
        num_samples=num_sequences,
        signal_length=train_dataset._signal_length,
        smnr_db=train_dataset._smnr_db,
    )
    kls = []
    for start in range(0, num_sequences, loop._batch_size):
        batch = test_dataset.collate_fn(
            [
                test_dataset[i]
                for i in range(start, min(start + loop._batch_size, num_sequences))
            ]
        )
        _, _, observations, Cws, kf_mean, kf_cov = batch
        _, _, _, kl = loop._train_once(
            observations.to(loop._device),
            Cws.to(loop._device),
            num_particles=loop._num_particles,
            posterior_mean=kf_mean,
            posterior_cov=kf_cov,
        )
        if kl is not None:
            kls.append(kl)
    return float(np.mean(kls))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'paths',
        nargs='+',
        help='Run prefixes (e.g. runs/linear/dsr_10db), or gin configs with --oracle.',
    )
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--num_sequences', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--oracle',
        action='store_true',
        help='Evaluate the true model with each config\'s resampling method.',
    )
    args = parser.parse_args()

    for path in args.paths:
        values = []
        if args.oracle:
            loop = build_loop(path, batch_size=64)
            set_true_model(loop)
            values.append(posterior_kl(loop, args.num_sequences, args.seed))
        else:
            for i in range(args.num_runs):
                run_dir = f'{path}_{i}'
                gin_path = os.path.join(run_dir, 'gin_config.gin')
                if not os.path.exists(gin_path):
                    print(f'  [warn] missing {gin_path}, skipping')
                    continue
                loop = build_loop(gin_path, batch_size=64)
                load_checkpoints(loop, run_dir)
                values.append(posterior_kl(loop, args.num_sequences, args.seed))
        if not values:
            continue
        spread = f' +- {np.std(values, ddof=1):.2f}' if len(values) > 1 else ''
        print(
            f'{path:<45} KLD KF {np.mean(values):7.2f}{spread}'
            f'   ({len(values)} run{"s" if len(values) > 1 else ""})'
        )


if __name__ == '__main__':
    main()
