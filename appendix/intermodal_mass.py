"""Inter-modal mass of resampled particles for a static two-component Gaussian.

Paper appendix "Multimodal Posteriors and Inter-modal Mass", Table 13. From the
repository root:

    python -m appendix.intermodal_mass                # published: --axis first
    python -m appendix.intermodal_mass --axis last
    python -m appendix.intermodal_mass --axis random

--axis sets the direction that separates the modes. `first` (coordinate 0) is
the published setting; it is the best case for Morton sorting, because
morton_sort ranks coordinate 0 most significant, so the first bit of every
Morton key is the mode label for any d. `last` and `random` (a random rotation
per repetition) remove that alignment. Hard SR, soft resampling and unsorted
DSR do not depend on the direction.
"""

import argparse

import torch

import resampling as resampling_lib

METHODS = ('hard', 'soft', 'dsr_sorted', 'dsr_unsorted', 'erot')
AXES = ('first', 'last', 'random')


def orient(
    particles: torch.Tensor, axis: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turns particles separated along coordinate 0 to the separation `axis`.

    Args:
        particles: Particles (n_reps, N, dim) separated along coordinate 0.
        axis: 'first' (unchanged), 'last' (coordinate 0 moved to the end) or
            'random' (a random rotation per repetition).

    Returns:
        The particles and the unit separation direction (n_reps, dim).
    """
    n_reps, _, dim = particles.shape
    eye = torch.eye(dim, device=particles.device, dtype=particles.dtype)
    if axis == 'first':
        return particles, eye[0].expand(n_reps, dim)
    if axis == 'last':
        return particles.roll(-1, dims=-1), eye[-1].expand(n_reps, dim)
    if axis == 'random':
        gaussian = torch.randn(
            n_reps, dim, dim, device=particles.device, dtype=particles.dtype
        )
        rotation, _ = torch.linalg.qr(gaussian)
        # x @ R moves e_0 to the row R[0]: (x @ R) . R[0] = x_0.
        return particles @ rotation, rotation[:, 0, :]
    raise ValueError(f'Unknown axis: {axis}')


def two_gaussian_target(
    n_reps: int,
    N: int,
    dim: int,
    sep: float,
    sigma: float,
    frac_left: float = 0.30,
    device: str = 'cpu',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Particles from two Gaussians separated along coordinate 0.

    The modes are at -+ sep / 2 along coordinate 0; coordinates 1, ...,
    dim - 1 are noise without mode information. Each mode has mass 0.5.
    Particles are shuffled, so any locality a resampler exploits comes from
    its own ordering.

    Args:
        n_reps: Number of independent particle sets to draw.
        N: Number of particles in each particle set.
        dim: Dimension of the particles.
        sep: Distance between the two mode centers along coordinate 0.
        sigma: Standard deviation of each mode, in every coordinate.
        frac_left: Fraction of the particles drawn from the left mode.
        device: Torch device of the returned tensors.

    Returns:
        Particles (n_reps, N, dim) and weights (n_reps, N).
    """
    n_left = int(round(frac_left * N))
    particles = sigma * torch.randn(n_reps, N, dim, device=device)
    particles[:, :n_left, 0] -= sep / 2.0
    particles[:, n_left:, 0] += sep / 2.0

    weights = torch.zeros(n_reps, N, device=device)
    weights[:, :n_left] = 0.5 / n_left
    weights[:, n_left:] = 0.5 / (N - n_left)
    weights = weights / weights.sum(1, keepdim=True)

    perm = torch.stack(
        [torch.randperm(N, device=device) for _ in range(n_reps)]
    )
    particles = torch.gather(
        particles, 1, perm.unsqueeze(-1).expand(n_reps, N, dim)
    )
    return particles, torch.gather(weights, 1, perm)


def hard_systematic(
    particles: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hard systematic resampling of a batch.

    Args:
        particles: B particle sets (B, N, D) of N particles each.
        weights: Normalized particle weights (B, N), summing to one over N.

    Returns:
        Resampled particles (B, N, D) and uniform weights (B, N).
    """
    B, N, D = particles.shape
    cdf = torch.cumsum(weights, 1)
    grid = torch.arange(N, device=particles.device, dtype=particles.dtype)
    u = (
        torch.rand(B, 1, device=particles.device, dtype=particles.dtype) / N
        + grid.view(1, N) / N
    )
    idx = torch.searchsorted(cdf.contiguous(), u.contiguous()).clamp(max=N - 1)
    out = torch.gather(particles, 1, idx.unsqueeze(-1).expand(B, N, D))
    return out, torch.full_like(weights, 1.0 / N)


@torch.no_grad()
def intermodal_mass(
    dim: int,
    N: int = 100,
    sep: float = 10.0,
    sigma: float = 0.3,
    tau: float = 0.1,
    eps: float = 0.5,
    alpha: float = 0.5,
    frac_left: float = 0.5,
    gap_frac: float = 0.6,
    n_reps: int = 10000,
    seed: int = 0,
    device: str = 'cpu',
    axis: str = 'first',
) -> dict[str, dict[str, float]]:
    """Resampled mass in the band |x . v| < gap_frac * sep / 2 between the modes.

    v is the unit direction that separates the modes. The target has
    essentially no mass in the band, so copy-based schemes (hard SR, soft)
    place none there and any mass is due to the relaxation.

    Args:
        dim: Dimension of the particles.
        N: Number of particles in each particle set.
        sep: Distance between the two mode centers along coordinate 0.
        sigma: Standard deviation of each mode, in every coordinate.
        tau: Temperature of the DSR relaxation.
        eps: Entropic regularization of EROT.
        alpha: Soft resampling samples alpha * w + (1 - alpha) * uniform.
        frac_left: Fraction of the particles drawn from the left mode.
        gap_frac: Half-width of the inter-modal band |x_0| < gap_frac * sep / 2,
            as a fraction of the distance from the center to a mode.
        n_reps: Number of independent particle sets, each resampled once.
        seed: Random seed for the particle sets and the resampling draws.
        device: Torch device to run on.
        axis: Direction that separates the modes, see orient; 'first'
            reproduces the published table bit for bit.

    Returns:
        {method: {'mass_mean', 'mass_std'}} over the repetitions.
    """
    torch.manual_seed(seed)
    particles, weights = two_gaussian_target(
        n_reps, N, dim, sep, sigma, frac_left, device
    )
    particles, direction = orient(particles, axis)

    # The order of the calls fixes the random stream each method draws from.
    outputs = {'hard': hard_systematic(particles, weights)}
    outputs['dsr_sorted'] = resampling_lib.resampler_systematic_morton(
        particles, weights, tau=tau, requires_grad=False
    )
    outputs['dsr_unsorted'] = resampling_lib.resampler_systematic(
        particles, weights, tau=tau, requires_grad=False
    )
    outputs['erot'] = resampling_lib.resampler_ot(
        particles, weights, eps=eps, max_iter=100, threshold=1e-3, device=device
    )[:2]
    outputs['soft'] = resampling_lib.soft_resampling(
        particles, weights, alpha=alpha
    )

    gap = gap_frac * sep / 2.0
    results = {}
    for name, (x, w) in outputs.items():
        w = w.detach() / w.detach().sum(1, keepdim=True)
        projection = torch.einsum('bnd,bd->bn', x.detach(), direction)
        mass = (w * (projection.abs() < gap)).sum(1)
        results[name] = {
            'mass_mean': float(mass.mean()),
            'mass_std': float(mass.std()),
        }
    return results


def sweep_dimensions(
    dims: tuple[int, ...] = (1, 2, 3, 5, 10), **kwargs
) -> dict[int, dict[str, dict[str, float]]]:
    """Prints and returns the inter-modal mass for each dimension.

    Args:
        dims: Dimensions of the particles, one table column each.
        **kwargs: Further arguments of intermodal_mass.

    Returns:
        Dimension -> output of intermodal_mass.
    """
    rows = {d: intermodal_mass(dim=d, **kwargs) for d in dims}
    width = 18
    print(f"{'method':<15}" + ''.join(f"{f'd={d}':>{width}}" for d in dims))
    print('-' * (15 + width * len(dims)))
    for method in METHODS:
        print(
            f'{method:<15}'
            + ''.join(
                f"{rows[d][method]['mass_mean']:>8.3f} ±"
                f" {rows[d][method]['mass_std']:<7.3f}"
                for d in dims
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--axis', choices=AXES, default='first')
    parser.add_argument(
        '--dims', type=int, nargs='+', default=[1, 2, 3, 5, 10]
    )
    parser.add_argument('--n_reps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(f'Separation axis: {args.axis}')
    sweep_dimensions(
        dims=tuple(args.dims),
        axis=args.axis,
        n_reps=args.n_reps,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
