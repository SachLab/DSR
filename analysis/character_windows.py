"""Which CharacterTrajectories recordings the experiments use (paper Sec. 5.4).

CharacterTrajectoriesDataset cuts non-overlapping windows of signal_length
samples, so a recording shorter than that gives no window. This prints, for the
training split (mixout[:2000], at most 1000 windows) and for the validation
split that the reported NMSE is computed on (mixout[2000:2400], the first 10
windows, as in train.py), how many windows each character contributes.

    python analysis/character_windows.py
    python analysis/character_windows.py --signal_length 100
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np
import scipy.io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataset as dataset_lib  # noqa: E402

# (first index, end index, maximum number of windows) as in dataset.py and
# train.py.
SPLITS = {'train': (0, 2000, 1000), 'valid': (2000, 2400, 10)}


def window_sources(
    lengths: np.ndarray, start: int, end: int, signal_length: int, cap: int
) -> list[int]:
    """Recording index of every window, in the order the dataset takes them."""
    sources = []
    for index in range(start, min(end, len(lengths))):
        for _ in range(0, lengths[index] - signal_length + 1, signal_length):
            if len(sources) < cap:
                sources.append(index)
    return sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal_length', type=int, default=178)
    parser.add_argument(
        '--data_path', type=str, default=dataset_lib.CHARACTERS_PATH
    )
    args = parser.parse_args()

    mat = scipy.io.loadmat(args.data_path)
    trajectories = mat['mixout'][0]
    consts = mat['consts'][0, 0]
    keys = [str(key[0]) for key in consts['key'].ravel()]
    labels = [keys[int(label) - 1] for label in consts['charlabels'].ravel()]
    lengths = np.array([trajectory.shape[1] for trajectory in trajectories])
    per_character = collections.defaultdict(set)
    for label, length in zip(labels, lengths):
        per_character[label].add(int(length))
    print(
        f'{len(trajectories)} recordings of {len(keys)} characters, '
        f'{lengths.min()}-{lengths.max()} samples'
    )
    print(
        'length per character: '
        + ', '.join(
            f'{c}={"/".join(map(str, sorted(v)))}'
            for c, v in sorted(per_character.items())
        )
    )
    for split, (start, end, cap) in SPLITS.items():
        sources = window_sources(lengths, start, end, args.signal_length, cap)
        counts = collections.Counter(labels[i] for i in sources)
        print(
            f'{split}: {len(sources)} windows of {args.signal_length} samples '
            f'from recordings {sources[0] if sources else "-"}-'
            f'{sources[-1] if sources else "-"}; per character: '
            + ', '.join(f'{c}={n}' for c, n in sorted(counts.items()))
        )


if __name__ == '__main__':
    main()
