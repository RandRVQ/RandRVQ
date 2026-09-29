"""Codebook usage of a trained model (Table III of the paper).

Every track of the MUSDB18 test set is encoded (with ``DAC.compress``, i.e. in
chunks of 1 s after loudness normalisation to -16 dB) and the occurrences of
the tokens of each quantization stage are counted. The perplexity of a
codebook is

    PP = exp( - sum_i p_i log p_i ),

where p_i is the empirical frequency of token i. It equals the codebook size
when all tokens are used uniformly, and takes low values in case of codebook
collapse. For random quantizers, tokens are counted in the big codebook; the
last line (``big codebook``) pools the tokens of all the random quantizers.

Usage:

    python scripts/compute_perplexity.py --path runs/randrvq1 --input data/musdb18/test
"""
from pathlib import Path

import argbind
import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.core import util
from tqdm import tqdm

from train import DAC


def perplexity(counts: np.ndarray) -> float:
    """Perplexity of the empirical distribution given by ``counts``."""
    p = counts / counts.sum()
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


@argbind.bind(without_prefix=True)
@torch.no_grad()
def compute_perplexity(
    path: str = "ckpt",
    input: str = "data/musdb18/test",
    model_tag: str = "best",
    from_weights: bool = False,
    device: str = "cuda",
    output: str = None,
):
    """Print the perplexity of every codebook of a model.

    Parameters
    ----------
    path : str
        Folder of the training run (``--save_path`` of ``train.py``).
    input : str
        Folder of full-length test tracks.
    model_tag : str
        Checkpoint to use.
    from_weights : bool
        Load the weights with the current code instead of the torch package.
    device : str
        Device used for the encoding.
    output : str, optional
        If given, the token counts are saved to this ``.npz`` file.
    """
    generator, _ = DAC.load_from_folder(
        folder=f"{path}/{model_tag}", map_location="cpu", package=not from_weights
    )
    generator.to(device).eval()
    quantizer = generator.quantizer

    # One histogram per quantization stage, in the codebook of that stage
    # (the big codebook for random stages).
    counts = [np.zeros(q.codebook_size, dtype=np.int64) for q in quantizer.quantizers]

    for audio_file in tqdm(util.find_audio(input)):
        codes = generator.compress(AudioSignal(audio_file)).codes  # (C x N x T)
        for k in range(codes.shape[1]):
            counts[k] += np.bincount(
                codes[:, k, :].flatten().numpy(), minlength=len(counts[k])
            )

    print(f"{'codebook':<14}{'size':>8}{'perplexity':>14}{'ratio':>8}")
    for k, c in enumerate(counts):
        pp = perplexity(c)
        print(f"{k + 1:<14}{len(c):>8}{pp:>14.1f}{pp / len(c):>8.2f}")

    random_counts = [
        c for q, c in zip(quantizer.quantizers, counts) if getattr(q, "is_random", False)
    ]
    if random_counts:
        pooled = np.sum(random_counts, axis=0)
        pp = perplexity(pooled)
        print(f"{'big codebook':<14}{len(pooled):>8}{pp:>14.1f}{pp / len(pooled):>8.2f}")

    if output is not None:
        np.savez(output, *counts)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        compute_perplexity()
