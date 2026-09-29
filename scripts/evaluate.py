"""Objective evaluation of reconstructions (Tables I and II of the paper).

For every test file ``<input>/<name>.wav`` and every reconstruction
``<output>/<name>_draw<k>.wav`` written by ``get_samples.py``, the following
metrics are computed at 44.1 kHz:

* ``mel``: multi-scale mel-spectrogram loss (lower is better);
* ``stft``: multi-scale STFT loss (lower is better);
* ``waveform``: L1 loss on the waveform (lower is better);
* ``sisdr``: scale-invariant signal-to-distortion ratio in dB (higher is better);
* ``visqol``: ViSQOL v3 in audio mode (higher is better). Requires the
  ``visqol`` Python package; pass ``--skip_visqol`` to skip it.

The losses are those of DAC with their default parameters. The per-file results
are written to ``<output>/metrics.csv`` and a summary is printed:

* ``mean``: average over all files and draws;
* ``draw std``: standard deviation across the random draws, computed for each
  file and averaged over the files. This is the ``±`` of Table II.

Usage:

    python scripts/evaluate.py --input samples/input --output samples/randrvq1
"""
import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import argbind
import numpy as np
import torch
from audiotools import AudioSignal
from audiotools import metrics
from audiotools.core import util

from randrvq.nn import loss

METRICS = ["mel", "stft", "waveform", "sisdr", "visqol"]


def get_metrics(signal_path: Path, recons_path: Path, use_visqol: bool = True):
    """Compute the metrics between a test file and one of its reconstructions."""
    signal = AudioSignal(signal_path).resample(44100)
    recons = AudioSignal(recons_path).resample(44100)
    output = {
        "file": signal_path.name,
        "reconstruction": recons_path.name,
        "mel": loss.MelSpectrogramLoss()(signal, recons).item(),
        "stft": loss.MultiScaleSTFTLoss()(signal, recons).item(),
        "waveform": loss.L1Loss()(signal, recons).item(),
        # SISDRLoss returns the negative SI-SDR.
        "sisdr": -loss.SISDRLoss()(signal, recons).item(),
    }
    if use_visqol:
        output["visqol"] = metrics.quality.visqol(signal, recons).item()
    return output


def summarize(rows: list):
    """Print the mean of each metric and its standard deviation across draws."""
    keys = [k for k in METRICS if k in rows[0]]
    by_file = {}
    for row in rows:
        by_file.setdefault(row["file"], []).append(row)

    print(f"{'metric':<10}{'mean':>12}{'draw std':>12}")
    for k in keys:
        values = np.array([row[k] for row in rows])
        stds = [
            np.std([row[k] for row in file_rows], ddof=1)
            for file_rows in by_file.values()
            if len(file_rows) > 1
        ]
        draw_std = f"{np.mean(stds):>12.4f}" if stds else f"{'-':>12}"
        print(f"{k:<10}{values.mean():>12.4f}{draw_std}")


@argbind.bind(without_prefix=True)
@torch.no_grad()
def evaluate(
    input: str = "samples/input",
    output: str = "samples/output",
    n_proc: int = 8,
    skip_visqol: bool = False,
):
    """Evaluate all the reconstructions found in ``output``.

    Parameters
    ----------
    input : str
        Folder of the test set.
    output : str
        Folder of the reconstructions (``get_samples.py``).
    n_proc : int
        Number of parallel processes.
    skip_visqol : bool
        Do not compute ViSQOL (which is slow and requires the ``visqol`` package).
    """
    output = Path(output)
    pairs = [
        (audio_file, recons)
        for audio_file in util.find_audio(input)
        for recons in sorted(output.glob(f"{audio_file.stem}_draw*.wav"))
    ]
    if not pairs:
        raise FileNotFoundError(f"No reconstruction found in {output}.")

    with ProcessPoolExecutor(n_proc, mp.get_context("fork")) as pool:
        futures = [pool.submit(get_metrics, x, y, not skip_visqol) for x, y in pairs]
        rows = [f.result() for f in futures]

    with open(output / "metrics.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(pairs)} reconstructions evaluated, results in {output / 'metrics.csv'}")
    summarize(rows)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        evaluate()
