"""Reconstruct the test set with a trained model.

Each file ``<input>/<name>.wav`` is encoded and decoded, and the reconstruction
is written to ``<output>/<name>_draw<k>.wav``. Since the random quantizers draw
new sub-codebooks at every forward pass, the reconstruction of a Random RVQ
model is itself random: ``n_draws`` reconstructions are computed per file (5 in
the paper). For a model without random quantizers, use ``n_draws = 1``.

Usage:

    # Table II: 5 random draws with all the quantizers
    python scripts/get_samples.py --path runs/randrvq1 --input samples/input \
        --output samples/randrvq1 --n_draws 5

    # Table I, "Baseline 5q": baseline model using its first 5 quantizers only
    python scripts/get_samples.py --path runs/baseline --input samples/input \
        --output samples/baseline_5q --n_quantizers 5
"""
from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal
from audiotools.core import util
from audiotools.ml.decorators import Tracker
from train import Accelerator
from train import DAC


def load_model(
    accel: Accelerator,
    tracker: Tracker,
    save_path: str,
    tag: str = "best",
    from_weights: bool = False,
):
    """Load the generator saved by ``scripts/train.py`` in ``save_path/tag``.

    If ``from_weights`` is False, the model is loaded from its torch package
    (which embeds the source code it was trained with); otherwise it is rebuilt
    with the code of this repository and its weights are loaded.
    """
    folder = f"{save_path}/{tag}"
    tracker.print(f"Loading model from {Path(folder).absolute()}")
    generator, _ = DAC.load_from_folder(
        folder=folder, map_location="cpu", package=not from_weights
    )
    generator = accel.prepare_model(generator)
    return generator


@torch.no_grad()
def process(signal, accel, generator, **kwargs):
    """Reconstruct ``signal`` and match the loudness of the input."""
    signal = signal.to(accel.device)
    recons = generator(signal.audio_data, signal.sample_rate, **kwargs)["audio"]
    recons = AudioSignal(recons, signal.sample_rate)
    recons = recons.normalize(signal.loudness())
    return recons.cpu()


@argbind.bind(without_prefix=True)
@torch.no_grad()
def get_samples(
    accel,
    path: str = "ckpt",
    input: str = "samples/input",
    output: str = "samples/output",
    model_tag: str = "best",
    from_weights: bool = False,
    n_quantizers: int = None,
    n_draws: int = 1,
):
    """Write the reconstructions of the test set.

    Parameters
    ----------
    path : str
        Folder of the training run (``--save_path`` of ``train.py``).
    input : str
        Folder of the test set (see ``save_test_set.py``).
    output : str
        Output folder.
    model_tag : str
        Checkpoint to use (``best``, ``latest``, ``100k``, ...).
    from_weights : bool
        Load the weights with the current code instead of the torch package.
    n_quantizers : int, optional
        Number of quantization stages to use (all of them by default).
    n_draws : int
        Number of reconstructions per file (random draws of the sub-codebooks).
    """
    tracker = Tracker(log_file=f"{path}/eval.txt", rank=accel.local_rank)
    generator = load_model(
        accel, tracker, save_path=path, tag=model_tag, from_weights=from_weights
    )
    generator.eval()
    kwargs = {"n_quantizers": n_quantizers}

    audio_files = util.find_audio(input)

    global process
    process = tracker.track("process", n_draws * len(audio_files))(process)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    with tracker.live:
        for audio_file in audio_files:
            signal = AudioSignal(audio_file)
            for draw in range(1, n_draws + 1):
                recons = process(signal, accel, generator, **kwargs)
                recons.write(output / f"{audio_file.stem}_draw{draw}.wav")

        tracker.done("test", f"N={len(audio_files)}")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        with Accelerator() as accel:
            get_samples(accel)
