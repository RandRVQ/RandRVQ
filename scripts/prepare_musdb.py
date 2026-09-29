"""Export the mixtures of MUSDB18 as one audio file per track.

The models of the paper are trained on the mixtures of MUSDB18
(https://doi.org/10.5281/zenodo.1117372): 100 training tracks and 50 test
tracks at 44.1 kHz. The training code expects one folder per split containing
one audio file per track (the stems are not used):

    data/musdb18/train/<track>.wav
    data/musdb18/test/<track>.wav

Both releases of the dataset are supported:

* MUSDB18-HQ: ``<root>/{train,test}/<track>/mixture.wav`` (files are copied);
* MUSDB18: ``<root>/{train,test}/<track>.stem.mp4``. The mixture is the first
  audio stream of each stem file; it is decoded with ffmpeg, which must be
  installed.

Usage:

    python scripts/prepare_musdb.py --musdb_root /path/to/musdb18hq --output data/musdb18
"""
import argparse
import shutil
import subprocess
from pathlib import Path


def export_split(split_dir: Path, output_dir: Path):
    """Export the mixtures of one split (train or test) of MUSDB18."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_tracks = 0

    # MUSDB18-HQ: one folder per track, containing mixture.wav
    for mixture in sorted(split_dir.glob("*/mixture.wav")):
        shutil.copy(mixture, output_dir / f"{mixture.parent.name}.wav")
        n_tracks += 1

    # MUSDB18: one .stem.mp4 file per track, the mixture is the first stream
    for stem in sorted(split_dir.glob("*.stem.mp4")):
        track = stem.name[: -len(".stem.mp4")]
        subprocess.run(
            [
                "ffmpeg", "-loglevel", "error", "-y",
                "-i", str(stem),
                "-map", "0:0",
                "-ar", "44100",
                str(output_dir / f"{track}.wav"),
            ],
            check=True,
        )
        n_tracks += 1

    print(f"{split_dir.name}: exported {n_tracks} tracks to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--musdb_root", required=True, help="Root of MUSDB18 or MUSDB18-HQ."
    )
    parser.add_argument(
        "--output", default="data/musdb18", help="Output folder (default: %(default)s)."
    )
    args = parser.parse_args()

    for split in ["train", "test"]:
        export_split(Path(args.musdb_root) / split, Path(args.output) / split)


if __name__ == "__main__":
    main()
