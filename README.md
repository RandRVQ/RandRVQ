# RandRVQ: Using Random Codebooks for Audio Neural AutoEncoders

This repository contains the code of the paper

> Benoît Giniès, Xiaoyu Bie, Olivier Fercoq, Gaël Richard,
> **Using Random Codebooks for Audio Neural AutoEncoders**,
> 32nd European Signal Processing Conference (EUSIPCO), 2024, pp. 311–315.

Audio examples are available at <https://randrvq.github.io/>.

**This code is based on the [Descript Audio Codec (DAC)](https://github.com/descriptinc/descript-audio-codec)**
by Kumar et al. ([paper](https://arxiv.org/abs/2306.06546)), released under the
MIT license. The encoder, decoder, discriminator, losses and training loop are
those of DAC. The contribution of this work is the random residual vector
quantizer in [`randrvq/nn/quantize.py`](randrvq/nn/quantize.py).

## Method

In a residual vector quantizer (RVQ), each quantization stage encodes the
residual left by the previous stages with a trained codebook. In **Random RVQ**,
the last `n_r` stages do not own a trained codebook. At every forward pass, a
large codebook `B_big` of `N_big` tokens is randomly partitioned into disjoint
sub-codebooks of `s` tokens, one per random stage. `B_big` is drawn from
`N(0, I)` and never trained. Each random stage selects the nearest token of its
sub-codebook (Algorithm 1 of the paper).

| Paper notation | Argument (`DAC` / config file)  |
|----------------|---------------------------------|
| `n_t + n_r`    | `DAC.n_codebooks` (9)           |
| `n_r`          | `DAC.n_random_quantizers`       |
| `N_big`        | `DAC.size_big_codebook`         |
| `s`            | `DAC.sampling_size`             |
| Collapse miti. | `DAC.trained_collapse_mitigant`, `DAC.random_collapse_mitigant` |

"Collapse mitigants" are DAC's factorised (projected to dimension 8) and
L2-normalised codes. They can be switched off separately for the trained and
the random stages. With `n_random_quantizers = 0`, the quantizer is exactly
DAC's.

Because the sub-codebooks are redrawn at every forward pass, **encoding is
random, also at inference**. Codes of random stages index the big codebook, so
**decoding from codes is deterministic**.

## Repository structure

```
randrvq/
  nn/quantize.py        Random residual vector quantizer (contribution of the paper)
  model/dac.py          DAC model using the random RVQ
  model/base.py         DAC compression utilities (unchanged)
  model/discriminator.py, nn/layers.py, nn/loss.py   (unchanged from DAC)
conf/                   Configuration of every model of the paper
scripts/
  prepare_musdb.py      Export the MUSDB18 mixtures
  train.py              Training (DAC's training script)
  save_test_set.py      Extract the 1000 test excerpts
  get_samples.py        Reconstruct the test set (with several random draws)
  evaluate.py           Objective metrics (Tables I and II)
  compute_perplexity.py Codebook perplexities (Table III)
tests/                  Unit tests of the quantizer
```

## Installation

```bash
git clone https://github.com/RandRVQ/RandRVQ.git
cd RandRVQ
pip install -e .
```

The code was tested with Python 3.11, PyTorch 2.1.2, descript-audiotools 0.7.2
and argbind 0.3.7. Computing ViSQOL requires the `visqol` Python bindings
(see [google/visqol](https://github.com/google/visqol)); the other metrics do
not.

## Data

All models are trained on the mixtures of [MUSDB18](https://doi.org/10.5281/zenodo.1117372)
(100 training tracks, 50 test tracks, 44.1 kHz). Export them as one audio file per
track. Both MUSDB18 (`.stem.mp4`, requires ffmpeg) and MUSDB18-HQ are supported:

```bash
python scripts/prepare_musdb.py --musdb_root /path/to/musdb18 --output data/musdb18
```

This creates `data/musdb18/train/*.wav` and `data/musdb18/test/*.wav`, the paths
used in [`conf/base.yml`](conf/base.yml). The test tracks are also used for
validation during training.

## Training

Each model of the paper has its own configuration file. They all include
[`conf/base.yml`](conf/base.yml): 250k iterations, batch size 12, single GPU.

| Model (paper)  | Config                  | `n_r` | `N_big` | `s`  | Collapse mitigants        |
|----------------|-------------------------|-------|---------|------|---------------------------|
| Baseline       | `conf/baseline.yml`     | 0     | –       | –    | yes                       |
| Collapse       | `conf/collapse.yml`     | 0     | –       | –    | no                        |
| RandRVQ1       | `conf/randrvq1.yml`     | 4     | 8192    | 1024 | yes                       |
| RandRVQ2       | `conf/randrvq2.yml`     | 4     | 8192    | 1024 | no (random stages only)   |
| RandRVQ3       | `conf/randrvq3.yml`     | 4     | 8192    | 256  | no (random stages only)   |
| RandRVQ4       | `conf/randrvq4.yml`     | 4     | 16384   | 512  | yes                       |
| RandRVQ5       | `conf/randrvq5.yml`     | 6     | 16384   | 256  | yes                       |

```bash
python scripts/train.py --args.load conf/randrvq1.yml --save_path runs/randrvq1
```

Checkpoints are saved in `runs/randrvq1/{latest,best,10k,50k,...}` and
TensorBoard logs in `runs/randrvq1/logs`. Use `--resume` to resume a run.
Training took about 1.5 days per model on one GPU.

## Evaluation

**1. Test set.** Extract the 1000 excerpts of 10 s from the MUSDB18 test tracks.
The excerpts are deterministic, so this is done once for all models:

```bash
python scripts/save_test_set.py --args.load conf/base.yml --output samples/input
```

**2. Reconstructions.** The paper reports metrics averaged over 5 random draws
for the Random RVQ models:

```bash
python scripts/get_samples.py --path runs/randrvq1 --input samples/input \
    --output samples/randrvq1 --n_draws 5
```

For deterministic models (Baseline, Collapse), use `--n_draws 1`. The
*Baseline 5q* column of Table I is the baseline model with only its first 5
quantizers:

```bash
python scripts/get_samples.py --path runs/baseline --input samples/input \
    --output samples/baseline_5q --n_quantizers 5
```

**3. Objective metrics (Tables I and II).** The script computes the mel, STFT and
waveform losses, SI-SDR and ViSQOL at 44.1 kHz. Pass `--skip_visqol` if ViSQOL
is not installed.

```bash
python scripts/evaluate.py --input samples/input --output samples/randrvq1
```

Per-file results are written to `samples/randrvq1/metrics.csv`. The printed
`draw std` is the standard deviation across random draws, averaged over the
files: this is the `±` of Table II.

**4. Codebook perplexities (Table III).** The full test tracks are encoded and
the perplexity of the token distribution of each codebook is computed. The
`big codebook` line pools the tokens of all random stages.

```bash
python scripts/compute_perplexity.py --path runs/randrvq1 --input data/musdb18/test
```

All the evaluation scripts load the `best` checkpoint by default
(`--model_tag` to change it). By default, the model is loaded from its
`torch.package` archive; add `--from_weights` to rebuild it with the code of
this repository and load its weights.

## Using the model

```python
import randrvq
from audiotools import AudioSignal

model = randrvq.DAC.load("runs/randrvq1/best/dac/weights.pth").eval()
signal = AudioSignal("audio.wav")

codes = model.compress(signal)      # random: sub-codebooks are drawn at encoding
recons = model.decompress(codes)    # deterministic
```

## Tests

```bash
pip install -e ".[dev]"
python -m pytest tests
```

## Citation

```bibtex
@inproceedings{ginies2024randrvq,
  title     = {Using Random Codebooks for Audio Neural AutoEncoders},
  author    = {Gini{\`e}s, Beno{\^\i}t and Bie, Xiaoyu and Fercoq, Olivier and Richard, Ga{\"e}l},
  booktitle = {32nd European Signal Processing Conference (EUSIPCO)},
  pages     = {311--315},
  year      = {2024}
}
```

Please also cite DAC, on which this code is based:

```bibtex
@inproceedings{kumar2023high,
  title     = {High-Fidelity Audio Compression with Improved {RVQGAN}},
  author    = {Kumar, Rithesh and Seetharaman, Prem and Luebs, Alejandro and Kumar, Ishaan and Kumar, Kundan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}
```

## License and acknowledgements

This code is released under the MIT license (see [LICENSE](LICENSE)). It is
derived from the [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec),
© 2023 Descript, also released under the MIT license.

This work was funded by the European Union (ERC, HI-Audio, 101052978). Views
and opinions expressed are however those of the author(s) only and do not
necessarily reflect those of the European Union or the European Research
Council. Neither the European Union nor the granting authority can be held
responsible for them.
