"""Unit tests of the random residual vector quantizer (CPU only).

Run with ``python -m pytest tests``.
"""
import pytest
import torch

from randrvq import DAC
from randrvq.nn.quantize import RandomResidualVectorQuantize


def make_rvq(**kwargs):
    config = dict(
        input_dim=64,
        n_codebooks=9,
        codebook_size=32,
        codebook_dim=8,
        n_random_quantizers=4,
        size_big_codebook=512,
        sampling_size=64,
    )
    config.update(kwargs)
    return RandomResidualVectorQuantize(**config)


@pytest.mark.parametrize("random_collapse_mitigant", [True, False])
@pytest.mark.parametrize("trained_collapse_mitigant", [True, False])
def test_shapes_and_code_ranges(trained_collapse_mitigant, random_collapse_mitigant):
    rvq = make_rvq(
        trained_collapse_mitigant=trained_collapse_mitigant,
        random_collapse_mitigant=random_collapse_mitigant,
    ).eval()
    z = torch.randn(2, 64, 10)
    z_q, codes, latents, _, _ = rvq(z)

    assert z_q.shape == z.shape
    assert codes.shape == (2, 9, 10)
    assert latents.shape[1] == sum(q.codebook_dim for q in rvq.quantizers)
    # Trained codes index the small codebooks, random codes the big codebook.
    assert codes[:, :5].max() < 32
    assert codes[:, 5:].max() < 512


def test_random_sub_codebooks_are_disjoint():
    rvq = make_rvq(size_big_codebook=256, sampling_size=64)
    samplings = rvq.draw_sub_codebooks(torch.device("cpu"))
    assert samplings[:5] == [None] * 5
    indices = torch.cat(samplings[5:])
    assert len(indices) == 4 * 64
    assert len(indices.unique()) == len(indices)


def test_random_codes_use_the_drawn_sub_codebook():
    rvq = make_rvq().eval()
    z = torch.randn(1, 64, 20)
    torch.manual_seed(0)
    _, codes, _, _, _ = rvq(z)
    torch.manual_seed(0)
    samplings = rvq.draw_sub_codebooks(z.device)
    for k in range(5, 9):
        assert torch.isin(codes[:, k], samplings[k]).all()


def test_decoding_from_codes_matches_encoding():
    rvq = make_rvq().eval()
    z = torch.randn(2, 64, 10)
    z_q, codes, _, _, _ = rvq(z)
    z_q_from_codes, _, _ = rvq.from_codes(codes)
    assert torch.allclose(z_q, z_q_from_codes, atol=1e-5)


def test_big_codebook_is_frozen_and_shared():
    rvq = make_rvq().train()
    z = torch.randn(4, 64, 10)
    _, _, _, commitment_loss, codebook_loss = rvq(z)
    (commitment_loss + codebook_loss).backward()

    assert rvq.big_codebook.weight.grad is None
    assert all(q.codebook is rvq.big_codebook for q in rvq.quantizers[5:])
    assert rvq.quantizers[0].codebook.weight.grad is not None


def test_no_random_quantizer_is_dac():
    rvq = make_rvq(n_random_quantizers=0)
    assert not hasattr(rvq, "big_codebook")
    assert not any(q.is_random for q in rvq.quantizers)


def test_big_codebook_too_small():
    with pytest.raises(AssertionError):
        make_rvq(size_big_codebook=100, sampling_size=64)


def test_dac_forward_and_kwargs():
    model = DAC(
        encoder_dim=8,
        decoder_dim=32,
        n_random_quantizers=4,
        size_big_codebook=4096,
        sampling_size=512,
        random_collapse_mitigant=False,
    ).eval()
    x = torch.randn(1, 1, 4096)
    out = model(x, 44100)
    assert out["audio"].shape == x.shape
    assert out["codes"].shape[1] == 9

    # All constructor arguments are stored as attributes, so that audiotools
    # can save the kwargs required to rebuild the model from its weights.
    for name in [
        "n_random_quantizers",
        "size_big_codebook",
        "sampling_size",
        "trained_collapse_mitigant",
        "random_collapse_mitigant",
    ]:
        assert hasattr(model, name)
