"""Random Residual Vector Quantization (Random RVQ).

This module generalises the residual vector quantizer of the Descript Audio
Codec (DAC, https://github.com/descriptinc/descript-audio-codec) as described
in "Using Random Codebooks for Audio Neural AutoEncoders" (EUSIPCO 2024).

The residual quantizer is made of ``n_codebooks`` stages:

* the first ``n_codebooks - n_random_quantizers`` stages are standard, trained
  vector quantizers (exactly as in DAC);
* the last ``n_random_quantizers`` stages are *random* quantizers: they do not
  own a codebook. At every forward pass, a single random permutation of a large,
  frozen codebook ``B_big`` (N_big tokens drawn from N(0, I)) is computed and cut
  into disjoint chunks of ``sampling_size`` tokens. Random quantizer ``k``
  performs its nearest-neighbour search in chunk ``k`` only, which guarantees
  that the sub-codebooks of the different random quantizers never overlap
  (Algorithm 1 of the paper).

With ``n_random_quantizers = 0`` the module is strictly equivalent to DAC's
``ResidualVectorQuantize`` (this is how the *Baseline* and *Collapse* models of
the paper are obtained).

Codebook-collapse mitigants
---------------------------
DAC uses two tricks borrowed from Improved VQGAN
(https://arxiv.org/abs/2110.04627) to avoid codebook collapse:

1. factorised codes: the latent is projected into a low-dimensional space
   (``codebook_dim``) by a 1x1 convolution before the nearest-neighbour search,
   and projected back afterwards;
2. L2-normalised codes: latents and codewords are L2-normalised before the
   search, so that the euclidean distance becomes a cosine similarity.

These mitigants can be switched on/off independently for the trained
quantizers (``trained_collapse_mitigant``) and for the random quantizers
(``random_collapse_mitigant``). When they are switched off, the search is
performed directly in the latent space (``input_dim``) with plain euclidean
distance.

Note on the codes
-----------------
The codes returned for a random quantizer are indices in the *big* codebook
(in ``[0, N_big)``), not in the sampled sub-codebook. Decoding from codes is
therefore deterministic: randomness only affects the encoding.
"""
from typing import List
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from randrvq.nn.layers import WNConv1d


class VectorQuantize(nn.Module):
    """Single vector-quantization stage.

    Implementation of VQ similar to Karpathy's repo
    (https://github.com/karpathy/deep-vector-quantization), adapted from DAC.

    The stage works in two modes:

    * **trained** (default): the stage owns a learnable codebook of
      ``codebook_size`` tokens;
    * **random**: the stage is given a reference to a frozen ``big_codebook``
      shared with other stages, and the indices of the sub-codebook it may use
      are passed at every forward pass (``sampling`` argument).

    Parameters
    ----------
    input_dim : int
        Dimension of the latent space (number of channels of the encoder output).
    codebook_size : int
        Number of tokens of the learnable codebook (trained mode only).
    codebook_dim : int
        Dimension of the tokens when ``collapse_mitigant`` is True (factorised
        codes). Ignored otherwise, as the tokens then live in the latent space.
    collapse_mitigant : bool
        Whether to use the factorised + L2-normalised codes of DAC.
    big_codebook : nn.Embedding, optional
        Frozen codebook shared between random stages. If given, the stage works
        in random mode.
    """

    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        collapse_mitigant: bool = True,
        big_codebook: nn.Embedding = None,
    ):
        super().__init__()
        self.collapse_mitigant = collapse_mitigant
        self.is_random = big_codebook is not None

        # Without factorised codes, the search happens in the latent space.
        self.codebook_dim = codebook_dim if collapse_mitigant else input_dim

        if collapse_mitigant:
            # Factorised codes (ViT-VQGAN): project to / from a low-dim space.
            self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
            self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()

        if self.is_random:
            # Shared, frozen codebook (the same module is referenced by every
            # random stage, so its weights are stored only once).
            self.codebook = big_codebook
        else:
            self.codebook = nn.Embedding(codebook_size, self.codebook_dim)
        self.codebook_size = self.codebook.num_embeddings

    def forward(self, z: torch.Tensor, sampling: torch.Tensor = None):
        """Quantize the input tensor.

        Parameters
        ----------
        z : Tensor[B x D x T]
            Input (residual) latent.
        sampling : Tensor[S], optional
            Indices of the tokens of the big codebook this stage may use
            (random mode only).

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input.
        Tensor[B]
            Commitment loss to train encoder to predict vectors closer to
            codebook entries.
        Tensor[B]
            Codebook loss to update the codebook. For a random stage the codebook
            is frozen, hence this loss does not produce any gradient.
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input).
        Tensor[B x D' x T]
            Projected latents (continuous representation of input before
            quantization).
        """
        z_e = self.in_proj(z)  # z_e : (B x D' x T)
        z_q, indices = self.decode_latents(z_e, sampling)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        # No-op in forward pass, straight-through gradient estimator in backward.
        z_q = z_e + (z_q - z_e).detach()
        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id: torch.Tensor):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id: torch.Tensor):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents: torch.Tensor, sampling: torch.Tensor = None):
        """Nearest-neighbour search of ``latents`` in the (sub-)codebook.

        Returns the quantized latents and the corresponding indices. For a random
        stage, the returned indices refer to the big codebook.
        """
        if self.is_random and sampling is None:
            raise ValueError("A random quantizer needs the `sampling` indices.")

        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # (N x D')
        if self.is_random:
            codebook = codebook[sampling]  # (S x D') random sub-codebook

        if self.collapse_mitigant:
            # L2-normalise encodings and codebook (ViT-VQGAN).
            encodings = F.normalize(encodings)
            codebook = F.normalize(codebook)

        # Euclidean distance with the codebook.
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        if self.is_random:
            # Map sub-codebook indices back to big-codebook indices.
            indices = sampling[indices]

        z_q = self.decode_code(indices)
        return z_q, indices


class RandomResidualVectorQuantize(nn.Module):
    """Residual vector quantizer with trained and random stages.

    Residual VQ was introduced in SoundStream (https://arxiv.org/abs/2107.03312).
    See the module docstring for a description of the random stages.

    Parameters
    ----------
    input_dim : int
        Dimension of the latent space.
    n_codebooks : int
        Total number of quantization stages (trained + random).
    codebook_size : int
        Number of tokens of each trained codebook.
    codebook_dim : int or list of int
        Dimension of the factorised codes (per stage if a list is given).
    quantizer_dropout : float
        Fraction of the batch for which a random number of stages is used
        during training (DAC's quantizer dropout).
    n_random_quantizers : int
        Number of random stages (the last stages of the residual pile).
    size_big_codebook : int
        Number of tokens N_big of the frozen big codebook.
    sampling_size : int
        Number of tokens s drawn from the big codebook for each random stage.
        ``n_random_quantizers * sampling_size`` must not exceed
        ``size_big_codebook`` (the sub-codebooks are disjoint).
    trained_collapse_mitigant : bool
        Use DAC's collapse mitigants in the trained stages.
    random_collapse_mitigant : bool
        Use DAC's collapse mitigants in the random stages.
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, List[int]] = 8,
        quantizer_dropout: float = 0.0,
        n_random_quantizers: int = 0,
        size_big_codebook: int = 8192,
        sampling_size: int = 1024,
        trained_collapse_mitigant: bool = True,
        random_collapse_mitigant: bool = True,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        assert 0 <= n_random_quantizers <= n_codebooks, (
            "n_random_quantizers must be between 0 and n_codebooks."
        )
        assert n_random_quantizers * sampling_size <= size_big_codebook, (
            "The big codebook is too small to draw disjoint sub-codebooks: "
            "n_random_quantizers * sampling_size > size_big_codebook."
        )

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.quantizer_dropout = quantizer_dropout

        self.n_trained_quantizers = n_codebooks - n_random_quantizers
        self.n_random_quantizers = n_random_quantizers
        self.size_big_codebook = size_big_codebook
        self.sampling_size = sampling_size

        quantizers = [
            VectorQuantize(
                input_dim,
                codebook_size,
                codebook_dim[i],
                collapse_mitigant=trained_collapse_mitigant,
            )
            for i in range(self.n_trained_quantizers)
        ]

        if n_random_quantizers > 0:
            # Big codebook, sampled from N(0, I) (default nn.Embedding init) and
            # never trained.
            big_dim = codebook_dim[-1] if random_collapse_mitigant else input_dim
            self.big_codebook = nn.Embedding(size_big_codebook, big_dim)
            self.big_codebook.weight.requires_grad_(False)

            quantizers += [
                VectorQuantize(
                    input_dim,
                    sampling_size,
                    codebook_dim[i],
                    collapse_mitigant=random_collapse_mitigant,
                    big_codebook=self.big_codebook,
                )
                for i in range(self.n_trained_quantizers, n_codebooks)
            ]

        self.quantizers = nn.ModuleList(quantizers)

    def draw_sub_codebooks(self, device: torch.device) -> List[torch.Tensor]:
        """Draw the disjoint sub-codebooks of the random stages.

        A single random permutation of the big codebook is cut into
        ``n_random_quantizers`` chunks of ``sampling_size`` indices.

        Returns
        -------
        list of Tensor[S]
            One tensor of big-codebook indices per quantization stage
            (``None`` for trained stages).
        """
        samplings = [None] * self.n_trained_quantizers
        if self.n_random_quantizers > 0:
            permutation = torch.randperm(self.size_big_codebook, device=device)
            s = self.sampling_size
            samplings += [
                permutation[k * s : (k + 1) * s]
                for k in range(self.n_random_quantizers)
            ]
        return samplings

    def forward(self, z: torch.Tensor, n_quantizers: int = None):
        """Quantize the input tensor with the residual pile.

        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            Number of stages to use (``n_quantizers < self.n_codebooks``, e.g.
            for bitrate scalability). In training mode, this argument is ignored
            and quantizer dropout is applied instead.

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input.
        Tensor[B x N x T]
            Codebook indices for each stage (quantized discrete representation).
        Tensor[B x N*D' x T]
            Projected latents (continuous representation before quantization).
        Tensor[1]
            Commitment loss.
        Tensor[1]
            Codebook loss.
        """
        # New random sub-codebooks at every forward pass (training and inference).
        samplings = self.draw_sub_codebooks(z.device)

        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            # Quantizer dropout: a random number of stages for part of the batch.
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual, samplings[i]
            )

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        """Reconstruct the continuous representation from the codes.

        Decoding is deterministic, as random-stage codes index the big codebook.

        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input.

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input.
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the continuous
        representation after quantization (new random sub-codebooks are drawn).

        Parameters
        ----------
        latents : Tensor[B x N x T]
            Continuous representation of input after projection.

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space.
        Tensor[B x D x T]
            Quantized representation of latent space.
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[
            0
        ]
        samplings = self.draw_sub_codebooks(latents.device)
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(
                latents[:, j:k, :], samplings[i]
            )
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)


if __name__ == "__main__":
    rvq = RandomResidualVectorQuantize(
        n_random_quantizers=4, size_big_codebook=8192, sampling_size=1024
    )
    x = torch.randn(16, 512, 80)
    z_q, codes, latents, _, _ = rvq(x)
    print(z_q.shape, codes.shape, latents.shape)
