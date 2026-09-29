"""RandRVQ: Descript Audio Codec with random codebooks.

Code for "Using Random Codebooks for Audio Neural AutoEncoders" (EUSIPCO 2024),
built on top of the Descript Audio Codec
(https://github.com/descriptinc/descript-audio-codec).
"""
__version__ = "1.0.0"

import audiotools

# Let audiotools package the source code of this library with saved models.
audiotools.ml.BaseModel.INTERN += ["randrvq.**"]
audiotools.ml.BaseModel.EXTERN += ["einops"]


from . import nn
from . import model
from .model import DAC
from .model import DACFile
