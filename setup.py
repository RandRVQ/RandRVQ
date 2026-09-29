from setuptools import find_packages
from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="randrvq",
    version="1.0.0",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    description="Random codebooks for residual vector quantization in neural audio codecs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Benoît Giniès, Xiaoyu Bie, Olivier Fercoq, Gaël Richard",
    url="https://github.com/RandRVQ/RandRVQ",
    license="MIT",
    packages=find_packages(include=["randrvq", "randrvq.*"]),
    keywords=["audio", "compression", "vector quantization", "machine learning"],
    install_requires=[
        "argbind>=0.3.7",
        "descript-audiotools>=0.7.2",
        "einops",
        "numpy",
        "torch",
        "torchaudio",
        "tqdm",
        "tensorboard",
    ],
    extras_require={"dev": ["pytest"]},
)
