import codecs
from pathlib import Path
from setuptools import setup, find_packages

from exptorch import __version__


readme_filename = Path(__file__).absolute().parent / "README.md"
with codecs.open(readme_filename, encoding="utf-8") as file:
    long_description = file.read()


setup(
    name="exptorch",
    version=__version__,
    description="basic pytorch wrapper and experiment scheduler",
    long_description=long_description,
    maintainer="Paul Aurel Diederichs",
    packages=find_packages(exclude=["tests"]),
    install_requires=["torch", "numpy", "tensorboard", "tqdm"],
    extras_require=dict(
        dev=[
            "black",
            "codespell",
            "flake8",
            "pydocstyle",
            "pylint",
            "pytest",
            "pytest-cov",
            "torchvision",
        ],
    ),
)
