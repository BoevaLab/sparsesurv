"""Install package."""
import io
import re

from pkg_resources import parse_requirements
from setuptools import find_packages, setup


def read_version(filepath: str) -> str:
    """Read the __version__ variable from the file.

    Args:
        filepath: probably the path to the root __init__.py

    Returns:
        the version
    """
    match = re.search(
        r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        io.open(filepath, encoding="utf_8_sig").read(),
    )
    if match is None:
        raise SystemExit("Version number not found.")
    return match.group(1)


# ease installation during development
vcs = re.compile(r"(git|svn|hg|bzr)\+")
try:
    with open("requirements.txt") as fp:
        VCS_REQUIREMENTS = [
            str(requirement)
            for requirement in parse_requirements(fp)
            if vcs.search(str(requirement))
        ]
except FileNotFoundError:
    # requires verbose flags to show
    print("requirements.txt not found.")
    VCS_REQUIREMENTS = []

# TODO: Update these values according to the name of the module.
setup(
    name="sparsesurv",
    version=read_version("sparsesurv/__init__.py"),  # single place for version
    license="BSD-3-Clause",
    description="Toolkit for sparse survival analysis.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/BoevaLab/sparsesurv",
    author="David Wissel, Nikita Janakarajan",
    author_email="dwissel@inf.ethz.ch, jnikita@inf.ethz.ch",
    # the following exclusion is to prevent shipping of tests.
    # if you do include them, add pytest to the required packages.
    packages=find_packages(".", exclude=["*tests*"]),
    package_data={"sparsesurv": ["py.typed"]},
    extras_require={
        "vcs": VCS_REQUIREMENTS,
        "test": ["pytest", "pytest-cov"],
        "dev": [
            # tests
            "pytest",
            "pytest-cov",
            # checks
            "black==21.5b0",
            "flake8",
            "mypy",
            # docs
            "sphinx",
            "sphinx-autodoc-typehints",
            "better-apidoc",
            "six",
            "sphinx_rtd_theme",
            "myst-parser",
        ],
    },
    classifiers=[
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: BSD License",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=[
        "numpy>=1.23.4",
        "pandas>=1.5.1",
        "celer>=0.7.2",
        "numba>=0.56.4",
        "scikit-learn>=1.2.1",
    ],
    keywords=[
        "Survival",
        "Knowledge Distillation",
        "Sparse Survival",
        "High Dimensional Data",
    ],
)
