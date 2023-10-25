[![License: BSD3](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
![GitHub all releases](https://img.shields.io/github/downloads/niklexical/sparsesurv/total)

# sparsesurv
`sparsesurv` [1] is a toolbox for high-dimensional survival analysis. Currently, the package is focused exclusively on knowledge distillation for sparse survival analysis, sometimes also called preconditoning [2, 3]. In the future, we plan to also extend `sparsesurv` to other techniques useful for (high-dimensional) survival analysis that are not commonly available in Python.

## Installation
The easiest way to install `sparsesurv` is currently via PyPi:

```
pip install sparsesurv
```

If you want to install directly from Github, you can also install by cloning the repo, or directly piping the repo to pip:

```
git clone https://github.com/BoevaLab/sparsesurv/
cd sparsesurv
pip install .
```

```
pip install git+https://github.com/BoevaLab/sparsesurv.git
```

If there is sufficient interest, we may also provide a conda package in the future.

## Bug reports and feature requests
If you have a bug report to make or a feature request for something you would like included in `sparsesurv` in the future, please open a [Github issue](https://github.com/BoevaLab/sparsesurv/issues).

## General questions
If you have general __questions__, meaning you are unsure about the usage of `sparsesurv`, or have other questions about the package that do not seem like a bug or feature request, please use [Github discussions](https://github.com/BoevaLab/sparsesurv/discussions/).

## Documentation and user guides
Documentation and user guides are available on [Github pages](TODO ADD LINK).

## Contributing
We always welcome new contributors to `sparsesurv`. If you're interested in contributing, get in touch with us (see Contact) or have a look at the open issues.

## Contact
[Nikita Janakarajan](jnikita@ethz.ch)

[David Wissel](dwissel@ethz.ch)

## Citation
If you use any or part of this package, please cite our work.
[TODO - add bibtext]


## References
[1] David Wissel, Nikita Janakarajan, Daniel Rowson, Julius Schulte, Xintian Yuan, Valentina Boeva. "sparsesurv: Sparse survival models via knowledge distillation." (2023, under review).

[2] Paul, Debashis, et al. "“Preconditioning” for feature selection and regression in high-dimensional problems." (2008): 1595-1618.

[3] Pavone, Federico, et al. "Using reference models in variable selection." Computational Statistics 38.1 (2023): 349-371.
