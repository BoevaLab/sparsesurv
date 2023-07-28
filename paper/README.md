# *sparsesurv*: Knowledge distillation for sparse semiparametric survival models
## Abstract
Regularized models that perform integrated feature selection, such as the Lasso, have found broad application in high-dimensional time-to-event settings. Genomic and transcriptomic cancer survival datasets, where the $p >> n$ regime is commonly observed are typical use cases for these methods. Despite the broad usage of such models, they can overregularize and are sensitive to the choice of the regularization hyperparameter. We propose *sparsesurv*, a Python package for fitting sparse semiparametric right-censored survival models via knowledge distillation (KD) [1]. KD was originally proposed to improve the performance of a single neural network by training on the predictions of an ensemble of neural networks. In statistics, KD has also been used for feature selection and was proposed several years before KD under the name of preconditioning [2]. Both preconditioning and KD have shown great promise, both for model compression and feature selection. Despite its simple premise, proper implementation of KD for survival models requires care. It is unclear which scoring functions should be used for the choice of regularization hyperparameter. Certain semiparametric methods are not implemented in commonly used languages such as Python, which makes estimating the desired teacher impossible. Lastly, having estimated the student, there is still a need for estimating the survival function, necessitating the usage of additional packages or even hand-written code. Our package contains linear variants of the semiparametric Accelerated Failure Time (AFT) and the Extended Hazards (EH) model to remedy their unavailability in Python. We implement easy-to-use KD methods for the Cox PH model, the AFT model, and the EH model and enhance them with a flexible choice of the scoring function and survival function estimators. We compare our distilled implementation of the Cox PH model to previous works on sparse linear Cox PH models and show that KD-based approaches achieve better discriminative performance across the regularization path while making the choice of the regularization hyperparameter significantly easier.


## Reproducibility
### From scratch
Since installing via conda and R in bash scripts can be finicky and anyway requires user input, we guide you through the process below, after which you may reproduce all of our results by executing our reproduction script.

#### Python
Please run the following in a terminal and give user input as appropriate (e.g., confirming that you want to create a new conda env).

```sh
conda create -n noise_resistance python==3.10.0
conda activate noise_resistance
pip install -r requirements.txt
pip install -e .
```

#### R
We require you to have R 4.2.2 installed - we recommend to use [Rig](https://github.com/r-lib/rig) to manage different R versions.

Supposing you already have R 4.2.2, you may simply run the below in a terminal.

```sh
Rscript -e "install.packages('renv');require(renv);renv::activate();renv::restore()"
```

#### Running experiments
Once both the necessary R and Python packages are installed, you may reproduce all of our work (including data downloads, preprocessing, etc) by running the below in a terminal (make sure to activate the respective conda environment if it is not already active). 

```sh
bash reproduce.sh
```

### Results
All of our results, including preprocessed data, computed performance metrics and predicted survival functions for all models and experiments are available on [Zenodo](TODO UPDATE ZENODO LINK).

## Questions
In case of any questions, please reach out to david.wissel@inf.ethz.ch or open an issue in this repo.

## Citation
Our manuscript is still under review.

## References
[1] Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).
[2] Paul, Debashis, et al. "" Preconditioning" for Feature Selection and Regression in High-Dimensional Problems." The Annals of Statistics (2008): 1595-1618.
