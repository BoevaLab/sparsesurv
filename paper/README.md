# *sparsesurv*: Sparse survival models via knowledge distillation
## Abstract
Sparse survival models are statistical models that select a subset of predictor variables while modeling the time until an event occurs, which can subsequently help interpretability and transportability. The subset of important features is typically obtained with regularized models, such as the Cox Proportional Hazards model with Lasso regularization, which limit the number of non-zero coefficients. However, such models can be sensitive to the choice of regularization hyperparameter. In this work, we demonstrate how knowledge distillation, a powerful technique in machine learning that aims to transfer knowledge from a complex teacher model to a simpler student model, can be leveraged to learn sparse models while mitigating the challenge above. We present sparsesurv, a Python package that contains a set of teacher-student model pairs, including the semi-parametric accelerated failure time and the extended hazards models as teachers, which currently do not have Python implementations. It also contains in-house survival function estimators, removing the need for external packages. Sparsesurv is validated against R-based Elastic Net regularized linear Cox proportional hazards models as implemented in the commonly used *glmnet* package. Our results reveal that knowledge distillation-based approaches achieve better discriminative performance across the regularization path while making the choice of the regularization hyperparameter significantly easier. All of these features, combined with an *sklearn*-like API, make sparsesurv an easy-to-use Python package that enables survival analysis for high-dimensional datasets and allows fitting sparse survival models via knowledge distillation.

## Reproducibility
### From scratch
Since installing via conda and R in bash scripts can be finicky and anyway requires user input, we guide you through the process below, after which you may reproduce all of our results by executing our reproduction script.

#### Python
Please run the following in a terminal and give user input as appropriate (e.g., confirming that you want to create a new conda env).

```sh
conda create -n sparsesurv_paper python==3.10.0
conda activate sparsesurv_paper
pip install -r requirements.txt
pip install -e ..
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
All of our results, including preprocessed data, computed performance metrics and predicted survival functions for all models and experiments are available on [Zenodo](https://zenodo.org/record/8280015).

## Questions
In case of any questions, please reach out to david.wissel@inf.ethz.ch or open an issue in this repo.

## Citation
Our manuscript is still under review.

## References
[1] Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).
[2] Paul, Debashis, et al. "" Preconditioning" for Feature Selection and Regression in High-Dimensional Problems." The Annals of Statistics (2008): 1595-1618.
