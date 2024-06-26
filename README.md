# sparsesurv: A Python package for fitting sparse survival models via knowledge distillation

## Abstract

Sparse survival models are statistical models that select a subset of predictor variables while modeling the time until an event occurs, which can subsequently help interpretability and transportability. The subset of important features is often obtained with regularized models, such as the Cox Proportional Hazards model with Lasso regularization, which limit the number of non-zero coefficients. However, such models can be sensitive to the choice of regularization hyperparameter. In this work, we develop a software package and demonstrate how knowledge distillation, a powerful technique in machine learning that aims to transfer knowledge from a complex teacher model to a simpler student model, can be leveraged to learn sparse survival models while mitigating this challenge. For this purpose, we present sparsesurv, a Python package that contains a set of teacher-student model pairs, including the semi-parametric accelerated failure time and the extended hazards models as teachers, which currently do not have Python implementations. It also contains in-house survival function estimators, removing the need for external packages. Sparsesurv is validated against R-based Elastic Net regularized linear Cox proportional hazards models as implemented in the commonly used glmnet package. Our results reveal that knowledge distillation-based approaches achieve competitive discriminative performance relative to glmnet across the regularization path while making the choice of the regularization hyperparameter significantly easier. All of these features, combined with an sklearn-like API, make sparsesurv an easy-to-use Python package that enables survival analysis for high-dimensional datasets through fitting sparse survival models via knowledge distillation.

## Reproducibility

### From scratch

```
snakemake --use-conda --conda-frontend mamba --cores 12
```

### Results

All of our results, including preprocessed data, computed performance metrics and predicted survival functions for all models and experiments are available on [Zenodo](https://zenodo.org/doi/10.5281/zenodo.8280014).

## Questions

In case of any questions, please reach out to david.wissel@inf.ethz.ch or open an issue in this repo.

## Citation

Our manuscript is still under review.

## References

[1] Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).
[2] Paul, Debashis, et al. "" Preconditioning" for Feature Selection and Regression in High-Dimensional Problems." The Annals of Statistics (2008): 1595-1618.
