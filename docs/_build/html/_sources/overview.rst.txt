Overview
========
*sparsesurv* [Wissel2023]_ is a toolbox for high-dimensional survival analysis. Currently, the package is focused exclusively on knowledge distillation for sparse survival analysis, sometimes also called preconditioning or reference models ([Paul2008]_, [Pavone2023]_). In the future, we plan to also extend sparsesurv to other techniques useful for (high-dimensional) survival analysis that are not commonly available in Python.

Below, we will briefly walk you introduce you to some concepts you should understand before using sparsesurv. Once you're ready to move on, you can check out the basic usage of sparsesurv under :ref:`basic-usage-label`.

Datasets
________

We leverage RNA-seq data generated for the PANCANATLAS project [Weinstein2013]_, which can be downloaded from the
official PANCANATLAS website. We use “overall survival” as the endpoint of all datasets, as recommended by [Liu2018]_,
also available on the official PANCANATLAS website.


Methods
____________
.. toctree::
    :maxdepth: 2
    
    notebooks/01_intro
    