#!/bin/bash

Rscript scripts/r/plot_figure_1.R

Rscript scripts/r/plot_figure_S1.R
Rscript scripts/r/plot_figure_S2.R
Rscript scripts/r/plot_figure_S3.R

Rscript scripts/r/make_table_S2.R
Rscript scripts/r/make_table_S3.R
Rscript scripts/r/make_table_S4.R

python scripts/py/make_table_S1.py
