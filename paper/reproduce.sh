#!/bin/bash

bash scripts/sh/create_paths.sh &> create_paths.log
bash scripts/sh/download_data.sh &> download_data.log
bash scripts/sh/preprocess_data.sh &> preprocess_data.log
bash scripts/sh/make_splits.sh &> make_splits.log
bash scripts/sh/rerun_experiments.sh &> reproduce_experiments.log
