#!/bin/bash

for model in "breslow" "cox_nnet"; do
    for cancer in "BLCA" "BRCA" "HNSC" "KIRC" "LGG" \
        "LIHC" "LUAD" "LUSC" "OV" "STAD"; do
        mkdir -p ./results/kd/$model/$cancer
        mkdir -p ./results/kd/$model/$cancer/path

    done
done

for model in "breslow"; do
    for cancer in "BLCA" "BRCA" "HNSC" "KIRC" "LGG" \
        "LIHC" "LUAD" "LUSC" "OV" "STAD"; do
        mkdir -p ./results/non_kd/$model/$cancer
        mkdir -p ./results/non_kd/$model/$cancer/path
    done
done

mkdir -p ./results/metrics/
