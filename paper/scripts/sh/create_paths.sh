#!/bin/bash

for model in "efron" "breslow" "aft" "eh"; do
    for cancer in "BLCA" "BRCA" "HNSC" "KIRC" "LGG" \
        "LIHC" "LUAD" "LUSC" "OV" "STAD" \
        "PAAD" "SKCM"; do
        mkdir -p ./results/pc/$model/$cancer

    done
done

for model in "breslow"; do
    for cancer in "BLCA" "BRCA" "HNSC" "KIRC" "LGG" \
        "LIHC" "LUAD" "LUSC" "OV" "STAD" \
        "PAAD" "SKCM"; do
        mkdir -p ./results/non_pc/$model/$cancer
    done
done

mkdir -p ./results/metrics/
